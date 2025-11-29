# ======================================================
# Hand Bridge Flask Server (5초 녹화 + GRU 모델 호환)
# ======================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, os, torch, traceback, cv2, re
from PIL import Image
import numpy as np
import builtins
from collections import deque

# ------------------------------------------------------
# 1. Vocab & Tokenizer (Pickle 로딩 필수)
# ------------------------------------------------------
class Vocabulary:
    def __init__(self, tokenizer=None, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {}; self.stoi = {}
        self.min_freq = min_freq; self.pad_idx = 0; self.sos_idx = 1; self.eos_idx = 2
    def __len__(self): return len(self.itos)

def simple_tokenizer(text): return text.split(' ')

builtins.Vocabulary = Vocabulary
builtins.simple_tokenizer = simple_tokenizer

# ------------------------------------------------------
# 2. 모델 Import (수정된 inference.py 사용)
# ------------------------------------------------------
from inference import (
    vocab, encoder_session, decoder_session,          
    gec_model, gec_tokenizer, stt_model, emo,         
    onnx_predict                                      
)

# ------------------------------------------------------
# 3. MediaPipe 설정
# ------------------------------------------------------
import mediapipe as mp
mp_holistic = mp.solutions.holistic
try:
    holistic_processor = mp_holistic.Holistic(
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    print("✅ [app] MediaPipe 로드 완료")
except:
    holistic_processor = None

# ------------------------------------------------------
# 4. 서버 설정
# ------------------------------------------------------
app = Flask(__name__)
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

# 🔥 [핵심 설정] 5초 녹화 (30FPS * 5초 = 150프레임)
TARGET_FRAMES = 150  
MODEL_INPUT_LEN = 30 # 모델 입력 길이
server_buffer = []   # 프레임 저장소

# ------------------------------------------------------
# 5. 유틸리티 (전처리 & 압축)
# ------------------------------------------------------
def _extract_kps(frame_bgr, holistic):
    if not holistic: return np.zeros(150, dtype=np.float32)
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = holistic.process(img)
    
    pose = np.zeros(66, dtype=np.float32)
    if res.pose_landmarks:
        for i, lm in enumerate(res.pose_landmarks.landmark): pose[i*2], pose[i*2+1] = lm.x, lm.y
    lh = np.zeros(42, dtype=np.float32)
    if res.left_hand_landmarks:
        for i, lm in enumerate(res.left_hand_landmarks.landmark): lh[i*2], lh[i*2+1] = lm.x, lm.y
    rh = np.zeros(42, dtype=np.float32)
    if res.right_hand_landmarks:
        for i, lm in enumerate(res.right_hand_landmarks.landmark): rh[i*2], rh[i*2+1] = lm.x, lm.y
        
    kps = np.concatenate([pose, lh, rh])
    # 감지 실패 시 0으로 채움 (녹화 시간 유지를 위해)
    if np.sum(np.abs(kps)) < 0.01: return np.zeros(150, dtype=np.float32)
    return kps

def _resample(buffer, target_len=30):
    """150개 프레임을 30개로 균일 압축"""
    arr = np.array(buffer, dtype=np.float32)
    if len(arr) == 0: return np.zeros((target_len, 150), dtype=np.float32)
    indices = np.linspace(0, len(arr)-1, target_len, dtype=int)
    return arr[indices]

def _prepare(arr):
    """Motion Feature 추가 (30,150) -> (1,30,300)"""
    mot = np.zeros_like(arr)
    if len(arr) > 1: mot[1:] = arr[1:] - arr[:-1]
    return np.expand_dims(np.concatenate([arr, mot], axis=1), axis=0)

# ------------------------------------------------------
# 6. API 라우트
# ------------------------------------------------------
@app.route('/')
def serve_index(): return send_from_directory(FRONTEND_DIR, 'index.html')
@app.route('/demo.html')
def serve_demo(): return send_from_directory(FRONTEND_DIR, 'demo.html')
@app.route('/assets/<path:filename>')
def serve_assets(filename): return send_from_directory(os.path.join(FRONTEND_DIR, 'assets'), filename)

@app.route("/api/reset", methods=["POST"])
def reset():
    global server_buffer
    server_buffer = []
    print("🔄 버퍼 리셋됨")
    return jsonify({"msg": "reset_ok"})

@app.route("/api/sign_infer", methods=["POST"])
def sign_infer():
    global server_buffer
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "frame" not in data: return jsonify({"error": "No frame"}), 400

        # 이미지 디코딩
        f_b64 = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
        img = Image.open(io.BytesIO(base64.b64decode(f_b64))).convert("RGB")
        kps = _extract_kps(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), holistic_processor)
        
        server_buffer.append(kps)
        curr = len(server_buffer)
        progress = int((curr / TARGET_FRAMES) * 100)

        # 1. 녹화 중 (5초 미만)
        if curr < TARGET_FRAMES:
            msg = f"📸 녹화 중... ({progress}%)"
            if progress > 80: msg = "🟠 분석 준비 중..."
            return jsonify({"status": "recording", "progress": progress, "message": msg})

        # ========================================================
        # 2. 5초 녹화 완료 -> 분석 시작
        # ========================================================
        print("✅ 5초 데이터 확보! 분석 시작...")
        
        # (1) 데이터 압축 (150 -> 30) 및 전처리
        resampled = _resample(server_buffer)
        inp = _prepare(resampled) # (1, 30, 300)
        
        raw_text = "..."
        
        # (2) 추론 (inference.py의 onnx_predict 호출)
        if encoder_session and vocab:
            sos_id = vocab.stoi.get("<SOS>", 1)
            eos_id = vocab.stoi.get("<EOS>", 2)
            
            # GRU 호환된 함수 호출 (Cell State 필요 없음)
            pred_idx = onnx_predict(
                encoder_session, decoder_session, inp, 
                max_output_len=50, sos_idx=sos_id, eos_idx=eos_id
            )
            
            tokens = [vocab.itos.get(i, "") for i in pred_idx]
            # 특수 토큰 제거하고 원본 라벨 그대로 출력
            raw_text = " ".join([t for t in tokens if t not in ["<SOS>", "<PAD>", "<EOS>"]]).strip()

        # (3) 문맥 교정 (선택 사항)
        corrected_text = raw_text
        if gec_model and raw_text:
            try:
                inp_g = gec_tokenizer(raw_text, return_tensors="pt").to(DEVICE)
                out_g = gec_model.generate(**inp_g, max_length=50)
                corrected_text = gec_tokenizer.decode(out_g[0], skip_special_tokens=True)
            except: pass

        server_buffer = [] # 버퍼 초기화
        
        print(f"👉 예측 결과: {raw_text}")
        
        return jsonify({
            "status": "done", 
            "progress": 100, 
            "translation": raw_text,       # 학습 라벨 그대로
            "corrected": corrected_text    # 문법 교정본
        })

    except Exception as e:
        traceback.print_exc()
        server_buffer = []
        return jsonify({"error": str(e)}), 500

# 음성 API
@app.route("/api/voice_infer", methods=["POST"])
def voice_infer():
    try:
        file = request.files.get("audio")
        if not file: return jsonify({"error": "No audio"}), 400
        save_path = os.path.join(BASE_DIR, "temp_audio.wav")
        file.save(save_path)
        
        rec_text = ""
        if stt_model:
            res = stt_model.transcribe(save_path, language="ko")
            rec_text = res.get("text", "").strip()
            
        emotion = "neutral"
        if emo: 
            emotion, conf, _ = emo.infer_from_file(save_path)
            
        if os.path.exists(save_path): os.remove(save_path)
        return jsonify({"recognized_text": rec_text, "emotion": emotion})
    except Exception as e: return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"\n🚀 서버 실행 중 (5초 녹화 + GRU 모델)")
    app.run(host="0.0.0.0", port=8000)