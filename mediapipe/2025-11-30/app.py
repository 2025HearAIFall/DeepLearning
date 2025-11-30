# ======================================================
# Hand Bridge Flask Server (NameError FIX)
# ======================================================

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import base64, io, os, torch, traceback, cv2, re
from PIL import Image
import numpy as np
import builtins
from collections import deque
import mediapipe as mp # <-- 🔥 FIX: mediapipe import를 최상단으로 이동!

# ------------------------------------------------------
# 1. Path & Vocab Setup
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', 'frontend'))

class Vocabulary:
    def __init__(self, tokenizer=None, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {}; self.stoi = {}
        self.min_freq = min_freq; self.pad_idx = 0; self.sos_idx = 1; self.eos_idx = 2
    def __len__(self): return len(self.itos)

def simple_tokenizer(text): return text.split(' ')

builtins.Vocabulary = Vocabulary
builtins.simple_tokenizer = simple_tokenizer

# ------------------------------------------------------
# 2. Load Models
# ------------------------------------------------------
from inference import (
    vocab, encoder_session, decoder_session,          
    gec_model, gec_tokenizer, stt_model, emo,         
    onnx_predict                                      
)

# ------------------------------------------------------
# 3. MediaPipe Setup (NameError FIX 적용)
# ------------------------------------------------------
mp_holistic = mp.solutions.holistic # mp가 이미 import되었으므로 사용 가능

try:
    holistic_processor = mp_holistic.Holistic(
        static_image_mode=True, 
        model_complexity=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    print("✅ [app] MediaPipe loaded (Static Mode)")
except Exception as e:
    holistic_processor = None
    print(f"❌ [app] MediaPipe failed: {e}")

# ------------------------------------------------------
# 4. Server Config
# ------------------------------------------------------
app = Flask(__name__)
CORS(app)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TARGET_FRAMES = 50  
MODEL_INPUT_LEN = 30 
server_buffer = []   

# ------------------------------------------------------
# 5. Utilities (기존 유지)
# ------------------------------------------------------
def _extract_kps(frame_bgr, holistic):
    if not holistic: return np.zeros(150, dtype=np.float32)
    
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_RGB2BGR)
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
    if np.sum(np.abs(kps)) < 0.01: return np.zeros(150, dtype=np.float32)
    return kps

def _resample(buffer, target_len=30):
    arr = np.array(buffer, dtype=np.float32)
    if len(arr) == 0: return np.zeros((target_len, 150), dtype=np.float32)
    indices = np.linspace(0, len(arr)-1, target_len, dtype=int)
    return arr[indices]

def _prepare(arr):
    mot = np.zeros_like(arr)
    if len(arr) > 1: mot[1:] = arr[1:] - arr[:-1]
    return np.expand_dims(np.concatenate([arr, mot], axis=1), axis=0)

# ------------------------------------------------------
# 6. Routes
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
    print("🔄 Buffer reset")
    return jsonify({"msg": "reset_ok"})

@app.route("/api/sign_infer", methods=["POST"])
def sign_infer():
    global server_buffer
    try:
        data = request.get_json(force=True, silent=True)
        if not data or "frame" not in data: return jsonify({"error": "No frame"}), 400

        f_b64 = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
        img = Image.open(io.BytesIO(base64.b64decode(f_b64))).convert("RGB")
        
        kps = _extract_kps(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR), holistic_processor)
        
        server_buffer.append(kps)
        curr = len(server_buffer)
        progress = int((curr / TARGET_FRAMES) * 100)

        if curr < TARGET_FRAMES:
            msg = f"📸 Recording... ({progress}%)"
            return jsonify({"status": "recording", "progress": progress, "message": msg})

        print("✅ 5s Data Collected! Analyzing...")
        
        resampled = _resample(server_buffer)
        inp = _prepare(resampled)
        
        raw_text = "..."
        if encoder_session and vocab:
            sos_id = vocab.stoi.get("<SOS>", 1)
            eos_id = vocab.stoi.get("<EOS>", 2)
            
            pred_idx = onnx_predict(encoder_session, decoder_session, inp, 50, sos_id, eos_id)
            tokens = [vocab.itos.get(i, "") for i in pred_idx]
            raw_text = " ".join([t for t in tokens if t not in ["<SOS>", "<PAD>", "<EOS>"]]).strip()

        corrected_text = raw_text
        if gec_model and raw_text:
            try:
                inp_g = gec_tokenizer(raw_text, return_tensors="pt").to(DEVICE)
                out_g = gec_model.generate(**inp_g, max_length=50)
                corrected_text = gec_tokenizer.decode(out_g[0], skip_special_tokens=True)
            except: pass

        server_buffer = [] 
        print(f"👉 Result: {raw_text}")
        
        return jsonify({
            "status": "done", 
            "progress": 100, 
            "translation": raw_text,
            "corrected": corrected_text
        })

    except Exception as e:
        traceback.print_exc()
        server_buffer = []
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice_infer", methods=["POST"])
def voice_infer():
    try:
        file = request.files.get("audio")
        if not file: return jsonify({"error": "No audio"}), 400
        
        # -----------------------------------------------------
        # 수정됨: 브라우저에서 보낸 파일명(확장자 .webm 예상)을 사용하여 저장
        # -----------------------------------------------------
        filename = file.filename if file.filename else "temp_audio.webm"
        save_path = os.path.join(BASE_DIR, filename)
        
        file.save(save_path)
        print(f"🎤 Audio saved to: {save_path}")
        
        rec_text = ""
        if stt_model:
            # Whisper 모델이 .webm 파일을 처리할 수 있도록 가정
            res = stt_model.transcribe(save_path, language="ko")
            rec_text = res.get("text", "").strip()
            
        emotion = "neutral"
        if emo: 
            try:
                emotion, conf, _ = emo.infer_from_file(save_path)
            except Exception as e:
                print(f"⚠️ Emotion inference failed: {e}")
                emotion = "neutral"
            
        if os.path.exists(save_path): os.remove(save_path)
        
        return jsonify({"recognized_text": rec_text, "emotion": emotion})
    except Exception as e: 
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print(f"\n🚀 Server running (Final Fixes Applied)")
    app.run(host="0.0.0.0", port=8000)