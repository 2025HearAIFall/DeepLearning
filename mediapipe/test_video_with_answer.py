# test_video_with_json_v3.py (v4 - 슬라이딩 윈도우 수정)
# -----------------------------------------------------------------------------
# [목표]: 'train_from_npy.py' (INPUT_SIZE=300)로 훈련된 모델을 테스트합니다.
#
# [치명적 버그 수정]:
# - 긴 비디오(10초)에서 30프레임만 샘플링(np.linspace)하여
#   '가만히 서 있는' 부분만 분석하던 오류를 수정.
# - 실시간 추론과 동일하게 30프레임 '슬라이딩 윈도우' 방식으로 변경하여
#   모든 구간(3초~6초 동작 포함)을 예측하도록 수정.
# -----------------------------------------------------------------------------

import cv2
import mediapipe as mp
import onnxruntime as ort
import numpy as np
import pickle
from collections import deque
import sys
import os
import json 
from pathlib import Path

# --- [사용자 설정] ---
VIDEO_FILENAME = "NIA_SL_SEN0142_REAL01_F.mp4" 
GROUND_TRUTH_JSON = "NIA_SL_SEN0142_REAL01_F_morpheme.json" 
# ------------------------


# --- [1. 설정값] ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(f"[INFO] 스크립트 기준 경로 (BASE_DIR): {BASE_DIR}")

VIDEO_FILE_PATH = os.path.join(BASE_DIR, VIDEO_FILENAME)
JSON_FILE_PATH = os.path.join(BASE_DIR, GROUND_TRUTH_JSON)

# --- [수정] 하이퍼파라미터 (INPUT_SIZE = 300) ---
MAX_LEN_A = 30
# (Pose:33 + Hands:42) * 2D = 150
BASE_FEATURES_MEDIAPIPE = (33 + 21 + 21) * 2 # 150
# 150 * 2 (모션) = 300
INPUT_SIZE_A = BASE_FEATURES_MEDIAPIPE * 2           # 300
START_TOKEN_IDX = 1 # (SOS)
END_TOKEN_IDX = 2   # (EOS)
MAX_OUTPUT_LEN_A = 50
# -------------------------------------------

# --- [수정] ONNX 모델 경로 ---
# 'model_a_face_excluded_best.pth'를 변환한 파일
ONNX_ENCODER_PATH = os.path.join(BASE_DIR, 'model_a_v2_encoder.onnx')
ONNX_DECODER_PATH = os.path.join(BASE_DIR, 'model_a_v2_decoder.onnx')
VOCAB_PATH = os.path.join(BASE_DIR, 'vocab.pkl')
# -------------------------------------------

# --- Pickle 오류 수정을 위해 클래스 정의 추가 (모델 A) ---
def simple_tokenizer(text):
    return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
    def __len__(self): return len(self.itos)
    def build_vocab(self, sentence_list): pass
    def numericalize(self, text): pass
    @property
    def pad_idx(self): return self.stoi["<PAD>"]
    @property
    def sos_idx(self): return self.stoi["<SOS>"]
    @property
    def eos_idx(self): return self.stoi["<EOS>"]
    @property
    def unk_idx(self): return self.stoi["<UNK>"]

if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
# ----------------------------------------------------

# -----------------------------------------------------------------------------
# 모델 A 로드
# -----------------------------------------------------------------------------
print(f"모델 A (V3 - 얼굴 제외) 및 Vocab 로드 중...")
try:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary 로드 완료 (크기: {len(vocab)})")
    encoder_session = ort.InferenceSession(ONNX_ENCODER_PATH, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(ONNX_DECODER_PATH, providers=['CPUExecutionProvider'])
    print(f"모델 A (V3 - ONNX) 로드 완료. (Encoder: {ONNX_ENCODER_PATH})")
except Exception as e:
    print(f"[오류] 모델 A(V3) 또는 Vocab 로드 실패: {e}"); 
    print(f">>> {ONNX_ENCODER_PATH} 파일이 맞는지 확인하세요.")
    exit()

# -----------------------------------------------------------------------------
# 정답 라벨 로드 (JSON 사용)
# -----------------------------------------------------------------------------
def get_ground_truth_from_json(json_path):
    print(f"정답 JSON 로드 중: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        sentence = ""
        
        if 'sentence' in data:
            sentence = data['sentence']
        elif 'data' in data and isinstance(data['data'], list):
            morphemes = [item['attributes'][0]['name'] for item in data['data'] if 'attributes' in item and item['attributes']]
            sentence = " ".join(morphemes)
        else:
            raise ValueError("JSON에서 'sentence' 키 또는 'data' 리스트 구조를 찾을 수 없습니다.")

        if not sentence:
             raise ValueError("JSON은 찾았으나 문장이 비어있습니다.")

        print(f"✅ 정답 문장 찾음: {sentence}")
        return sentence
        
    except FileNotFoundError:
        print(f"[경고] 정답 JSON 파일을 찾을 수 없습니다: {json_path}")
        return "N/A (JSON 없음)"
    except Exception as e:
        print(f"[경고] 정답 JSON 로드 중 오류: {e}")
        return "N/A (JSON 오류)"

# -----------------------------------------------------------------------------

print("MediaPipe Holistic 모델 로드 중...")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# -----------------------------------------------------------------------------
# [수정] 파이프라인 A 유틸리티 (얼굴 특징 제외)
# -----------------------------------------------------------------------------
def _extract_keypoints_from_frame(frame, holistic) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    pose = np.zeros(33 * 2, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*2] = lm.x; pose[i*2 + 1] = lm.y

    # [수정] 얼굴(Face) 특징은 추출하지 않습니다 (train_from_npy.py와 일치)
    # face = np.zeros(468 * 2, dtype=np.float32) ...

    lh = np.zeros(21 * 2, dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*2] = lm.x; lh[i*2 + 1] = lm.y
    rh = np.zeros(21 * 2, dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*2] = lm.x; rh[i*2 + 1] = lm.y
            
    # [수정] 반환값에서 face를 제외합니다.
    return np.concatenate([pose, lh, rh]) # (150,)

def onnx_predict_realtime(encoder_sess, decoder_sess, src_seq_np):
    try:
        # [확인] src_seq_np는 (1, 30, 300)이어야 함
        encoder_inputs = {'input_keypoints': src_seq_np}
        encoder_outputs, hidden, cell = encoder_sess.run(None, encoder_inputs)
        trg_input = np.array([START_TOKEN_IDX], dtype=np.int64) 
        output_tokens = []
        for _ in range(MAX_OUTPUT_LEN_A):
            decoder_inputs = {
                'input_token': trg_input, 'in_hidden': hidden, 
                'in_cell': cell, 'encoder_outputs': encoder_outputs
            }
            logits, hidden, cell = decoder_sess.run(None, decoder_inputs)
            top1_idx_array = np.argmax(logits, axis=1)
            top1_item = top1_idx_array[0]
            if top1_item == END_TOKEN_IDX: break
            output_tokens.append(top1_item)
            trg_input = np.array([top1_item], dtype=np.int64)
        return output_tokens
    except Exception as e:
        print(f"[오류] ONNX predict 중 오류: {e}"); 
        if 'src_seq_np' in locals():
            print(f"  -> 실제 입력된 텐서 크기: {src_seq_np.shape}")
        input_name = encoder_sess.get_inputs()[0].name
        expected_shape = encoder_sess.get_inputs()[0].shape
        print(f"  -> ONNX 모델이 기대하는 입력 이름: '{input_name}'")
        print(f"  -> ONNX 모델이 기대하는 입력 크기: {expected_shape} (batch, 30, 300 이어야 함)")
        return []

# -----------------------------------------------------------------------------
# [수정] 파이프라인 A (슬라이딩 윈도우로 변경)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# [수정] 파이프라인 A (학습 방식과 동일하게 '전체 샘플링 1회' 예측)
# -----------------------------------------------------------------------------
def run_whole_video_inference_A(GROUND_TRUTH_SENTENCE):
    print("\n" + "="*30)
    print(f"--- 🚀 파이프라인 A (비디오 전체 샘플링 후 1회 예측) 시작 ---")
    print(f"파일: {VIDEO_FILE_PATH}")
    print("="*30)

    if not os.path.exists(VIDEO_FILE_PATH):
        print(f"[오류] 동영상 파일을 찾을 수 없습니다: {VIDEO_FILE_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_FILE_PATH)
    
    # [수정] 큐(deque) 대신, 모든 프레임과 키포인트를 저장할 리스트
    all_display_frames = []
    all_keypoints_list = []

    print("--- 1단계: 동영상 전체 프레임 '수집' 중... ---")
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame_count += 1
        all_display_frames.append(frame.copy()) 
        
        # 1. 키포인트 추출 후 리스트에 저장
        keypoints = _extract_keypoints_from_frame(frame, holistic)
        all_keypoints_list.append(keypoints) # (150,)

    cap.release()
    holistic.close()
    
    if not all_keypoints_list:
        print("[경고] 분석된 프레임이 없습니다.")
        return
        
    print(f"--- 1단계: 수집 완료 (총 {frame_count} 프레임) ---")


    # --- [수정] 2단계: '단 한 번'의 예측을 위한 30프레임 샘플링 ---
    print(f"--- 2단계: {frame_count}개 프레임을 30개로 샘플링하여 '단 1회' 예측... ---")
    
    num_frames = len(all_keypoints_list)
    sampled_features = []

    if num_frames > MAX_LEN_A:
        # [중요] train_mediapipe.py의 샘플링 로직과 동일하게
        # (예: 100프레임 -> 30프레임으로 균등하게 샘플링)
        indices = np.linspace(0, num_frames - 1, MAX_LEN_A, dtype=int)
        sampled_features = [all_keypoints_list[i] for i in indices]
    else:
        # 30프레임보다 짧으면 그냥 사용
        sampled_features = all_keypoints_list
        
    # (Keypoints 150D -> Input 300D)
    sequence = np.array(sampled_features, dtype=np.float32) # (<=30, 150)
    
    # 패딩 (30프레임보다 짧은 비디오 대응)
    if sequence.shape[0] < MAX_LEN_A:
        padding_shape = (MAX_LEN_A - sequence.shape[0], sequence.shape[1])
        padding = np.zeros(padding_shape, dtype=np.float32)
        sequence = np.vstack([sequence, padding]) # (30, 150)

    # 모션 벡터 생성
    positions = sequence # (30, 150)
    motions = np.zeros_like(positions) # (30, 150)
    if len(positions) > 1:
        motions[1:] = positions[1:] - positions[:-1]
    
    final_sequence = np.concatenate([positions, motions], axis=1)  # (30, 300)
    final_sequence_batch = np.expand_dims(final_sequence, axis=0) # (1, 30, 300)

    # [중요] 비디오 전체에 대해 '단 1회' 예측 수행
    predicted_indices = onnx_predict_realtime(
        encoder_session, decoder_session, final_sequence_batch
    )
    raw_tokens = [vocab.itos.get(idx, "<UNK>") for idx in predicted_indices]
    
    # 이것이 최종 1회 예측 결과입니다.
    final_prediction_one_shot = " ".join(raw_tokens)
    
    print(f"--- [최종 예측 결과 (1-Shot)]: {final_prediction_one_shot} ---")


    # --- [수정] 3단계: 재생 (정답 vs 최종 예측 1개) ---
    print("--- 3단계: 분석 결과 재생 시작... ('q' 키로 종료) ---")
    
    color_correct = (0, 255, 0) # 초록색 (정답)
    color_model = (0, 255, 255) # 노란색 (모델)
    font_size = 1.0
    font_thickness = 2
    
    # 0번 프레임부터 재생
    for i, frame in enumerate(all_display_frames):
        
        # [수정] 매번 바뀌는 예측값이 아닌, 위에서 확정된 '최종 예측' 변수를 사용합니다.
        current_sentence = final_prediction_one_shot
        
        # 1. 정답 표시
        cv2.putText(frame, 
                    f"Correct: {GROUND_TRUTH_SENTENCE}", 
                    (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, 
                    color_correct, 
                    font_thickness)
        
        # 2. 모델 예측 표시 (고정된 최종 결과)
        cv2.putText(frame, 
                    f"Model: {current_sentence}", 
                    (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    font_size, 
                    color_model, 
                    font_thickness)
        
        cv2.imshow('Video File Translation (Ground Truth vs Model)', frame)

        if cv2.waitKey(33) & 0xFF == ord('q'): # 30fps 속도
            break

    cv2.destroyAllWindows()
    print("--- 3단계: 재생 완료 ---")

# =============================================================================
# 메인 실행 블록
# =============================================================================
if __name__ == "__main__":
    
    print("\n" + "="*50)
    print("✅ 모든 모델 로드 완료.")
    print("="*50)
    
    video_path_abs = os.path.abspath(VIDEO_FILE_PATH)
    json_path_abs = os.path.abspath(JSON_FILE_PATH)
    
    GROUND_TRUTH_SENTENCE = get_ground_truth_from_json(json_path_abs)
    
    if os.path.exists(video_path_abs):
        run_whole_video_inference_A(GROUND_TRUTH_SENTENCE)
    else:
        print(f"[오류] 메인 실행: 비디오 파일을 찾을 수 없습니다: {video_path_abs}")

    print("\n모든 파이프라인이 종료되었습니다.")
