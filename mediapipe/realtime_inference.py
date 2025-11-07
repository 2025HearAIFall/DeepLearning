# realtime_inference.py (Pickle 오류 수정)
# -----------------------------------------------------------------------------
# [목표]: 웹캠(1개 뷰)을 실시간으로 입력받아 V2 ONNX 모델로 수어 문장을 번역합니다.
# [수정]: Pickle이 '__main__'에서 Vocabulary 클래스를 찾지 못하는 오류 수정
# -----------------------------------------------------------------------------

import cv2
import mediapipe as mp
import onnxruntime as ort
import numpy as np
import pickle
from collections import deque
import sys # [신규] Pickle 오류 수정을 위해 임포트

# --- [1. 설정값] ---
MAX_LEN = 30
BASE_FEATURES_MEDIAPIPE = (33 + 468 + 21 + 21) * 2 # 1086
INPUT_SIZE = BASE_FEATURES_MEDIAPIPE * 2           # 2172

ONNX_ENCODER_PATH = 'model_a_v2_encoder.onnx'
ONNX_DECODER_PATH = 'model_a_v2_decoder.onnx'
VOCAB_PATH = 'vocab.pkl'

START_TOKEN_IDX = 1 # (SOS)
END_TOKEN_IDX = 2   # (EOS)
MAX_OUTPUT_LEN = 50 

# --- [신규] Pickle 오류 수정을 위해 클래스 정의 추가 ---
# (train_mediapipe.py에서 복사)
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

# [신규] Pickle 'AttributeError' 해결
# 'vocab.pkl'이 '__main__.Vocabulary'를 찾으므로, 수동으로 경로를 지정해줍니다.
if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
# ----------------------------------------------------


# --- [2. ONNX 모델 및 Vocab 로드] ---
print("ONNX 모델 및 Vocab 로드 중...")
try:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary 로드 완료 (크기: {len(vocab)})")

    encoder_session = ort.InferenceSession(ONNX_ENCODER_PATH, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(ONNX_DECODER_PATH, providers=['CPUExecutionProvider'])
    print("ONNX 모델 로드 완료.")
    
except Exception as e:
    print(f"[오류] 모델 또는 Vocab 로드 실패: {e}")
    print(">>> 1단계 (V2 학습)와 2단계 (V2 ONNX 변환)를 먼저 실행해야 합니다.")
    exit()

# --- [3. MediaPipe 모듈 로드] ---
print("MediaPipe Holistic 모델 로드 중...")
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)

# --- [4. 유틸리티 함수] ---

def _extract_keypoints_from_frame(frame, holistic) -> np.ndarray:
    """단일 프레임에서 1086개의 2D 키포인트를 추출합니다."""
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    
    pose = np.zeros(33 * 2, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*2] = lm.x; pose[i*2 + 1] = lm.y
    face = np.zeros(468 * 2, dtype=np.float32)
    if results.face_landmarks:
        for i, lm in enumerate(results.face_landmarks.landmark):
            face[i*2] = lm.x; face[i*2 + 1] = lm.y
    lh = np.zeros(21 * 2, dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*2] = lm.x; lh[i*2 + 1] = lm.y
    rh = np.zeros(21 * 2, dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*2] = lm.x; rh[i*2 + 1] = lm.y
    return np.concatenate([pose, face, lh, rh]) # (1086,)

def onnx_predict_realtime(encoder_sess, decoder_sess, src_seq_np):
    """V2(단일 뷰) ONNX 모델로 추론합니다."""
    # src_seq_np: (1, 30, 2172)
    
    try:
        # Encoder 실행
        encoder_inputs = {'input_keypoints': src_seq_np}
        encoder_outputs, hidden, cell = encoder_sess.run(None, encoder_inputs)
        
        # Decoder 초기 입력 (SOS 토큰)
        trg_input = np.array([[START_TOKEN_IDX]], dtype=np.int64)
        
        output_tokens = []
        for _ in range(MAX_OUTPUT_LEN):
            decoder_inputs = {
                'input_token': trg_input, 
                'in_hidden': hidden, 
                'in_cell': cell, 
                'encoder_outputs': encoder_outputs
            }
            logits, hidden, cell = decoder_sess.run(None, decoder_inputs)
            
            top1_idx_array = np.argmax(logits, axis=1)
            top1_item = top1_idx_array[0]
            
            if top1_item == END_TOKEN_IDX:
                break
                
            output_tokens.append(top1_item)
            trg_input = np.array([[top1_item]], dtype=np.int64)
            
        return output_tokens
    except Exception as e:
        print(f"[오류] ONNX predict 중 오류: {e}")
        return []

# --- [5. 실시간 추론 루프] ---
print("웹캠을 시작합니다... ('q' 키를 누르면 종료됩니다)")
cap = cv2.VideoCapture(0) # 0번 카메라

# MAX_LEN (30) 프레임의 키포인트를 저장할 큐
keypoint_queue = deque(maxlen=MAX_LEN)
predicted_sentence = ""

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("카메라 프레임을 읽을 수 없습니다.")
        break

    # 1. MediaPipe 키포인트 추출
    keypoints_1086 = _extract_keypoints_from_frame(frame, holistic)
    
    # 2. 큐에 현재 키포인트 추가
    keypoint_queue.append(keypoints_1086)

    # 3. 추론 (큐가 30프레임으로 찼을 때만)
    if len(keypoint_queue) == MAX_LEN:
        
        # (30, 1086) 배열 생성
        sequence = np.array(keypoint_queue, dtype=np.float32)
        
        # (30, 2172) 배열로 변환 (모션 벡터 추가)
        positions = sequence
        motions = np.zeros_like(positions)
        if len(positions) > 1:
            motions[1:] = positions[1:] - positions[:-1]
        final_sequence = np.concatenate([positions, motions], axis=1)
        
        # (1, 30, 2172) 배치 차원 추가
        final_sequence_batch = np.expand_dims(final_sequence, axis=0)

        # 4. ONNX 추론 실행
        predicted_indices = onnx_predict_realtime(
            encoder_session, decoder_session, final_sequence_batch
        )
        
        # 5. 결과 디코딩
        raw_tokens = [vocab.itos.get(idx, "<UNK>") for idx in predicted_indices]
        predicted_sentence = " ".join(raw_tokens)
        
        # (추론 효율을 위해 큐의 일부를 비움 - 예: 5 프레임)
        for _ in range(5):
             keypoint_queue.popleft()


    # 6. 화면에 표시
    frame = cv2.flip(frame, 1)
    
    status_text = f"Frames: {len(keypoint_queue)}/{MAX_LEN}"
    color = (0, 255, 0) if len(keypoint_queue) == MAX_LEN else (0, 0, 255)
    cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, predicted_sentence, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow('Real-time Sign Language Translation (V2)', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- [6. 종료] ---
cap.release()
cv2.destroyAllWindows()
holistic.close()
print("실시간 추론을 종료합니다.")
