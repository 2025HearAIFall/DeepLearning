# realtime_v2_integrated.py
# -----------------------------------------------------------------------------
# [통합 스크립트 - 실시간 전용]
#
# [기준]: 'realtime_inference.py' (v2 모델 로드)
# [통합]: 'inference_mediapipe.py'의 모델 B (문맥 복원) 기능
# [변경]: .csv 파일이 필요한 파일 테스트 모드 (1번)를 완전히 제거하고
#         실행 즉시 실시간 웹캠 추론(모델 A+B)만 시작하도록 수정.
# -----------------------------------------------------------------------------

import cv2
import mediapipe as mp
import onnxruntime as ort
import numpy as np
import pickle
from collections import deque
import sys 

# --- [통합] 모델 B (문맥 복원)을 위한 임포트 ---
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch 

# --- [1. 설정값] ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# (모델 A: 수어) --- [MediaPipe 기준] ---
MAX_LEN = 30
BASE_FEATURES_MEDIAPIPE = (33 + 468 + 21 + 21) * 2 # 1086
INPUT_SIZE = BASE_FEATURES_MEDIAPIPE * 2           # 2172

# --- [수정] v2 모델 경로로 변경 ---
ONNX_ENCODER_PATH = 'model_a_v2_encoder.onnx'
ONNX_DECODER_PATH = 'model_a_v2_decoder.onnx'
VOCAB_PATH = 'vocab.pkl'

# (모델 B: 문맥 복원)
MODEL_B_PATH_OR_NAME = "."

START_TOKEN_IDX = 1 # (SOS)
END_TOKEN_IDX = 2   # (EOS)
MAX_OUTPUT_LEN = 50 

# --- Pickle 오류 수정을 위해 클래스 정의 추가 ---
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

# [중요] Pickle 'AttributeError' 해결
if '__main__' in sys.modules:
    setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
# ----------------------------------------------------


# --- [2. ONNX 모델 및 Vocab 로드] ---
print("모델 A (V2-ONNX) 및 Vocab 로드 중...")
try:
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    print(f"Vocabulary 로드 완료 (크기: {len(vocab)})")
    
    # [수정] V2 경로 변수 사용
    encoder_session = ort.InferenceSession(ONNX_ENCODER_PATH, providers=['CPUExecutionProvider'])
    decoder_session = ort.InferenceSession(ONNX_DECODER_PATH, providers=['CPUExecutionProvider'])
    print("모델 A (V2-ONNX) 로드 완료.")
    
except Exception as e:
    print(f"[오류] 모델 A 또는 Vocab 로드 실패: {e}")
    print(f">>> {ONNX_ENCODER_PATH}, {ONNX_DECODER_PATH}, {VOCAB_PATH} 파일이 있는지 확인하세요.")
    encoder_session, decoder_session, vocab = None, None, None
    exit() # 모델 A가 없으면 실행 불가

# --- [통합] 모델 B (문맥 복원) 로드 ---
print("모델 B (문맥 복원) 로딩 중...")
try:
    gec_tokenizer = AutoTokenizer.from_pretrained(MODEL_B_PATH_OR_NAME)
    gec_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_B_PATH_OR_NAME).to(DEVICE)
    print("모델 B 로드 완료.")
except Exception as e:
    print(f"[오류] 모델 B 로드 실패: {e}")
    print(">>> 모델 B (Hugging Face) 파일들이 '.' 경로에 있는지 확인하세요.")
    gec_model = None; gec_tokenizer = None
    exit() # 모델 B가 없으면 통합 의미가 없으므로 종료 (또는 이 exit()를 제거하고 원본만 보도록 해도 됨)

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

# --- [5. 실시간 추론 루프 (모델 B 통합)] ---
def run_realtime_inference():
    print("\n" + "="*30)
    print("--- 🚀 실시간 추론 모드 시작 ---")
    print("="*30)

    print("웹캠을 시작합니다... ('q' 키를 누르면 종료됩니다)")
    cap = cv2.VideoCapture(0) # 0번 카메라

    # MAX_LEN (30) 프레임의 키포인트를 저장할 큐
    keypoint_queue = deque(maxlen=MAX_LEN)
    predicted_sentence = ""
    corrected_sentence = "" # [통합] 교정된 문장

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

            # 4. ONNX 추론 실행 (모델 A)
            predicted_indices = onnx_predict_realtime(
                encoder_session, decoder_session, final_sequence_batch
            )
            
            # 5. 결과 디코딩 (모델 A)
            raw_tokens = [vocab.itos.get(idx, "<UNK>") for idx in predicted_indices]
            predicted_sentence = " ".join(raw_tokens)
            
            # [통합] 5-1. 문맥 복원 (모델 B)
            if gec_model and gec_tokenizer and len(raw_tokens) >= 3:
                try:
                    inputs = gec_tokenizer(predicted_sentence, return_tensors="pt").to(DEVICE)
                    outputs = gec_model.generate(
                        **inputs, max_length=128, num_beams=5,
                        repetition_penalty=1.2, no_repeat_ngram_size=2,
                        early_stopping=True
                    )
                    corrected_sentence = gec_tokenizer.decode(outputs[0], skip_special_tokens=True)
                except Exception as e:
                    print(f"[오류] 모델 B 실시간 추론 실패: {e}")
                    corrected_sentence = predicted_sentence # 오류 시 원본 사용
            else:
                corrected_sentence = predicted_sentence # 모델이 없거나 단어가 짧으면 원본 사용
            
            # (추론 효율을 위해 큐의 일부를 비움 - 예: 5 프레임)
            for _ in range(5):
                keypoint_queue.popleft()

        # 6. 화면에 표시
        frame = cv2.flip(frame, 1)
        
        status_text = f"Frames: {len(keypoint_queue)}/{MAX_LEN}"
        color = (0, 255, 0) if len(keypoint_queue) == MAX_LEN else (0, 0, 255)
        cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # [수정] 원본 번역과 교정된 번역을 모두 표시
        cv2.putText(frame, f"Raw: {predicted_sentence}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"Fix: {corrected_sentence}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow('Real-time Sign Language Translation (Model A + B)', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # --- [6. 종료] ---
    cap.release()
    cv2.destroyAllWindows()
    holistic.close()
    print("실시간 추론을 종료합니다.")

# =============================================================================
# [통합] 메인 실행 블록
# =============================================================================
if __name__ == "__main__":
    
    # 모델 A와 B가 모두 로드되었는지 최종 확인
    if encoder_session and decoder_session and vocab and gec_model and gec_tokenizer:
        print("\n" + "="*50)
        print("✅ 모델 A (V2-ONNX) 및 모델 B (문맥 복원) 로드 완료.")
        print("="*50)
        
        # [수정] 메뉴 없이 실시간 추론 바로 시작
        run_realtime_inference()
        
    else:
        print("\n[치명적 오류] 모델 A 또는 모델 B 로드에 실패하여 실시간 추론을 시작할 수 없습니다.")
        print("스크립트 상단의 오류 메시지를 확인하고 필요한 파일이 있는지 확인하세요.")
        print("스크립트를 종료합니다.")