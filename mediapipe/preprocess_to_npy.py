# preprocess_to_npy.py (v3.1, 로그 제거 및 최적화)
# -----------------------------------------------------------------------------
# [수정 사항]
# 1. 시끄러운 TensorFlow/MediaPipe 경고 로그를 숨겼습니다.
# 2. 작업 진행 중 에러가 나도 멈추지 않고 다음 파일로 넘어가도록 예외처리를 강화했습니다.
# -----------------------------------------------------------------------------

import os
# [중요] TensorFlow/MediaPipe 로그 숨기기 설정 (import 전에 해야 함)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 

import sys
import logging

# absl 로그 숨기기 (MediaPipe 내부 로깅)
logging.getLogger('absl').setLevel(logging.ERROR)

import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    mp_holistic = mp.solutions.holistic
except ImportError:
    print("[오류] mediapipe/opencv 설치 필요")
    exit()

# --- [설정] ---
MAX_LEN = 30
BASE_FEATURES_MEDIAPIPE = (33 + 21 + 21) * 2 # 150 (Face 제외)
INPUT_SIZE = BASE_FEATURES_MEDIAPIPE * 2     # 300

# [작업 목록 정의]
TASKS = [
    ("train_dataset.csv", "npy_train", "train_npy_index.csv"),
    ("val_dataset.csv",   "npy_val",   "val_npy_index.csv")
]

# -------------------------

def _extract_keypoints_from_frame(frame, holistic) -> np.ndarray:
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = holistic.process(image)
    image.flags.writeable = True
    
    # 1. Raw 데이터 추출 (기존과 동일)
    pose = np.zeros(33 * 2, dtype=np.float32)
    if results.pose_landmarks:
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pose[i*2] = lm.x
            pose[i*2 + 1] = lm.y
            
    lh = np.zeros(21 * 2, dtype=np.float32)
    if results.left_hand_landmarks:
        for i, lm in enumerate(results.left_hand_landmarks.landmark):
            lh[i*2] = lm.x
            lh[i*2 + 1] = lm.y
            
    rh = np.zeros(21 * 2, dtype=np.float32)
    if results.right_hand_landmarks:
        for i, lm in enumerate(results.right_hand_landmarks.landmark):
            rh[i*2] = lm.x
            rh[i*2 + 1] = lm.y
            
    # --- [🔥 핵심 추가: 중심점 기준 상대 좌표 변환] ---
    # MediaPipe Pose 23: Left Hip, 24: Right Hip
    # 인덱스 계산: 23*2=46, 24*2=48
    lx, ly = pose[46], pose[47]
    rx, ry = pose[48], pose[49]
    
    # 골반이 감지되었다면 그 중심을 (0,0)으로 잡음
    if lx > 0 and rx > 0: 
        cx, cy = (lx + rx) / 2, (ly + ry) / 2
    else:
        cx, cy = 0.5, 0.5 # 감지 안 되면 화면 중앙 기준
        
    # 전체 키포인트 통합
    kps = np.concatenate([pose, lh, rh]) # (150,)
    
    # (x, y) 쌍으로 묶어서 일괄 빼기
    kps_reshaped = kps.reshape(-1, 2)
    kps_reshaped[:, 0] -= cx
    kps_reshaped[:, 1] -= cy
    
    # 다시 1차원으로 풀기 (클리핑 범위 조정: -1.0 ~ 1.0)
    kps = kps_reshaped.flatten()
    kps = np.clip(kps, -1.0, 1.0) # 상대 좌표이므로 음수 가능
    
    return kps.astype(np.float32)

def process_video_to_npy(video_path: str, holistic, max_len: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    frames_features = []
    
    if not cap.isOpened():
        return np.zeros((max_len, INPUT_SIZE), dtype=np.float32)
        
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        
        # 프레임이 비어있는 경우 방지
        if frame is None: continue

        keypoints = _extract_keypoints_from_frame(frame, holistic)
        frames_features.append(keypoints)
        
    cap.release()

    if not frames_features:
        return np.zeros((max_len, INPUT_SIZE), dtype=np.float32)

    # 샘플링 (프레임 수 맞추기)
    num_frames = len(frames_features)
    if num_frames > max_len:
        indices = np.linspace(0, num_frames - 1, max_len, dtype=int)
        sampled = [frames_features[i] for i in indices]
    else:
        sampled = frames_features

    seq = np.array(sampled, dtype=np.float32)
    
    # Motion feature (현재 프레임 - 이전 프레임)
    motions = np.zeros_like(seq)
    if len(seq) > 1:
        motions[1:] = seq[1:] - seq[:-1]
    
    final = np.concatenate([seq, motions], axis=1)

    # Padding (모자란 프레임 0으로 채우기)
    if final.shape[0] < max_len:
        pad = np.zeros((max_len - final.shape[0], final.shape[1]), dtype=np.float32)
        final = np.vstack([final, pad])

    return final

def process_dataset_task(in_csv, save_dir, out_csv, holistic):
    print(f"\n>>> 작업 시작: {in_csv} -> {save_dir}")
    
    if not os.path.exists(in_csv):
        print(f"[스킵] 입력 파일이 없습니다: {in_csv}")
        return

    os.makedirs(save_dir, exist_ok=True)
    df = pd.read_csv(in_csv)
    
    new_rows = []
    
    # tqdm 설정 (로그 때문에 줄바꿈 되는 현상 방지용 leave=True)
    pbar = tqdm(df.iterrows(), total=len(df), desc=f"Processing", unit="video")
    
    for idx, row in pbar:
        video_path = row['video_path']
        sentence = row['sentence']
        
        # 파일명을 인덱스 기준으로 (0.npy, 1.npy...)
        npy_filename = f"{idx}.npy"
        npy_rel_path = os.path.join(save_dir, npy_filename)
        npy_abs_path = os.path.abspath(npy_rel_path)
        
        # 이미 생성된 파일은 건너뛰기 (이어하기 기능)
        if os.path.exists(npy_abs_path):
             new_rows.append({
                "npy_path": str(Path(npy_rel_path)).replace('\\', '/'),
                "sentence": sentence
            })
             continue

        try:
            tensor = process_video_to_npy(video_path, holistic, MAX_LEN)
            np.save(npy_abs_path, tensor)
            
            new_rows.append({
                "npy_path": str(Path(npy_rel_path)).replace('\\', '/'),
                "sentence": sentence
            })
        except Exception as e:
            # 에러가 나더라도 멈추지 않고 로그만 남김
            # pbar.write를 써야 진행바가 깨지지 않음
            pbar.write(f"[Error] {video_path}: {e}")
            continue
        
    # 결과 CSV 저장
    if new_rows:
        pd.DataFrame(new_rows).to_csv(out_csv, index=False, encoding='utf-8-sig')
        print(f"[완료] {out_csv} 생성됨 ({len(new_rows)}건)")
    else:
        print(f"[주의] {out_csv}에 저장된 데이터가 없습니다.")

def main():
    # Holistic 모델 로드
    # static_image_mode=False (동영상 모드, 더 빠르고 부드러움)
    # model_complexity=1 (기본값, 속도와 정확도 균형)
    holistic = mp_holistic.Holistic(
        static_image_mode=False, 
        model_complexity=1,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    )
    
    for input_csv, npy_dir, output_csv in TASKS:
        process_dataset_task(input_csv, npy_dir, output_csv, holistic)
        
    holistic.close()
    print("\n모든 작업 완료.")

if __name__ == "__main__":
    main()