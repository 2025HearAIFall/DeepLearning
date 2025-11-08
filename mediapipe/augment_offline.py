# augment_offline.py
# -----------------------------------------------------------------------------
# [목표]: 'all_index_npy.csv'에 있는 .npy 파일을
#         15배로 증강하여 'npy_data_augmented' 폴더에 저장합니다.
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- [사용자 설정] ---
INPUT_INDEX_FILE = 'all_index_npy.csv'
OUTPUT_INDEX_FILE = 'all_index_augmented.csv'
OUTPUT_NPY_DIR = 'npy_data_augmented'
AUGMENTATION_FACTOR = 15 # 15배 증강
# ------------------------

# --- [설정값] (train_from_npy.py와 동일) ---
MAX_LEN = 30
INPUT_SIZE = (33 + 21 + 21) * 2 * 2 # 300

def augment_noise(keypoints_seq):
    """(train_from_npy.py에서 복사)
    키포인트 시퀀스에 미세한 노이즈(Jitter)를 추가합니다."""
    # 표준편차 0.005의 노이즈 생성
    noise = np.random.normal(0, 0.005, keypoints_seq.shape).astype(np.float32)
    augmented_seq = keypoints_seq + noise
    return augmented_seq

if __name__ == '__main__':
    print(f"'{INPUT_INDEX_FILE}' 로딩 중...")
    try:
        df = pd.read_csv(INPUT_INDEX_FILE)
    except FileNotFoundError:
        print(f"[오류] '{INPUT_INDEX_FILE}' 파일을 찾을 수 없습니다.")
        print(">>> 'preprocess_to_npy.py'를 먼저 실행해야 합니다.")
        exit()

    if not os.path.exists(OUTPUT_NPY_DIR):
        os.makedirs(OUTPUT_NPY_DIR)
        print(f"'{OUTPUT_NPY_DIR}' 폴더 생성 완료.")

    new_data = [] # 새로운 CSV 파일에 저장될 내용

    print(f"총 {len(df)}개의 원본 파일을 {AUGMENTATION_FACTOR}배 증강합니다...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_npy_path = row['npy_path']
        sentence = row['sentence']
        
        try:
            # 1. 원본 .npy 로드
            original_sequence = np.load(original_npy_path)
        except Exception as e:
            print(f"[경고] 원본 '{original_npy_path}' 로드 실패. 건너뜁니다. ({e})")
            continue
            
        # 2. 원본 파일 1개 저장 (증강 0번)
        base_filename = os.path.basename(original_npy_path).replace('.npy', '')
        
        # (2-1) 원본 저장
        new_filename_original = f"{base_filename}_aug_0.npy"
        new_path_original = os.path.join(OUTPUT_NPY_DIR, new_filename_original)
        np.save(new_path_original, original_sequence)
        new_data.append({'npy_path': new_path_original, 'sentence': sentence})

        # (2-2) 14개 추가 증강 (총 15개)
        for i in range(1, AUGMENTATION_FACTOR):
            # 3. 노이즈 주입
            augmented_sequence = augment_noise(original_sequence)
            
            # 4. 새 .npy 파일로 저장
            new_filename = f"{base_filename}_aug_{i}.npy"
            new_path = os.path.join(OUTPUT_NPY_DIR, new_filename)
            np.save(new_path, augmented_sequence)
            
            # 5. 새 CSV 목록에 추가
            new_data.append({'npy_path': new_path, 'sentence': sentence})

    # 6. 새 인덱스 파일(.csv) 저장
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(OUTPUT_INDEX_FILE, index=False, encoding='utf-8-sig')
    
    print("\n--- 오프라인 증강 완료 ---")
    print(f"총 {len(new_df)}개의 증강된 데이터 생성 완료.")
    print(f"NPY 폴더: '{OUTPUT_NPY_DIR}'")
    print(f"인덱스 파일: '{OUTPUT_INDEX_FILE}'")