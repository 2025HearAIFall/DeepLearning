# augment_offline.py (v2.1 정확도 향상 버전)
# -----------------------------------------------------------------------------
# [목표]: Train 데이터를 읽어 스케일링(크기 변화) + 노이즈(떨림) 증강 수행
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from pathlib import Path

# --- [사용자 설정] ---
INPUT_INDEX_FILE = 'train_npy_index.csv'      # preprocess_to_npy.py 결과물
OUTPUT_INDEX_FILE = 'train_augmented.csv'     # 최종 학습용 리스트
OUTPUT_NPY_DIR = 'npy_train_augmented'        # 증강된 파일 저장 폴더

AUGMENTATION_FACTOR = 10  # 원본 포함 10배로 증강 (원본 1 + 증강 9)
# ------------------------

def augment_noise(keypoints_seq):
    """
    [노이즈 추가]
    키포인트 좌표에 미세한 떨림(Jittering)을 추가합니다.
    표준편차 0.005 (0~1 정규화 기준) 정도의 노이즈
    """
    noise = np.random.normal(0, 0.005, keypoints_seq.shape).astype(np.float32)
    return keypoints_seq + noise

def augment_scale(keypoints_seq):
    """
    [스케일링]
    사람의 크기가 화면마다 다른 것을 모사하기 위해
    전체 좌표에 0.85 ~ 1.15 배의 랜덤 값을 곱합니다.
    (상대 좌표계, 절대 좌표계 모두 유효)
    """
    scale_factor = np.random.uniform(0.85, 1.15)
    return keypoints_seq * scale_factor

def main():
    print(f"Loading '{INPUT_INDEX_FILE}'...")
    
    if not os.path.exists(INPUT_INDEX_FILE):
        print(f"[오류] '{INPUT_INDEX_FILE}' 파일을 찾을 수 없습니다.")
        print(">>> 'preprocess_to_npy.py'를 먼저 실행했는지 확인하세요.")
        return

    df = pd.read_csv(INPUT_INDEX_FILE)

    if not os.path.exists(OUTPUT_NPY_DIR):
        os.makedirs(OUTPUT_NPY_DIR)
        print(f"Created directory: '{OUTPUT_NPY_DIR}'")

    new_data = [] 

    print(f"Total {len(df)} videos -> Augmenting x{AUGMENTATION_FACTOR}...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        original_npy_path = row['npy_path']
        sentence = row['sentence']
        
        try:
            # 1. 원본 로드
            original_sequence = np.load(original_npy_path)
        except Exception as e:
            print(f"[Skip] Load failed: {original_npy_path} ({e})")
            continue
            
        base_filename = Path(original_npy_path).stem 
        
        # (2-1) 원본 저장 (aug_0)
        new_filename_0 = f"{base_filename}_aug_0.npy"
        new_path_0 = os.path.join(OUTPUT_NPY_DIR, new_filename_0)
        new_path_0 = str(Path(new_path_0)).replace('\\', '/')
        
        np.save(new_path_0, original_sequence)
        new_data.append({'npy_path': new_path_0, 'sentence': sentence})

        # (2-2) 증강본 생성 및 저장 (aug_1 ~ aug_N-1)
        for i in range(1, AUGMENTATION_FACTOR):
            # 복사본 생성
            aug_seq = original_sequence.copy()
            
            # 50% 확률로 스케일링 적용 (크기 변화)
            if np.random.rand() < 0.5:
                aug_seq = augment_scale(aug_seq)
            
            # 노이즈는 항상 적용 (센서 잡음 모사)
            aug_seq = augment_noise(aug_seq)
            
            # 저장
            new_filename = f"{base_filename}_aug_{i}.npy"
            new_path = os.path.join(OUTPUT_NPY_DIR, new_filename)
            new_path = str(Path(new_path)).replace('\\', '/')
            
            np.save(new_path, aug_seq)
            new_data.append({'npy_path': new_path, 'sentence': sentence})

    # CSV 저장
    new_df = pd.DataFrame(new_data)
    new_df.to_csv(OUTPUT_INDEX_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*40)
    print(f"✅ 증강 완료!")
    print(f" - 결과 파일: {OUTPUT_INDEX_FILE}")
    print(f" - 데이터 수: {len(new_df)}개 (원본 {len(df)}개 * {AUGMENTATION_FACTOR})")
    print("="*40)

if __name__ == '__main__':
    main()