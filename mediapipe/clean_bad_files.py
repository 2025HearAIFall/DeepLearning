import numpy as np
import pandas as pd
import os

# 설정: preprocess_to_npy.py에서 생성한 파일 경로
TRAIN_NPY_INDEX = 'train_npy_index.csv'

def clean_data():
    if not os.path.exists(TRAIN_NPY_INDEX):
        print(f"파일이 없습니다: {TRAIN_NPY_INDEX}")
        return

    df = pd.read_csv(TRAIN_NPY_INDEX)
    valid_rows = []
    removed_count = 0

    print(f"총 {len(df)}개 파일 검사 시작...")

    for idx, row in df.iterrows():
        npy_path = row['npy_path']
        
        try:
            # .npy 파일 로드
            data = np.load(npy_path)
            
            # 모든 값이 0인지 확인 (깨진 파일은 0으로 채워졌을 가능성이 높음)
            if np.all(data == 0):
                print(f"[삭제] 데이터가 비어있음(All Zeros): {npy_path}")
                os.remove(npy_path) # 파일 삭제
                removed_count += 1
            else:
                valid_rows.append(row)
                
        except Exception as e:
            print(f"[삭제] 파일 로드 실패: {npy_path} ({e})")
            if os.path.exists(npy_path):
                os.remove(npy_path)
            removed_count += 1

    # 깨끗해진 목록으로 CSV 다시 저장
    if removed_count > 0:
        new_df = pd.DataFrame(valid_rows)
        new_df.to_csv(TRAIN_NPY_INDEX, index=False, encoding='utf-8-sig')
        print(f"\n완료: 총 {removed_count}개의 불량 데이터를 삭제하고 인덱스를 업데이트했습니다.")
    else:
        print("\n완료: 불량 데이터가 발견되지 않았습니다. (이미 처리가 잘 되었거나, OpenCV가 에러만 뱉고 넘어감)")

if __name__ == "__main__":
    clean_data()