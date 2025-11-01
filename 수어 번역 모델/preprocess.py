# preprocess.py (최종: 단일 단어(문법 속성 포함)만 추출 / 경로 자동 탐색 / 탐색 진행 상황 게이지 추가)

import os
import csv
from tqdm import tqdm
import re
from collections import defaultdict
import glob
import json # 💡 JSON 파일을 읽기 위해 import

sets_to_process = {
    'training': './videos/1.Training',
    'validation': './videos/2.Validation'
}

def _build_label_map(root_dir):
    """
    Morpheme JSON 파일들을 재귀적으로 탐색하여
    'base_name'을 키로, '단일 단어 라벨'을 값으로 하는 딕셔너리를 생성합니다.
    (문법 속성(attribute)이 있을 경우 라벨에 포함합니다.)
    """
    print(f"-> '{root_dir}'에서 라벨(morpheme) 파일 스캔 중...")
    search_pattern = os.path.join(root_dir, "**", "*_morpheme.json")
    
    # [1/2] Morpheme 파일 탐색 (게이지 표시)
    morpheme_files_iter = glob.iglob(search_pattern, recursive=True)
    morpheme_files = list(tqdm(morpheme_files_iter, desc="[1/2] Morpheme 파일 찾는 중"))
    
    label_map = {}
    skipped_multi_word = 0
    
    # [2/2] Morpheme 파일 처리 (게이지 표시)
    for morpheme_path in tqdm(morpheme_files, desc="[2/2] Morpheme 파일 처리 중"):
        try:
            base_name = os.path.basename(morpheme_path).replace('_morpheme.json', '')

            with open(morpheme_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data_items = data.get('data')

            # 💡 [핵심 로직]
            # 'data' 키가 리스트이고, 항목이 1개일 때만 (단일 단어)
            if isinstance(data_items, list) and len(data_items) == 1:
                item_attrs = data_items[0].get('attributes', [{}])[0]
                label_name = item_attrs.get('name') # e.g., "왼쪽"
                grammar_attr = item_attrs.get('attribute') # e.g., ["1형태소..."]

                if not label_name: # 라벨 이름이 없으면 스킵
                    skipped_multi_word += 1
                    continue
                
                # 문법 속성이 있는지 확인
                if isinstance(grammar_attr, list) and len(grammar_attr) > 0:
                    grammar_str = " ".join(grammar_attr)
                    final_label = f"{label_name} ({grammar_str})" # e.g., "왼쪽 (1형태소...)"
                else:
                    final_label = label_name # e.g., "가락로"
                
                label_map[base_name] = final_label
                    
            else:
                # 항목이 없거나, 2개 이상인 '문장' 데이터 (e.g., real_sen)
                skipped_multi_word += 1
        
        except Exception as e:
            skipped_multi_word += 1
            pass # 오류가 나도 계속 진행
    
    print(f"-> 총 {len(morpheme_files)}개 morpheme 파일 발견.")
    print(f"-> {len(label_map)}개의 단일 단어 라벨(문법 속성 포함)을 맵핑했습니다. (문장/오류 {skipped_multi_word}개 스킵)")
    return label_map

def create_index_file_optimized(dataset_name, root_dir):
    print(f"\n'{dataset_name}' 데이터셋 인덱스 생성 (파일 경로 자동 탐색)...")
    
    output_filename = f'{dataset_name}_index.csv'
    
    if not os.path.isdir(root_dir):
        print(f"-> [❌ 오류] 최상위 경로를 찾을 수 없습니다: {root_dir}")
        return

    # 1. 라벨 맵 생성
    label_map = _build_label_map(root_dir)
    if not label_map:
        print(f"-> [❌ 오류] '{root_dir}'에서 유효한 (단일 단어) morpheme 파일을 찾을 수 없습니다.")
        return

    print(f"'{root_dir}' 경로 및 모든 하위 폴더에서 keypoint 파일을 탐색합니다...")
    search_pattern = os.path.join(root_dir, "**", "*_keypoints.json")
    
    # [1/3] Keypoint 파일 탐색 (게이지 표시)
    all_files_iter = glob.iglob(search_pattern, recursive=True)
    all_files = list(tqdm(all_files_iter, desc="[1/3] Keypoint 파일 찾는 중"))
    
    if not all_files:
        print(f"-> [❌ 오류] '{root_dir}' 경로에서 Keypoint 파일을 찾을 수 없습니다.")
        return
        
    print(f"✅ 총 {len(all_files)}개의 keypoint 파일을 찾았습니다. 이제 그룹화를 시작합니다...")

    # [2/3] Keypoint 파일 그룹화 (게이지 표시)
    gesture_groups = defaultdict(list)
    for file_path in tqdm(sorted(all_files), desc="[2/3] Keypoint 파일 그룹화"):
        filename = os.path.basename(file_path)
        base_name = '_'.join(filename.split('_')[:-2])
        gesture_groups[base_name].append(file_path)

    total_groups_written = 0
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['gesture_base_name', 'label', 'file_paths'])

        # [3/3] CSV 파일 생성 (게이지 표시)
        for base_name, file_list in tqdm(gesture_groups.items(), desc=f"[3/3] '{dataset_name}' CSV 생성"):
            
            label = label_map.get(base_name)
            
            # 라벨 맵에 있는 (단일 단어) 데이터만 CSV에 쓴다
            if label:
                paths_str = ";".join(file_list)
                writer.writerow([base_name, label, paths_str])
                total_groups_written += 1
    
    print(f"✅ 성공! 총 {len(gesture_groups)}개 keypoint 그룹 중 {total_groups_written}개의 유효한 동작 그룹과 파일 경로를 '{output_filename}' 파일로 저장했습니다.")


if __name__ == '__main__':
    for name, path in sets_to_process.items():
        create_index_file_optimized(name, path)