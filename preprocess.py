# preprocess.py (수정: 문장 추출)

import os
import csv
from tqdm import tqdm
import re
from collections import defaultdict
import glob
import json

sets_to_process = {
    'training': './videos/1.Training',
    'validation': './videos/2.Validation'
}

def _build_label_map(root_dir):
    """
    Morpheme JSON 파일들을 재귀적으로 탐색하여
    'base_name'을 키로, '완성된 문장'을 값으로 하는 딕셔너리를 생성합니다.
    """
    print(f"-> '{root_dir}'에서 라벨(morpheme) 파일 스캔 중...")
    search_pattern = os.path.join(root_dir, "**", "*_morpheme.json")
    
    morpheme_files_iter = glob.iglob(search_pattern, recursive=True)
    morpheme_files = list(tqdm(morpheme_files_iter, desc="[1/2] Morpheme 파일 찾는 중"))
    
    label_map = {}
    skipped_files = 0
    
    for morpheme_path in tqdm(morpheme_files, desc="[2/2] Morpheme 파일 처리 중"):
        try:
            base_name = os.path.basename(morpheme_path).replace('_morpheme.json', '')

            with open(morpheme_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            data_items = data.get('data')

            # 💡 [핵심 로직 수정]
            # 'data' 키가 리스트이고, 항목이 1개 이상일 때 (문장)
            if isinstance(data_items, list) and len(data_items) > 0:
                words = []
                for item in data_items:
                    # 각 형태소 정보에서 단어(name) 추출
                    item_attrs = item.get('attributes', [{}])[0]
                    word = item_attrs.get('name')
                    if word:
                        words.append(word)
                
                if not words: # 추출된 단어가 없으면 스킵
                    skipped_files += 1
                    continue
                
                # 단어들을 공백으로 연결하여 문장 생성
                final_sentence = " ".join(words)
                label_map[base_name] = final_sentence
                    
            else:
                # 항목이 없는 등 유효하지 않은 데이터
                skipped_files += 1
        
        except Exception as e:
            skipped_files += 1
            pass # 오류가 나도 계속 진행
    
    print(f"-> 총 {len(morpheme_files)}개 morpheme 파일 발견.")
    print(f"-> {len(label_map)}개의 문장 라벨을 맵핑했습니다. (유효하지 않은 파일 {skipped_files}개 스킵)")
    return label_map

def create_index_file_optimized(dataset_name, root_dir):
    print(f"\n'{dataset_name}' 데이터셋 인덱스 생성 (파일 경로 자동 탐색)...")
    
    output_filename = f'{dataset_name}_index.csv'
    
    if not os.path.isdir(root_dir):
        print(f"-> [❌ 오류] 최상위 경로를 찾을 수 없습니다: {root_dir}")
        return

    # 1. 라벨 맵(문장) 생성
    label_map = _build_label_map(root_dir)
    if not label_map:
        print(f"-> [❌ 오류] '{root_dir}'에서 유효한 (문장) morpheme 파일을 찾을 수 없습니다.")
        return

    print(f"'{root_dir}' 경로 및 모든 하위 폴더에서 keypoint 파일을 탐색합니다...")
    search_pattern = os.path.join(root_dir, "**", "*_keypoints.json")
    
    all_files_iter = glob.iglob(search_pattern, recursive=True)
    all_files = list(tqdm(all_files_iter, desc="[1/3] Keypoint 파일 찾는 중"))
    
    if not all_files:
        print(f"-> [❌ 오류] '{root_dir}' 경로에서 Keypoint 파일을 찾을 수 없습니다.")
        return
        
    print(f"✅ 총 {len(all_files)}개의 keypoint 파일을 찾았습니다. 이제 그룹화를 시작합니다...")

    gesture_groups = defaultdict(list)
    for file_path in tqdm(sorted(all_files), desc="[2/3] Keypoint 파일 그룹화"):
        filename = os.path.basename(file_path)
        base_name = '_'.join(filename.split('_')[:-2])
        gesture_groups[base_name].append(file_path)

    total_groups_written = 0
    with open(output_filename, 'w', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        # 💡 [수정] 'label' -> 'sentence' (명확한 이름)
        writer.writerow(['gesture_base_name', 'sentence', 'file_paths'])

        for base_name, file_list in tqdm(gesture_groups.items(), desc=f"[3/3] '{dataset_name}' CSV 생성"):
            
            # 💡 [수정] 라벨(문장) 가져오기
            sentence = label_map.get(base_name)
            
            # 라벨 맵에 있는 (문장) 데이터만 CSV에 쓴다
            if sentence:
                paths_str = ";".join(file_list)
                writer.writerow([base_name, sentence, paths_str])
                total_groups_written += 1
    
    print(f"✅ 성공! 총 {len(gesture_groups)}개 keypoint 그룹 중 {total_groups_written}개의 유효한 동작 그룹과 파일 경로를 '{output_filename}' 파일로 저장했습니다.")


if __name__ == '__main__':
    for name, path in sets_to_process.items():
        create_index_file_optimized(name, path)