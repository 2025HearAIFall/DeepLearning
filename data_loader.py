# data_loader.py (수정: Seq2Seq용)

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time

# 💡 Vocabulary 클래스는 train.py에서 정의하고 여기에 주입(inject)합니다.
# (파일 간 의존성을 줄이기 위해)

class SignLanguageDataset(Dataset):
    # 💡 [수정] label_maps 대신 vocab 객체와 max_target_len 추가
    def __init__(self, index_file_path, max_len=30, max_target_len=50, vocab=None):
        self.max_len = max_len
        self.max_target_len = max_target_len # 💡 타겟 문장 최대 길이
        self.data_info = pd.read_csv(index_file_path)
        
        # 💡 [수정] 라벨맵 대신 Vocabulary 객체 사용
        if vocab is None:
            raise ValueError("Vocabulary 객체가 제공되어야 합니다.")
        self.vocab = vocab
    
    def _reshape_and_get_xy(self, keypoints_data, expected_points):
        """키포인트 데이터를 (N, 3)으로 변환하고 x, y 좌표만 추출, 없으면 0으로 채움"""
        keypoints = np.array(keypoints_data).flatten()
        if keypoints.size == 0:
            return np.zeros((expected_points, 2), dtype=np.float32)
        
        keypoints = keypoints.reshape(-1, 3)
        return keypoints[:, :2]

    def _extract_and_normalize_keypoints(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        person_data = data.get('people', {})

        pose_xy = self._reshape_and_get_xy(
            person_data.get('pose_keypoints_2d', []), 25)
        face_xy = self._reshape_and_get_xy(
            person_data.get('face_keypoints_2d', []), 70)
        left_hand_xy = self._reshape_and_get_xy(
            person_data.get('hand_left_keypoints_2d', []), 21)
        right_hand_xy = self._reshape_and_get_xy(
            person_data.get('hand_right_keypoints_2d', []), 21)

        # 1. 중심점 이동 (목 'Neck' 기준)
        neck = pose_xy[1].copy()
        
        if np.sum(neck**2) > 1e-6:
            pose_xy -= neck
            face_xy -= neck
            left_hand_xy -= neck
            right_hand_xy -= neck
        
        # 2. 크기 정규화
        combined_coords = np.vstack([pose_xy, face_xy, left_hand_xy, right_hand_xy])
        max_abs_val = np.max(np.abs(combined_coords))
        
        if max_abs_val > 1e-6:
            pose_xy /= max_abs_val
            face_xy /= max_abs_val
            left_hand_xy /= max_abs_val
            right_hand_xy /= max_abs_val

        # 모든 정규화된 좌표를 1차원 벡터로 결합 (274)
        return np.concatenate([
            pose_xy.flatten(), 
            face_xy.flatten(), 
            left_hand_xy.flatten(), 
            right_hand_xy.flatten()
        ])

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        item = self.data_info.iloc[idx]
        
        # 💡 [수정] 'label' -> 'sentence' (preprocess.py와 일치)
        sentence_str = item['sentence']
        
        paths_str = item['file_paths']
        json_paths = paths_str.split(';')
        
        num_frames = len(json_paths)
        if num_frames > self.max_len:
            indices = np.linspace(0, num_frames - 1, self.max_len, dtype=int)
            sampled_paths = [json_paths[i] for i in indices]
        else:
            sampled_paths = json_paths

        sequence = []
        for path in sampled_paths:
            try:
                keypoints = self._extract_and_normalize_keypoints(path)
                sequence.append(keypoints)
            except Exception as e:
                continue
        
        num_features = 274
        
        # 💡 [수정] 타겟 문장(label) 처리
        # 문장을 토큰 인덱스로 변환 (e.g., "나는 밥을 먹었다")
        indices = self.vocab.numericalize(sentence_str)
        
        # <SOS>와 <EOS> 토큰 추가
        indices = [self.vocab.sos_idx] + indices + [self.vocab.eos_idx]
        
        # 타겟 문장 패딩
        target_len = len(indices)
        if target_len < self.max_target_len:
            indices.extend([self.vocab.pad_idx] * (self.max_target_len - target_len))
        else:
            indices = indices[:self.max_target_len] # 자르기
            
        target_tensor = torch.tensor(indices, dtype=torch.long)
        
        # --- 키포인트 시퀀스 처리 ---
        if not sequence:
            final_sequence = np.zeros((self.max_len, num_features * 2), dtype=np.float32)
            # 💡 [수정] (키포인트 시퀀스, 타겟 문장 시퀀스) 반환
            return torch.from_numpy(final_sequence), target_tensor
            
        sequence = np.array(sequence, dtype=np.float32)
        
        positions = sequence
        motions = np.zeros_like(positions)
        if len(positions) > 1:
            motions[1:] = positions[1:] - positions[:-1]
        
        final_sequence = np.concatenate([positions, motions], axis=1) # (seq_len, 548)

        seq_len = final_sequence.shape[0]
        if seq_len < self.max_len:
            padding = np.zeros((self.max_len - seq_len, final_sequence.shape[1]), dtype=np.float32)
            final_sequence = np.vstack([final_sequence, padding])

        # 💡 [수정] (키포인트 시퀀스, 타겟 문장 시퀀스) 반환
        return torch.from_numpy(final_sequence), target_tensor
