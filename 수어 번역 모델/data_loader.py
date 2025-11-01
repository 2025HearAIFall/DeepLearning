# data_loader.py (수정)

import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import time

class SignLanguageDataset(Dataset):
    # 💡 [수정] index_file_path 대신 data_frame을 받도록 변경
    def __init__(self, data_frame, max_len=30, label_maps=None):
        self.max_len = max_len
        # 💡 [수정] CSV를 읽는 대신, 전달받은 DataFrame을 바로 사용
        self.data_info = data_frame

        if label_maps is None:
            self.labels = sorted(self.data_info['label'].unique())
            self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        else:
            self.labels = sorted(label_maps['label_to_idx'].keys())
            self.label_to_idx = label_maps['label_to_idx']
    
    def _reshape_and_get_xy(self, keypoints_data, expected_points):
        """키포인트 데이터를 (N, 3)으로 변환하고 x, y 좌표만 추출, 없으면 0으로 채움"""
        keypoints = np.array(keypoints_data).flatten()
        if keypoints.size == 0:
            return np.zeros((expected_points, 2), dtype=np.float32)
        
        keypoints = keypoints.reshape(-1, 3)
        # 💡 신뢰도(confidence)를 제외하고 x, y 좌표만 사용
        return keypoints[:, :2]

    def _extract_and_normalize_keypoints(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 💡 [수정] people 데이터를 안전하게 가져옵니다.
        #    (JSON 구조에 따라 person_data = data.get('people', [{}])[0]
        #     또는 data.get('people', {}) 사용. 제공된 JSON 기준으로는 후자가 맞습니다.)
        person_data = data.get('people', {})

        # 💡 [수정] Pose(상체), Face, Hands 키포인트를 모두 추출 (각 2D 좌표)
        # OpenPose 표준: Pose=25, Face=70, Hand=21
        pose_xy = self._reshape_and_get_xy(
            person_data.get('pose_keypoints_2d', []), 25)
        face_xy = self._reshape_and_get_xy(
            person_data.get('face_keypoints_2d', []), 70)
        left_hand_xy = self._reshape_and_get_xy(
            person_data.get('hand_left_keypoints_2d', []), 21)
        right_hand_xy = self._reshape_and_get_xy(
            person_data.get('hand_right_keypoints_2d', []), 21)

        # 1. 중심점 이동 (목 'Neck' 기준)
        # 💡 [수정] Pose keypoint 1번(Neck)을 중심점으로 사용
        neck = pose_xy[1].copy()
        
        if np.sum(neck**2) > 1e-6: # 목 좌표가 (0,0)이 아닐 경우
            pose_xy -= neck
            face_xy -= neck
            left_hand_xy -= neck
            right_hand_xy -= neck
        
        # 💡 2. 크기 정규화 (모든 좌표 통합)
        combined_coords = np.vstack([pose_xy, face_xy, left_hand_xy, right_hand_xy])
        max_abs_val = np.max(np.abs(combined_coords))
        
        if max_abs_val > 1e-6:
            pose_xy /= max_abs_val
            face_xy /= max_abs_val
            left_hand_xy /= max_abs_val
            right_hand_xy /= max_abs_val

        # 💡 [수정] 모든 정규화된 좌표를 1차원 벡터로 결합
        # (50 + 140 + 42 + 42) = 274
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
        label = item['label']
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
        
        # 💡 [수정] 입력 피처 크기 변경 (274)
        num_features = 274
        
        if not sequence:
            # 💡 [수정] 모션 피처 포함 (274 * 2 = 548)
            sequence = np.zeros((self.max_len, num_features * 2), dtype=np.float32)
            label_idx = self.label_to_idx[label]
            return torch.from_numpy(sequence), torch.tensor(label_idx, dtype=torch.long)
            
        sequence = np.array(sequence, dtype=np.float32)
        
        positions = sequence
        motions = np.zeros_like(positions)
        if len(positions) > 1:
            motions[1:] = positions[1:] - positions[:-1]
        
        # 💡 [수정] 최종 피처: 위치(274) + 모션(274) = 548
        final_sequence = np.concatenate([positions, motions], axis=1)

        seq_len = final_sequence.shape[0]
        if seq_len < self.max_len:
            padding = np.zeros((self.max_len - seq_len, final_sequence.shape[1]), dtype=np.float32)
            final_sequence = np.vstack([final_sequence, padding])

        label_idx = self.label_to_idx[label]
        return torch.from_numpy(final_sequence), torch.tensor(label_idx, dtype=torch.long)