import torch
import torch.nn as nn
import torch.optim as optim
# 💡 [수정] random_split 임포트 확인
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np

from data_loader import SignLanguageDataset # 💡 수정된 data_loader 임포트
from model import LSTMClassifier

# --- 하이퍼파라미터 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 15       # 💡 [수정] 조기 종료를 대비해 최대 Epoch 증가
MAX_LEN = 30
INPUT_SIZE = 548      # 💡 [수정] (Pose:50 + Face:140 + Hands:84) * 2 (pos+mot) = 548
HIDDEN_SIZE = 512     # 💡 [수정] 모델 표현력 증가
NUM_LAYERS = 3        # 💡 [수정] 모델 깊이 증가
NUM_WORKERS = 0
PATIENCE = 5          # 💡 [수정] 조기 종료를 위한 Patience 설정

# --- 💡 [수정] 데이터 준비 및 분할 (전략 변경) ---
print("데이터 인덱스 로딩...")
# 1. 학습/검증용 데이터 로드 (1.Training 폴더 기반)
train_valid_df = pd.read_csv('training_index.csv')
# 2. 테스트용 데이터 로드 (2.Validation 폴더 기반)
test_df = pd.read_csv('validation_index.csv')

# --- 💡 [수정] 통합 사전 생성 (중요!) ---
# 모델이 Train/Valid/Test에서 나올 수 있는 '모든' 단어를 알아야 하므로,
# '사전'을 만들 때만 두 데이터를 임시로 합칩니다.
print("통합 라벨 사전 생성...")
all_df_for_labels = pd.concat([train_valid_df, test_df], ignore_index=True)
all_labels = sorted(all_df_for_labels['label'].unique())

label_to_idx = {label: i for i, label in enumerate(all_labels)}
idx_to_label = {i: label for i, label in enumerate(all_labels)}
label_maps = {'label_to_idx': label_to_idx, 'idx_to_label': idx_to_label}
num_classes = len(all_labels)
print(f"통합된 총 클래스(단어) 수: {num_classes}")

# --- 💡 [수정] 데이터셋 생성 및 분할 (전략 변경) ---
print("데이터셋 로딩 및 분할 시작...")

# 1. 테스트 데이터셋(Test Dataset) 생성
# (2.Validation 폴더 데이터 사용)
test_dataset = SignLanguageDataset(
    data_frame=test_df,  # 💡 [수정] test_df 전달
    max_len=MAX_LEN,
    label_maps=label_maps
)
print(f"✅ 테스트 데이터셋(Test): {len(test_dataset)}개 (원본 'validation_index.csv' 기반)")

# 2. 학습/검증 통합 데이터셋(Train/Valid Dataset) 생성
# (1.Training 폴더 데이터 사용)
full_train_valid_dataset = SignLanguageDataset(
    data_frame=train_valid_df, # 💡 [수정] train_valid_df 전달
    max_len=MAX_LEN,
    label_maps=label_maps
)
total_tv_size = len(full_train_valid_dataset)
print(f"원본 학습/검증 데이터 총: {total_tv_size}개 (원본 'training_index.csv' 기반)")

# 3. 학습/검증 데이터셋을 80% / 20%로 분할 (예시)
# (1.Training 폴더 데이터를 학습용과 검증용으로 나눔)
train_size = int(0.8 * total_tv_size)
valid_size = total_tv_size - train_size

if valid_size == 0 and train_size > 0: # 1개라도 검증셋에 할당
    train_size -= 1
    valid_size = 1

print(f"분할 비율 (training_index 기준) -> 학습: {train_size} (80%), 검증: {valid_size} (20%)")

train_dataset, valid_dataset = random_split(
    full_train_valid_dataset, 
    [train_size, valid_size],
    generator=torch.Generator().manual_seed(42) # 💡 재현성을 위해 시드 고정
)

# --- 💡 [수정] 데이터 로더 생성 ---
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"분리된 학습(Train) 데이터 수: {len(train_dataset)}")
print(f"분리된 검증(Validation) 데이터 수: {len(valid_dataset)}")
print(f"분리된 테스트(Test) 데이터 수: {len(test_dataset)}")

# --- 모델, 손실 함수, 옵티마이저, 스케줄러 정의 ---
model = LSTMClassifier(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=num_classes
    # 💡 Dropout(0.6)은 model.py의 기본값으로 적용됨
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

# 💡 [수정] 조기 종료 변수 초기화
best_val_loss = np.inf
patience_counter = 0
best_model_path = 'sign_language_model_best.pth'

# --- 모델 학습 시작 ---
print(f"\n모델 학습을 시작합니다... (DEVICE: {DEVICE})")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    
    for keypoints, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
        keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(keypoints)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()

    # --- 검증 ---
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for keypoints, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
            keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(keypoints)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(valid_loader)
    accuracy = (100 * correct / total) if total > 0 else 0
    
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")
    
    scheduler.step(avg_val_loss)

    # 💡 [수정] 조기 종료 로직
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"🎉 New best model saved with Val Loss: {best_val_loss:.4f}")
    else:
        patience_counter += 1
        print(f"Patience: {patience_counter}/{PATIENCE}")
    
    if patience_counter >= PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

print("\n학습 완료!")

# --- 최종 모델 테스트 ---
print(f"\n최고 성능의 모델({best_model_path})을 로드하여 최종 테스트를 진행합니다...")
# 💡 [수정] 가장 성능이 좋았던 모델을 로드
try:
    model.load_state_dict(torch.load(best_model_path))
except FileNotFoundError:
    print(f"[경고] {best_model_path} 파일을 찾을 수 없습니다. 마지막 epoch의 모델로 테스트합니다.")
    
model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for keypoints, labels in tqdm(test_loader, desc="Final Test"):
        keypoints, labels = keypoints.to(DEVICE), labels.to(DEVICE)
        outputs = model(keypoints)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = (100 * test_correct / test_total) if test_total > 0 else 0
print(f"\n최종 테스트 정확도 (Test Accuracy): {test_accuracy:.2f}%")

print(f"최고 성능 모델이 '{best_model_path}' 파일로 저장되었습니다.")