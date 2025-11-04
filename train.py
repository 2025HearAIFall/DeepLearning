# train.py (수정: Seq2Seq 학습)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from collections import Counter

# 💡 [수정] data_loader, model 임포트
from data_loader import SignLanguageDataset
from model import Encoder, Decoder, Seq2Seq

# --- 토크나이저 및 Vocabulary 클래스 정의 ---
# 💡 (간단한 예시: 공백 기준 토크나이저. 형태소 분석기(Mecab) 사용 권장)
def simple_tokenizer(text):
    return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        
        # 0: <PAD>, 1: <SOS>, 2: <EOS>, 3: <UNK>
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
        
    def __len__(self):
        return len(self.itos)
    
    def build_vocab(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list:
            counter.update(self.tokenizer(sentence))
            
        idx = 4 # 0-3은 특수 토큰
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1
                
    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokens]
    
    @property
    def pad_idx(self): return self.stoi["<PAD>"]
    @property
    def sos_idx(self): return self.stoi["<SOS>"]
    @property
    def eos_idx(self): return self.stoi["<EOS>"]
    @property
    def unk_idx(self): return self.stoi["<UNK>"]


# --- 하이퍼파라미터 설정 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32       # 💡 Seq2Seq는 메모리를 많이 사용하므로 BATCH_SIZE 줄임
LEARNING_RATE = 0.001
NUM_EPOCHS = 100      # 💡 조기 종료를 위해 증가
MAX_LEN = 30          # 💡 [유지] 입력 키포인트 시퀀스 최대 길이
MAX_TARGET_LEN = 50   # 💡 [추가] 타겟 문장 시퀀스 최대 길이
INPUT_SIZE = 548      # 💡 [유지] (Pose+Face+Hands) * 2 (pos+mot) = 548
HIDDEN_SIZE = 512     # 💡 [유지]
NUM_LAYERS = 3        # 💡 [유지]
EMBED_SIZE = 256      # 💡 [추가] 타겟 단어 임베딩 크기
DROPOUT_PROB = 0.5    # 💡 [수정] Dropout 값 명시 (기존 0.6)
NUM_WORKERS = 0
PATIENCE = 10         # 💡 [수정] 조기 종료 Patience 증가

# --- 데이터 준비 및 Vocabulary 생성 ---
print("전체 데이터 인덱스 로딩 및 Vocabulary 생성...")
train_df = pd.read_csv('training_index.csv')
valid_df = pd.read_csv('validation_index.csv')

all_df = pd.concat([train_df, valid_df], ignore_index=True)
all_sentences = all_df['sentence'].tolist() # 💡 'label' -> 'sentence'

# 💡 Vocabulary 객체 생성 및 빌드
vocab = Vocabulary(simple_tokenizer, min_freq=2)
vocab.build_vocab(all_sentences)
TARGET_VOCAB_SIZE = len(vocab) # 💡 [추가]
PAD_IDX = vocab.pad_idx

print(f"생성된 Target Vocabulary 크기: {TARGET_VOCAB_SIZE}")

print("학습/검증 데이터 로딩 시작...")
full_train_dataset = SignLanguageDataset(
    index_file_path='training_index.csv', 
    max_len=MAX_LEN, 
    max_target_len=MAX_TARGET_LEN, # 💡 추가
    vocab=vocab                  # 💡 주입
)
train_size = int(0.8 * len(full_train_dataset))
valid_size = len(full_train_dataset) - train_size
train_dataset, valid_dataset = random_split(full_train_dataset, [train_size, valid_size])

print("테스트 데이터 로딩 시작...")
test_dataset = SignLanguageDataset(
    index_file_path='validation_index.csv', 
    max_len=MAX_LEN, 
    max_target_len=MAX_TARGET_LEN, # 💡 추가
    vocab=vocab                  # 💡 주입
)

# 💡 [주의] DataLoader는 이제 (keypoints, target_sentences) 튜플을 반환
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print(f"분리된 학습 데이터 수: {len(train_dataset)}, 검증 데이터 수: {len(valid_dataset)}")
print(f"별도 테스트 데이터 수: {len(test_dataset)}")

# --- 모델, 손실 함수, 옵티마이저, 스케줄러 정의 ---
# 💡 [수정] Seq2Seq 모델 정의
encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

# 💡 [수정] 패딩 토큰은 손실 계산에서 제외
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

best_val_loss = float('inf')
patience_counter = 0
best_model_path = 'sign_language_model_seq2seq_best.pth'

# --- 모델 학습 시작 ---
print(f"\nSeq2Seq 모델 학습을 시작합니다... (DEVICE: {DEVICE})")
for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0.0
    
    for (keypoints, targets) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
        keypoints, targets = keypoints.to(DEVICE), targets.to(DEVICE)
        
        # keypoints: (batch, max_len, 548)
        # targets: (batch, max_target_len)
        
        optimizer.zero_grad()
        
        # 💡 [수정] Seq2Seq 모델 forward
        # outputs: (batch, max_target_len, vocab_size)
        outputs = model(keypoints, targets, teacher_forcing_ratio=0.5)
        
        # 💡 [수정] 손실 계산 (CrossEntropyLoss는 2D 입력을 기대)
        # outputs: (batch * (max_target_len-1), vocab_size)
        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim) # <SOS> 제외
        
        # targets: (batch * (max_target_len-1))
        targets_flat = targets[:, 1:].reshape(-1) # <SOS> 제외
        
        loss = criterion(outputs_flat, targets_flat)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()

    # --- 검증 ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (keypoints, targets) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
            keypoints, targets = keypoints.to(DEVICE), targets.to(DEVICE)
            
            # 💡 [수정] 검증 시에는 Teacher Forcing 끔 (ratio=0.0)
            outputs = model(keypoints, targets, teacher_forcing_ratio=0.0)
            
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
            targets_flat = targets[:, 1:].reshape(-1)
            
            loss = criterion(outputs_flat, targets_flat)
            val_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(valid_loader)
    
    # 💡 [수정] 분류 정확도 대신 Perplexity (PPL) 출력
    try:
        train_ppl = math.exp(avg_train_loss)
        val_ppl = math.exp(avg_val_loss)
    except OverflowError:
        train_ppl = float('inf')
        val_ppl = float('inf')

    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}), Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")
    
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
try:
    model.load_state_dict(torch.load(best_model_path))
except FileNotFoundError:
    print(f"[경고] {best_model_path} 파일을 찾을 수 없습니다. 마지막 epoch의 모델로 테스트합니다.")
    
model.eval()
test_loss = 0.0
with torch.no_grad():
    for (keypoints, targets) in tqdm(test_loader, desc="Final Test"):
        keypoints, targets = keypoints.to(DEVICE), targets.to(DEVICE)
        outputs = model(keypoints, targets, teacher_forcing_ratio=0.0)
        
        output_dim = outputs.shape[-1]
        outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
        targets_flat = targets[:, 1:].reshape(-1)
        
        loss = criterion(outputs_flat, targets_flat)
        test_loss += loss.item()

avg_test_loss = test_loss / len(test_loader)
test_ppl = math.exp(avg_test_loss)
print(f"\n최종 테스트 손실 (Test Loss): {avg_test_loss:.4f}, 테스트 Perplexity (PPL): {test_ppl:.2f}")

print(f"최고 성능 모델이 '{best_model_path}' 파일로 저장되었습니다.")
# 💡 Vocabulary도 저장해야 실제 추론에서 사용할 수 있습니다.
import pickle
with open('vocab.pkl', 'wb') as f:
    pickle.dump(vocab, f)
print("Vocabulary가 'vocab.pkl' 파일로 저장되었습니다.")