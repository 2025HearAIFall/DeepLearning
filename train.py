# train.py (수정: if __name__ == '__main__' 블록 추가)

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
import pickle
import os 

from data_loader import SignLanguageDataset
from model import Encoder, Decoder, Seq2Seq, Attention # 💡 Attention 임포트

# --- 토크나이저 및 Vocabulary 클래스 정의 ---
# (이 부분은 import 되어야 하므로 밖에 둡니다)
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
            
        idx = 4 
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

# --------------------------------------------------
# 💡 [수정] 여기부터 파일 끝까지 모두 if __name__ == '__main__': 블록 안으로 이동
# --------------------------------------------------

if __name__ == '__main__':
    
    # --- 하이퍼파라미터 설정 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005  
    NUM_EPOCHS = 15 # 💡 [수정] Epoch 15로 조정 (로그 기준)
    MAX_LEN = 30
    MAX_TARGET_LEN = 50
    INPUT_SIZE = 548
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    EMBED_SIZE = 256
    DROPOUT_PROB = 0.6
    NUM_WORKERS = 0
    PATIENCE = 10 # 💡 [수정] Patience 10으로 조정

    # --- 데이터 준비 및 Vocabulary 생성 ---
    print("전체 데이터 인덱스 로딩 및 Vocabulary 생성...")
    try:
        train_df = pd.read_csv('training_index.csv')
        valid_df = pd.read_csv('validation_index.csv')
    except FileNotFoundError:
        print("[오류] training_index.csv 또는 validation_index.csv 파일을 찾을 수 없습니다.")
        exit()

    all_df = pd.concat([train_df, valid_df], ignore_index=True)
    all_sentences = all_df['sentence'].tolist()

    vocab = Vocabulary(simple_tokenizer, min_freq=2)
    vocab.build_vocab(all_sentences)
    TARGET_VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab.pad_idx

    print(f"생성된 Target Vocabulary 크기: {TARGET_VOCAB_SIZE}")

    print("통합 데이터셋 생성 및 분할...")

    COMBINED_INDEX_FILE = 'combined_index.csv'
    all_df.to_csv(COMBINED_INDEX_FILE, index=False, encoding='utf-8-sig')

    try:
        full_dataset = SignLanguageDataset(
            index_file_path=COMBINED_INDEX_FILE, 
            max_len=MAX_LEN, 
            max_target_len=MAX_TARGET_LEN,
            vocab=vocab
        )
    except Exception as e:
        print(f"데이터셋 로딩 중 오류 발생: {e}")
        exit()

    if os.path.exists(COMBINED_INDEX_FILE):
        os.remove(COMBINED_INDEX_FILE)

    total_size = len(full_dataset)
    train_size = int(total_size * 0.8)
    valid_size = int(total_size * 0.1)
    test_size = total_size - train_size - valid_size 

    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_dataset = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )

    print(f"총 데이터: {total_size}개")
    print(f"분리된 학습 데이터 수: {len(train_dataset)} (80%)")
    print(f"분리된 검증 데이터 수: {len(valid_dataset)} (10%)")
    print(f"분리된 테스트 데이터 수: {len(test_dataset)} (10%)")

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 모델, 손실 함수, 옵티마이저, 스케줄러 정의 ---
    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

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
            
            optimizer.zero_grad()
            
            outputs = model(keypoints, targets, teacher_forcing_ratio=0.5)
            
            output_dim = outputs.shape[-1]
            outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
            targets_flat = targets[:, 1:].reshape(-1)
            
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
                
                outputs = model(keypoints, targets, teacher_forcing_ratio=0.0)
                
                output_dim = outputs.shape[-1]
                outputs_flat = outputs[:, 1:, :].reshape(-1, output_dim)
                targets_flat = targets[:, 1:].reshape(-1)
                
                loss = criterion(outputs_flat, targets_flat)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(valid_loader)
        
        try:
            train_ppl = math.exp(avg_train_loss)
            val_ppl = math.exp(avg_val_loss)
        except OverflowError:
            train_ppl = float('inf')
            val_ppl = float('inf')

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f} (PPL: {train_ppl:.2f}), Val Loss: {avg_val_loss:.4f} (PPL: {val_ppl:.2f})")
        
        scheduler.step(avg_val_loss)

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
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
    except FileNotFoundError:
        print(f"[경고] {best_model_path} 파일을 찾을 수 없습니다. 마지막 epoch의 모델로 테스트합니다.")
    except Exception as e:
        print(f"[경고] 모델 로드 중 오류 발생: {e}. 마지막 epoch의 모델로 테스트합니다.")
        
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
    try:
        test_ppl = math.exp(avg_test_loss)
    except OverflowError:
        test_ppl = float('inf')
        
    print(f"\n최종 테스트 손실 (Test Loss): {avg_test_loss:.4f}, 테스트 Perplexity (PPL): {test_ppl:.2f}")

    print(f"최고 성능 모델이 '{best_model_path}' 파일로 저장되었습니다.")

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary가 'vocab.pkl' 파일로 저장되었습니다.")
