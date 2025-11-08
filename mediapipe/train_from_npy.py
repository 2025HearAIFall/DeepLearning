# train_from_npy.py (Offline Augmentation 버전으로 개조됨)
# -----------------------------------------------------------------------------
# [목표]: 'all_index_npy.csv'를 읽어들인 직후,
#         메모리 상에서 데이터를 30배로 증강시키고,
#         30배가 된 데이터로 훈련을 시작합니다.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import math
from collections import Counter
import pickle
import os 
import random 

# --- [사용자 설정] ---
INPUT_INDEX_FILE = 'all_index_npy.csv'
AUGMENTATION_FACTOR = 30 # 30배 증강
# ------------------------

# --- 하이퍼파라미터 (전역 변수로 이동) ---
MAX_LEN = 30
INPUT_SIZE = (33 + 468 + 21 + 21) * 2 * 2 # 2172

# --- 토크나이저 및 Vocabulary 클래스 정의 ---
# ... (이전과 동일) ...
def simple_tokenizer(text):
    return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
    def __len__(self): return len(self.itos)
    def build_vocab(self, sentence_list):
        counter = Counter()
        for sentence in sentence_list: counter.update(self.tokenizer(sentence))
        idx = 4 
        for word, freq in counter.items():
            if freq >= self.min_freq: self.stoi[word] = idx; self.itos[idx] = word; idx += 1
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
# [개조] NPY -> InMemory 데이터셋 클래스
# --------------------------------------------------
class SignLanguageDataset_InMemory(Dataset):
    def __init__(self, data_list, max_target_len=50, vocab=None):
        self.max_target_len = max_target_len
        # [개조] CSV가 아닌, (np.array, "문장") 튜플의 리스트를 직접 받음
        self.data_list = data_list 
        if vocab is None:
            raise ValueError("Vocabulary 객체가 제공되어야 합니다.")
        self.vocab = vocab
        
        # [제거됨] is_train (실시간 증강 안 함)

    def __len__(self):
        # [개조] data_info가 아닌 data_list의 길이 반환
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # [개조] CSV가 아닌 리스트에서 (텐서, 문장)을 바로 가져옴
        final_sequence, sentence_str = self.data_list[idx]

        # 타겟 문장 처리 (기존과 동일)
        indices = self.vocab.numericalize(sentence_str)
        indices = [self.vocab.sos_idx] + indices + [self.vocab.eos_idx]
        if len(indices) < self.max_target_len:
            indices = indices + [self.vocab.pad_idx] * (self.max_target_len - len(indices))
        else:
            indices = indices[:self.max_target_len]
        target_tensor = torch.tensor(indices, dtype=torch.long)

        return torch.from_numpy(final_sequence), target_tensor

# --- [개조] 증강 함수 (클래스 밖으로 이동) ---
def augment_noise(keypoints_seq):
    """키포인트 시퀀스에 미세한 노이즈(Jitter)를 추가합니다."""
    noise = np.random.normal(0, 0.005, keypoints_seq.shape).astype(np.float32)
    augmented_seq = keypoints_seq + noise
    return augmented_seq

# --------------------------------------------------
# [모델 클래스 정의] (Encoder, Attention, Decoder, Seq2Seq)
# ... (이전과 동일) ...
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size; self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.shape[1]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        energy_input = torch.cat((hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(energy_input))
        attention_scores = self.v(energy)
        return torch.softmax(attention_scores.squeeze(2), dim=1).unsqueeze(1)
class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.output_dim = output_dim; self.hidden_size = hidden_size; self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + emb_dim, hidden_size, num_layers, dropout=dropout_prob)
        self.fc_out = nn.Linear(hidden_size * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token))
        last_layer_hidden = hidden[-1]
        attn_weights = self.attention(last_layer_hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        context = context.permute(1, 0, 2)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        prediction_input = torch.cat((output.squeeze(0), embedded.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(prediction_input)
        return prediction, hidden, cell
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]; trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            teacher_force = torch.rand(1) < teacher_forcing_ratio
            top1 = output.argmax(1)
            input_token = trg[:, t] if teacher_force else top1
        return outputs
# --------------------------------------------------
# 메인 학습 블록
# --------------------------------------------------

if __name__ == '__main__':
    
    # --- 하이퍼파라미터 설정 ---
    # ... (이전과 동일) ...
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0005  
    NUM_EPOCHS = 30 
    MAX_TARGET_LEN = 50
    HIDDEN_SIZE = 512
    NUM_LAYERS = 3
    EMBED_SIZE = 256
    DROPOUT_PROB = 0.6
    NUM_WORKERS = 0 
    PATIENCE = 10 

    # --- [개조] 데이터 준비 및 Vocabulary 생성 ---
    print(f"'{INPUT_INDEX_FILE}' 로딩 및 Vocabulary 생성 (NPY용)...")
    try:
        all_df = pd.read_csv(INPUT_INDEX_FILE)
    except FileNotFoundError:
        print(f"[오류] '{INPUT_INDEX_FILE}' 파일을 찾을 수 없습니다.")
        exit()

    all_sentences = all_df['sentence'].tolist()

    vocab = Vocabulary(simple_tokenizer, min_freq=2)
    vocab.build_vocab(all_sentences)
    TARGET_VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab.pad_idx

    print(f"생성된 Target Vocabulary 크기: {TARGET_VOCAB_SIZE}")
    print(f"원본 데이터 수: {len(all_df)}개")

    # --- [신규] 오프라인(메모리) 증강 ---
    print(f"{AUGMENTATION_FACTOR}배 오프라인 증강을 시작합니다... (메모리로 로드 중)")
    
    augmented_data_list = []
    
    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Augmenting"):
        npy_path = row['npy_path']
        sentence = row['sentence']
        
        try:
            original_sequence = np.load(npy_path)
        except Exception as e:
            print(f"Warning: Cannot load npy file {npy_path}. Skipping. Error: {e}")
            continue
            
        # 1. 원본 1개 추가
        augmented_data_list.append((original_sequence, sentence))
        
        # 2. (AUGMENTATION_FACTOR - 1) 만큼 증강하여 추가
        for _ in range(AUGMENTATION_FACTOR - 1):
            augmented_sequence = augment_noise(original_sequence)
            augmented_data_list.append((augmented_sequence, sentence))

    print(f"증강 완료. 총 데이터 수: {len(augmented_data_list)}개")
    print("통합 데이터셋 생성 및 분할 (In-Memory)...")

    try:
        # [개조] InMemory 데이터셋 클래스 사용
        full_dataset = SignLanguageDataset_InMemory(
            data_list=augmented_data_list, 
            max_target_len=MAX_TARGET_LEN,
            vocab=vocab
        )
    except Exception as e:
        print(f"데이터셋 로딩 중 오류 발생: {e}")
        exit()

    # (데이터 분할 로직은 기존과 동일)
    total_size = len(full_dataset) # 1560 (10배)
    train_size = int(total_size * 0.8) # 약 1248
    valid_size = int(total_size * 0.1) # 약 156
    test_size = total_size - train_size - valid_size # 약 156

    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset, test_dataset = random_split(
        full_dataset, [train_size, valid_size, test_size], generator=generator
    )

    print(f"총 데이터: {total_size}개")
    print(f"분리된 학습 데이터 수: {len(train_dataset)} ({train_size})")
    print(f"분리된 검증 데이터 수: {len(valid_dataset)} ({valid_size})")
    print(f"분리된 테스트 데이터 수: {len(test_dataset)} ({test_size})")

    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 모델, 손실 함수, 옵티마이저, 스케줄러 정의 ---
    # ... (이전과 동일) ...
    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'model_a_mediapipe_best.pth'

    # --- 모델 학습 시작 ---
    print(f"\nSeq2Seq (Offline Augmented NPY) 모델 학습을 시작합니다... (DEVICE: {DEVICE})")
    print(f"Epoch 당 배치 수: {len(train_loader)} (이전 4/4에서 증가 확인)")
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # (학습 루프는 기존과 동일)
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
            train_ppl = float('inf'); val_ppl = float('inf')

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
        model.load_state_dict(torch.load(best_model_path))
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
    try: test_ppl = math.exp(avg_test_loss)
    except OverflowError: test_ppl = float('inf')
        
    print(f"\n최종 테스트 손실 (Test Loss): {avg_test_loss:.4f}, 테스트 Perplexity (PPL): {test_ppl:.2f}")

    print(f"최고 성능 모델이 '{best_model_path}' 파일로 저장되었습니다.")

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("Vocabulary가 'vocab.pkl' 파일로 저장되었습니다.")
