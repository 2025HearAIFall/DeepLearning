# train_from_npy.py (v4.2: Bi-GRU, Relative Normalization for Accuracy)
# -----------------------------------------------------------------------------
# [변경 사항]: 
# 1. Bidirectional GRU 및 Attention 구조 반영 (성능 핵심 개선)
# 2. Keypoint 정규화 (Nose 기준) 적용 (데이터 일관성 확보)
# 3. Dropout 확률을 0.7 -> 0.5로 조정
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import pandas as pd
import numpy as np
import random
import pickle
import os 
from collections import Counter

# --- 설정 ---
TRAIN_INDEX_FILE = 'train_augmented.csv'
VAL_INDEX_FILE   = 'val_npy_index.csv'

# 하이퍼파라미터
MAX_LEN = 30
INPUT_SIZE = (33 + 21 + 21) * 2 * 2 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
MAX_TARGET_LEN = 50
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBED_SIZE = 256
DROPOUT_PROB = 0.5  # [수정] 0.7에서 0.5로 낮춰 과소적합 방지

TEACHER_FORCING_START = 1.0
TEACHER_FORCING_END = 0.0

# --- Tokenizer & Vocab ---
def simple_tokenizer(text):
    return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
        
    def __len__(self): 
        return len(self.itos)

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

# --- Dataset (Normalization & Masking 적용) ---
class SignLanguageDataset_InMemory(Dataset):
    def __init__(self, df, max_target_len=50, vocab=None, augment=False):
        self.max_target_len = max_target_len
        self.vocab = vocab
        self.augment = augment
        self.data_list = []
        
        print(f"Loading {len(df)} npy files into memory...")
        for _, row in tqdm(df.iterrows(), total=len(df)):
            npy_path = row['npy_path']
            sentence = row['sentence']
            try:
                seq = np.load(npy_path)
                self.data_list.append((seq, sentence))
            except Exception:
                pass

    # [새로 추가] 코(Nose) 기준으로 상대 좌표로 변환 (위치 무관 학습)
    def normalize_pose(self, seq):
        # seq shape: (Length, 300) -> [x1, y1, x2, y2, ..., motion_x1, motion_y1, ...]
        
        # 원본 (150D Position) 부분만 추출하여 상대 좌표 계산
        position_part = seq[:, :150].copy()
        
        # 코(Nose)의 x, y 좌표 인덱스는 0, 1
        nose_x = position_part[:, 0].reshape(-1, 1)
        nose_y = position_part[:, 1].reshape(-1, 1)
        
        # 모든 x, y 좌표에서 코의 좌표를 빼줌 (상대 좌표화)
        for i in range(0, position_part.shape[1], 2):
            position_part[:, i] -= nose_x.squeeze()
            position_part[:, i+1] -= nose_y.squeeze()
            
        # Motion part는 이미 상대적 변화량이므로 그대로 유지
        motion_part = seq[:, 150:]
        
        return np.concatenate((position_part, motion_part), axis=1)

    # 프레임 마스킹
    def apply_frame_masking(self, seq, mask_prob=0.15, max_mask_len=5):
        seq_len = seq.shape[0]
        if seq_len < 10: return seq 
        
        if np.random.rand() < mask_prob:
            t0 = np.random.randint(0, seq_len - max_mask_len)
            # 마스킹 부분 0으로 채움
            seq[t0 : t0 + max_mask_len] = 0 
        return seq
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        final_sequence, sentence_str = self.data_list[idx]
        
        # 1. 정규화 적용 (학습/검증 모두 위치 정보 제거)
        final_sequence = self.normalize_pose(final_sequence)
        
        # 2. 학습용 데이터면 마스킹 적용
        if self.augment:
            # Masking은 원본 데이터를 건드리지 않도록 복사해서 적용
            final_sequence = final_sequence.copy() 
            final_sequence = self.apply_frame_masking(final_sequence)
            
        indices = self.vocab.numericalize(sentence_str)
        indices = [self.vocab.sos_idx] + indices + [self.vocab.eos_idx]
        
        # Padding
        if len(indices) < self.max_target_len:
            indices = indices + [self.vocab.pad_idx] * (self.max_target_len - len(indices))
        else:
            indices = indices[:self.max_target_len]
            
        target_tensor = torch.tensor(indices, dtype=torch.long)
        return torch.from_numpy(final_sequence), target_tensor

# --- Models (Bidirectional GRU Architecture) ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        # [수정] LSTM -> GRU, bidirectional=True
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        # 양방향 hidden state를 합치기 위한 FC layer
        self.fc = nn.Linear(hidden_size * 2, hidden_size) 

    def forward(self, x):
        # outputs: (batch, seq_len, hidden*2)
        # hidden: (num_layers*2, batch, hidden)
        outputs, hidden = self.gru(x)
        
        # 마지막 레이어의 양방향 hidden state를 합치고 tanh 적용 (Decoder의 초기 hidden state)
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden shape: (batch, hidden) -> Decoder에 맞게 (1, batch, hidden)으로 변환
        hidden = hidden.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)

        # GRU는 cell state가 없습니다.
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # [수정] Encoder가 Bi-GRU(hidden*2)이고 Decoder hidden은 단방향(hidden)
        self.attn = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: (num_layers, batch, hidden)
        # last_hidden: (batch, hidden)
        last_hidden = hidden[-1]
        src_len = encoder_outputs.shape[1]
        
        # hidden을 sequence length만큼 복사: (batch, src_len, hidden)
        last_hidden = last_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        # energy = tanh(W_e * [hidden; encoder_outputs])
        energy = torch.tanh(self.attn(torch.cat((last_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        # [수정] LSTM -> GRU, 입력 크기는 Context(hidden*2) + Embedding(emb_dim)
        self.gru = nn.GRU(hidden_size * 2 + emb_dim, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_prob)
        # 출력 크기: Output(hidden) + Embedded(emb) + Context(hidden*2)
        self.fc_out = nn.Linear(hidden_size + emb_dim + hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, input_token, hidden, encoder_outputs):
        # input_token: (batch) -> (batch, 1)
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token)) # (batch, 1, emb_dim)
        
        # Attention 적용 (hidden[-1]은 last layer hidden state)
        attn_weights = self.attention(hidden, encoder_outputs)
        
        # context: (batch, 1, hidden*2)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        # GRU 입력: Embedding + Context
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # Decoder GRU 실행
        # output: (batch, 1, hidden)
        output, hidden = self.gru(rnn_input, hidden)
        
        # 예측 (Output + Embedded + Context)
        prediction = self.fc_out(torch.cat((output.squeeze(1), embedded.squeeze(1), context.squeeze(1)), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # [수정] Encoder는 hidden state만 반환 (GRU)
        encoder_outputs, hidden = self.encoder(src) 
        
        input_token = trg[:, 0] # Start token <SOS>
        
        for t in range(1, trg_len):
            # [수정] Decoder는 hidden state만 받음
            output, hidden = self.decoder(input_token, hidden, encoder_outputs)
            outputs[:, t, :] = output
            
            top1 = output.argmax(1) 
            use_teacher_forcing = random.random() < teacher_forcing_ratio
            input_token = trg[:, t] if use_teacher_forcing else top1
            
        return outputs

# --- Main Execution ---
if __name__ == '__main__':
    if not os.path.exists(TRAIN_INDEX_FILE):
        print(f"[오류] Train 파일 없음: {TRAIN_INDEX_FILE}")
        exit()
    if not os.path.exists(VAL_INDEX_FILE):
        print(f"[오류] Val 파일 없음: {VAL_INDEX_FILE}")
        exit()

    train_df = pd.read_csv(TRAIN_INDEX_FILE)
    val_df = pd.read_csv(VAL_INDEX_FILE)

    print("Vocabulary 구축 (Train 기준)...")
    vocab = Vocabulary(simple_tokenizer, min_freq=2)
    vocab.build_vocab(train_df['sentence'].tolist())
    
    # Train셋만 augment=True
    train_dataset = SignLanguageDataset_InMemory(train_df, MAX_TARGET_LEN, vocab, augment=True)
    if len(val_df) > 0:
        val_dataset = SignLanguageDataset_InMemory(val_df, MAX_TARGET_LEN, vocab, augment=False)
    else:
        val_dataset = train_dataset

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Vocab Size: {len(vocab)}")
    
    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    decoder = Decoder(len(vocab), EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    
    # Label Smoothing 적용
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx, label_smoothing=0.1)
    
    # Weight Decay 적용
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_loss = float('inf')
    best_model_path = 'model_a_mediapipe_best.pth'

    print(f"\n🚀 Improved Training Started (Epochs: {NUM_EPOCHS})")

    for epoch in range(NUM_EPOCHS):
        current_ratio = max(0.0, TEACHER_FORCING_START - (epoch / NUM_EPOCHS))

        model.train()
        train_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Ep {epoch+1} (TF:{current_ratio:.2f})")
        for k, t in loop:
            k, t = k.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            out = model(k, t, teacher_forcing_ratio=current_ratio)
            
            loss = criterion(out[:,1:].reshape(-1, out.shape[-1]), t[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for k, t in valid_loader:
                k, t = k.to(DEVICE), t.to(DEVICE)
                # Teacher Forcing 0.0으로 설정
                out = model(k, t, teacher_forcing_ratio=0.0) 
                loss = criterion(out[:,1:].reshape(-1, out.shape[-1]), t[:,1:].reshape(-1))
                val_loss += loss.item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(valid_loader)
        
        print(f"   Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            print(f"   ✅ Best Model Saved ({best_val_loss:.4f})")

    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    print("\n완료.")