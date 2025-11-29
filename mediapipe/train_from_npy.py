# train_from_npy.py (v4.1: __len__ 오류 수정 및 성능 개선 포함)
# -----------------------------------------------------------------------------
# [개선 사항]: 
# 1. Label Smoothing, Weight Decay, Frame Masking 적용
# 2. Vocabulary 클래스 __len__ 메서드 복구
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
NUM_EPOCHS = 80
MAX_TARGET_LEN = 50
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBED_SIZE = 256
DROPOUT_PROB = 0.6  # 과적합 방지를 위해 상향

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
        
    # [🔥 복구된 부분] 이 메서드가 없어서 에러가 났었습니다.
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

# --- Dataset (Masking 적용) ---
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

    # 프레임 마스킹 (중간중간 가리기)
    def apply_frame_masking(self, seq, mask_prob=0.15, max_mask_len=5):
        seq_len = seq.shape[0]
        if seq_len < 10: return seq 
        
        if np.random.rand() < mask_prob:
            t0 = np.random.randint(0, seq_len - max_mask_len)
            seq[t0 : t0 + max_mask_len] = 0 
        return seq
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        final_sequence, sentence_str = self.data_list[idx]
        
        # 학습용 데이터면 마스킹 적용
        if self.augment:
            final_sequence = final_sequence.copy()
            final_sequence = self.apply_frame_masking(final_sequence)
            
        indices = self.vocab.numericalize(sentence_str)
        indices = [self.vocab.sos_idx] + indices + [self.vocab.eos_idx]
        
        if len(indices) < self.max_target_len:
            indices = indices + [self.vocab.pad_idx] * (self.max_target_len - len(indices))
        else:
            indices = indices[:self.max_target_len]
            
        target_tensor = torch.tensor(indices, dtype=torch.long)
        return torch.from_numpy(final_sequence), target_tensor

# --- Models ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
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
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        return torch.softmax(self.v(energy).squeeze(2), dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + emb_dim, hidden_size, num_layers, dropout=dropout_prob)
        self.fc_out = nn.Linear(hidden_size * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, input_token, hidden, cell, encoder_outputs):
        input_token = input_token.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_token))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs).permute(1, 0, 2)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(rnn_input, (hidden, cell))
        
        prediction = self.fc_out(torch.cat((output.squeeze(0), embedded.squeeze(0), context.squeeze(0)), dim=1))
        return prediction, hidden, cell

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
        encoder_outputs, hidden, cell = self.encoder(src)
        
        input_token = trg[:, 0]
        
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
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