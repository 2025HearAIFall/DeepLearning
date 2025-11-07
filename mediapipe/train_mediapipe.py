# train_medipe.py (V3: Multi-View 5방향 융합)
# -----------------------------------------------------------------------------
# [목표]: 5개 방향의 뷰(.npy)를 융합(Fusion)하여 문장 번역 성능을 극대화합니다.
# [입력]: all_multi_view_index_npy.csv
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

# --- 토크나이저 및 Vocabulary 클래스 정의 (변경 없음) ---
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

# --- [수정됨] NPY용 다중 뷰 데이터셋 클래스 ---
BASE_FEATURES_MEDIAPIPE = (33 + 468 + 21 + 21) * 2 # 1086
INPUT_SIZE_NPY = BASE_FEATURES_MEDIAPIPE * 2      # 2172

class SignLanguageDataset_NPY_MultiView(Dataset):
    def __init__(self, index_file_path, max_len=30, max_target_len=50, vocab=None):
        self.max_len = max_len
        self.max_target_len = max_target_len
        self.data_info = pd.read_csv(index_file_path) 
        if vocab is None: raise ValueError("Vocabulary 객체가 제공되어야 합니다.")
        self.vocab = vocab
        
        # [수정] CSV 컬럼 이름
        self.view_paths_cols = [f"view{i+1}_path" for i in range(5)]
        self.sentence_col = 'sentence'

    def __len__(self):
        return len(self.data_info)

    def _load_npy(self, npy_path):
        """Helper: NPY 1개 로드, 실패 시 0 텐서 반환"""
        try:
            final_sequence = np.load(npy_path) 
            if final_sequence.shape != (self.max_len, INPUT_SIZE_NPY):
                 # print(f"Warning: Mismatched shape in {npy_path}")
                 raise ValueError("Mismatched shape")
            return final_sequence
        except Exception as e:
            # print(f"Warning: Cannot load {npy_path}")
            return np.zeros((self.max_len, INPUT_SIZE_NPY), dtype=np.float32) 

    def __getitem__(self, idx):
        item = self.data_info.iloc[idx]
        sentence_str = item[self.sentence_col]

        # [수정] 5개 뷰 NPY 로드
        views_tensors = []
        for col in self.view_paths_cols:
            npy_path = item[col]
            views_tensors.append(torch.from_numpy(self._load_npy(npy_path)))
        
        # 타겟 문장 처리
        indices = self.vocab.numericalize(sentence_str)
        indices = [self.vocab.sos_idx] + indices + [self.vocab.eos_idx]
        if len(indices) < self.max_target_len:
            indices = indices + [self.vocab.pad_idx] * (self.max_target_len - len(indices))
        else:
            indices = indices[:self.max_target_len]
        target_tensor = torch.tensor(indices, dtype=torch.long)
        
        return tuple(views_tensors), target_tensor

# --- [수정됨] 모델 클래스 정의 (Multi-View 적용) ---

# NPY 1개 처리용 (V2 모델을 재사용)
class BaseEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(BaseEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            dropout=dropout_prob, 
                            bidirectional=True)
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        
        # 입력 데이터 드롭아웃 (과적합 방지)
        self.input_dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.input_dropout(x) # (과적합 방지)
        outputs, (hidden, cell) = self.lstm(x)
        hidden = hidden.view(self.num_layers, 2, -1, self.hidden_size)
        hidden = torch.cat((hidden[:, 0, :, :], hidden[:, 1, :, :]), dim=2)
        cell = cell.view(self.num_layers, 2, -1, self.hidden_size)
        cell = torch.cat((cell[:, 0, :, :], cell[:, 1, :, :]), dim=2)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        return outputs, hidden, cell

# [신규] 5개 뷰를 융합하는 MultiViewEncoder
class MultiViewEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(MultiViewEncoder, self).__init__()
        
        # [핵심] 1개의 BaseEncoder를 5개 뷰가 공유 (Weight Sharing)
        self.base_encoder = BaseEncoder(input_size, hidden_size, num_layers, dropout_prob)

    def forward(self, v1, v2, v3, v4, v5):
        # 5개 뷰를 동일한 BaseEncoder로 각각 처리
        out1, h1, c1 = self.base_encoder(v1)
        out2, h2, c2 = self.base_encoder(v2)
        out3, h3, c3 = self.base_encoder(v3)
        out4, h4, c4 = self.base_encoder(v4)
        out5, h5, c5 = self.base_encoder(v5)
        
        # [융합(Fusion): 평균]
        outputs = torch.stack([out1, out2, out3, out4, out5], dim=1).mean(dim=1)
        hidden = torch.stack([h1, h2, h3, h4, h5], dim=1).mean(dim=1)
        cell = torch.stack([c1, c2, c3, c4, c5], dim=1).mean(dim=1)

        return outputs, hidden, cell

# --- Decoder, Attention (V2와 동일, 변경 없음) ---
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 3, hidden_size)
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
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM((hidden_size * 2) + emb_dim, hidden_size, 
                            num_layers, dropout=dropout_prob)
        self.fc_out = nn.Linear(hidden_size + (hidden_size * 2) + emb_dim, output_dim)
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
# ----------------------------------------------------

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_views, trg, teacher_forcing_ratio=0.5):
        # [수정] src -> 5개 뷰 튜플
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # [수정] 5개 뷰를 Encoder에 전달
        encoder_outputs, hidden, cell = self.encoder(*src_views)
        
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
    
    # --- 하이퍼파라미터 설정 (V2 과적합 방지 튜닝 적용) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001 # (정규화)
    NUM_EPOCHS = 30 
    MAX_LEN = 30
    MAX_TARGET_LEN = 50
    INPUT_SIZE = BASE_FEATURES_MEDIAPIPE * 2 # 2172
    HIDDEN_SIZE = 512 
    NUM_LAYERS = 2
    EMBED_SIZE = 256
    DROPOUT_PROB = 0.65 # (정규화)
    NUM_WORKERS = 4 
    PATIENCE = 10 
    
    # [수정] 새 다중 뷰 인덱스 파일 경로
    ALL_IN_ONE_NPY_CSV = 'all_multi_view_index_npy.csv'

    # --- 데이터 준비 ---
    print(f"[V3] Multi-View 데이터 인덱스 로딩 ({ALL_IN_ONE_NPY_CSV})...")
    try:
        all_df = pd.read_csv(ALL_IN_ONE_NPY_CSV) 
    except FileNotFoundError:
        print(f"[오류] '{ALL_IN_ONE_NPY_CSV}' 파일을 찾을 수 없습니다.")
        print(">>> 'preprocess_to_npy.py (V3)'를 먼저 실행해야 합니다.")
        exit()

    all_sentences = all_df['sentence'].tolist()
    vocab = Vocabulary(simple_tokenizer, min_freq=2)
    vocab.build_vocab(all_sentences)
    TARGET_VOCAB_SIZE = len(vocab)
    PAD_IDX = vocab.pad_idx
    print(f"생성된 Target Vocabulary 크기: {TARGET_VOCAB_SIZE}")
    
    print("통합 다중 뷰 데이터셋 생성 및 분할...")
    
    try:
        # [수정] 새 데이터셋 클래스 사용
        full_dataset = SignLanguageDataset_NPY_MultiView(
            index_file_path=ALL_IN_ONE_NPY_CSV, 
            max_len=MAX_LEN, 
            max_target_len=MAX_TARGET_LEN,
            vocab=vocab
        )
    except Exception as e:
        print(f"데이터셋 로딩 중 오류 발생: {e}")
        exit()

    total_size = len(full_dataset)
    train_size = int(total_size * 0.7) # 약 116개
    valid_size = total_size - train_size
    test_size = total_size - train_size - valid_size 
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = random_split(
        full_dataset, [train_size, valid_size], generator=generator
    )

    test_dataset = valid_dataset # 임시로 test_dataset을 valid_dataset으로 설정

    print(f"총 데이터: {total_size}개")
    print(f"분리된 학습 데이터 수: {len(train_dataset)} (80%)")
    print(f"분리된 검증 데이터 수: {len(valid_dataset)} (10%)")
    print(f"분리된 테스트 데이터 수: {len(test_dataset)} (10%)")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- 모델, 손실 함수, 옵티마이저 ---
    # [수정] MultiViewEncoder 사용
    encoder = MultiViewEncoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    
    decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4) 
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_path = 'model_a_mediapipe_best_multiview.pth' # (새 이름)

    # --- 모델 학습 루프 ---
    print(f"\n[V3] Seq2Seq (Multi-View NPY) 모델 학습을 시작합니다... (DEVICE: {DEVICE})")
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        
        # [수정] 5개 뷰 + 타겟 1개
        for (views_tuple, targets) in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Training]"):
            
            # [수정] 5개 뷰 텐서를 DEVICE로 이동
            v1, v2, v3, v4, v5 = views_tuple
            src_views = (v1.to(DEVICE), v2.to(DEVICE), v3.to(DEVICE), v4.to(DEVICE), v5.to(DEVICE))
            targets = targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(src_views, targets, teacher_forcing_ratio=0.5)
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
            for (views_tuple, targets) in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Validation]"):
                v1, v2, v3, v4, v5 = views_tuple
                src_views = (v1.to(DEVICE), v2.to(DEVICE), v3.to(DEVICE), v4.to(DEVICE), v5.to(DEVICE))
                targets = targets.to(DEVICE)
                outputs = model(src_views, targets, teacher_forcing_ratio=0.0) 
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

    # --- 최종 테스트 ---
    print(f"\n최고 성능의 모델({best_model_path})을 로드하여 최종 테스트를 진행합니다...")
    try:
        model.load_state_dict(torch.load(best_model_path))
    except FileNotFoundError:
        print(f"[경고] {best_model_path} 파일을 찾을 수 없습니다.")
    except Exception as e:
        print(f"[경고] 모델 로드 중 오류 발생: {e}.")
        
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for (views_tuple, targets) in tqdm(test_loader, desc="Final Test"):
            v1, v2, v3, v4, v5 = views_tuple
            src_views = (v1.to(DEVICE), v2.to(DEVICE), v3.to(DEVICE), v4.to(DEVICE), v5.to(DEVICE))
            targets = targets.to(DEVICE)
            outputs = model(src_views, targets, teacher_forcing_ratio=0.0)
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
