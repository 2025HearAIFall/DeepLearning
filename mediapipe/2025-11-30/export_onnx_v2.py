# export_onnx_v3.py (Bidirectional GRU 호환)
# -----------------------------------------------------------------------------
# [목표]: Bidirectional GRU 구조에 맞춰 ONNX 변환
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn
import os
import pickle
import sys
from collections import Counter

# --- 설정 ---
TRAINED_MODEL_PATH = 'model_a_mediapipe_best.pth'
VOCAB_PATH = 'vocab.pkl'
ONNX_ENCODER_PATH = 'model_a_v3_encoder.onnx' # 파일명 v3로 변경
ONNX_DECODER_PATH = 'model_a_v3_decoder.onnx' # 파일명 v3로 변경

# 학습 설정과 동일해야 함
INPUT_SIZE = (33 + 21 + 21) * 2 * 2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBED_SIZE = 256
DROPOUT_PROB = 0.5 # 학습 코드와 일치
MAX_LEN = 30
DEVICE = "cpu"

# --- Vocab 클래스 (Load용) ---
def simple_tokenizer(text): return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}; self.stoi = {v: k for k, v in self.itos.items()}
    def __len__(self): return len(self.itos)
    def build_vocab(self, s): pass 
    def numericalize(self, t): pass

# Pickle 로드를 위해 등록
setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
setattr(sys.modules['__main__'], 'simple_tokenizer', simple_tokenizer)

# --- Model Classes (Bi-GRU 반영) ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        # [수정] GRU, bidirectional=True 사용
        self.gru = nn.GRU(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout_prob, bidirectional=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_size * 2, hidden_size) 

    def forward(self, x):
        # outputs: (batch, seq_len, hidden*2), hidden: (num_layers*2, batch, hidden)
        outputs, hidden = self.gru(x)
        
        # 마지막 레이어의 양방향 hidden state를 합쳐서 단방향 hidden state로 변환
        hidden = torch.tanh(self.fc(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)))
        # hidden shape: (batch, hidden) -> Decoder에 맞게 (num_layers, batch, hidden)으로 변환
        hidden = hidden.unsqueeze(0).repeat(self.gru.num_layers, 1, 1)

        # [수정] GRU는 cell state가 없습니다.
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # [수정] Bi-GRU 출력(hidden*2) + Decoder hidden(hidden)
        self.attn = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        last_hidden = hidden[-1]
        src_len = encoder_outputs.shape[1]
        
        last_hidden = last_hidden.unsqueeze(1).repeat(1, src_len, 1)
        
        energy = torch.tanh(self.attn(torch.cat((last_hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1).unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        # [수정] GRU 사용, 입력 크기는 Context(hidden*2) + Embedding(emb_dim)
        self.gru = nn.GRU(hidden_size * 2 + emb_dim, hidden_size, num_layers, 
                          batch_first=True, dropout=dropout_prob)
        # 출력 크기: Output(hidden) + Embedded(emb) + Context(hidden*2)
        self.fc_out = nn.Linear(hidden_size + emb_dim + hidden_size * 2, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_token, hidden, encoder_outputs):
        input_token = input_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(input_token))
        
        attn_weights = self.attention(hidden, encoder_outputs)
        context = torch.bmm(attn_weights, encoder_outputs)
        
        rnn_input = torch.cat((embedded, context), dim=2)
        
        # [수정] GRU는 cell state를 반환하지 않습니다.
        output, hidden = self.gru(rnn_input, hidden) 
        
        prediction = self.fc_out(torch.cat((output.squeeze(1), embedded.squeeze(1), context.squeeze(1)), dim=1))
        
        return prediction, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device
    # Forward는 ONNX Export에 사용되지 않으므로 pass 유지

# --- Execution ---
if __name__ == '__main__':
    if not os.path.exists(VOCAB_PATH) or not os.path.exists(TRAINED_MODEL_PATH):
        print("[오류] 파일 없음. train_from_npy.py 먼저 실행하세요.")
        exit()

    with open(VOCAB_PATH, 'rb') as f: vocab = pickle.load(f)
    TARGET_VOCAB_SIZE = len(vocab)
    print(f"Vocab Size: {TARGET_VOCAB_SIZE}")

    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE)
    model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()

    # 1. Encoder Export
    # outputs: encoder_outputs (Batch, Seq_len, H*2), hidden (Num_layers, Batch, H)
    dummy_enc = torch.randn(1, MAX_LEN, INPUT_SIZE)
    torch.onnx.export(model.encoder, dummy_enc, ONNX_ENCODER_PATH,
                      input_names=['input_keypoints'], output_names=['encoder_outputs', 'hidden'],
                      # [수정] cell state 제거
                      dynamic_axes={'input_keypoints':{0:'B'}, 'encoder_outputs':{0:'B'}, 'hidden':{1:'B'}}, 
                      opset_version=11)
    
    # 2. Decoder Export
    # inputs: input_token, in_hidden, encoder_outputs
    # outputs: logits, out_hidden
    dummy_tok = torch.tensor([1], dtype=torch.long)
    dummy_h = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    # [수정] cell state 제거
    dummy_enc_out = torch.randn(1, MAX_LEN, HIDDEN_SIZE * 2) 
    
    torch.onnx.export(model.decoder, (dummy_tok, dummy_h, dummy_enc_out), ONNX_DECODER_PATH,
                      input_names=['input_token', 'in_hidden', 'encoder_outputs'],
                      output_names=['logits', 'out_hidden'],
                      # [수정] cell state 제거
                      dynamic_axes={'input_token':{0:'B'}, 'encoder_outputs':{0:'B'}, 'logits':{0:'B'}}, 
                      opset_version=11)
    
    print("ONNX 변환 완료!")
    print(f"파일 저장: {ONNX_ENCODER_PATH} 및 {ONNX_DECODER_PATH}")