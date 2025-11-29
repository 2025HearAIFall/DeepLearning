# export_onnx_v2.py
# -----------------------------------------------------------------------------
# [목표]: model_a_mediapipe_best.pth -> ONNX 변환
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
ONNX_ENCODER_PATH = 'model_a_v2_encoder.onnx'
ONNX_DECODER_PATH = 'model_a_v2_decoder.onnx'

INPUT_SIZE = (33 + 21 + 21) * 2 * 2
HIDDEN_SIZE = 256
NUM_LAYERS = 2
EMBED_SIZE = 256
DROPOUT_PROB = 0.7
MAX_LEN = 30
DEVICE = "cpu"

# --- Vocab 클래스 (Load용) ---
def simple_tokenizer(text): return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer; self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}; self.stoi = {v: k for k, v in self.itos.items()}
    def __len__(self): return len(self.itos)
    # 아래 메서드들은 pickle 로드시 필요하므로 정의 유지
    def build_vocab(self, s): pass 
    def numericalize(self, t): pass
    @property
    def pad_idx(self): return self.stoi["<PAD>"]
    @property
    def sos_idx(self): return self.stoi["<SOS>"]
    @property
    def eos_idx(self): return self.stoi["<EOS>"]
    @property
    def unk_idx(self): return self.stoi["<UNK>"]

# Pickle 로드를 위해 등록
setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
setattr(sys.modules['__main__'], 'simple_tokenizer', simple_tokenizer)

# --- Model Classes ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob)
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
        output, (hidden, cell) = self.lstm(torch.cat((embedded, context), dim=2), (hidden, cell))
        prediction = self.fc_out(torch.cat((output.squeeze(0), embedded.squeeze(0), context.squeeze(0)), dim=1))
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder; self.decoder = decoder; self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5): pass

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

    # Encoder Export
    dummy_enc = torch.randn(1, MAX_LEN, INPUT_SIZE)
    torch.onnx.export(model.encoder, dummy_enc, ONNX_ENCODER_PATH,
                      input_names=['input_keypoints'], output_names=['encoder_outputs', 'hidden', 'cell'],
                      dynamic_axes={'input_keypoints':{0:'B'}, 'encoder_outputs':{0:'B'}, 'hidden':{1:'B'}, 'cell':{1:'B'}}, opset_version=11)
    
    # Decoder Export
    dummy_tok = torch.tensor([1], dtype=torch.long)
    dummy_h = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    dummy_c = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE)
    dummy_enc_out = torch.randn(1, MAX_LEN, HIDDEN_SIZE)
    
    torch.onnx.export(model.decoder, (dummy_tok, dummy_h, dummy_c, dummy_enc_out), ONNX_DECODER_PATH,
                      input_names=['input_token', 'in_hidden', 'in_cell', 'encoder_outputs'],
                      output_names=['logits', 'out_hidden', 'out_cell'],
                      dynamic_axes={'input_token':{0:'B'}, 'encoder_outputs':{0:'B'}, 'logits':{0:'B'}}, opset_version=11)
    
    print("ONNX 변환 완료!")