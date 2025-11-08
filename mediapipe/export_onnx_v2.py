# export_to_onnx.py
# -----------------------------------------------------------------------------
# [목표]: 'train_from_npy.py'로 훈련된 'model_a_mediapipe_best.pth' 파일을
#         추론(inference)에 사용할 수 있는 ONNX 포맷으로 변환합니다.
#
# [작동 방식]:
# 1. train_from_npy.py와 동일한 모델 구조(Encoder, Decoder, Seq2Seq)를 정의합니다.
# 2. 훈련된 'model_a_mediapipe_best.pth' 가중치를 로드합니다.
# 3. 모델을 Encoder 부분과 Decoder 부분으로 분리합니다.
# 4. 'model_a_v2_encoder.onnx'와 'model_a_v2_decoder.onnx'로 저장합니다.
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn
import os
import pickle
import sys # sys.modules에 등록하기 위해 추가

# --- [1. 설정값] ---
# train_from_npy.py의 설정과 정확히 일치해야 합니다.

# [입력 파일]
TRAINED_MODEL_PATH = 'model_a_mediapipe_best.pth'
VOCAB_PATH = 'vocab.pkl' # 어휘 사전 크기를 알기 위해 로드

# [출력 파일] (inference.py, test_video...py 등이 사용하는 이름)
ONNX_ENCODER_PATH = 'model_a_v2_encoder.onnx'
ONNX_DECODER_PATH = 'model_a_v2_decoder.onnx'

# [모델 하이퍼파라미터] (train_from_npy.py와 동일하게 설정)
INPUT_SIZE = (33 + 468 + 21 + 21) * 2 * 2 # 2172
HIDDEN_SIZE = 512
NUM_LAYERS = 3
EMBED_SIZE = 256
DROPOUT_PROB = 0.6
MAX_LEN = 30 # Encoder의 더미 입력 시퀀스 길이
DEVICE = "cpu" # ONNX 변환은 CPU로 진행

# --- [2. Pickle 로드를 위한 Vocabulary 클래스 정의] ---
# (train_from_npy.py에서 복사)
from collections import Counter
def simple_tokenizer(text):
    return text.split(' ')

class Vocabulary:
    def __init__(self, tokenizer, min_freq=2):
        self.tokenizer = tokenizer
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}
        self.min_freq = min_freq
    def __len__(self): return len(self.itos)
    def build_vocab(self, sentence_list): pass
    def numericalize(self, text): pass
    @property
    def pad_idx(self): return self.stoi["<PAD>"]
    @property
    def sos_idx(self): return self.stoi["<SOS>"]
    @property
    def eos_idx(self): return self.stoi["<EOS>"]
    @property
    def unk_idx(self): return self.stoi["<UNK>"]

# [중요] Pickle 로드를 위해 __main__ 모듈에 클래스 등록
# (inference.py의 builtins 방식과 유사하게 처리)
setattr(sys.modules['__main__'], 'Vocabulary', Vocabulary)
setattr(sys.modules['__main__'], 'simple_tokenizer', simple_tokenizer)


# --- [3. 모델 클래스 정의] ---
# (train_from_npy.py에서 그대로 복사)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
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
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.attention = Attention(hidden_size)
        self.lstm = nn.LSTM(hidden_size + emb_dim, hidden_size, num_layers, dropout=dropout_prob)
        self.fc_out = nn.Linear(hidden_size * 2 + emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, input_token, hidden, cell, encoder_outputs):
        # ONNX 추론을 위해 1D 입력 (배치 크기 1)을 가정
        # (train_from_npy.py의 forward와 약간 다름 - ONNX export용)
        input_token = input_token.unsqueeze(0) # (1) -> (1, 1) [Batch, Seq]
        embedded = self.dropout(self.embedding(input_token)) # (1, 1, emb_dim)
        
        last_layer_hidden = hidden[-1] # (1, hidden_size)
        
        attn_weights = self.attention(last_layer_hidden, encoder_outputs) # (1, 1, seq_len)
        context = torch.bmm(attn_weights, encoder_outputs) # (1, 1, hidden_size)
        context = context.permute(1, 0, 2) # (1, 1, hidden_size)
        
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        prediction_input = torch.cat((output.squeeze(0), embedded.squeeze(0), context.squeeze(0)), dim=1)
        prediction = self.fc_out(prediction_input) # (1, output_dim)
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 변환 시에는 이 foward는 사용되지 않음
        pass

# --- [4. ONNX 변환 실행] ---
if __name__ == '__main__':
    
    print(f"'{VOCAB_PATH}' 로드 중...")
    try:
        # [중요] builtins가 아닌 __main__에 등록했으므로 
        # pickle.load()가 Vocabulary 클래스를 찾을 수 있음
        with open(VOCAB_PATH, 'rb') as f:
            vocab = pickle.load(f)
        TARGET_VOCAB_SIZE = len(vocab)
        print(f"어휘 사전 크기: {TARGET_VOCAB_SIZE}")
    except FileNotFoundError:
        print(f"[오류] '{VOCAB_PATH}' 파일을 찾을 수 없습니다.")
        print(">>> 'train_from_npy.py'를 먼저 실행하여 vocab.pkl을 생성해야 합니다.")
        exit()
    except Exception as e:
        print(f"[오류] {VOCAB_PATH} 로드 중 오류: {e}")
        exit()

    # 1. 전체 Seq2Seq 모델 정의 및 가중치 로드
    print("전체 Seq2Seq 모델 구조 생성 중...")
    encoder = Encoder(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    decoder = Decoder(TARGET_VOCAB_SIZE, EMBED_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT_PROB)
    model = Seq2Seq(encoder, decoder, DEVICE).to(DEVICE)

    print(f"훈련된 가중치 '{TRAINED_MODEL_PATH}' 로드 중...")
    try:
        # [수정] 훈련 로그에서 경고가 발생했으므로 weights_only=True 추가
        model.load_state_dict(torch.load(TRAINED_MODEL_PATH, map_location=DEVICE, weights_only=True))
    except FileNotFoundError:
        print(f"[오류] '{TRAINED_MODEL_PATH}' 파일을 찾을 수 없습니다.")
        print(">>> 'train_from_npy.py'를 먼저 실행하여 .pth 파일을 생성해야 합니다.")
        exit()
    except Exception as e:
        print(f"[오류] 가중치 로드 중 오류: {e}")
        print(">>> 모델 파라미터(INPUT_SIZE 등)가 훈련 때와 일치하는지 확인하세요.")
        exit()

    model.eval()
    print("가중치 로드 완료. ONNX 변환을 시작합니다...")

    # 2. Encoder 변환
    encoder_model = model.encoder
    # 더미 입력 (배치크기 1, 시퀀스 30, 입력크기 2172)
    dummy_input_encoder = torch.randn(1, MAX_LEN, INPUT_SIZE).to(DEVICE)
    
    torch.onnx.export(
        encoder_model,
        dummy_input_encoder,
        ONNX_ENCODER_PATH,
        input_names=['input_keypoints'], #
        output_names=['encoder_outputs', 'hidden', 'cell'], #
        dynamic_axes={
            'input_keypoints': {0: 'batch_size'},
            'encoder_outputs': {0: 'batch_size'},
            'hidden': {1: 'batch_size'},
            'cell': {1: 'batch_size'}
        },
        opset_version=11
    )
    print(f"✅ Encoder 변환 성공: '{ONNX_ENCODER_PATH}'")

    # 3. Decoder 변환
    decoder_model = model.decoder
    # 더미 입력 (추론 시 Decoder는 1프레임/1토큰씩 입력받음)
    dummy_input_token = torch.tensor([vocab.sos_idx], dtype=torch.long).to(DEVICE) # (1,)
    dummy_hidden = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE)
    dummy_cell = torch.randn(NUM_LAYERS, 1, HIDDEN_SIZE).to(DEVICE)
    dummy_encoder_outputs = torch.randn(1, MAX_LEN, HIDDEN_SIZE).to(DEVICE)

    torch.onnx.export(
        decoder_model,
        (dummy_input_token, dummy_hidden, dummy_cell, dummy_encoder_outputs),
        ONNX_DECODER_PATH,
        input_names=['input_token', 'in_hidden', 'in_cell', 'encoder_outputs'], #
        output_names=['logits', 'out_hidden', 'out_cell'], #
        dynamic_axes={
            'input_token': {0: 'batch_size'},
            'in_hidden': {1: 'batch_size'},
            'in_cell': {1: 'batch_size'},
            'encoder_outputs': {0: 'batch_size'},
            'logits': {0: 'batch_size'},
            'out_hidden': {1: 'batch_size'},
            'out_cell': {1: 'batch_size'}
        },
        opset_version=11
    )
    print(f"✅ Decoder 변환 성공: '{ONNX_DECODER_PATH}'")

    print("\n--- ONNX 변환 완료 ---")
    print("이제 `test_video_batch_v2.py` 또는 `inference.py`를 실행할 수 있습니다.")