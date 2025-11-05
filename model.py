# model.py (수정: Seq2Seq + Attention)

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class Encoder(nn.Module):
    """ 수어 키포인트 시퀀스를 인코딩합니다. """
    def __init__(self, input_size, hidden_size, num_layers, dropout_prob):
        super(Encoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            bidirectional=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_size * 2) 💡 어텐션을 위해 반환
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # cell: (num_layers * 2, batch_size, hidden_size)
        
        # 양방향 LSTM의 마지막 hidden/cell state를 Decoder의 초기 state로 변환
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()

        hidden = hidden.view(hidden.size(0), self.num_layers, self.hidden_size * 2)
        cell = cell.view(cell.size(0), self.num_layers, self.hidden_size * 2)
        
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()

        # 💡 (encoder_outputs, hidden, cell) 반환
        return lstm_out, hidden, cell

class Attention(nn.Module):
    """ 💡 Bahdanau (Additive) Attention """
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        
        self.hidden_size = hidden_size
        # 디코더 은닉 상태, 인코더 은닉 상태, V (에너지 계산용)
        self.W_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_encoder = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.V = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden: (num_layers, batch_size, hidden_size)
        # encoder_outputs: (batch_size, src_len, hidden_size * 2)
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        
        # 💡 디코더의 최상위 계층 은닉 상태만 사용
        # (batch_size, hidden_size) -> (batch_size, 1, hidden_size)
        decoder_hidden = decoder_hidden[-1].unsqueeze(1) 

        # (batch_size, 1, hidden_size) -> (batch_size, src_len, hidden_size)
        hidden_expanded = self.W_hidden(decoder_hidden).repeat(1, src_len, 1)
        
        # (batch_size, src_len, hidden_size * 2) -> (batch_size, src_len, hidden_size)
        encoder_features = self.W_encoder(encoder_outputs)
        
        # 💡 에너지 계산
        # (batch_size, src_len, hidden_size)
        energy = torch.tanh(hidden_expanded + encoder_features)
        
        # (batch_size, src_len, 1) -> (batch_size, src_len)
        attention_scores = self.V(energy).squeeze(2)
        
        # (batch_size, src_len)
        weights = F.softmax(attention_scores, dim=1)
        
        # 💡 문맥 벡터 계산
        # weights: (batch_size, 1, src_len)
        # encoder_outputs: (batch_size, src_len, hidden_size * 2)
        # context: (batch_size, 1, hidden_size * 2)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)
        
        return context, weights

class Decoder(nn.Module):
    """ Encoder의 context vector를 받아 문장을 디코딩합니다. (Attention 포함) """
    def __init__(self, output_size, embed_size, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        
        self.output_size = output_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(output_size, embed_size)
        
        # 💡 어텐션 모듈 추가
        self.attention = Attention(hidden_size)
        
        self.lstm = nn.LSTM(
            embed_size + (hidden_size * 2), # 💡 입력: (Embedding + Context Vector)
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        # 💡 출력층 입력: (LSTM Hidden + Context + Embedding)
        self.fc_out = nn.Linear(
            hidden_size + (hidden_size * 2) + embed_size, 
            output_size
        )
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_input, hidden, cell, encoder_outputs):
        # x_input: (batch_size, 1) - 이전 스텝의 예측 단어
        # hidden, cell: (num_layers, batch_size, hidden_size) - 이전 스텝의 state
        # encoder_outputs: (batch_size, src_len, hidden_size * 2) - 💡 인코더 전체 출력
        
        # (batch_size, 1) -> (batch_size, 1, embed_size)
        embedded = self.dropout(self.embedding(x_input))
        
        # 💡 1. 어텐션 계산 (이전 hidden state와 인코더 출력 사용)
        # context: (batch_size, 1, hidden_size * 2)
        context, attn_weights = self.attention(hidden, encoder_outputs)
        
        # 💡 2. LSTM 입력 준비 (임베딩 + 문맥 벡터)
        # (batch_size, 1, embed_size + hidden_size * 2)
        lstm_input = torch.cat((embedded, context), dim=2)
        
        # 💡 3. LSTM 실행
        # output: (batch_size, 1, hidden_size)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden, cell))
        
        # 💡 4. 최종 예측 (3가지 정보 결합)
        output = output.squeeze(1)     # (batch_size, hidden_size)
        context = context.squeeze(1)   # (batch_size, hidden_size * 2)
        embedded = embedded.squeeze(1) # (batch_size, embed_size)
        
        # (batch_size, hidden_size + hidden_size*2 + embed_size)
        prediction_input = torch.cat((output, context, embedded), dim=1)
        
        # (batch_size, output_size)
        prediction = self.fc_out(prediction_input)
        
        # (batch_size, output_size), (num_layers, batch_size, hidden_size) * 2
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """ Encoder와 Decoder를 결합하는 Seq2Seq (Attention) 모델 """
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_seq, trg_seq, teacher_forcing_ratio=0.5):
        # src_seq: (batch_size, src_len, input_size)
        # trg_seq: (batch_size, trg_len)
        
        batch_size = trg_seq.shape[0]
        trg_len = trg_seq.shape[1]
        trg_vocab_size = self.decoder.output_size
        
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Encoder
        # 💡 encoder_outputs 추가
        encoder_outputs, hidden, cell = self.encoder(src_seq)
        
        # 2. Decoder
        trg_input = trg_seq[:, 0].unsqueeze(1) # <SOS> 토큰
        
        for t in range(1, trg_len):
            # 💡 encoder_outputs를 매 스텝 전달
            output, hidden, cell = self.decoder(trg_input, hidden, cell, encoder_outputs)
            
            outputs[:, t, :] = output
            
            use_teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            trg_input = (trg_seq[:, t] if use_teacher_force else top1).unsqueeze(1)
            
        return outputs
