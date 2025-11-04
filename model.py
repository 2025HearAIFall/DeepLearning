# model.py (수정: Seq2Seq Encoder-Decoder)

import torch
import torch.nn as nn
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
            bidirectional=True, # 💡 양방향 LSTM
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        
        # 💡 양방향 LSTM의 hidden/cell을 단방향 Decoder에 맞게 변환
        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # lstm_out: (batch_size, seq_len, hidden_size * 2)
        # hidden: (num_layers * 2, batch_size, hidden_size)
        # cell: (num_layers * 2, batch_size, hidden_size)
        
        # 💡 양방향 LSTM의 마지막 hidden/cell state를 Decoder의 초기 state로 사용
        # (num_layers * 2, batch_size, hidden_size) -> (num_layers, batch_size, hidden_size * 2)
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()

        # (batch_size, num_layers * 2, hidden_size) -> (batch_size, num_layers, hidden_size * 2)
        hidden = hidden.view(hidden.size(0), self.num_layers, self.hidden_size * 2)
        cell = cell.view(cell.size(0), self.num_layers, self.hidden_size * 2)
        
        # (batch_size, num_layers, hidden_size * 2) -> (batch_size, num_layers, hidden_size)
        hidden = torch.tanh(self.fc_hidden(hidden))
        cell = torch.tanh(self.fc_cell(cell))
        
        # (batch_size, num_layers, hidden_size) -> (num_layers, batch_size, hidden_size)
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = cell.permute(1, 0, 2).contiguous()

        # 💡 Decoder로 전달될 context vector
        # (num_layers, batch_size, hidden_size)
        return hidden, cell

class Decoder(nn.Module):
    """ Encoder의 context vector를 받아 문장을 디코딩합니다. """
    def __init__(self, output_size, embed_size, hidden_size, num_layers, dropout_prob):
        super(Decoder, self).__init__()
        
        self.output_size = output_size # Target vocab size
        
        self.embedding = nn.Embedding(output_size, embed_size)
        
        self.lstm = nn.LSTM(
            embed_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            # 💡 Decoder는 단방향
            dropout=dropout_prob if num_layers > 1 else 0
        )
        
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x_input, hidden, cell):
        # x_input: (batch_size, 1) - 이전 스텝의 예측 단어
        # hidden, cell: (num_layers, batch_size, hidden_size) - 이전 스텝의 state
        
        # (batch_size, 1) -> (batch_size, 1, embed_size)
        embedded = self.dropout(self.embedding(x_input))
        
        # (batch_size, 1, embed_size) -> (batch_size, 1, hidden_size)
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        
        # (batch_size, 1, hidden_size) -> (batch_size, output_size)
        prediction = self.fc_out(output.squeeze(1))
        
        # (batch_size, output_size), (num_layers, batch_size, hidden_size) * 2
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """ Encoder와 Decoder를 결합하는 Seq2Seq 모델 """
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
        
        # Decoder 출력을 저장할 텐서
        # (batch_size, trg_len, trg_vocab_size)
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)
        
        # 1. Encoder
        hidden, cell = self.encoder(src_seq)
        
        # 2. Decoder
        # 첫 번째 입력은 <SOS> 토큰
        # (batch_size, 1)
        trg_input = trg_seq[:, 0].unsqueeze(1)
        
        for t in range(1, trg_len): # <SOS> 다음 단어부터 예측
            output, hidden, cell = self.decoder(trg_input, hidden, cell)
            
            # (batch_size, trg_vocab_size)
            outputs[:, t, :] = output
            
            # Teacher Forcing: 학습 시 실제 타겟 단어를 다음 입력으로 사용
            use_teacher_force = random.random() < teacher_forcing_ratio
            
            # 가장 확률이 높은 단어
            top1 = output.argmax(1)
            
            trg_input = (trg_seq[:, t] if use_teacher_force else top1).unsqueeze(1)
            
        return outputs