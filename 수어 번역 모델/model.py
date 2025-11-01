# model.py (수정)

import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    # 💡 [수정] dropout_prob 기본값을 0.6으로 변경
    def __init__(self, input_size, hidden_size, num_layers, num_classes, dropout_prob=0.6):
        super(LSTMClassifier, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            bidirectional=True
        )
        
        self.dropout = nn.Dropout(dropout_prob)
        # 💡 [수정] LSTM이 양방향(bidirectional=True)이므로 fc 레이어 입력은 hidden_size * 2 입니다.
        #    이 부분은 기존 코드와 동일하게 유지합니다.
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # 💡 [수정] 마지막 시점 대신, 모든 시점의 출력을 평균(Average Pooling)합니다.
        # (batch_size, seq_len, hidden_size * 2) -> (batch_size, hidden_size * 2)
        mean_pool_out = torch.mean(lstm_out, dim=1)
        
        out = self.dropout(mean_pool_out)
        out = self.fc(out)
        return out