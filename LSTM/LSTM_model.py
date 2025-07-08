from torch import nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.input_size = input_size # 输入
        self.hidden_size = hidden_size # 隐藏状态
        self.output_size = output_size # 输出
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)
        self.Wc = nn.Linear(input_size + hidden_size, hidden_size)  # 12 -> 10（以下标注均为隐藏层大小10，特征向量2，batch_size为1的情况）
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):# (batch_size, seq_len, features)
        batch_size = input.size(0)
        seq_len = input.size(1)
        hidden_state = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32)  # 初始化隐藏状态 (1,10)
        cell_state = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32) # 初始化细胞状态
        outputs = [] # 每个细胞的输出

        for i in range(seq_len):
            combined = torch.cat((input[:, i, :], hidden_state), dim=1)  # 输入+隐藏状态 (1,2)（时间步，特征）+(batch_size,10)=(batch_size,12)
            f_t = self.sigmoid(self.Wf(combined))  # 遗忘门 (1,10)
            i_t = self.sigmoid(self.Wi(combined)) # 输入门 (1,10)
            o_t = self.sigmoid(self.Wo(combined)) # 输出门 (1,10)
            c_hat_t = self.tanh(self.Wc(combined)) # 候选记忆 (1,10)
            # cell_state = f_t * cell_state + (1-f_t) * c_hat_t#(1,10)
            cell_state = f_t * cell_state + i_t * c_hat_t  # 更新细胞状态/长期记忆 (1,10)
            hidden_state = o_t * self.tanh(cell_state)  # 更新隐藏状态/短期记忆 (1,10)
            outputs.append(hidden_state.unsqueeze(1))  # outputs中每个元素的维度为(1,+1,10)

        outputs = torch.cat(outputs, dim=1)  # 拼接成seq_len个outputs (1,seq_len,10)
        final_output = self.output_layer(outputs)  # (1,seq_len,output_size)
        return final_output, (hidden_state, cell_state)  # hidden_state、cell_state大小没变
