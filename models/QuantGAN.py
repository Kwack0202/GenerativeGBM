from common_imports import *

from common_imports import *
import torch.nn as nn
import torch

# TemporalBlock with explicit same-padding for odd kernel_size
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        # Compute padding: for odd kernel_size, this guarantees same output length.
        padding = dilation * (kernel_size - 1) // 2  
        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding=padding)
        self.relu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding=padding)
        self.relu2 = nn.PReLU()

        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)

        # If input and output channels differ, apply a 1x1 convolution
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv1.weight, 0, 0.01)
        nn.init.normal_(self.conv2.weight, 0, 0.01)
        if self.downsample is not None:
            nn.init.normal_(self.downsample.weight, 0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

# TCN using the revised TemporalBlock
class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=80, num_layers=4, kernel_size=3):
        """
        :param input_size: 입력 feature 차원
        :param output_size: 출력 feature 차원
        :param n_hidden: 각 레이어의 은닉 채널 수
        :param num_layers: TemporalBlock의 레이어 수
        :param kernel_size: 커널 크기 (홀수여야 함)
        """
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            num_inputs = input_size if i == 0 else n_hidden
            # 일반적으로 TCN에서는 dilation factor를 지수적으로 증가시킵니다.
            dilation = 2 ** i  
            layers.append(TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation))
        self.net = nn.Sequential(*layers)
        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv.weight, 0, 0.01)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        y = self.net(x.transpose(1, 2))
        # Final 1x1 convolution and transpose back to (batch, seq_len, output_size)
        return self.conv(y).transpose(1, 2)

# QuantGenerator using TCN backbone
class QuantGenerator(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantGenerator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        # 최종 출력에 tanh 활성화 적용
        return torch.tanh(self.net(x))

# QuantDiscriminator using TCN backbone
class QuantDiscriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(QuantDiscriminator, self).__init__()
        self.net = TCN(input_size, output_size)

    def forward(self, x):
        # 최종 출력에 sigmoid 활성화 적용 (판별기)
        return torch.sigmoid(self.net(x))
