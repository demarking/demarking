import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset


# 定义训练自编码器的数据集
class MyDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]  # 0是one-hot向量，1是ipd

    def __len__(self):
        return len(self.data)


# 定义训练鉴别器的数据集
class DisDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)


# 定义鉴别器结构
class Discriminator(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 400), nn.ReLU(),
                                 nn.Linear(400, 1), nn.Sigmoid())

    def forward(self, input_data):
        return self.net(input_data)


class Discriminator2(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 4096), nn.ReLU(),
                                 nn.Linear(4096, 512), nn.ReLU(),
                                 nn.Linear(512, 1), nn.Sigmoid())

    def forward(self, input_data):
        return self.net(input_data)


# 定义编码器结构
class Encoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 1000), nn.ReLU(),
                                     nn.Linear(1000, 2000), nn.ReLU(),
                                     nn.Linear(2000, 2000), nn.ReLU(),
                                     nn.Linear(2000, output_dim), nn.ReLU())

    def forward(self, input_data):
        return self.encoder(input_data)


class Encoder2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.encoder = nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU(),
                                     nn.Linear(500, 2000), nn.ReLU(),
                                     nn.Linear(2000, output_dim), nn.ReLU())

    def forward(self, input_data):
        return self.encoder(input_data)


# 定义解码器结构
class Decoder(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        kernel_size = 10
        length = 10 * (input_dim - (kernel_size - 1) - (kernel_size - 1))
        self.decoder = nn.Sequential(nn.Conv1d(1, 50, kernel_size, 1),
                                     nn.ReLU(),
                                     nn.Conv1d(50, 10, kernel_size, 1),
                                     nn.ReLU(), nn.Flatten(),
                                     nn.Linear(length, 256), nn.ReLU(),
                                     nn.Linear(256, output_dim))

    def forward(self, input_data):
        return self.decoder(input_data)


class Decoder2(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.decoder = nn.Sequential(nn.Linear(input_dim, 1000), nn.ReLU(),
                                     nn.Linear(1000, 3000), nn.ReLU(),
                                     nn.Linear(3000, output_dim))

    def forward(self, input_data):
        return self.decoder(input_data)


# 定义自编码器结构
class AE(nn.Module):

    def __init__(self, input_dim, flow_length, device):
        super().__init__()
        self.encoder = Encoder(input_dim, flow_length)
        self.decoder = Decoder(flow_length, input_dim)
        self.device = device

    def forward(self, input_data, ipd):

        delay = self.encoder(input_data)
        code = delay + ipd

        noise = torch.FloatTensor(np.random.laplace(5, 3, code.shape)).to(
            self.device)
        code = code + noise
        code = code.unsqueeze(1)
        output = self.decoder(code)

        return delay, code.squeeze(1), output


class AE2(nn.Module):

    def __init__(self, input_dim, flow_length, device):
        super().__init__()
        self.encoder = Encoder2(input_dim, flow_length)
        self.decoder = Decoder2(flow_length, input_dim)
        self.device = device

    def forward(self, input_data, ipd):

        delay = self.encoder(input_data)
        code = delay + ipd

        noise = torch.FloatTensor(np.random.laplace(5, 3, code.shape)).to(
            self.device)
        code = code + noise
        output = self.decoder(code)

        return delay, code, output


# 定义Encoder和Decoder联合训练的损失函数
class AELoss(nn.Module):

    def __init__(self, device, encoder_w=1.0, decoder_w=1.0):
        super().__init__()
        self.device = device
        self.encoder_w = encoder_w
        self.decoder_w = decoder_w

    def forward(self, input_data, code, output_data):
        label = torch.zeros(code.size()).to(self.device)
        encoder_loss = nn.functional.l1_loss(code, label)
        decoder_loss = nn.functional.cross_entropy(output_data, input_data)
        return self.encoder_w * encoder_loss + self.decoder_w * decoder_loss


class AELoss2(nn.Module):

    def __init__(self, device, encoder_w=1.0, decoder_w=1.0):
        super().__init__()
        self.device = device
        self.encoder_w = encoder_w
        self.decoder_w = decoder_w

    def forward(self, input_data, code, output_data):
        label = torch.zeros(code.size()).to(self.device)
        encoder_loss = nn.functional.l1_loss(code, label)
        decoder_loss = nn.functional.cross_entropy(output_data, input_data)
        return self.encoder_w * encoder_loss + self.decoder_w * decoder_loss


# 定义防御变换的结构
class Defence(nn.Module):

    def __init__(self, input_dim, max_mu, min_mu, sigma):
        super().__init__()
        self.max_mu = max_mu
        self.min_mu = min_mu
        self.sigma = sigma
        self.defence = nn.Sequential(nn.Linear(input_dim, 1024),
                                     nn.LeakyReLU(), nn.Linear(1024, 2048),
                                     nn.LeakyReLU(), nn.Linear(2048, 512),
                                     nn.LeakyReLU(), nn.Linear(512, input_dim),
                                     nn.LeakyReLU())

    def remapping(self, ipd):
        ipd_std = torch.std(ipd, dim=1, keepdim=True)
        sigma = torch.full_like(ipd_std, self.sigma)
        new_ipd = ipd * (torch.min(ipd_std, sigma) / ipd_std)

        ipd_mean = torch.mean(new_ipd, dim=1, keepdim=True)
        mu_maxn = torch.max(ipd_mean - self.max_mu, torch.zeros_like(ipd_mean))
        mu_minn = torch.min(ipd_mean - self.min_mu, torch.zeros_like(ipd_mean))
        new_ipd = new_ipd - mu_maxn - mu_minn

        return new_ipd

    def forward(self, input_data):
        ipd = self.defence(input_data)
        return self.remapping(ipd)


# 定义defence的损失函数
class DefLoss(nn.Module):

    def __init__(self, batch_size, w1=1.0, w2=1.0):
        super().__init__()
        self.batch_size = batch_size
        self.w1 = w1
        self.w2 = w2

    def forward(self, score, input1, input2):
        label = torch.zeros_like(score)
        loss1 = nn.functional.l1_loss(score, label)

        target = torch.full((self.batch_size, ), -1).to(input1.device)
        loss2 = nn.functional.cosine_embedding_loss(input1, input2, target)
        return self.w1 * loss1 + self.w2 * loss2
