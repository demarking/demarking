# 数学操作
import math
import numpy as np

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 读写数据
import pickle

# 可视化
from tqdm import tqdm

# 引用模型结构
from structure import *
from tool_funcs import *


def train(train_loader):
    # 加载自编码器
    ae = torch.load(model_path + f"10bits_black{flow_length}.pth").to(device)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False

    defence = Defence(flow_length, 30, 60, 20).to(device)
    discriminator = Discriminator2(flow_length).to(device)

    def_criterion = DefLoss(batch_size)
    dis_criterion = nn.L1Loss()

    def_opt = torch.optim.Adam(defence.parameters(), learning_rate)
    dis_opt = torch.optim.Adam(discriminator.parameters(), learning_rate)

    loss_per_epoch = []

    for epoch in range(n_epochs):
        defence.train()
        loss_record = []

        train_pbar = tqdm(train_loader)
        for x, ipd in train_pbar:
            # Prepare data for discriminator
            laplace_data = torch.FloatTensor(
                np.random.laplace(40, 5, (batch_size, flow_length))).to(device)
            fake_label = torch.ones((batch_size, 1)).to(device)
            true_label = torch.zeros((batch_size, 1)).to(device)

            # 生成带水印的流
            x, ipd = x.to(device), ipd.to(device)
            delay, water_flow, label = ae(x, ipd)
            label = nn.functional.softmax(label, dim=1)

            new_ipd = defence(water_flow)  # Defense Transformation
            score = discriminator(new_ipd)  # Differentiate

            # y = ae.decoder(new_ipd.unsqueeze(1))  # 由于白盒解码器使用了CNN，这里需要增加一个维度
            y = ae.decoder(new_ipd)  # 黑盒模型没有使用CNN，不需要增加一个维度
            y = torch.softmax(y, dim=1)

            def_opt.zero_grad()
            loss = def_criterion(score, y, label)  # label是不经过defence变换直接通过解码器的
            loss.backward()
            def_opt.step()
            loss_record.append(loss.detach().item())

            # 训练鉴别器
            dis_opt.zero_grad()
            ipd_score = discriminator(new_ipd.detach())
            lap_score = discriminator(laplace_data)
            dis_loss = (dis_criterion(ipd_score, fake_label) +
                        dis_criterion(lap_score, true_label)) / 2
            dis_loss.backward()
            dis_opt.step()

            train_pbar.set_description(f'Epoch [{epoch+1}/{n_epochs}]')
            postfix = {
                'def_loss': loss.detach().item(),
                'dis_loss': dis_loss.detach().item(),
            }
            train_pbar.set_postfix(postfix)

        mean_train_loss = sum(loss_record) / len(loss_record)
        loss_per_epoch.append(mean_train_loss)
        torch.save(defence, model_path + f"def_10bits_black{flow_length}.pth")
        print('Saving model with loss {:.3f}...'.format(mean_train_loss))

    draw_loss_curve(loss_per_epoch, 'def_loss.jpg')


# 超参数
device = 'cuda' if torch.cuda.is_available() else 'cpu'
learning_rate = 1e-5
n_epochs = 2
batch_size = 512
input_dim = 1024
flow_length = 200
train_rate = 0.8
model_path = './models/'
data_path = '/data_2_mnt/gejian/'

# 加载数据
with open(data_path + f'data2_{flow_length}.pkl', 'rb') as file:
    data = pickle.load(file)
length = len(data)
splitter = int(length * train_rate)

train_data, test_data = data[:splitter], data[splitter:]
train_dataset, test_data_set = MyDataset(train_data), MyDataset(test_data)
train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          pin_memory=True)
test_loader = DataLoader(test_data_set,
                         batch_size=batch_size,
                         shuffle=True,
                         pin_memory=True)

train(train_loader)
