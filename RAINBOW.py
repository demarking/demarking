import os
import torch
import glob
import pandas as pd
import numpy as np
import random
from structure import *

a = 10
length = 1200
threshold = 0.54


# Calculate normalized correlation
def normalization(a, b):
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    return dot / (norm_a * norm_b)


# Calculate true/false positive rate
def rate(a):
    return a.count(True) / len(a)


# Perform defensive transformation
def defense(ipd):
    global model, flow_length, device
    count = len(ipd) // flow_length
    new_ipd = np.array([])

    for i in range(count):
        ipd_slice = ipd[i * flow_length:(i + 1) * flow_length]
        ipd_slice = torch.Tensor(ipd_slice).to(device).reshape(1, flow_length)
        new_ipd_slice = model(ipd_slice).detach().cpu().numpy()
        new_ipd = np.concatenate((new_ipd, new_ipd_slice.squeeze(0)))
    return new_ipd


def experiment_once(ipd):
    global res_without_w_with_def, res_without_w, res_with_w, res_with_defense
    w = [random.choice([a, -a])
         for _ in range(length)]  # Generate random delay
    delta = np.random.laplace(0, 10, length)  # Generate random jitter

    received_ipd = ipd + delta  # without watermarking
    watermarked_ipd = ipd + w + delta  # with watermarking
    new_ipd_with_w = defense(
        watermarked_ipd) + delta  # with watermarking, after defense
    new_ipd_without_w = defense(
        received_ipd) + delta  # without watermarking, after defense

    y1 = received_ipd - ipd
    y2 = watermarked_ipd - ipd
    y3 = new_ipd_with_w - ipd
    y4 = new_ipd_without_w - ipd

    normal_y1 = normalization(y1, w)  # without watermarking
    normal_y2 = normalization(y2, w)  # with watermarking
    normal_y3 = normalization(y3, w)  # with watermarking, after defense
    normal_y4 = normalization(y4, w)  # without watermarking, after defense

    res_without_w.append(normal_y1 > threshold)
    res_with_w.append(normal_y2 > threshold)
    res_with_defense.append(normal_y3 > threshold)
    res_without_w_with_def.append(normal_y4 > threshold)


# Load defense model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
flow_length = 200
data_path = './log1'
model_path = './models/'
model = torch.load(model_path +
                   f'def_10bits_black{flow_length}.pth').to(device)
model.eval()

res_without_w_with_def = []
res_without_w, res_with_w, res_with_defense = [], [], []
files = glob.glob(os.path.join(data_path, '*.csv'))
for file in files:
    data_df = pd.read_csv(file, header=None)
    timestamp = data_df[0].values
    ipd = np.diff(timestamp)[:length]
    print(np.mean(ipd) * 1000, np.std(ipd * 1000))
    experiment_once(ipd)

print(
    f'TP without defense: {rate(res_with_w)}, TP with defense: {rate(res_with_defense)}'
)
print(
    f'FP without defense: {rate(res_without_w)}, FP with defense: {rate(res_without_w_with_def)}'
)
