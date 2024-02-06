import time
import torch
import torch.nn as nn
import numpy as np

from structure import *
from tool_funcs import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = './models/'
defense_length = 100
counts = 10000

defense = torch.load(model_path +
                     f'def_10bits_black{defense_length}.pth').to(device)
defense.eval()

ipd = torch.Tensor(np.random.poisson(100, (1, defense_length))).to(device)

start_time = time.time()
for _ in range(counts):
    new_ipd = defense(ipd)
end_time = time.time()

cost = (end_time - start_time) * 1000 / counts
print(
    f'Computational cost of def_10bits_black{defense_length}.pth is {cost} milliseconds'
)
