import numpy as np
import pandas as pd
import math
import torch
import random
import glob
import os

n = 3  # redundance
l = 8  # bit length
T = 400  # time interval(ms)

is_extracted = []
ber = []
is_extracted_with_defense = []
ber_with_defense = []


# Caculate the bit error rate
def get_ber(message, extraction):
    message, extraction = np.array(message), np.array(extraction)
    error = (message != extraction)
    ber = np.sum(error) / len(message)
    return ber


def defense(timestamp):
    global model, flow_length, device

    ipd = np.diff(timestamp)
    old_len = len(ipd)
    pad_len = math.ceil(old_len / flow_length) * flow_length - old_len
    padded_ipd = np.pad(ipd, (0, pad_len), mode='constant', constant_values=0)

    new_ipd = np.array([])
    for i in range(0, len(padded_ipd), flow_length):
        ipd_slice = padded_ipd[i:i + flow_length]
        ipd_slice = torch.Tensor(ipd_slice).to(device).reshape(1, flow_length)
        new_ipd_slice = model(ipd_slice).detach().cpu().numpy()
        new_ipd = np.concatenate((new_ipd, new_ipd_slice.squeeze(0)))
    new_ipd = new_ipd[:old_len]

    timestamp_with_defense = [0]
    for ipd in new_ipd:
        timestamp_with_defense.append(timestamp_with_defense[-1] + ipd)

    return timestamp_with_defense


def extract(timestamp):
    packets_per_slot = [0] * (l * n * 3)
    for t in timestamp:
        slot_num = int(t // T)
        if slot_num >= len(packets_per_slot):
            break
        packets_per_slot[slot_num] += 1

    bits = []
    for i in range(0, len(packets_per_slot), 3):
        if packets_per_slot[i] >= packets_per_slot[i + 1]:
            bits.append(1)
        else:
            bits.append(0)

    extraction = []
    for i in range(0, len(bits), 3):
        ones = bits[i:i + 3].count(1)
        if ones >= 2:
            extraction.append(1)
        else:
            extraction.append(0)

    return extraction


# Caculate which bit and which slot the timestamp belongs to
def get_bit_and_slot(t):
    slot_num = t // T
    bit_num = slot_num // (3 * n)
    return int(bit_num), int(slot_num % 3)


def experiment_once(timestamp):
    message = [random.randint(0, 1) for _ in range(l)]

    # embed
    for i in range(len(timestamp)):
        bit_num, slot = get_bit_and_slot(timestamp[i])
        if bit_num >= len(message):
            break
        if slot == 0 and message[bit_num] == 0:
            timestamp[i] += T
        elif slot == 1 and message[bit_num] == 1:
            timestamp[i] += T

    # defense
    timestamp_with_defense = defense(timestamp)

    # extract
    extraction = extract(timestamp)
    extraction_with_defense = extract(timestamp_with_defense)

    is_extracted.append(message == extraction)
    ber.append(get_ber(message, extraction))
    is_extracted_with_defense.append(message == extraction_with_defense)
    ber_with_defense.append(get_ber(message, extraction_with_defense))


total_time = T * 3 * n * l
device = 'cuda' if torch.cuda.is_available() else 'cpu'
flow_length = 200
data_path = './log1'
model_path = './models/'
model = torch.load(model_path +
                   f'def_10bits_black{flow_length}.pth').to(device)
model.eval()

files = glob.glob(os.path.join(data_path, '*.csv'))
for file in files:
    data_df = pd.read_csv(file, header=None)
    timestamp = data_df[0].values
    timestamp = (timestamp - timestamp[0]) * 1000
    if timestamp[-1] < total_time:
        continue

    experiment_once(timestamp)

print("ER without defense: ", is_extracted.count(True) / len(is_extracted))
print("BER without defense: ", sum(ber) / len(ber))
print("ER with defense: ",
      is_extracted_with_defense.count(True) / len(is_extracted_with_defense))
print("BER with defense: ", sum(ber_with_defense) / len(ber_with_defense))
