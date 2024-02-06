import os
import glob
import pandas as pd
import numpy as np
import torch
import random
import math

r = 20  # Number of subintervals
m = 5  # Number of slots per subinterval
tau = 0.5  # Packet detection threshold
eta = 12  # Mark detection threshold
n = 32  # Number of base and mark intervals
T = 2  # Interval length
q = 2.5  # Quantization multiplier

sub_len = T / r  # The length of a subinterval
slot_len = sub_len / m  # The length of a slot


def generate_permutation():
    dic = {}
    for i in range(m):
        pi = [random.randint(0, m - 1) for _ in range(r)]
        dic[i] = pi
    return dic


# Separate subintervals from the flow
def subinterval_slice(start_time, packets):
    slice = []
    for t in packets:
        if t > start_time + T:
            break
        if t >= start_time and t <= start_time + T:
            slice.append(t)
    return np.array(slice)


# Calculate the quantified centroid
def get_s(start_time, subinterval):
    centroid = np.mean(subinterval - start_time)
    s = math.floor(q * m * centroid / T) % m
    return s


def get_s_prime(start_time, subinterval):
    centroid = np.mean(subinterval - start_time)
    x = m * q * centroid / T
    if x - math.floor(x) >= 0.5:
        return math.ceil(x) % m
    else:
        return (math.floor(x) - 1) % m


# Calculate the subinterval and slot where the package is located
def get_packet_pos(start_time, timestamp):
    global sub_len, slot_len

    sub_index = math.floor((timestamp - start_time) / sub_len)
    slot_index = math.floor(((timestamp - start_time) % sub_len) / slot_len)
    return sub_index, slot_index


def embed(packets, start_time, s):
    global pi_dic

    pi = pi_dic[s]

    for target_sub in range(r):
        target_slot = pi[target_sub]
        slot_head = start_time + sub_len * target_sub + slot_len * target_slot
        new_timestamp = slot_head + slot_len / 2  # The midpoint of the selected slot

        for i in range(len(packets)):
            if packets[i] > start_time + T:
                break
            if packets[i] < start_time:
                continue

            sub_index, slot_index = get_packet_pos(start_time, packets[i])
            if sub_index > target_sub:
                break
            if sub_index == target_sub - 1:
                if slot_index > pi[target_sub - 1]:
                    packets[i] = new_timestamp  # Move to the selected  slot
            if sub_index == target_sub:
                if slot_index < target_slot:
                    packets[i] = new_timestamp  # Move to the selected  slot


def detect(start_time, s, packets):
    global pi_dic, tau

    pi = pi_dic[s]
    valid, total = 0, 0

    for i in range(len(packets)):
        if packets[i] >= start_time + T:
            break
        if packets[i] < start_time:
            continue

        total += 1
        sub_index, slot_index = get_packet_pos(start_time, packets[i])
        if slot_index == pi[sub_index]:
            valid += 1

    if total == 0:
        return False
    else:
        return (valid / total) > tau


def defence(packets):
    global model, flow_length, device

    ipd = np.diff(packets)
    zero_len = math.ceil(len(ipd) / flow_length) * flow_length - len(ipd)
    ipd = np.concatenate((ipd, np.zeros(zero_len)))  # Zero padding
    ipd = ipd * 1000  # Convert to milliseconds

    count = len(ipd) // flow_length
    new_ipd = np.array([])
    new_packets = [packets[0]]

    for i in range(count):
        ipd_slice = ipd[i * flow_length:(i + 1) * flow_length]
        ipd_slice = torch.Tensor(ipd_slice).to(device).reshape(1, flow_length)
        new_ipd_slice = model(ipd_slice).detach().cpu().numpy()
        new_ipd = np.concatenate((new_ipd, new_ipd_slice.squeeze(0)))

    for i in range(len(packets) - 1):
        new_packets.append(new_packets[-1] + new_ipd[i] / 1000)

    return new_packets


def experiment_once(packets):
    global res

    # Embed watermarks
    watermarked_packets = packets.copy()
    for i in range(n):
        base_start_time = i * 2 * T
        mark_start_time = base_start_time + T
        base_interval = subinterval_slice(base_start_time, packets)
        if len(base_interval) == 0:
            continue
        s = get_s(base_start_time, base_interval)

        embed(watermarked_packets, mark_start_time, s)

    # Add random noise
    noise = np.random.normal(loc=5, size=(len(watermarked_packets), ))
    watermarked_packets = np.sort(watermarked_packets + noise / 1000)
    new_packets = defence(packets)

    # Detect watermarks
    d = 0
    for i in range(n):
        base_start_time = i * 2 * T
        mark_start_time = base_start_time + T
        base_interval = subinterval_slice(base_start_time, new_packets)
        if len(base_interval) == 0:
            continue
        s = get_s(base_start_time, base_interval)
        s_prime = get_s_prime(base_start_time, base_interval)

        if detect(mark_start_time, s, new_packets) or detect(
                mark_start_time, s_prime, new_packets):
            d += 1

    # print(d)
    res.append(d > eta)
    # print(d > eta)


pi_dic = generate_permutation()

# Load defense model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
flow_length = 200
model_path = './models/'
model = torch.load(model_path +
                   f'def_10bits_black{flow_length}.pth').to(device)
model.eval()

# Read timestamps
data_path = './log1'
packets = []
files = glob.glob(os.path.join(data_path, '*.csv'))
for file in files:
    data_df = pd.read_csv(file, header=None)
    timestamp = data_df[0].values
    packets.append(timestamp)

packets = np.sort(np.concatenate(packets))
packets = packets - packets[0]

res = []
num = int(packets[-1] // (n * 2 * T))
ptr = 0
for i in range(num):
    pkts = []
    start_time = i * (n * 2 * T)
    end_time = (i + 1) * (n * 2 * T)
    while packets[ptr] >= start_time and packets[ptr] < end_time:
        pkts.append(packets[ptr])
        ptr += 1
    if len(pkts) > 0:
        pkts = np.array(pkts)
        pkts = pkts - pkts[0]

        experiment_once(pkts)
print(res.count(True) / len(res))
