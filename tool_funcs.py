import torch
import pandas as pd
import matplotlib.pyplot as plt


# 绘制损失曲线
def draw_loss_curve(loss_per_epoch, file_name):
    plt.figure()
    plt.plot(list(range(1, len(loss_per_epoch) + 1)), loss_per_epoch)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.savefig('./pic/' + file_name)


# 绘制概率分布图
def draw_hist(probs, file_name):
    # bins参数控制直方图的柱子数量，density参数将概率标准化为密度
    plt.figure()
    plt.hist(probs, bins='auto', edgecolor='black', density=True)
    plt.title("Probability Distribution")
    plt.xlabel("Probability")
    plt.ylabel("Density")
    plt.savefig('./pic/' + file_name)


# 保存数据到csv文件中
def write_csv(head, content, file_name):
    dic = {}
    for i in range(len(head)):
        dic[head[i]] = content[i]
    df = pd.DataFrame(dic)
    df.index = df.index + 1
    df.to_csv(file_name, encoding='gbk')


def tensorlist2numpy(t):
    t = torch.cat(t).cpu()
    if (len(t.shape) == 2):
        t.squeeze(1)
    return t.numpy()


def accuracy(raw, pred):
    cnt = 0
    length = len(raw)
    for i in range(length):
        if raw[i] == pred[i]:
            cnt += 1
    return cnt / length


def bit_error_rate(embed_bit_num, raw, pred):
    error_bits = 0
    total_bits = len(raw) * embed_bit_num
    for i in range(len(raw)):
        x_bin = bin(raw[i])[2:].zfill(embed_bit_num)
        y_bin = bin(pred[i])[2:].zfill(embed_bit_num)

        error_count = sum(bit1 != bit2 for bit1, bit2 in zip(x_bin, y_bin))
        error_bits += error_count
    return error_bits / total_bits
