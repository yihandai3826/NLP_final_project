import re
import matplotlib.pyplot as plt
import matplotlib

# 切换到非交互式后端
matplotlib.use('Agg')  # 或 'TkAgg'

def extract_and_plot(filename):
    epochs = []
    losses = []

    pattern = r"\{'loss': ([\d\.]+), 'learning_rate': ([\deE\+\-\.]+), 'epoch': ([\d\.]+)\}"

    with open(filename, 'r') as file:
        for line in file:
            match = re.search(pattern, line)
            if match:
                loss = float(match.group(1))
                epoch = float(match.group(3))
                losses.append(loss)
                epochs.append(epoch)

    if not epochs or not losses:
        print("没有匹配的数据！")
        return

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, marker="o", label="Loss vs. Epoch")
    plt.title("Loss vs. Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()

    # 保存为文件
    plt.savefig('output.png')  # 保存为 PNG
    print("图表已保存为 output.png")

# 调用函数
extract_and_plot('output.txt')