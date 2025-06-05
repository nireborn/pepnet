import os

import numpy as np
from tqdm import tqdm

from pepnet.utils import get_data_dir
from pepnet.models import prediction
import pandas as pd
from pepnet.data import DataLoader

# 全局参数
path_to_sequence_csv = '/share/home/wuj/kwd/pepnet/data/input_proteins.txt'  # 默认路径
save_dir = 'outputs/'
model_architecture = 'transformer'

# 获取当前工作目录
current_directory = os.getcwd()
print(f"当前工作目录是: {current_directory}")

if not os.path.exists(save_dir):
    os.makedirs(save_dir, exist_ok=True)
print(f"当前结果存储目录为: {current_directory}")


# 读取蛋白序列
def read_protein_sequences(file_path):
    with open(file_path, 'r') as f:
        sequences = [line.strip() for line in f.readlines()]
    return sequences


input_df = read_protein_sequences(path_to_sequence_csv)
print("[INFO] 读取蛋白质序列成功！")

window_size = 13
eval_sequences = []
for input_sequence in input_df:
    if len(input_sequence) > window_size:
        eval_sequences.extend([input_sequence[i:i + window_size] for i in range(len(input_sequence) - window_size + 1)])
print("评估序列为：", eval_sequences)  # 列表形式

data_dir = get_data_dir()
data_path = os.path.join(data_dir, "ppc_negative.csv")
print("[INFO] 加载编码数据集：", data_path)


# 预测并输出每个滑动窗口的出现次数（预测值）
k_pred_cscores = []  # 初始化为空列表

# 使用 tqdm 显示进度条
for i, seq in enumerate(tqdm(eval_sequences, desc="滑动窗口进度：")):
    # 使用 pepnet 的 prediction 方法进行预测
    print("[INFO] 正在加载预测模型！")
    k_pred_zscore = prediction(data_path,
                               seq,  # 这里传入的是一个包含单个序列的列表
                               save_dir,
                               checkpoint_dir='weights/',
                               predictor_model_type=model_architecture)
    print("[INFO] 预测完成！")
    # 将每次的预测结果添加到 k_pred_cscores 中
    k_pred_cscores.append(k_pred_zscore.flatten())  # 确保数据展平

# 将列表转换为 NumPy 数组
k_pred_cscores = np.array(k_pred_cscores)
print("得分", k_pred_cscores)

# 输出每个序列的预测值
for i, seq in enumerate(eval_sequences):
    print(f"序列: {seq}, 预测出现次数: {k_pred_cscores[i][0]}")

# 初始化文件路径
output_file = "output_predictions.tsv"  # 输出文件路径

# 打开文件进行写入
with open(output_file, 'w') as f:
    # 写入标题
    f.write("Seq, Prediction\n")

    for i, seq in enumerate(eval_sequences):
        prediction_score = k_pred_cscores[i][0]  # 获取预测值

        # 保存序列前7个字符和预测值到文件
        f.write(f"{seq[:7]}\t{prediction_score}\n")

print(f"结果已保存到 {output_file}")
