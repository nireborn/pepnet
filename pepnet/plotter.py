# plotter.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import tensorflow as tf


# -------------------------------------------
def plot_parity(y_true: np.ndarray,
                y_pred: np.ndarray,
                save_dir: str,
                title: str = "True vs. Predicted",
                fname: str = "parity_plot.svg"):
    """
    单输出回归的散点 + 回归线 + 皮尔森 R
    """
    assert y_true.shape == y_pred.shape and y_true.ndim == 2 and y_true.shape[1] == 1
    y_true = y_true.squeeze()
    y_pred = y_pred.squeeze()

    # 计算皮尔森相关
    r, _ = stats.pearsonr(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    sns.regplot(x=y_true, y=y_pred, scatter_kws=dict(alpha=.6))
    plt.title(f"{title}\n$R$ = {r:.2f}", weight="bold")
    plt.xlabel("True value")
    plt.ylabel("Predicted value")
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname))
    plt.close()


# -------------------------------------------
def save_rmse(y_true: np.ndarray, y_pred: np.ndarray, save_dir: str):
    """
    计算整体 RMSE 并保存到 txt
    """
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred))).numpy()

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "rmse.txt"), "w") as f:
        f.write(str(rmse))
    return rmse


# -------------------------------------------
def aa_frequency_logo(seqs: list[str],
                      save_dir: str,
                      fname: str = "aa_logo.png"):
    """
    统计肽段 10 位 (P5···P5') 的氨基酸频率并画 logomaker logo
    - `seqs` 为等长已对齐肽序列列表 (长度==10)
    """
    import logomaker

    assert len(seqs) > 0 and all(len(s) == len(seqs[0]) for s in seqs), \
        "所有肽段长度必须一致"

    L = len(seqs[0])
    aa = "ACDEFGHIKLMNPQRSTVWY"
    freq = pd.DataFrame(0, index=list(aa), columns=range(L))

    for s in seqs:
        for pos, res in enumerate(s):
            if res in aa:
                freq.at[res, pos] += 1

    freq = freq / len(seqs)  # 归一化
    freq.columns = [f"P{5 - i}" if i < 5 else f"P{i - 4}'" for i in range(L)]

    # 画 logo
    logo = logomaker.Logo(freq,
                          color_scheme="NajafabadiEtAl2017",
                          font_name="Arial Rounded MT Bold",
                          figsize=(6, 2))
    logo.style_spines(visible=False)
    logo.style_xticks(rotation=90)
    logo.ax.set_ylabel("Frequency")
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, fname), dpi=300)
    plt.close()
