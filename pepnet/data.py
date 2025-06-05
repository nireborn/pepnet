import os
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader(object):
    def __init__(self, data_path, dataset, seed=0, model='transformer', test_split=0.2, use_dataloader=None):
        self.dataset = dataset
        self.seed = seed
        self.model = model
        self.data_path = data_path
        self.test_split = test_split

        np.random.seed(self.seed)
        random.seed(self.seed)

        # 设置划分保存路径（每个数据集对应一个文件夹）
        self.out_path = os.path.join('splits/', self.dataset + '/')
        print("[INFO] 数据集所在位置：", self.out_path)

        print("[INFO] 正在检测是否有已划分好的数据集……")
        if os.path.exists(self.out_path):
            print("[INFO] 检测到已有划分后的文件，正在直接加载数据集！")

            self.X = list(get_data(self.out_path + 'X_all.csv', names=['sequence']).index)  # 设置列名为sequence，并将X变为列表形式
            # print(type(self.X).__name__)    list

            self.y = list(get_data(self.out_path + 'y_all.csv', names=['count']).index)  # 设置列名count,（样本数,1）
            # print(type(self.y).__name__)    list

            self.sequences = self.X
            if test_split > 0:
                self.X_train = list(get_data(self.out_path + 'X_train.csv', names=['sequence']).index)
                self.y_train = list(get_data(self.out_path + 'y_train.csv', names=['count']).index)
                self.X_test = list(get_data(self.out_path + 'X_test.csv', names=['sequence']).index)
                self.y_test = list(get_data(self.out_path + 'y_test.csv', names=['count']).index)

        else:
            print("[INFO] 首次使用该数据集，正在从原始数据构建训练样本……")

            print("[INFO] 正在创建数据存放目录……")
            os.makedirs(self.out_path)

            data = self.load_count_data()
            self.sequences = data.index.to_list()
            self.X = self.sequences
            self.value = np.around(data.values, 2)
            self.y = self.value.flatten().tolist()
            np.savetxt(self.out_path + 'X_all.csv', self.X, delimiter=",", fmt='%s')
            np.savetxt(self.out_path + 'y_all.csv', self.y, delimiter=",", fmt='%.2f')

            if test_split > 0:
                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.test_split, random_state=self.seed)
                np.savetxt(self.out_path + 'X_train.csv', self.X_train, delimiter=",", fmt='%s')
                np.savetxt(self.out_path + 'y_train.csv', self.y_train, delimiter=",", fmt='%.2f')
                np.savetxt(self.out_path + 'X_test.csv', self.X_test, delimiter=",", fmt='%s')
                np.savetxt(self.out_path + 'y_test.csv', self.y_test, delimiter=",", fmt='%.2f')

        # 构建 vocab
        if not use_dataloader:  # use_dataloader为None时执行
            self.char2idx, self.idx2char = self.create_vocab()
            # print("Vocab 映射表: \n", self.char2idx)
        else:
            self.char2idx = use_dataloader.char2idx
            self.idx2char = use_dataloader.idx2char
            # print("Vocab 映射表：\n", self.char2idx)

    def load_count_data(self):
        """
        加载出现次数
        """
        df_c = pd.read_csv(self.data_path,sep='\t', header=0)
        df_c = df_c.dropna(how='all', axis=1)
        # 先删除重复出现的，且是负样本的Sequence行
        mask_dup = df_c['Sequence'].duplicated(keep=False)
        df_c = df_c.loc[~(mask_dup & (df_c['SpectralCount'] == 0))].reset_index(drop=True)
        df_c = pd.pivot_table(df_c, index=["Sequence"])  # 对序列重复的行，进行一个平均值计算
        return df_c

    def create_vocab(self):
        """
        构建字符级 vocab 映射
        """
        seqs_joined = "".join(self.sequences)

        # 由于字符中包含了'-'[PAD]，将 '-' 放在最前面
        vocab = sorted(set(seqs_joined), key=lambda x: (x != '-', x))

        if self.model == 'transformer':
            self.CLS = '!'
            vocab += self.CLS
            # print("create_vocab函数的vocab：", vocab)

        # print("vocab =", vocab)
        char2idx = {u: i for i, u in enumerate(vocab)}  # 字典类型
        idx2char = np.array(vocab)  # (22,)  ndarray
        return char2idx, idx2char


def tokenize_sequences(sequences, dataloader):
    char2idx, idx2char = dataloader.char2idx, dataloader.idx2char
    tokenized = [np.array([char2idx[aa] for aa in sequence]) for sequence in sequences]

    return np.stack(tokenized)


def get_batch(x, y, batch_size, dataloader, test=False, transformer=False, is_fir=False, selected_ind=None):
    if test == True:
        batch_X = x
        batch_Y = y

    else:  # 训练模式下
        if is_fir:
            selected_inds = np.random.choice(len(x), size=batch_size)
            batch_X = [x[s] for s in selected_inds]  # 列表
            batch_Y = [y[s] for s in selected_inds]  # 列表
        else:
            selected_inds = selected_ind
            batch_X = [x[s] for s in selected_inds]  # 列表
            batch_Y = [y[s] for s in selected_inds]  # 列表

    tokenized_batch_X = tokenize_sequences(batch_X, dataloader)
    batch_Y = np.asarray(batch_Y)
    batch_Y = batch_Y[:, np.newaxis]

    if transformer:
        cls_idx = dataloader.char2idx[dataloader.CLS]
        tokenized_batch_X = np.stack([np.append(np.array(cls_idx), s) for s in tokenized_batch_X])

    if is_fir and not test:
        # 正样本时多返回 selected_inds
        return tokenized_batch_X, batch_Y, selected_inds
    else:
        return tokenized_batch_X, batch_Y


# 从CSV文件读取数据
def get_data(path, index_col=0, names=None):  # 默认为第0列作为行索引
    if names is not None:  # 如果有列名，则为列名，但必须和列数相同
        data = pd.read_csv(path, index_col=index_col, names=names)
    else:
        data = pd.read_csv(path, index_col=index_col)
    return data  # 返回值是DataFrame
