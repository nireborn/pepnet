import argparse
import datetime
import os
import numpy as np
import pandas as pd

import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score

import pepnet.data
from pepnet.models import TransformerSchedule
from pepnet.data import DataLoader, get_batch, tokenize_sequences
from pepnet.models import TransformerEncoder
from pepnet.utils import get_data_dir

# import pepnet.plotter as plotter

# ---------- 全局超参数 ----------
model_type = 'transformer'  # 模型选择：transformer
num_epochs = 50  # 设置了50个epoch
batch_size = 128  # 正样本的batch大小，一次选取128个正样本
d_model = 128  # 隐藏层维度128
alpha = 0.99
split = 0.8  # 训练集划分
ensemble_num = 5  # 集成训练5个模型
# ---------------------------------


# 数据路径
data_dir = get_data_dir()
data_path_pn = os.path.join(data_dir, "ppc_pn.csv")
print("data_path_pn：", data_path_pn)

# 设置随机种子进行集成训练
random_seed = list(range(ensemble_num))
print("random_seed：", random_seed)  # [0, 1, 2, 3, 4]


def main() -> None:
    # 加载ppc正样本和负样本全部数据，正样本是输入，负样本全部输入作为编码use_dataloader
    ppc_pn = pepnet.data.DataLoader(data_path_pn, dataset='ppc_pn', seed=0, model=model_type, test_split=0.2)
    print("[INFO] 加载ppc_neg数据集")

    # 集成训练
    for ensemble in range(ensemble_num):
        # 划分训练集和验证集，类型为list
        X_pn_train, X_pn_valid, y_pn_train, y_pn_valid = train_test_split(ppc_pn.X_train, ppc_pn.y_train, test_size=1 - split, random_state=random_seed[ensemble])

        vocab_size = len(ppc_pn.char2idx)
        print("[INFO] vocab：", ppc_pn.char2idx)
        print("[INFO] vocab大小：", vocab_size)

        num_samples = len(X_pn_train)
        num_valid_samples = len(X_pn_valid)
        print("训练集样本数:", num_samples, "验证集样本数: ", num_valid_samples)

        run_name = "run-%d" % ensemble
        print('--- 开始训练集成编号: %s' % run_name)

        # 构建预测模型参数：
        num_layers = 4
        embedding_dim = 128
        num_heads = 8
        dropout = 0.01
        output_dim = 1

        # 二分类模型：Classification
        model_cla = TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=num_heads,
            dff=d_model,
            vocab_size=vocab_size,
            dropout_rate=dropout,
            output_dim=output_dim,
            pool_outputs=True,
            mask_zero=True)
        model_cla.final_layer = tf.keras.layers.Dense(
            output_dim,
            activation='sigmoid',
            bias_initializer=tf.keras.initializers.Constant(0.0)
        )


        lr_cla = TransformerSchedule(d_model)
        model_label_cla = '/' + model_type + '_' + str(ensemble) + '_cla'
        model_cla.build((batch_size, None))
        model_cla.summary()
        loss_fn_cla = tf.keras.losses.BinaryCrossentropy()  # 二元交叉熵损失函数（BCE，Binary Cross‑Entropy）
        optimizer_cla = tf.keras.optimizers.Adam(lr_cla)
        # optimizer_cla = tf.keras.optimizers.Adam(learning_rate=1e-3)

        # 回归模型：regression
        model_reg = TransformerEncoder(
            num_layers=num_layers,  # 设置了4层Encoder
            d_model=embedding_dim,  # d_model即token编码维度为128
            num_heads=num_heads,  # 8个头的注意力机制
            dff=d_model,  # 前馈网络的隐藏层维度128
            vocab_size=vocab_size,  # 编码长度（无输出维度时，这个长度就是输出维度）
            dropout_rate=dropout,  # 丢弃率为1%
            output_dim=output_dim,  # 输出维度为1，即只有一个预测值
            pool_outputs=True,
            mask_zero=True)

        lr_reg = TransformerSchedule(d_model)
        model_label_reg = '/' + model_type + '_' + str(ensemble) + '_reg'
        model_reg.build((batch_size, None))
        model_reg.summary()
        optimizer_reg = tf.optimizers.Adam(lr_reg)

        # 二分类任务：
        @tf.function
        def train_step_cla(x, y):
            '''
            :param x: (batch_size, seq_len)
            :param y: (batch, 1)
            :return:
            '''
            with tf.GradientTape() as tape:
                y_hat = model_cla(x, training=True)  # (batch_size, 1)
                y = tf.cast(y, tf.float32)
                loss = loss_fn_cla(y, y_hat)

            # 反向传播 + 更新参数
            grads = tape.gradient(loss, model_cla.trainable_variables)
            tf.print("梯度均值(layer0):", tf.reduce_mean(grads[0]))
            optimizer_cla.apply_gradients(zip(grads, model_cla.trainable_variables))
            # 计算准确率ACC（分类对的样本数/总样本数）
            preds = tf.cast(y_hat >= 0.5, tf.float32)
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
            return loss, batch_acc, y_hat

        # 回归任务：
        @tf.function
        def train_step_reg(x, y):
            with tf.GradientTape() as tape:
                y_hat = model_reg(x)
                y = tf.cast(y, tf.float64)
                loss = model_reg.compute_loss(y, y_hat)
            grads = tape.gradient(loss, model_reg.trainable_variables)  # compute gradient
            optimizer_reg.apply_gradients(zip(grads, model_reg.trainable_variables))  # update
            return loss, y_hat

        def smooth(prev, val):
            if prev is not None:
                new = (1 - alpha) * val + alpha * prev
            else:
                new = val
            return new

        # LOGGING
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join('save' + model_label_cla, '{}_PREDICTOR'.format(current_time))
        os.makedirs(save_dir)
        train_log_dir = os.path.join('logs' + model_label_cla, '{}_PREDICTOR_train'.format(current_time))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs' + model_label_cla, '{}_PREDICTOR_val'.format(current_time))
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        global_step = 0
        running_loss = None
        # running_rmse = None
        running_acc = None
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print("Epoch：", epoch)
            pbar = tqdm(range(num_samples // batch_size))  # 样本数//batch_size
            for iter in pbar:
                # 获取batch的训练数据x和y
                x, y, _ = get_batch(X_pn_train, y_pn_train, batch_size, ppc_pn, transformer=True, is_fir=True)
                y = (y > 0).astype("float32")
                tf.print("Batch 正例率:", tf.reduce_mean(tf.cast(y > 0, tf.float32)))
                # print("y：", y)
                # print("y shape:", y.shape)
                loss, acc, y_hat = train_step_cla(x, y)
                # rmse = model.compute_rmse(y, y_hat)  # compute train rmse

                running_loss = smooth(running_loss, loss.numpy())
                running_acc = smooth(running_acc, acc.numpy())
                # running_rmse = smooth(running_rmse, rmse.numpy())

                global_step += 1

                # saving
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=global_step)
                    # tf.summary.scalar('rmse', rmse, step=global_step)
                    tf.summary.scalar('acc', acc, step=global_step)

            if epoch > 0:
                print("运行验证数据集……")
                vbar = tqdm(range(len(X_pn_valid) // batch_size))
                val_loss = []
                # val_rmse = []
                val_loss_sum = 0.0
                metric_val = tf.keras.metrics.BinaryAccuracy()

                for v_iter in vbar:
                    xv, yv, _ = pepnet.data.get_batch(X_pn_valid, y_pn_valid, batch_size, ppc_pn, transformer=True, is_fir=True)
                    yv = np.asarray(yv > 0).reshape(-1, 1).astype("float32")
                    yv_hat = model_cla(xv, training=False)

                    val_loss_sum += loss_fn_cla(yv, yv_hat).numpy() * len(yv)
                    metric_val.update_state(yv, yv_hat)

                    val_loss = val_loss_sum / len(X_pn_valid)
                    val_acc = metric_val.result().numpy()
                    print(f"验证 BCE={val_loss:.4f}  acc={val_acc:.4f}")

                    true_vals = yv.reshape(-1)
                    pred_vals = yv_hat.numpy().reshape(-1)
                    df_pred = pd.DataFrame({
                        'true': true_vals,
                        'pred': pred_vals
                    })
                    print(df_pred.head(5))

                    # val_loss.append(model_cla.compute_loss(yv, yv_hat) * batch_size)  # compute loss
                    # val_acc.append(model_cla.compute_rmse(yv, yv_hat) * batch_size)  # compute val rmse

                # val_loss = np.sum(val_loss) / len(X_pn_valid)
                # val_acc = np.sum(val_acc) / len(X_pn_valid)
                # print(f"验证LOSS: {val_loss:.4f}")
                # print(f"验证RMSE: {val_acc:.4f}")

                with val_summary_writer.as_default():
                    tf.summary.scalar('loss', val_loss, step=epoch)
                    tf.summary.scalar('acc', val_acc, step=epoch)

                    if val_loss < best_val_loss:
                        print(f"保存验证loss: {val_loss:.4f}")
                        print(f"保存验证acc: {val_acc:.4f}")
                        model_cla.save_weights(os.path.join(save_dir, "{}.weights.h5".format("model")))
                        best_val_loss = val_loss
                        print("最佳验证损失:", best_val_loss)

        save_file = save_dir + '/best_loss.csv'
        with open(save_file, 'w') as f:
            f.write(str(best_val_loss))
            print("保存最佳验证损失:", best_val_loss)

        ensemble_dir = save_dir
        checkpoint_path_final = os.path.join(ensemble_dir, "model.weights.h5")

        if os.path.exists(checkpoint_path_final):
            print(f"模型权重文件存在： {checkpoint_path_final}.")
        else:
            print(f"模型权重文件不存在： {checkpoint_path_final}.请检查文件！")

        # 对最终模型进行重新构建后评估
        if model_type == 'transformer':
            model = TransformerEncoder(
                num_layers=4,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=d_model,
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=1,
                pool_outputs=True)
            model.final_layer = tf.keras.layers.Dense(1, activation='sigmoid')
        model.build((len(ppc_pn.X_test), None))
        model.summary()
        model.load_weights(checkpoint_path_final)

        xt, yt = pepnet.data.get_batch(ppc_pn.X_test, ppc_pn.y_test, len(ppc_pn.X_test), ppc_pn, test=True, transformer=True, is_fir=True)
        yt = (yt > 0).astype("float32")
        yt_hat = model(xt, training=False)

        # 计算测试准确率 & PR‑AUC
        acc_test = accuracy_score(yt, (yt.numpy() >= 0.5))
        ap_test = average_precision_score(yt, yt_hat.numpy())
        print(f"测试集ACC = {acc_test:.4f},  PR‑AUC = {ap_test:.4f}")
        # test_rmse = model.compute_rmse(yt, yt_hat, axis=0)
        # print(test_rmse)

        # Save embeddings for later
        embeddings = model.last_layer_embeddings
        np.save(os.path.join(ensemble_dir, 'test_weighted_cluster_embeddings.npy'), np.array(embeddings))


if __name__ == "__main__":
    main()
    print("模型训练结束！")
