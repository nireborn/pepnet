import argparse
import datetime
import os
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
import tensorflow as tf
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score
from tensorflow.keras.utils import plot_model
import pepnet.data
from pepnet.models import TransformerSchedule
from pepnet.data import DataLoader, get_batch, tokenize_sequences
from pepnet.models import TransformerEncoder
from pepnet.utils import get_data_dir
from pepnet.plot import evaluate_thresholds_and_plot
import matplotlib.pyplot as plt
import pepnet.plotter as plotter

for length in range(5,15,2):
    # ---------- 全局超参数 ----------
    model_type = 'transformer'  # 模型选择：transformer
    num_epochs = 50  # 设置了50个epoch[调试为2]
    batch_size = 512  # 正样本的batch大小，一次选取128个正样本
    d_model = 128  # 隐藏层维度128
    split = 0.8  # 训练集划分
    ensemble_num = 5  # 集成训练5个模型[调试为1]
    # ---------------------------------


    # 数据路径
    data_dir = get_data_dir()
    data_path_pn = os.path.join(data_dir, f"Dataset_{length}_Merged.tsv")
    print("data_path_pn（positiveAndNegative）：", data_path_pn)

    # 设置随机种子进行集成训练
    random_seed = list(range(ensemble_num))
    print("random_seed：", random_seed)  # [0, 1, 2, 3, 4]


    # 加载ppc正样本和负样本全部数据，正样本是输入，负样本全部输入作为编码use_dataloader
    print("[INFO] 正在加载ppc_neg数据集中……")
    ppc_pn = pepnet.data.DataLoader(data_path_pn, dataset=f'ppc_pn_{length}', seed=0, model=model_type, test_split=0.2)
    print("[INFO] 加载ppc_neg数据集完成。")

    # 集成训练
    for ensemble in range(ensemble_num):
        # 划分训练集和验证集
        X_pn_train, X_pn_valid, y_pn_train, y_pn_valid = train_test_split(ppc_pn.X_train, ppc_pn.y_train, test_size=1 - split, random_state=random_seed[ensemble])
        vocab_size = len(ppc_pn.char2idx)
        print("[INFO] vocab：", ppc_pn.char2idx)
        print("[INFO] vocab大小：", vocab_size)

        num_train_samples = len(X_pn_train)
        num_valid_samples = len(X_pn_valid)
        num_test_samples = len(ppc_pn.X_test)
        print("训练集样本数:", num_train_samples, "验证集样本数: ", num_valid_samples, "测试集样本数: ", num_test_samples)

        run_name = "run-%d" % ensemble
        print('--- 开始训练集成编号: %s --- ' % run_name)

        # 构建预测模型参数：
        num_layers = 4
        embedding_dim = 128
        num_heads = 8
        dropout = 0.1
        output_dim = 1
        dff = 4 * embedding_dim
        # 二分类模型：Classification
        model_cla = TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout,
            output_dim=output_dim,
            pool_outputs=True,
            mask_zero=True)
        model_cla.final_layer = tf.keras.layers.Dense(
            output_dim,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        )

        model_cla.build((batch_size, length+1))  # 基于(batch_size, 6/8/10/12/14)的形状，初始化模型参数
        model_cla.summary()  # 查看模型结构
        # Model: "transformer_encoder"
        # _________________________________________________________________
        #  Layer (type)                Output Shape              Param #
        # =================================================================
        #  encoder (Encoder)           multiple                  2641664
        #
        #  dense_9 (Dense)             multiple                  129
        #
        # =================================================================
        # Total params: 2,641,793
        # Trainable params: 2,641,793
        # Non-trainable params: 0
        # _________________________________________________________________

        lr_cla = TransformerSchedule(d_model)
        optimizer_cla = tf.keras.optimizers.Adam(lr_cla)
        # optimizer_cla = tf.keras.optimizers.Adam(learning_rate=0.005)

        # 查看optimizer_cla的配置：
        # config = optimizer_cla.get_config()
        # print(config)  {'name': 'Adam', 'learning_rate': 0.05, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False}

        loss_fn_cla = tf.keras.losses.BinaryCrossentropy(from_logits=False)  # 二元交叉熵损失函数（BCE，Binary Cross‑Entropy）

        # 二分类任务：
        @tf.function
        def train_step_cla(x, y):
            with tf.GradientTape() as tape:
                y_hat = model_cla(x, training=True)  # 输入x为(batch_size, seq_len)，输出y_hat为(batch_size, 1)
                y = tf.cast(y, tf.float32)  # (batch_size,1)
                loss = loss_fn_cla(y, y_hat)
                # 查看训练过程中的预测值和真实值
                tf.print("y=", tf.squeeze(y))
                tf.print("y_hat=", tf.squeeze(y_hat))
                tf.print("loss =", loss)

                # 全局步数
                step = optimizer_cla.iterations
                # 对于常数 lr，直接访问属性；不加括号
                lr = optimizer_cla.learning_rate
                tf.print("Global step:", step, "  lr:", lr)

            # 反向传播 + 更新参数
            grads = tape.gradient(loss, model_cla.trainable_variables)
            tf.print("梯度均值(layer0):", tf.reduce_mean(grads[0]))
            optimizer_cla.apply_gradients(zip(grads, model_cla.trainable_variables))

            # 计算准确率ACC（分类对的样本数/总样本数）
            preds = tf.cast(y_hat >= 0.5, tf.float32)
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(preds, y), tf.float32))
            return loss, batch_acc, y_hat

        # LOGGING
        model_label_cla = '/' + model_type + '_' + str(ensemble) + '_cla' + f'_{length}'
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join('save' + model_label_cla, '{}_PREDICTOR'.format(current_time))
        os.makedirs(save_dir)
        train_log_dir = os.path.join('logs' + model_label_cla, '{}_PREDICTOR_train'.format(current_time))
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_log_dir = os.path.join('logs' + model_label_cla, '{}_PREDICTOR_val'.format(current_time))
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # 训练模型
        global_step = 0
        # running_loss = None
        # running_acc = None
        best_val_loss = float('inf')
        train_losses_per_epoch = []
        val_losses_per_epoch = []
        val_losses_per_epoch.append(0.5)
        for epoch in range(num_epochs):
            print("Epoch：", epoch)
            pbar = tqdm(range(num_train_samples // batch_size))  # 样本数//batch_size
            epoch_losses = []
            for iter in pbar:
                # 获取batch的训练数据x和y
                x, y, _ = get_batch(X_pn_train, y_pn_train, batch_size, ppc_pn, transformer=True, is_fir=True)
                y = (y > 0).astype("float32")
                loss, batch_acc, y_hat = train_step_cla(x, y)
                epoch_losses.append(loss.numpy())
                tf.print("Batch 正例率:", tf.reduce_mean(tf.cast(y > 0, tf.float32)))

                global_step += 1
                # saving
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=global_step)
                    tf.summary.scalar('acc', batch_acc, step=global_step)

            avg_epoch_loss = np.mean(epoch_losses)
            train_losses_per_epoch.append(avg_epoch_loss)

            if epoch > 0:
                print("运行验证数据集……")
                vbar = tqdm(range(num_valid_samples // batch_size))
                val_loss = []
                val_loss_sum = 0.0
                metric_val = tf.keras.metrics.BinaryAccuracy()

                for v_iter in vbar:
                    xv, yv, _ = pepnet.data.get_batch(X_pn_valid, y_pn_valid, batch_size, ppc_pn, transformer=True, is_fir=True)
                    yv = (yv > 0).astype("float32")
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
                    print(df_pred)

                val_losses_per_epoch.append(val_loss)
                val_acc = metric_val.result().numpy()
                print(f"[EPOCH {epoch}] 验证损失: {val_loss:.4f}, ACC: {val_acc:.4f}")

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

        plt.figure()
        plt.plot(range(1, num_epochs + 1), train_losses_per_epoch, label="Train Loss")
        plt.plot(range(1, num_epochs + 1), val_losses_per_epoch, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, "train_val_loss_curve.png"))

        # 对最终模型进行重新构建后评估
        if model_type == 'transformer':
            model = TransformerEncoder(
                num_layers=4,
                d_model=embedding_dim,
                num_heads=num_heads,
                dff=dff,
                vocab_size=vocab_size,
                dropout_rate=dropout,
                output_dim=1,
                pool_outputs=True)
            model.final_layer = tf.keras.layers.Dense(output_dim, activation='sigmoid')
        model.build((len(ppc_pn.X_test), length+1))
        model.summary()
        model.load_weights(checkpoint_path_final)

        xt, yt = pepnet.data.get_batch(ppc_pn.X_test, ppc_pn.y_test, len(ppc_pn.X_test), ppc_pn, test=True, transformer=True, is_fir=True)
        yt = (yt > 0).astype("float32")
        yt_hat = model(xt, training=False)

        # 先把预测张量转成 NumPy 数组
        y_pred = yt_hat.numpy()
        # 对预测概率做 0.5 阈值，得到 0/1 预测标签
        y_pred_labels = (y_pred >= 0.5).astype(int)
        # 计算测试集的 ACC 和 PR-AUC
        acc_test = accuracy_score(yt, y_pred_labels)
        ap_test = average_precision_score(yt, y_pred)
        print(f"测试集ACC = {acc_test:.4f},  PR‑AUC = {ap_test:.4f}")

        embeddings = model.last_layer_embeddings
        np.save(os.path.join(ensemble_dir, 'test_weighted_cluster_embeddings.npy'), np.array(embeddings))

        output_dir = os.path.join(save_dir, "metrics")
        best_thresh, accs, pr_aucs, losses = evaluate_thresholds_and_plot(yt, y_pred, output_dir=output_dir)
        print(f"最佳 PR-AUC 阈值: {best_thresh:.2f}")


print("模型训练结束！")
