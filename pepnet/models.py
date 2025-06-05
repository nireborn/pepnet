import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import L2

import pepnet.data


# from cleavenet import analysis, plotter

class TransformerEncoder(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, dropout_rate=0.1, output_dim=1, pool_outputs=False, mask_zero=False):
        super().__init__()
        self.pool_outputs = pool_outputs
        self.encoder = Encoder(num_layers=num_layers,
                               d_model=d_model,
                               num_heads=num_heads,
                               dff=dff,
                               vocab_size=vocab_size,
                               dropout_rate=dropout_rate,
                               mask_zero=mask_zero)
        self.final_layer = tf.keras.layers.Dense(output_dim)
        self.vocab_size = vocab_size

    def call(self, x, training=False):
        x = self.encoder(x, training=training)  # (batch_size, target_len, d_model)
        self.last_layer_embeddings = x
        if self.pool_outputs:
            x = tf.squeeze(x[:, 0:1, :], axis=1)  # (batch_size, d_model)
        logits = self.final_layer(x)  # (batch_size, 1)
        return logits


class TransformerSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        # print(tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2))
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]  # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :] / depth  # (1, depth)
    angle_rates = 1 / (10000 ** depths)  # (1, depth)
    angle_rads = positions * angle_rates  # (pos, depth)
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model, max_seq_length=14, mask_zero=True, label=False, start_idx=21):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=mask_zero)
        self.pos_encoding = positional_encoding(length=max_seq_length, depth=d_model)
        self.label = label
        self.start_idx = start_idx
        if label:
            self.label_embedding = tf.keras.layers.Dense(d_model)

    def call(self, x):
        if self.label:
            x, label = x
            label_emb = []
            # print("label", label)
            for i in range(len(label)):
                if label[i][0] == self.start_idx:  # if start token
                    label_emb_temp = self.embedding(label[i])
                else:
                    label_emb_temp = self.label_embedding(tf.expand_dims(label[i], 0))
                # print("embedded", label_emb_temp)
                label_emb.append(label_emb_temp)
            label_emb = tf.stack(label_emb)
        x = self.embedding(x)
        if self.label:
            if x.shape[1] == 0:
                x = label_emb
            else:
                x = tf.concat([label_emb, x], 1)
        length = tf.shape(x)[1]
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x


### Transformer Components ###
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


class GlobalSelfAttention(BaseAttention):
    def call(self, x, training=False, mask=None):
        attn_output = self.mha(
            query=x,
            value=x,
            key=x,
            attention_mask=mask,
            training=training)
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, d_model, dff, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation=activation),
            tf.keras.layers.Dense(d_model),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x, training=False):
        x = self.add([x, self.seq(x, training=training)])
        x = self.layer_norm(x)
        return x


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, dff, activation='relu', dropout_rate=0.1):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=d_model,
            dropout=dropout_rate)
        self.ffn = FeedForward(d_model, dff, activation=activation, dropout_rate=dropout_rate)

    def call(self, x, training=False, mask=None):
        x = self.self_attention(x, training=training, mask=mask)
        x = self.ffn(x, training=training)
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, dff, vocab_size, activation='relu', dropout_rate=0.1, mask_zero=False):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model, mask_zero=mask_zero)
        self.enc_layers = [
            EncoderLayer(d_model=d_model,
                         num_heads=num_heads,
                         dff=dff,
                         activation=activation,
                         dropout_rate=dropout_rate)
            for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False, mask=None):
        # `x` is token-IDs shape: (batch, seq_len)
        x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training=training, mask=mask)
        return x  # Shape `(batch_size, seq_len, d_model)`.


def load_predictor_model(model_type, checkpoint_path, mask_zero=False):
    # checkpoint_path模型权重的路径，用于加载训练好的模型。
    if model_type == 'transformer':
        dff = 512
        num_layers = 4
        num_heads = 8
        dropout = 0
        vocab_size = 22
        embedding_dim = 128
        model = pepnet.models.TransformerEncoder(
            num_layers=num_layers,
            d_model=embedding_dim,  # d_model,
            num_heads=num_heads,
            dff=dff,
            vocab_size=vocab_size,
            dropout_rate=dropout,
            output_dim=1,
            pool_outputs=True,
            mask_zero=mask_zero)
        model.final_layer = tf.keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer=tf.keras.initializers.GlorotNormal(),
            bias_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        )
    fake_batch = np.array([[21, 20, 14, 8, 9, 13, 10, 9, 16, 17, 3, 1, 2, 15]])
    model(fake_batch, training=False)  # build model in TF
    model.summary()
    model.load_weights(checkpoint_path)  # load weights
    return model


def prediction(data_path_ppc, protein_sequence, generated_dir, checkpoint_dir='weights/', predictor_model_type='transformer'):
    # 根据模型类型选择集成模型
    if not os.path.exists(generated_dir):
        os.mkdir(generated_dir)
    if predictor_model_type == 'transformer':
        ensembles = ['transformer_0_cla/',
                     'transformer_1_cla/',
                     'transformer_2_cla/',
                     'transformer_3_cla/',
                     'transformer_4_cla/'
                     ]

    # ppc_pn
    ppc_pn = pepnet.data.DataLoader(data_path_ppc, dataset='ppc_pn', seed=0, model=predictor_model_type, test_split=0.2)

    # 直接将输入的蛋白序列转为肽段（长度为13）
    peptides = [protein_sequence]

    # 对肽段进行编码
    x_all = pepnet.data.tokenize_sequences(peptides, ppc_pn)
    print("肽序列编码后为：", x_all)
    if predictor_model_type == 'transformer':
        cls_idx = ppc_pn.char2idx[ppc_pn.CLS]
        x_all = np.stack([np.append(np.array(cls_idx), s) for s in x_all])

    predictions = []

    # 对每个集成模型进行预测
    for e_num, ensemble in enumerate(ensembles):
        print("Running", e_num, ensemble)
        checkpoint_path = os.path.join(checkpoint_dir, ensemble, "model.weights.h5")

        # 构建并加载模型
        model = pepnet.models.load_predictor_model(model_type=predictor_model_type, checkpoint_path=checkpoint_path, mask_zero=True)

        # 进行前向预测
        y_hat = model(x_all, training=False)

        predictions.append(y_hat)
        if e_num == (len(ensembles) - 1):  # save embeddings from last ensemble model for plotting later
            embeddings = model.last_layer_embeddings
            np.save(os.path.join(generated_dir, 'embeddings.npy'), np.array(embeddings))
    # 将所有预测结果堆叠在一起
    predictions = np.stack(np.array(predictions))
    print("predictions：", predictions)
    # print(predictions.shape)

    # 根据需要生成最终的评分（例如平均值），对5个集成模型取平均值
    final_predictions = np.mean(predictions, axis=0)
    print("final_predictions：", final_predictions)

    return final_predictions
