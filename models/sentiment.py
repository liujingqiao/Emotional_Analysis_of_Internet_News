from keras.layers import Input, Dense, Embedding, CuDNNLSTM, Bidirectional,\
Conv2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras import Model
from keras import backend as K
from sklearn.model_selection import KFold
from emb_utils import w2v_embedding, make_vocab, batch_iter, f1, seq_padding
from scipy import stats
import tensorflow as tf
import numpy as np


class Sentiment:
    def __init__(self, data):
        self.data = data
        self.vocab = make_vocab(data)
        self.embedding = w2v_embedding(data, self.vocab)

    def create_model(self):
        embedding = self.embedding
        # (batch_size, seq_len)
        inputs = Input(shape=(None,))
        emb_layer = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding], trainable=False)
        emb = emb_layer(inputs)  # (batch_size, seq_len, dim)

        # ========biLSTM========= #
        bilstm_1 = Bidirectional(CuDNNLSTM(units=100, return_sequences=True))(emb)  # (batch_size, seq_len, 100)
        bilstm_2 = Bidirectional(CuDNNLSTM(units=100, return_sequences=True))(
            bilstm_1)  # (batch_size, seq_len, 100)
        bilstm_2_last = bilstm_2[:, -1, :]  # (batch_size, 100)
        bilstm_2_expand = K.expand_dims(bilstm_2_last, axis=-1)  # (batch_size, 100, 1)

        # ========Attention========= #
        # (batch_size, seq_len, 100) * (batch_size, 100, 1) -> (batch_size, seq_len)
        attention = K.squeeze(K.dot(bilstm_2, bilstm_2_expand), 2)
        attn_weights = K.softmax(attention, 1)  # (batch_size, seq_len)
        context = K.dot(K.reshape(bilstm_2, (-1, 100, 300)), K.expand_dims(attn_weights, 2))
        context = K.squeeze(context, 2)
        # ========highway========= #
        #         for i in range(2):
        #             context = self.high_way(context)

        output = Dense(units=3, activation='softmax')(context)
        model = Model(inputs=[inputs], outputs=[output])
        return model

    def train(self):
        train, test, label = self.data['x_train'], self.data['x_test'], self.data['y_train']
        folds, batch_size = 5, 8
        y_vals = np.zeros(len(label))
        y_test = np.zeros((len(label), folds))
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(label)):
            model = self.create_model()
            model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3))
            patient, best_score = 0, 0
            x_trn, y_trn = train[train_index], label[train_index]
            x_val, y_val = train[val_index], label[val_index]
            for epoch in range(100):
                generator = batch_iter(x_trn, y_trn, self.vocab)
                for x_batch, y_batch in generator:
                    model.train_on_batch([x_batch], [np.eye(3)[y_batch]])
                x_val_tok = seq_padding(x_val, self.vocab)
                y_val_pre = model.predict(x_val_tok)
                y_val_pre = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
                score = f1(y_val, y_val_pre)
                # ==========EarlyStop=========== #
                if score > best_score:
                    patient = 0
                    best_score = score
                    y_vals[val_index] = y_val_pre
                    model.save_weights('data/temp/weight')
                print('epoch:{}, score:{}, best_score:{}'.format(epoch, score, best_score))
                patient += 1
                if patient >= 10:
                    break
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('data/temp/weight')
            test_tok = seq_padding(test, self.vocab)
            predict = model.predict([test_tok])
            y_test[:, fold_n] = np.argmax(predict, -1)
            print("=" * 50)
        y_test = stats.mode(y_test, axis=1)[0].reshape(-1)  # 投票决定结果
        print("=" * 50)
        print('final score: ', f1(label, y_vals))

        return y_test, y_val






