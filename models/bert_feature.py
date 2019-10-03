#################### bert.py #####################
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from keras.layers import Input, Lambda, Dense, CuDNNLSTM, Bidirectional
from keras import Model
from keras import backend as K
from keras.optimizers import Adam
from scipy import stats
from bert_utils import sentence2token, batch_iter, f1
import gc

config_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'


class Bert:
    def __init__(self, data, bert):
        self.data = data
        self.bert = bert

    def create_model(self):
        bert_emb = Input(shape=(None, 758,), dtype=tf.float32)
        bilstm = Bidirectional(CuDNNLSTM(units=100, return_sequences=False))(bert_emb)  # (batch_size, seq_len, 100)
        output = Dense(3, activation='softmax')(bilstm)
        model = Model([bert_emb], [output])
        gc.collect()
        return model

    def train(self):
        x1_train, x2_train, y_train = self.data['x1_train'][:], self.data['x2_train'][:], self.data['y_train'][:]
        x1_test, x2_test = self.data['x1_test'][:], self.data['x2_test'][:]
        folds, batch_size = 5, 8
        y_vals = np.zeros(len(y_train))
        y_test = np.zeros((len(x2_test), folds))
        model = self.create_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5))
        model.save_weights('origin')
        # ==========folds折交叉验证=========== #
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            patient, best_score = 0, 0
            if fold_n > 0:
                model.load_weights('origin')

            x1_trn, x2_trn, y_trn = x1_train[train_index], x2_train[train_index], y_train[train_index]
            x1_val, x2_val, y_val = x1_train[val_index], x2_train[val_index], y_train[val_index]
            for epoch in range(8):
                # ==========批量训练=========== #
                generator = batch_iter(x1_trn, y_trn, x2_trn, max_len=512, batch_size=batch_size)
                for x1_tok, x2_tok, lab in generator:
                    bert_feature = self.bert.predict([np.array(x1_tok), np.array(x2_tok)])
                    print(bert_feature.shape)
                    model.train_on_batch([bert_feature], np.eye(3)[lab])
                x1_val_tok, x2_val_tok = sentence2token(x1_val, x2_val)
                bert_feature = self.bert.predict([np.array(x1_val_tok), np.array(x2_val_tok)])
                y_val_pre = model.predict([x1_val_tok, x2_val_tok])
                y_val_pre = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
                _, score = f1(y_val, y_val_pre)
                # ==========EarlyStop=========== #
                if score > best_score:
                    patient = 0
                    best_score = score
                    y_vals[val_index] = y_val_pre
                    model.save_weights('weight')
                print('epoch:{}, score:{}, best_score:{}'.format(epoch, score, best_score))
                patient += 1
                if patient >= 5:
                    break
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('weight')
            x1_test_tok, x2_test_tok = sentence2token(x1_test, x2_test)
            bert_feature = self.bert.predict([np.array(x1_test_tok), np.array(x2_test_tok)])
            predict = model.predict([x1_test_tok, x2_test_tok])
            y_test[:, fold_n] = np.argmax(predict, -1)
            print("=" * 50)
            del generator, x1_trn, x2_trn, y_trn, x1_val, x2_val, y_val, x1_tok, x2_tok, lab
            gc.collect()
        y_test = stats.mode(y_test, axis=1)[0].reshape(-1)  # 投票决定结果
        print('final score: ', f1(y_train, y_vals))

        return y_test, y_vals