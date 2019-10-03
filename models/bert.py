from bert_utils import f1, sentence2token, batch_iter
import numpy as np
import gc
import tensorflow as tf
from sklearn.model_selection import KFold
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Input, Lambda, Dense
from keras import Model
from keras.optimizers import Adam
from scipy import stats

config_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'


class Bert:
    def __init__(self, data):
        self.data = data
        self.bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
        for l in self.bert.layers:
            l.trainable = True

    def create_model(self):
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,), dtype=tf.int32)
        x = self.bert([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        output = Dense(3, activation='softmax')(x)
        model = Model([x1_in, x2_in], output)
        # print(model.summary())
        gc.collect()
        return model

    def train(self):
        x1_train, x2_train, y_train = self.data['x1_train'][:], self.data['x2_train'][:], self.data['y_train'][:]
        x1_test, x2_test = self.data['x1_test'][:], self.data['x2_test'][:]
        folds, batch_size = 5, 8
        y_vals = np.zeros(len(y_train))
        y_test = np.zeros((len(x2_test), folds))
        # ===========初始化模型=========== #
        model = self.create_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-2))
        model.save_weights('data/temp/origin')
        # ==========folds折交叉验证=========== #
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            patient, best_score = 0, 0
            if fold_n > 0:
                model.load_weights('data/temp/origin')

            x1_trn, x2_trn, y_trn = x1_train[train_index], x2_train[train_index], y_train[train_index]
            x1_val, x2_val, y_val = x1_train[val_index], x2_train[val_index], y_train[val_index]
            for epoch in range(100):
                # ==========批量训练=========== #
                generator = batch_iter(x1_trn, y_trn, x2_trn, max_len=256, batch_size=batch_size)
                for x1_tok, x2_tok, lab in generator:
                    model.train_on_batch([x1_tok, x2_tok], np.eye(3)[lab])
                x1_val_tok, x2_val_tok = sentence2token(x1_val, x2_val)
                y_val_pre = model.predict([x1_val_tok, x2_val_tok])
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
                if patient >= 5:
                    break
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('data/temp/weight')
            x1_test_tok, x2_test_tok = sentence2token(x1_test, x2_test)
            predict = model.predict([x1_test_tok, x2_test_tok])
            y_test[:, fold_n] = np.argmax(predict, -1)
            print("=" * 50)
            del generator, x1_trn, x2_trn, y_trn, x1_val, x2_val, y_val, x1_tok, x2_tok, lab
            gc.collect()
        y_test = stats.mode(y_test, axis=1)[0].reshape(-1)  # 投票决定结果
        print("=" * 50)
        print('final score: ', f1(y_train, y_vals))

        return y_test