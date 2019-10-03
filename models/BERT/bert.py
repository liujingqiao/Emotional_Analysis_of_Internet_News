#################### bert.py #####################
import numpy as np
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from keras.layers import Input, Lambda, Dense, concatenate, LSTM, Bidirectional
from keras import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from models.BERT.utils import *
import gc


class Bert:
    def __init__(self, data, bert):
        self.data = data
        self.bert = bert

    def create_model(self):
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,), dtype=tf.int32)
        x = self.bert([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类

        output = Dense(3, activation='softmax')(x)
        model = Model([x1_in, x2_in], output)
        model.summary()
        gc.collect()
        return model

    def train(self):
        x1_train, x2_train, y_train = self.data['x1_train'][:], self.data['x2_train'][:], self.data['y_train'][:]
        x1_test, x2_test = self.data['x1_test'][:], self.data['x2_test'][:]
        folds, batch_size = 5, 2
        y_vals = np.zeros((len(y_train), 3))
        y_test = np.zeros((len(x2_test), 3))
        y_vals_vote = np.zeros(len(y_train))
        model = self.create_model()
        model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-5))
        model.save_weights('origin')
        # ==========folds折交叉验证=========== #
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(y_train)):
            train_index = enhance(train_index, y_train, folds)
            patient, best_score = 0, 0
            if fold_n > 0:
                model.load_weights('origin')

            x1_trn, x2_trn, y_trn = x1_train[train_index], x2_train[train_index], y_train[train_index]
            x1_val, x2_val, y_val = x1_train[val_index], x2_train[val_index], y_train[val_index]
            for epoch in range(8):
                # ==========批量训练=========== #
                generator = batch_iter(x1_trn, y_trn, x2_trn, max_len=512, batch_size=batch_size)
                for x1_tok, x2_tok, lab in generator:
                    model.train_on_batch([x1_tok, x2_tok], np.eye(3)[lab])
                x1_val_tok, x2_val_tok = sentence2token(x1_val, x2_val)
                y_val_pre = model.predict([x1_val_tok, x2_val_tok])
                y_val_vote = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
                _, score = f1(y_val, y_val_vote)
                # ==========EarlyStop=========== #
                if score > best_score:
                    patient = 0
                    best_score = score
                    y_vals_vote[val_index] = y_val_vote
                    y_vals[val_index, :] = y_val_pre
                    model.save_weights('weight')
                print('epoch:{}, score:{}, best_score:{}'.format(epoch, score, best_score))
                patient += 1
                if patient >= 5:
                    break
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('weight')
            x1_test_tok, x2_test_tok = sentence2token(x1_test, x2_test)
            predict = model.predict([x1_test_tok, x2_test_tok])
            y_test += predict
            print("=" * 50)
            del generator, x1_trn, x2_trn, y_trn, x1_val, x2_val, y_val, x1_tok, x2_tok, lab
            gc.collect()
        y_test_vote = np.argmax(y_test, -1)
        print('final score: ', f1(y_train, y_vals_vote))
        y_test /= folds
        return y_test_vote, y_vals_vote, y_test, y_vals