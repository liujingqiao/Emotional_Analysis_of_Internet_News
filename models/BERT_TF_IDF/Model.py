#################### bert.py #####################
import numpy as np
from models.BERT_TF_IDF.utils import *
from keras.layers import *
from keras import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from scipy import stats
from tqdm import tqdm
import time
import gc


class Bert:
    def __init__(self, train, test, bert, feature):
        self.train = train
        self.test = test
        self.feature = feature
        self.bert = bert
        self.feature_dim = len(feature)

    def create_model(self):
        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))
        x = self.bert([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        x_drop = Dropout(0.5)(x)

        ###############fature#############
        x3_in = Input(shape=(self.feature_dim,))
        dense = BatchNormalization()(x3_in)
        for i in range(8):
            dense = Highway()(dense)

        output = concatenate([x_drop, dense], axis=-1)
        for i in range(3):
            output = Dense(1024, activation='relu')(output)
        output = Dense(3, activation='softmax')(output)
        model = Model([x1_in, x2_in, x3_in], output)
        model.summary()
        return model

    def forward(self):
        x1_train, x2_train, y_train = self.train['title'][:].values, self.train['content'][:].values, self.train[
                                                                                                          'label'][
                                                                                                      :].values
        x1_test, x2_test = self.test['title'][:].values, self.test['content'][:].values
        x3_train, x3_test = self.train[self.feature][:].values, self.test[self.feature][:].values
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
            x3_trn, x3_val = x3_train[train_index], x3_train[val_index]
            for epoch in range(8):
                start = time.time()
                # ==========批量训练=========== #
                generator = batch_iter(x1_trn, y_trn, s2=x2_trn, s3=x3_trn, max_len=490, batch_size=batch_size)
                for x1_tok, x2_tok, x3_tok, lab in generator:
                    model.train_on_batch([x1_tok, x2_tok, x3_tok], np.eye(3)[lab])
                x1_val_tok, x2_val_tok = sentence2token(x1_val, x2_val, max_len=490)
                y_val_pre = model.predict([x1_val_tok, x2_val_tok, x3_val])
                y_val_vote = np.argmax(y_val_pre, -1)  # 最大的值所在的索引作为预测结果
                _, score = f1(y_val, y_val_vote)
                # ==========EarlyStop=========== #
                if score > best_score:
                    patient = 0
                    best_score = score
                    y_vals_vote[val_index] = y_val_vote
                    y_vals[val_index, :] = y_val_pre
                    model.save_weights('weight')
                cost = time.time() - start
                print('epoch:{}, score:{}, best_score:{}, cost_time:{}'.format(epoch, score, best_score, cost))
                if patient >= 3:
                    break
                patient += 1
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('weight')
            x1_test_tok, x2_test_tok = sentence2token(x1_test, x2_test, max_len=490)
            predict = model.predict([x1_test_tok, x2_test_tok, x3_test])
            y_test += predict
            print("=" * 50)
            gc.collect()
        y_test_vote = np.argmax(y_test, -1)
        print('final score: ', f1(y_train, y_vals_vote))
        y_test /= folds
        return y_test_vote, y_vals_vote, y_test, y_vals