from keras.layers import Input, Dense, Embedding, CuDNNLSTM, Bidirectional, CuDNNGRU, TimeDistributed
from keras.optimizers import Adam
from keras import Model
from layers import attention
from sklearn.model_selection import KFold
from scipy import stats
import numpy as np
MAXSEN, MAXWORD = 40, 55

class HAN:
    def __init__(self, data, vocab, embedding):
        self.data = data
        self.vocab = vocab
        self.embedding = embedding

    def create_model(self):
        embedding = self.embedding
        sentence_input = Input(shape=(MAXWORD,), dtype='int32')
        embedding_layer = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding], trainable=False)
        embedded_sequences = embedding_layer(sentence_input)
        l_lstm = Bidirectional(CuDNNGRU(100, return_sequences=True))(embedded_sequences)
        l_att = attention(100)(l_lstm)
        sentEncoder = Model(sentence_input, l_att)

        review_input = Input(shape=(MAXSEN, MAXWORD), dtype='int32')
        review_encoder = TimeDistributed(sentEncoder)(review_input)
        l_lstm_sent = Bidirectional(CuDNNGRU(100, return_sequences=True))(review_encoder)
        l_att_sent = attention(100)(l_lstm_sent)
        preds = Dense(3, activation='softmax')(l_att_sent)
        model = Model(review_input, preds)

    def train(self):
        train, test, label = self.data['x_train'], self.data['x_test'], self.data['y_train']
        folds, batch_size = 2, 256
        y_vals = np.zeros(len(label))
        y_test = np.zeros((len(test), folds))
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(label)):
            model = self.create_model()
            model.compile(loss='categorical_crossentropy', optimizer=Adam(1e-3))
            patient, best_score = 0, 0
            x_trn, y_trn = train[train_index], label[train_index]
            x_val, y_val = train[val_index], label[val_index]
            for epoch in range(10):
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
                    model.save_weights('weight')
                print('epoch:{}, score:{}, best_score:{}'.format(epoch, score, best_score))
                patient += 1
                if patient >= 5:
                    break
            # ==========加载最优模型预测测试集=========== #
            model.load_weights('weight')
            test_tok = seq_padding(test, self.vocab)
            predict = model.predict([test_tok])
            y_test[:, fold_n] = np.argmax(predict, -1)
            print("=" * 50)
        y_test = stats.mode(y_test, axis=1)[0].reshape(-1)  # 投票决定结果
        print("=" * 50)
        print('final score: ', f1(label, y_vals))
        return y_test, y_vals