from keras.layers import Input, Dense, Embedding, CuDNNLSTM, Bidirectional, GlobalAveragePooling1D, \
    Conv1D, GlobalMaxPooling1D, Concatenate
from keras import Model
from keras_radam.training import RAdamOptimizer
from sklearn.model_selection import KFold
from models.RCNN.utils import batch_iter, f1, seq_padding
from scipy import stats
import numpy as np


class Sentiment:
    def __init__(self, data, vocab, embedding):
        self.data = data
        self.vocab = vocab
        self.embedding = embedding

    def concat_embedding(self, w2v, glove):
        return np.concatenate([w2v, glove], axis=-1)

    def create_model(self):
        embedding = self.embedding
        # (batch_size, seq_len)
        inputs = Input(shape=(None,))
        emb_layer = Embedding(embedding.shape[0], embedding.shape[1], weights=[embedding], trainable=True)
        emb = emb_layer(inputs)  # (batch_size, seq_len, dim)
        bilstm_1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(emb)
        bilstm_2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(bilstm_1)
        x = Concatenate()([bilstm_1, bilstm_2])
        convs = []
        for kernel_size in range(1, 5):
            conv = Conv1D(256, kernel_size, activation='relu')(x)
            convs.append(conv)
        poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv in convs]
        x = Concatenate()(poolings)

        output = Dense(units=3, activation='softmax')(x)

        model = Model(inputs=[inputs], outputs=[output])
        return model

    def train(self):
        train, test, label = self.data['x_train'], self.data['x_test'], self.data['y_train']
        folds, batch_size = 5, 256
        y_vals = np.zeros(len(label))
        y_test = np.zeros((len(test), folds))
        kf = KFold(n_splits=folds, shuffle=True, random_state=10)
        for fold_n, (train_index, val_index) in enumerate(kf.split(label)):
            model = self.create_model()
            model.compile(loss='categorical_crossentropy', optimizer=RAdamOptimizer(learning_rate=1e-4))
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
                    model.save_weights('weight')
                print('epoch:{}, score:{}, best_score:{}'.format(epoch, score, best_score))
                patient += 1
                if patient >= 8:
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