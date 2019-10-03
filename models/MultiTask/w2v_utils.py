###################word2vec###########################
from gensim.models import Word2Vec
import numpy as np

train_path = "../input/newssentiment/Train_DataSet.csv"
test_path = "../input/newssentiment/Test_DataSet.csv"
label_path = "../input/newssentiment/Train_DataSet_Label.csv"
model_path = "../input/news-sentiment-corpus/word2vec.model"

class W2V:

    def make_vocab(self, data):
        vocab = dict()
        for article in [data['w2v_train'], data['w2v_test']]:
            for sentence in article:
                for word in sentence:
                    if word not in vocab:
                        vocab[word] = len(vocab)+1
        return vocab


    def w2v_embedding(self, vocab):
        model = Word2Vec.load(model_path)
        embedding = np.zeros((len(vocab)+1,300))
        for word, token in vocab.items():
            try:
                embedding[token] = model.wv[word]
            except:
                continue
        return embedding


    def seq_padding(self, x, vocab):
        max_len = 500
        data = []
        for i, sen in enumerate(x):
            t = []
            for word in sen:
                t.append(vocab[word])
            slen = len(sen)
            if slen < max_len:
                t = [0]*(max_len - slen)+t
                data.append(t)
            else:
                data.append(t[:max_len-250] + t[-250:])
        return np.array(data)

