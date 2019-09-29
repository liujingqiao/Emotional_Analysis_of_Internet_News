###################utils.py###########################
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
import numpy as np
import jieba
import re

train_path = "../input/newssentiment/Train_DataSet.csv"
test_path = "../input/newssentiment/Test_DataSet.csv"
label_path = "../input/newssentiment/Train_DataSet_Label.csv"
model_path = "../input/news-sentiment-corpus/word2vec.model"

def cleaning(text):
    regex = u"[0-9]{1,}"
    p = re.compile(regex)
    text = p.sub('##', text)
    english = re.compile(r"[a-zA-Z]{1,}")
    text = english.sub('', text)
    non_chinese = re.compile('[^\u4E00-\u9FA5]{6,}')
    text = non_chinese.sub('', text)
    text = ' '.join(jieba.cut(text)).split()
    return text


def load_data():
    data = dict()
    id_train, x_train = [], []
    id_test, x_test = [], []
    label = []
    with open(train_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            id_train.append(line[:32])
            line = cleaning(line[33:])
            x_train.append(line)
    with open(test_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            id_test.append(line[:32])
            line = cleaning(line[33:])
            x_test.append(line)
    with open(label_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            label.append(line[1])
    label = list(map(int, label))
    data['x_train'], data['id_train'] = np.array(x_train), np.array(id_train)
    data['x_test'], data['id_test'] = np.array(x_test), np.array(id_test)
    data['y_train'] = np.array(label)
    return data


def make_vocab(data):
    vocab = dict()
    for article in [data['x_train'], data['x_test']]:
        for sentence in article:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab)+1
    return vocab


def w2v_embedding(data, vocab):
    corpus = []
    corpus.extend(list(data['x_train']))
    corpus.extend(list(data['x_test']))
#     model = Word2Vec(corpus, size=150, window=5, workers=6, min_count=1)
    model = Word2Vec.load(model_path)
    embedding = np.zeros((len(vocab)+1,300))
    for word, token in vocab.items():
        try:
            embedding[token] = model.wv[word]
        except:
            continue
    return embedding


def seq_padding(x, vocab):
    max_len = 1000
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


def batch_iter(X, label, vocab,
               batch_size=16, shuffle=True, seed=None):
    X, label = np.array(X), np.array(label)
    data_size = len(X)
    index = np.arange(data_size)
    num_batches_per_epoch = int((len(X) - 1) / batch_size) + 1

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        idx = index[start_index: end_index]
        x_batch, y_batch = X[idx], label[idx]
        yield seq_padding(x_batch, vocab), y_batch


def check(data):
    slen = np.zeros(len(data['x1_train']))
    for i, line in enumerate(data['x1_train']):
        slen[i] = len(data['x1_train'][i])
    print("0.9:{}, 0.95:{}, 0.99:{}, max:{}:".format(
        np.quantile(slen, 0.9), np.quantile(slen, 0.95), np.quantile(slen,0.99), np.max(slen)))


def f1(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average='macro')
    return  macro