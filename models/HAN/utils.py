from gensim.models import Word2Vec
from sklearn.metrics import f1_score
import jieba
import re
import gc
import copy

# 80->99 num_sen and num_word
# 136.0 136.0; 69.0 73.0; 46.0 56.0; 35.0 47.0; 29.0 41.0
train_path = "../input/newssentiment/Train_DataSet.csv"
test_path = "../input/newssentiment/Test_DataSet.csv"
label_path = "../input/newssentiment/Train_DataSet_Label.csv"
word2vec_path = '../input/news-sentiment-corpus/word2vec.model'
max_sen, max_word = 30, 60


def cleaning(text):
    regex = u"[0-9]{1,}"
    p = re.compile(regex)
    text = p.sub('', text)
    english = re.compile(r"[a-zA-Z]{1,}")
    text = english.sub('', text)
    non_chinese = re.compile('[^\u4E00-\u9FA5]{6,}')
    text = non_chinese.sub('', text)
    return text


def load_data():
    data = dict()
    id_train, x_train = [], []
    id_test, x_test = [], []
    label = []
    num_sent, num_word = [], []
    with open(train_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            id_train.append(line[:32])
            line = cleaning(line[33:].replace(',', '。'))
            line = line.split('。')
            num_sent.append(len(line))
            for j, sen in enumerate(line):
                line[j] = ' '.join(jieba.cut(sen)).split()
                num_word.append(len(line[j]))
            x_train.append(line)
    with open(test_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            id_test.append(line[:32])
            line = cleaning(line[33:].replace(',', '。'))
            line = line.split('。')
            for j, sen in enumerate(line):
                line[j] = ' '.join(jieba.cut(sen)).split()
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


def w2v_embedding(data):
    corpus, vocab = [], dict()
    for data in [data['x_train'], data['x_test']]:
        for doc in data:
            corpus.extend(doc)
            for sen in doc:
                for word in sen:
                    if word not in vocab:
                        vocab[word] = len(vocab) + 1
    #     model = Word2Vec(corpus, size=150, window=5, workers=6, min_count=1)
    model = Word2Vec.load(word2vec_path)
    embedding = np.zeros((len(vocab) + 1, 300))
    miss, exist = 0, 0
    for word, token in vocab.items():
        try:
            embedding[token] = model.wv[word]
            exist += 1
        except:
            miss += 1
            continue
    print("miss:{}, exist:{}".format(miss, exist))
    return embedding, vocab


def seq_padding(x, vocab):
    data, x_copy = [], copy.deepcopy(x)
    for i, doc in enumerate(x_copy):
        # =========固定句子长度===========#
        for j, sen in enumerate(doc):
            num_word = len(sen)
            # x[i][j][k]表示第i篇文章第j句话的第k个词
            for k, word in enumerate(sen):
                x_copy[i][j][k] = vocab[word]
            if num_word < max_word:
                x_copy[i][j] = [0] * (max_word - num_word) + x_copy[i][j]
            else:
                x_copy[i][j] = x_copy[i][j][:max_word - 20] + x_copy[i][j][-20:]
        # ==========固定句子数量==========#
        num_sen = len(doc)
        if num_sen < max_sen:
            pad = [0] * max_word
            for j in range(max_sen - num_sen):
                x_copy[i].append(pad)
        else:
            x_copy[i] = x_copy[i][:max_sen - 15] + x_copy[i][-15:]

    data = []
    for doc in x_copy:
        data.append(doc)
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
        np.quantile(slen, 0.9), np.quantile(slen, 0.95), np.quantile(slen, 0.99), np.max(slen)))


def f1(y_true, y_pred):
    macro = f1_score(y_true, y_pred, average='macro')
    return macro