import numpy as np
import re
import jieba
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

drop = ['5cd2edc53c584034bc8a56040cb2ed95', 'e528b35269fc4fba956f00d9b0f3efb3', '16ea18af0b5f4603b195cf9f62121237',
        '7dcb12cca55649198685f919bf654c2c',
        'c6792e08d2cc431b93e286f005d782ac', '90b3928ed61e48f08853b968a571bab5', 'a75e1768157041bbbb36d6a0b8930355',
        '3f391793acb248c787a95ae8279bba21',
        'e7f1aa4595ea4a29a66089545ad8e325', 'e9055d76030b4c9aa17d5ffa94e68799', 'b19ba952226e4d69ac18ef9ac40f6ebc',
        'ede76656f66f4c239c063856c554ce86']
vocab_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
train_path = "../input/newssentiment/Train_DataSet.csv"
test_path = "../input/newssentiment/Test_DataSet.csv"
label_path = "../input/newssentiment/Train_DataSet_Label.csv"


def idf_vector(train, test):
    corpus = []
    corpus.extend(train['title_split'])
    corpus.extend(train['content_split'])
    corpus.extend(test['title_split'])
    corpus.extend(test['content_split'])

    tfidf = TfidfVectorizer(token_pattern='\\b\\w+\\b')
    tfidf.fit(corpus)
    # tfidf
    for data in [train, test]:
        vector = tfidf.transform(data['title_split'])
        svd_enc = TruncatedSVD(n_components=100, random_state=4520)
        title_svd = svd_enc.fit_transform(vector)
        vector = tfidf.transform(data['content_split'])
        svd_enc = TruncatedSVD(n_components=200, random_state=4520)
        content_svd = svd_enc.fit_transform(vector)
        svd = np.concatenate((title_svd, content_svd), axis=1)
        for i in range(svd.shape[1]):
            col = 'idf_{}'.format(i)
            data[col] = svd[:, i]
    return train, test


def cleaning(text):
    regex = u"[0-9]{1,}"
    p = re.compile(regex)
    text = p.sub('', text)
    english = re.compile(r"[a-zA-Z]{1,}")
    text = english.sub('', text)
    non_chinese = re.compile('[^\u4E00-\u9FA5]{7,}')
    text = non_chinese.sub('', text)
    return text


def load_data():
    data = dict()
    id_train, x1_train, x2_train = [], [], []
    id_test, x1_test, x2_test = [], [], []
    label = []
    with open(train_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            try:
                line = line.strip().split(',')
                id_train.append(line[0])
                x1_train.append(cleaning(line[1]))
                x2_train.append(cleaning(line[2]))
            except:
                print(line)
                x2_train.append('空值')
    with open(test_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            try:
                line = line.strip().split(',')
                id_test.append(line[0])
                x1_test.append(cleaning(line[1]))
                x2_test.append(cleaning(line[2]))
            except:
                print(line)
                x2_test.append('空值')
    with open(label_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            label.append(line[1])
    label = list(map(int, label))
    train = pd.DataFrame({'id': id_train, 'title': x1_train, 'content': x2_train, 'label': label})
    test = pd.DataFrame({'id': id_test, 'title': x1_test, 'content': x2_test})
    train = train[~train['id'].isin(drop)]
    train.drop_duplicates(subset=['title', 'content'], inplace=True)
    for i, text in enumerate([train, test]):
        text['title_split'] = text['title'].apply(lambda x: ' '.join(jieba.cut(x)))
        text['content_split'] = text['content'].apply(lambda x: ' '.join(jieba.cut(x)))
    train, test = idf_vector(train, test)

    return train, test


def seq_padding(data, max_len=256, padding=0):
    max_len -= 3
    for i in range(len(data)):
        t = []
        slen = len(data[i])
        if slen >= max_len:
            t = data[i][-200:]
            data[i] = data[i][:max_len - 200] + t
        if slen < max_len:
            data[i] += [padding] * (max_len - slen)
    return data


from keras_bert import load_trained_model_from_checkpoint, Tokenizer


class MyTokenizer(Tokenizer):
    def _tokenize(self, text):
        token = []
        for c in text:
            if c in self._token_dict:
                token.append(c)
            elif self._is_space(c):
                token.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                token.append('[UNK]')  # 剩余的字符是[UNK]
        return token


def token_dict():
    import codecs
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    return token_dict


vocab = token_dict()
tokenizer = MyTokenizer(vocab)


def batch_iter(s1, label, s2=None, s3=None, max_len=512,
               batch_size=16, shuffle=True, seed=None, mode=0):
    vocab = token_dict()
    tokenizer = MyTokenizer(vocab)
    s1, label = np.array(s1), np.array(label)
    if type(s2).__name__ != 'NoneType':
        s2 = np.array(s2)
    data_size = len(s1)
    index = np.arange(data_size)
    num_batches_per_epoch = int((len(s1) - 1) / batch_size) + 1

    if shuffle:
        if seed:
            np.random.seed(seed)
        np.random.shuffle(index)

    for batch_num in range(num_batches_per_epoch):
        if mode == 1:
            np.random.shuffle(index)
            idx = index[0:batch_size]
        else:
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            idx = index[start_index: end_index]
        if type(s2).__name__ != 'NoneType':
            xs1, xs2 = sentence2token(s1[idx], s2[idx], max_len=max_len)
        else:
            xs1, xs2 = sentence2token(s1[idx], max_len=max_len)
        xs3 = s3[idx]
        yield xs1, xs2, xs3, label[idx]


def sentence2token(s1, s2=None, max_len=512):
    xs1, xs2 = [], []
    for i in range(len(s1)):
        if type(s2).__name__ != 'NoneType':
            x1, x2 = tokenizer.encode(first=s1[i], second=s2[i])
        else:
            x1, x2 = tokenizer.encode(first=s1[i])
        xs1.append(x1)
        xs2.append(x2)
    return seq_padding(xs1, max_len=max_len), seq_padding(xs2, max_len=max_len)


def f1(y_true, y_pred):
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return micro_f1, macro_f1


def enhance(index, label, folds):
    c1 = np.argwhere(label[index] == 0).reshape(-1)
    c2 = np.argwhere(label[index] == 2).reshape(-1)
    list_index = list(index)
    for j, c in enumerate([c1, c2]):
        # =========数据增强======== #
        if j == 0:
            list_index.extend(list(index[c]) * 3)
        if j == 1:
            list_index.extend(list(index[c])[:int(900 / folds)])
    np.random.shuffle(list_index)
    return list_index