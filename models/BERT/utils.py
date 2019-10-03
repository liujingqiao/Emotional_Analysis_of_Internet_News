################## utils.py ##################
import numpy as np
import re
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
vocab_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
train_path = "../input/newssentiment/Train_DataSet.csv"
test_path = "../input/newssentiment/Test_DataSet.csv"
label_path = "../input/newssentiment/Train_DataSet_Label.csv"

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
                x2_train.append([])
    with open(test_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            try:
                line = line.strip().split(',')
                id_test.append(line[0])
                x1_test.append(cleaning(line[1]))
                x2_test.append(cleaning(line[2]))
            except:
                x2_test.append([])
    with open(label_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            label.append(line[1])
    label = list(map(int, label))
    data['x1_train'], data['x2_train'], data['id_train'] = np.array(x1_train), np.array(x2_train), np.array(id_train)
    data['x1_test'], data['x2_test'], data['id_test'] = np.array(x1_test), np.array(x2_test), np.array(id_test)
    data['y_train'] = np.array(label)
    return data

def seq_padding(data, max_len=256, padding=0):
    max_len -= 3
    for i in range(len(data)):
        t = []
        slen = len(data[i])
        if slen >= max_len:
            t = data[i][-200:]
            data[i] = data[i][:max_len-200] + t
#             data[i] = data[i][:max_len]
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
def batch_iter(s1, label, s2=None, max_len=512,
               batch_size=16, shuffle=True, seed=None):
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
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        idx = index[start_index: end_index]
        if type(s2).__name__ != 'NoneType':
            xs1, xs2 = sentence2token(s1[idx], s2[idx], max_len=max_len)
        else:
            xs1, xs2 = sentence2token(s1[idx], max_len=max_len)
        yield xs1, xs2, label[idx]


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