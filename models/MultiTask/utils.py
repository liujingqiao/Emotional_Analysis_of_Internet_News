######################utils.py#############################
import numpy as np
import re
import jieba
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from models.MultiTask.w2v_utils import W2V
vocab_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/vocab.txt'
train_path = "data/Train_DataSet.csv"
test_path = "data/Test_DataSet.csv"
label_path = "data/Train_DataSet_Label.csv"
word2vec = W2V()


def cleaning(text):
    p = re.compile(u"[0-9]{1,}")
    text = p.sub('', text)
    english = re.compile(r"[a-zA-Z]{1,}")
    text = english.sub('', text)
    non_chinese = re.compile('[^\u4E00-\u9FA5]{7,}')
    text = non_chinese.sub('', text)
    split_text = ' '.join(jieba.cut(text)).split()
    return text, split_text

def load_data():
    data = dict()
    """
    x1_train, x2_train作为bert的输入
    w2v_train 作为word2vec的输入（标题和内容合并为一个句子）
    """
    id_train, x1_train, x2_train, w2v_train = [], [], [], []
    id_test, x1_test, x2_test, w2v_test = [], [], [], []
    label = []
    #=============训练集=================#
    with open(train_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            try:
                line = line.strip().split(',')
                id_train.append(line[0])
                line1_char, line1_word = cleaning(line[1])
                line2_char, line2_word = cleaning(line[2])
                x1_train.append(line1_char)
                x2_train.append(line2_char)
                w2v_train.append(line1_word+['。']+line2_word)
            except:
                x2_train.append([])
    #=============测试集=================#
    with open(test_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines[1:]):
            try:
                line = line.strip().split(',')
                id_test.append(line[0])
                line1_char, line1_word = cleaning(line[1]) #字级用于bert，词级用于word2vec
                line2_char, line2_word = cleaning(line[2])
                x1_test.append(line1_char)
                x2_test.append(line2_char)
                w2v_test.append(line1_word+['。']+line2_word)
            except:
                x1_test.append([])
                x2_test.append([])
                w2v_test.append([])
    #=============标签=================#
    with open(label_path, 'r', encoding='utf-8') as fr:
        lines = fr.readlines()
        for line in lines[1:]:
            line = line.strip().split(',')
            label.append(line[1])
    label = list(map(int, label))
    data['x1_train'], data['x2_train'], data['id_train'] = np.array(x1_train), np.array(x2_train), np.array(id_train)
    data['x1_test'], data['x2_test'], data['id_test'] = np.array(x1_test), np.array(x2_test), np.array(id_test)
    data['w2v_train'], data['w2v_test'] = np.array(w2v_train), np.array(w2v_test)
    data['y_train'] = np.array(label)
    return data

def seq_padding(data, max_len=256, padding=0):
    """
    固定句子长度
    """
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
def batch_iter(s1, w2v, label, w2v_vocab, s2=None, max_len=512,
               batch_size=16, shuffle=True, seed=None):
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
            xs1, xs2, w2v_batch = sentence2token(s1[idx], w2v[idx], w2v_vocab, s2[idx], max_len=max_len)
        else:
            xs1, xs2, w2v_batch = sentence2token(s1[idx], w2v[idx], w2v_vocab, max_len=max_len)
        yield xs1, xs2, w2v_batch, label[idx]


def sentence2token(s1, w2v, w2v_vocab, s2=None, max_len=512):
    xs1, xs2 = [], []
    for i in range(len(s1)):
        if type(s2).__name__ != 'NoneType':
            x1, x2 = tokenizer.encode(first=s1[i], second=s2[i])
        else:
            x1, x2 = tokenizer.encode(first=s1[i])
        xs1.append(x1)
        xs2.append(x2)
    w2v = word2vec.seq_padding(w2v, w2v_vocab)
    return seq_padding(xs1, max_len=max_len), seq_padding(xs2, max_len=max_len), w2v


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