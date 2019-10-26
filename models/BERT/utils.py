# -*- coding:utf-8 -*-
################## utils.py ##################
import numpy as np
import jieba
import re
import synonyms
import time
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
vocab_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
train_path = "data/Train_DataSet.csv"
test_path = "data/Test_DataSet.csv"
label_path = "data/Train_DataSet_Label.csv"
stop_words = ['———','》），','）÷（１－','”，','）、','＝（',':','→','℃','&','*','一一','~~~~','’','.','『','.一','./','--','』','＝″','【','［＊］','｝＞','［⑤］］','［①Ｄ］','ｃ］','ｎｇ昉','＊','//','［','］','［②ｅ］','［②ｇ］','＝｛','}','，也','‘','Ａ','［①⑥］','［②Ｂ］','［①ａ］','［④ａ］','［①③］','［③ｈ］','③］','１．','－－','［②ｂ］','’‘','×××','［①⑧］','０：２','＝［','［⑤ｂ］','［②ｃ］','［④ｂ］','［②③］','［③ａ］','［④ｃ］','［①⑤］','［①⑦］','［①ｇ］','∈［','［①⑨］','［①④］','［①ｃ］','［②ｆ］','［②⑧］','［②①］','［①Ｃ］','［③ｃ］','［③ｇ］','［②⑤］','［②②］','一.','［①ｈ］','.数','［］','［①Ｂ］','数/','［①ｉ］','［③ｅ］','［①①］','［④ｄ］','［④ｅ］','［③ｂ］','［⑤ａ］','［①Ａ］','［②⑧］','［②⑦］','［①ｄ］','［②ｊ］','〕〔','］［','://','′∈','［②④','［⑤ｅ］','１２％','ｂ］','...','...................','…………………………………………………③','ＺＸＦＩＴＬ','［③Ｆ］','」','［①ｏ］','］∧′＝［','∪φ∈','′｜','｛－','②ｃ','｝','［③①］','Ｒ．Ｌ．','［①Ｅ］','Ψ','－［＊］－','↑','.日','［②ｄ］','［②','［②⑦］','［②②］','［③ｅ］','［①ｉ］','［①Ｂ］','［①ｈ］','［①ｄ］','［①ｇ］','［①②］','［②ａ］','ｆ］','［⑩］','ａ］','［①ｅ］','［②ｈ］','［②⑥］','［③ｄ］','［②⑩］','ｅ］','〉','】','元／吨','［②⑩］','２．３％','５：０','［①］','::','［②］','［③］','［④］','［⑤］','［⑥］','［⑦］','［⑧］','［⑨］','……','——','?','、','。','“','”','《','》','！','，','：','；','？','．',',','．','?','·','———','──','?','—','<','>','（','）','〔','〕','[',']','(',')','-','+','～','×','／','/','①','②','③','④','⑤','⑥','⑦','⑧','⑨','⑩','Ⅲ','В','"',';','#','@','γ','μ','φ','φ．','×','Δ','■','▲','sub','exp','sup','sub','Lex','＃','％','＆','＇','＋','＋ξ','＋＋','－','－β','＜','＜±','＜Δ','＜λ','＜φ','＜＜','=','＝','＝☆','＝－','＞','＞λ','＿','～±','～＋','［⑤ｆ］','［⑤ｄ］','［②ｉ］','≈','［②Ｇ］','［①ｆ］','ＬＩ','㈧','［－','......','〉','［③⑩］','第二','一番','一直','一个','一些','许多','种','有的是','也就是说','末##末','啊','阿','哎','哎呀','哎哟','唉','俺','俺们','按','按照','吧','吧哒','把','罢了','被','本','本着','比','比方','比如','鄙人','彼','彼此','边','别','别的','别说','并','并且','不比','不成','不单','不但','不独','不管','不光','不过','不仅','不拘','不论','不怕','不然','不如','不特','不惟','不问','不只','朝','朝着','趁','趁着','乘','冲','除','除此之外','除非','除了','此','此间','此外','从','从而','打','待','但','但是','当','当着','到','得','的','的话','等','等等','地','第','叮咚','对','对于','多','多少','而','而况','而且','而是','而外','而言','而已','尔后','反过来','反过来说','反之','非但','非徒','否则','嘎','嘎登','该','赶','个','各','各个','各位','各种','各自','给','根据','跟','故','故此','固然','关于','管','归','果然','果真','过','哈','哈哈','呵','和','何','何处','何况','何时','嘿','哼','哼唷','呼哧','乎','哗','还是','还有','换句话说','换言之','或','或是','或者','极了','及','及其','及至','即','即便','即或','即令','即若','即使','几','几时','己','既','既然','既是','继而','加之','假如','假若','假使','鉴于','将','较','较之','叫','接着','结果','借','紧接着','进而','尽','尽管','经','经过','就','就是','就是说','据','具体地说','具体说来','开始','开外','靠','咳','可','可见','可是','可以','况且','啦','来','来着','离','例如','哩','连','连同','两者','了','临','另','另外','另一方面','论','嘛','吗','慢说','漫说','冒','么','每','每当','们','莫若','某','某个','某些','拿','哪','哪边','哪儿','哪个','哪里','哪年','哪怕','哪天','哪些','哪样','那','那边','那儿','那个','那会儿','那里','那么','那么些','那么样','那时','那些','那样','乃','乃至','呢','能','你','你们','您','宁','宁可','宁肯','宁愿','哦','呕','啪达','旁人','呸','凭','凭借','其','其次','其二','其他','其它','其一','其余','其中','起','起见','起见','岂但','恰恰相反','前后','前者','且','然而','然后','然则','让','人家','任','任何','任凭','如','如此','如果','如何','如其','如若','如上所述','若','若非','若是','啥','上下','尚且','设若','设使','甚而','甚么','甚至','省得','时候','什么','什么样','使得','是','是的','首先','谁','谁知','顺','顺着','似的','虽','虽然','虽说','虽则','随','随着','所','所以','他','他们','他人','它','它们','她','她们','倘','倘或','倘然','倘若','倘使','腾','替','通过','同','同时','哇','万一','往','望','为','为何','为了','为什么','为着','喂','嗡嗡','我','我们','呜','呜呼','乌乎','无论','无宁','毋宁','嘻','吓','相对而言','像','向','向着','嘘','呀','焉','沿','沿着','要','要不','要不然','要不是','要么','要是','也','也罢','也好','一','一般','一旦','一方面','一来','一切','一样','一则','依','依照','矣','以','以便','以及','以免','以至','以至于','以致','抑或','因','因此','因而','因为','哟','用','由','由此可见','由于','有','有的','有关','有些','又','于','于是','于是乎','与','与此同时','与否','与其','越是','云云','哉','再说','再者','在','在下','咱','咱们','则','怎','怎么','怎么办','怎么样','怎样','咋','照','照着','者','这','这边','这儿','这个','这会儿','这就是说','这里','这么','这么点儿','这么些','这么样','这时','这些','这样','正如','吱','之','之类','之所以','之一','只是','只限','只要','只有','至','至于','诸位','着','着呢','自','自从','自个儿','自各儿','自己','自家','自身','综上所述','总的来看','总的来说','总的说来','总而言之','总之','纵','纵令','纵然','纵使','遵照','作为','兮','呃','呗','咚','咦','喏','啐','喔唷','嗬','嗯','嗳']

def cleaning(text):
    regex = u"[0-9]{1,}"
    p = re.compile(regex)
    text = p.sub('', text)
    english = re.compile(r"[a-zA-Z]{1,}")
    text = english.sub('', text)
    non_chinese = re.compile('[^\u4E00-\u9FA5]{7,}')
    text = non_chinese.sub('', text)
    return text
def synonym_replacement(words):
    # ============1. 去掉停用词并打同义乱替换顺序================#
    words = list(jieba.cut(words))
    new_words = words.copy()
    if len(words) > 512:
        new_words = new_words[:256] + new_words[-256:]
    # 忽略停用词
    random_word_list = list(set([word for word in new_words if word not in stop_words]))
    np.random.shuffle(random_word_list)
    n = len(random_word_list)*0.6
    num_replaced = 0
    # ============2. 遍历句子替换n个词的同义词=============#
    for random_word in random_word_list:
        synonym = synonyms.nearby(random_word)[0][1]
        if len(synonym) >= 1:
            synonym = np.random.choice(synonym)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:
            break

    new_words = ''.join(new_words).replace(' ','')
    return new_words

def augment(x1, x2, label):
    idx_0 = np.argwhere(label==0).reshape(-1)
    idx_1 = np.random.permutation(np.argwhere(label==1).reshape(-1))
    idx_2 = np.random.permutation(np.argwhere(label==2).reshape(-1))

    for _ in range(4):
        for i, idx in enumerate(idx_0):
            print(x1[idx])
            time_start = time.time()
            x1 = np.append(x1,synonym_replacement(x1[idx]))
            print(x1[-1])
            x2 = np.append(x2,synonym_replacement(x2[idx]))
            time_end=time.time()
            print('cost',time_end-time_start)
        label = np.append(label, [0]*len(idx_0))
    for idx in idx_1[:700]:
        x1 = np.append(x1,synonym_replacement(x1[idx]))
        x2 = np.append(x2,synonym_replacement(x2[idx]))
        label = np.append(label, 1)
    for idx in idx_2[:1600]:
        x1 = np.append(x1,synonym_replacement(x1[idx]))
        x2 = np.append(x2,synonym_replacement(x2[idx]))
        label = np.append(label, 2)
    idx = np.random.permutation(np.arange(len(x1)))
    return x1[idx], x2[idx], label[idx]
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

#
# vocab = token_dict()
# tokenizer = MyTokenizer(vocab)
# def batch_iter(s1, label, s2=None, max_len=512,
#                batch_size=16, shuffle=True, seed=None):
#     vocab = token_dict()
#     tokenizer = MyTokenizer(vocab)
#     s1, label = np.array(s1), np.array(label)
#     if type(s2).__name__ != 'NoneType':
#         s2 = np.array(s2)
#     data_size = len(s1)
#     index = np.arange(data_size)
#     num_batches_per_epoch = int((len(s1) - 1) / batch_size) + 1
#
#     if shuffle:
#         if seed:
#             np.random.seed(seed)
#         np.random.shuffle(index)
#
#     for batch_num in range(num_batches_per_epoch):
#         start_index = batch_num * batch_size
#         end_index = min((batch_num + 1) * batch_size, data_size)
#         idx = index[start_index: end_index]
#         if type(s2).__name__ != 'NoneType':
#             xs1, xs2 = sentence2token(s1[idx], s2[idx], max_len=max_len)
#         else:
#             xs1, xs2 = sentence2token(s1[idx], max_len=max_len)
#         yield xs1, xs2, label[idx]
#
#
# def sentence2token(s1, s2=None, max_len=512):
#     xs1, xs2 = [], []
#     for i in range(len(s1)):
#         if type(s2).__name__ != 'NoneType':
#             x1, x2 = tokenizer.encode(first=s1[i], second=s2[i])
#         else:
#             x1, x2 = tokenizer.encode(first=s1[i])
#         xs1.append(x1)
#         xs2.append(x2)
#     return seq_padding(xs1, max_len=max_len), seq_padding(xs2, max_len=max_len)
#
#
# def f1(y_true, y_pred):
#     micro_f1 = f1_score(y_true, y_pred, average='micro')
#     macro_f1 = f1_score(y_true, y_pred, average='macro')
#     return micro_f1, macro_f1
#
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