# test_bert_v1.csv-1-google-bert-head
# val_bert_v1.csv-1-google-bert
# test_bert_v2.csv-2-wwm-bert-head
# val_bert_v2.csv-2-wwm-bert


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, RepeatedKFold
from keras.layers import Input, Embedding, Dense, GlobalMaxPool1D
from keras import Model
from keras.optimizers import Adam
from scipy import stats
from bert_utils import f1

t1 = pd.read_csv('../data/stacking_data/test_roberta_muti_v1.csv')
t2 = pd.read_csv('../data/stacking_data/test_team_v1.csv')
t3 = pd.read_csv('../data/stacking_data/test_wwm_v1.csv')
t4 = pd.read_csv('../data/stacking_data/test_bert_muti_v1.csv')
t = (t1[['c1', 'c2', 'c3']] + t2[['c1', 'c2', 'c3']]*2.5+ t3[['c1', 'c2', 'c3']]+t4[['c1', 'c2', 'c3']]).values
t = pd.DataFrame({'id':t1['id'], 'label': np.argmax(t, axis=-1)})
t.to_csv('../data/stacking_data/combination.csv', index=False)

# train_bert_v1 = pd.read_csv("../data/stacking_data/val_bert_v1.csv")
# train = pd.DataFrame({"id":train_bert_v1['id'], "v1":train_bert_v1['label']})
# train['v2'] = pd.read_csv("../data/stacking_data/val_bert_v2.csv", usecols=['label'])['label']
# train['v3'] = pd.read_csv("../data/stacking_data/val_han.csv", usecols=['label'])['label']
# train['v4'] = pd.read_csv("../data/stacking_data/val_w2v.csv", usecols=['label'])['label']
# train['label'] = pd.read_csv("../data/Train_DataSet_Label.csv", usecols=['label'])['label']
#
# test_bert_v1 = pd.read_csv("../data/stacking_data/test_bert_v1.csv")
# test = pd.DataFrame({"id":test_bert_v1['id'], "v1":test_bert_v1['label']})
# test['v2'] = pd.read_csv("../data/stacking_data/test_bert_v2.csv", usecols=['label'])['label']
# test['v3'] = pd.read_csv("../data/stacking_data/test_han.csv", usecols=['label'])['label']
# test['v4'] = pd.read_csv("../data/stacking_data/test_w2v.csv", usecols=['label'])['label']
# test['v5'] = pd.read_csv("../data/stacking_data/best.csv", usecols=['label'])['label']
#
# def make_one_hot(data):
#     cols = data.columns
#     one_hot = np.eye(3)
#     t = None
#     for i, col in enumerate(cols):
#         if i == 0:
#             t = one_hot[data[col]]
#         else:
#             t = np.concatenate([t, one_hot[data[col]]], axis=-1)
#     return t
#
#
#
# test['label'] = stats.mode(test[['v1', 'v2', 'v5', 'v3', 'v4']].values, axis=1)[0].reshape(-1)
# test[['id','label']].to_csv('../data/stacking_data/combination.csv', index=False)
# import lightgbm as lgb
# params={
#     'learning_rate':0.0001,
#     'lambda_l1':0.1,
#     'lambda_l2':0.2,
#     'max_depth':4,
#     'objective':'multiclass',
#     'num_class':3,
# }
# train_one_hot = make_one_hot(train[['v1', 'v2', 'v3', 'v4']])
# test_one_hot = make_one_hot(test[['v1', 'v2', 'v3', 'v4']])
# folds, repeate, batch_size = 10, 5, 8
# y_vals = np.zeros((len(train_one_hot), repeate))
# y_test = np.zeros((len(test_one_hot), folds*repeate))
# # kf = KFold(n_splits=folds, shuffle=True, random_state=10)
# kf = RepeatedKFold(n_splits=folds, n_repeats=3, random_state=4520)
# for fold_n, (train_index, val_index) in enumerate(kf.split(train['label'])):
#     patient, best_score = 0, 0
#     x_trn, y_trn = train_one_hot[train_index], train['label'][train_index]
#     x_val, y_val = train_one_hot[val_index], train['label'][val_index]
#
#     train_data = lgb.Dataset(x_trn, label=y_trn)
#     validation_data = lgb.Dataset(x_val, label=y_val)
#     clf=lgb.train(params, train_data, valid_sets=[validation_data], early_stopping_rounds = 300)
#     y_val_pre = clf.predict(x_val)
#     y_val_pre = np.argmax(y_val_pre, -1)
#     y_vals[val_index, int(fold_n/folds)] = y_val_pre
#     _, score = f1(y_val, y_val_pre)
#     predict = clf.predict(test_one_hot, num_iteration=clf.best_iteration)
#     y_test[:, fold_n] = np.argmax(predict, -1)
#     print("=" * 50)
# y_test = stats.mode(y_test, axis=1)[0].reshape(-1)  # 投票决定结果
# y_vals = stats.mode(y_vals, axis=1)[0].reshape(-1)  # 投票决定结果
# test['label'] = list(map(int, y_test))
# test[['id','label']].to_csv('../data/stacking_data/combination.csv', index=False)
# print('final score: ', f1(train['label'], y_vals))