from models.BERT.bert import *
import pandas as pd


data = load_data()
config_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '../input/roeberta/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
for i, l in enumerate(bert.layers):
    if i < 4:
        l.trainable = False
    else:
        l.trainable = True
# ========== 建模 ==========
model = Bert(data, bert)
y_test_vote, y_vals_vote, y_test, y_vals = model.train()
y_test_vote = list(map(int,y_test_vote))
y_vals_vote = list(map(int,y_vals_vote))
# ========== 输出结果 ==========
output = pd.DataFrame({'id': data['id_test'], 'label': y_test_vote})
output.to_csv('test_roberta_v1.csv', index=False)
output = pd.DataFrame({'id': data['id_train'], 'label': y_vals_vote})
output.to_csv('val_roberta_v1.csv', index=False)
output = pd.DataFrame({'id': data['id_test'], 'c1': y_test[:,0], 'c2': y_test[:,1], 'c3': y_test[:,2]})
output.to_csv('test_bert_muti_v1.csv', index=False)
output = pd.DataFrame({'id': data['id_train'], 'c1': y_vals[:,0], 'c2': y_vals[:,1], 'c3': y_vals[:,2]})
output.to_csv('val_bert_muti_v1.csv', index=False)

del data, model
gc.collect()