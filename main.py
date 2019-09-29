import han_utils
import pandas as pd
from models.han import Han
from models.sentiment import Sentiment
from keras_bert import load_trained_model_from_checkpoint

config_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'D:/Data/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'

if __name__ == '__main__':
    data = han_utils.load_data()
    embedding, vocab = han_utils.w2v_embedding(data)
    # bert = load_trained_model_from_checkpoint(config_path, checkpoint_path)
    # ========== 建模 ==========
    model = Han(data, vocab, embedding)
    y_test, y_val = model.train()
