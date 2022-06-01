from model import RNN, GRU, LSTM, BiLSTM
from word2vec import create_embedding
import numpy as np
from gensim.models import Word2Vec
from trainer import train, test
from dataloader import dataloader
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ====================            ====================
    # ==================== task: Text ====================
    # ====================            ====================

    data_path = './data/online_shopping_10_cats.csv'
    embedding_path = './embedding/word2vec.bin'

    # create_embedding(data_path, embedding_path)

    embedding_model = Word2Vec.load(embedding_path)
    key_to_index = embedding_model.wv.key_to_index
    index_to_key = embedding_model.wv.index_to_key

    vocab_size = len(index_to_key)  # 词典大小
    vector_size = 300  # 每个词向量的维度

    weight = np.zeros((vocab_size, vector_size))  # 词嵌入矩阵
    for i in range(vocab_size):
        weight[i][:] = embedding_model.wv[index_to_key[i]]
    weight = torch.Tensor(weight).to(device)

    label_dict = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9}

    log_path = './logs/RNN'
    model_save_path = './model/RNN.pt'
    model = RNN(weight, input_size=vector_size, hidden_size=128, output_size=len(label_dict), device=device)

    # log_path = './logs/GRU'
    # model_save_path = './model/GRU.pt'
    # model = GRU(weight, input_size=vector_size, hidden_size=128, output_size=len(label_dict), device=device)

    # log_path = './logs/LSTM'
    # model_save_path = './model/LSTM.pt'
    # model = LSTM(weight, input_size=vector_size, hidden_size=256, output_size=len(label_dict), device=device)

    # log_path = './logs/BiLSTM'
    # model_save_path = './model/BiLSTM.pt'
    # model = BiLSTM(weight, input_size=vector_size, hidden_size=256, output_size=len(label_dict), device=device)

    train_dataloader, dev_dataloader, test_dataloader = dataloader(data_path, index_to_key, key_to_index, label_dict,
                                                                   batch_size=256,
                                                                   words_length=64)

    train(model, train_dataloader, dev_dataloader, log_path=log_path, model_save_path=model_save_path, device=device,
          epochs=100)
    acc, loss = test(model, test_dataloader, device=device)
    print(f"[Test]  acc = {acc:5.4f}, loss = {loss:9.4f}")

    # ====================               ====================
    # ==================== task: Weather ====================
    # ====================               ====================

    pass
