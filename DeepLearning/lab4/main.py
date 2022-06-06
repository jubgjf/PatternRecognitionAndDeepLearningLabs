from model import TextRNN, TextGRU, TextLSTM, TextBiLSTM, WeatherLSTM
from word2vec import create_embedding
import numpy as np
from gensim.models import Word2Vec
from trainer import text_train, text_test, weather_train, weather_test
from dataloader import text_dataloader, weather_dataloader
from weather_split import weather_split
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    task = 'Weather'  # 'Text' or 'Weather'

    if task == 'Text':
        # ====================            ====================
        # ==================== task: Text ====================
        # ====================            ====================

        data_path = './data/online_shopping_10_cats.csv'
        embedding_path = './embedding/word2vec.bin'

        create_embedding(data_path, embedding_path)

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

        log_path = './logs/Text_RNN'
        model_save_path = './model/Text_RNN.pkl'
        model = TextRNN(weight, input_size=vector_size, hidden_size=128, output_size=len(label_dict), device=device)
        train_dataloader, dev_dataloader, test_dataloader = text_dataloader(data_path, index_to_key, key_to_index,
                                                                            label_dict,
                                                                            batch_size=256,
                                                                            words_length=64)
        text_train(model, train_dataloader, dev_dataloader, log_path=log_path, model_save_path=model_save_path,
                   device=device,
                   epochs=100)
        acc, recall, micro_f1, macro_f1, loss = text_test(model, test_dataloader, device=device)
        print(f"[Test] acc = {acc:6.4f}, recall = {recall:6.4f}, "
              f"micro f1 = {micro_f1:6.4f}, macro f1 = {macro_f1:6.4f}, "
              f"loss = {loss:6.4f}")

        log_path = './logs/Text_GRU'
        model_save_path = './model/Text_GRU.pkl'
        model = TextGRU(weight, input_size=vector_size, hidden_size=128, output_size=len(label_dict), device=device)
        train_dataloader, dev_dataloader, test_dataloader = text_dataloader(data_path, index_to_key, key_to_index,
                                                                            label_dict,
                                                                            batch_size=256,
                                                                            words_length=64)
        text_train(model, train_dataloader, dev_dataloader, log_path=log_path, model_save_path=model_save_path,
                   device=device,
                   epochs=100)
        acc, recall, micro_f1, macro_f1, loss = text_test(model, test_dataloader, device=device)
        print(f"[Test] acc = {acc:6.4f}, recall = {recall:6.4f}, "
              f"micro f1 = {micro_f1:6.4f}, macro f1 = {macro_f1:6.4f}, "
              f"loss = {loss:6.4f}")

        log_path = './logs/Text_LSTM'
        model_save_path = './model/Text_LSTM.pkl'
        model = TextLSTM(weight, input_size=vector_size, hidden_size=256, output_size=len(label_dict), device=device)
        train_dataloader, dev_dataloader, test_dataloader = text_dataloader(data_path, index_to_key, key_to_index,
                                                                            label_dict,
                                                                            batch_size=256,
                                                                            words_length=128)
        text_train(model, train_dataloader, dev_dataloader, log_path=log_path, model_save_path=model_save_path,
                   device=device,
                   epochs=100)
        acc, recall, micro_f1, macro_f1, loss = text_test(model, test_dataloader, device=device)
        print(f"[Test] acc = {acc:6.4f}, recall = {recall:6.4f}, "
              f"micro f1 = {micro_f1:6.4f}, macro f1 = {macro_f1:6.4f}, "
              f"loss = {loss:6.4f}")

        log_path = './logs/Text_BiLSTM'
        model_save_path = './model/Text_BiLSTM.pkl'
        model = TextBiLSTM(weight, input_size=vector_size, hidden_size=256, output_size=len(label_dict), device=device)
        train_dataloader, dev_dataloader, test_dataloader = text_dataloader(data_path, index_to_key, key_to_index,
                                                                            label_dict,
                                                                            batch_size=256,
                                                                            words_length=128)
        text_train(model, train_dataloader, dev_dataloader, log_path=log_path, model_save_path=model_save_path,
                   device=device,
                   epochs=100)
        acc, recall, micro_f1, macro_f1, loss = text_test(model, test_dataloader, device=device)
        print(f"[Test] acc = {acc:6.4f}, recall = {recall:6.4f}, "
              f"micro f1 = {micro_f1:6.4f}, macro f1 = {macro_f1:6.4f}, "
              f"loss = {loss:6.4f}")

    elif task == 'Weather':
        # ====================               ====================
        # ==================== task: Weather ====================
        # ====================               ====================

        data_path = './data/jena_climate_2009_2016.csv'
        splited_data_paths = ('./data/jena_climate_2009_2016_train.csv', './data/jena_climate_2009_2016_test.csv')
        log_path = './logs/Weather_LSTM'
        model_save_path = './model/Weather_LSTM.pkl'

        weather_split(data_path, splited_data_paths)

        model = WeatherLSTM(input_size=17, hidden_size=256, output_size=288, device=device).to(device)
        train_dataloader, test_dataloader = weather_dataloader(splited_data_paths, batch_size=128)

        weather_train(model, train_dataloader, log_path=log_path, model_save_path=model_save_path, device=device,
                      epochs=20)

        model.load_state_dict(torch.load(model_save_path))
        model.to(device)
        weather_test(model, test_dataloader, device)
