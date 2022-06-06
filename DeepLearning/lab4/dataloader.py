import jieba
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


def text_dataloader(input_path, index_to_key, key_to_index, label_dict, batch_size, words_length):
    """
    获取 train/dev/test 的 dataloader

    Args:
        input_path: 原始数据路径
        index_to_key: index 到 key 的映射
        key_to_index: key 到 index 的映射
        label_dict: 字符串的 label 到整数的 label 的映射
        batch_size: batch_size
        words_length: 句子长度，可能使用截取或填充强行使句子到这个长度

    Returns:
        返回一个 tuple，内容为 train_dataloader, dev_dataloader, test_dataloader
    """

    all_sentences = pd.read_csv(input_path).dropna(subset=['review'])

    train_sentences = all_sentences.iloc[
        [i for i in range(len(all_sentences)) if i % 5 == 1 or i % 5 == 2 or i % 5 == 3]]
    dev_sentences = all_sentences.iloc[[i for i in range(len(all_sentences)) if i % 5 == 4]]
    test_sentences = all_sentences.iloc[[i for i in range(len(all_sentences)) if i % 5 == 0]]

    train_dataset = TextDataset(train_sentences, index_to_key, key_to_index, label_dict, words_length)
    dev_dataset = TextDataset(dev_sentences, index_to_key, key_to_index, label_dict, words_length)
    test_dataset = TextDataset(test_sentences, index_to_key, key_to_index, label_dict, words_length)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, dev_dataloader, test_dataloader


def weather_dataloader(input_paths, batch_size):
    """
    获取 train/test 的 dataloader

    Args:
        input_paths: 原始数据路径，其中 [0] 为 train，[1] 为 detest
        batch_size: batch_size

    Returns:
        返回一个 tuple，内容为 train_dataloader, test_dataloader
    """

    train_weather = pd.read_csv(input_paths[0])
    test_weather = pd.read_csv(input_paths[1])

    train_weather = [data[1] for data in list(train_weather.groupby('Week Number')) if len(data[1]) == 1008]
    test_weather = [data[1] for data in list(test_weather.groupby('Week Number')) if len(data[1]) == 1008]

    train_dataset = WeatherDataset(train_weather)
    test_dataset = WeatherDataset(test_weather)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


class TextDataset(Dataset):
    def __init__(self, sentences, index_to_key, key_to_index, label_dict, words_length):
        super(TextDataset, self).__init__()
        self.text = sentences
        self.index_to_key = index_to_key
        self.key_to_index = key_to_index
        self.label_dict = label_dict
        self.padding_size = words_length

    def __getitem__(self, item):
        line = self.text.iloc[item]
        label = line['cat']
        sentence = line['review']

        words = [word for word in jieba.cut(sentence, cut_all=False)]
        vectors = []
        for word in words:
            if word in self.index_to_key:
                vectors.append(self.key_to_index[word])
            else:
                vectors.append(self.key_to_index['[UNK]'])

        if len(vectors) < self.padding_size:
            # 句子不够长，填充到 words_length 长度
            vectors.extend([self.key_to_index['[PAD]'] for i in range(self.padding_size - len(vectors))])
        else:
            # 截取到 words_length 长度
            vectors = vectors[0:self.padding_size]

        return torch.LongTensor(vectors), self.label_dict[label]

    def __len__(self):
        return len(self.text)


class WeatherDataset(Dataset):
    def __init__(self, weather):
        super(WeatherDataset, self).__init__()
        self.weather = weather

    def __getitem__(self, item):
        weather = self.weather[item]

        weekdays = weather.loc[weather['WeekDay'] <= 5]
        weekends = weather.loc[weather['WeekDay'] >= 6]

        date = list(weekends['Date Time'])
        train_data = weekdays.drop(['Date Time'], axis=1)
        test_data = weekends['T (degC)']

        return torch.Tensor(train_data.values), torch.Tensor(test_data.values), date

    def __len__(self):
        return len(self.weather)
