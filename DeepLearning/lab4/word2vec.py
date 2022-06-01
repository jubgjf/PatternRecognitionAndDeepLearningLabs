import jieba
from gensim.models import Word2Vec


def create_embedding(data_path: str, embedding_path: str):
    """
    生成并保存词嵌入模型

    Args:
        data_path: 原始数据路径
        embedding_path: 保存的词嵌入模型的路径
    """

    all_data = []
    with open(data_path, 'r') as f:
        lines = f.readlines()
    lines.remove('cat,label,review\n')  # 去掉 csv 的第一行表头
    for line in lines:
        label, emotion = line.strip().split(",")[0], line.strip().split(",")[1]
        data = line.strip()[len(label + "," + emotion + ","):]
        all_data.append(data)

    corpus = []
    for data in all_data:
        corpus.append([word for word in jieba.cut(data)])
    corpus.append(['[UNK]', '[PAD]'])  # 未登录词 和 填充

    model = Word2Vec(vector_size=300, window=5, min_count=1, seed=2)
    model.build_vocab(corpus_iterable=corpus)
    model.train(corpus_iterable=corpus, total_examples=len(corpus), epochs=10)
    model.save(embedding_path)
