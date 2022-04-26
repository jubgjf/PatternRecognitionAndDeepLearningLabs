from dataloader import dataloader
import torch


def train(model: torch.nn.Module, device: torch.device, batch_size: int):
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数选用交叉熵
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    for data, label in dataloader(data_type="train", batch_size=batch_size):
        data = data.to(device)
        label = label.to(device)

        predict_label = model(data)  # 预测结果: predict_label.shape = [20, 10] (`20` = batch_size, `10` = 0~9 的概率)

        loss = loss_func(predict_label, label)  # 预测与正确的差值
        loss.backward()  # 反向传播
        optimizer.step()  # 进行一次梯度下降
        optimizer.zero_grad()  # 梯度清零


def test(model: torch.nn.Module, device: torch.device, batch_size: int, epoch: int):
    correct_count = 0  # 预测正确数量
    total_count = 0  # 总数据数量

    with torch.no_grad():  # 训练集中不需要反向传播
        for data, label in dataloader(data_type="test", batch_size=batch_size):
            data = data.to(device)
            label = label.to(device)

            predict_label = model(data)
            predict_ans = torch.argmax(predict_label, dim=1)  # 预测结果，取的是 0~9 概率中的最大值
            correct_count += torch.sum(predict_ans == label)  # 本次正确的数量
            total_count += batch_size
            acc = correct_count / total_count  # 正确率

    if epoch == -1:
        print(f'Accuracy: {acc}')
    else:
        print(f'Epoch: {epoch}, Accuracy: {acc}')

    return acc
