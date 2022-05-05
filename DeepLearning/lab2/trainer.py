import torch
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.utils.data.dataloader import DataLoader


def train(model: torch.nn.Module, train_dataloader: DataLoader, dev_dataloader: DataLoader, device: torch.device,
          epochs: int):
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数选用交叉熵
    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-5)

    writer = SummaryWriter(logdir="./logs")

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)

            predict_label = model(data)  # 预测结果

            loss = loss_func(predict_label, label)  # 预测与正确的差值
            loss.backward()  # 反向传播
            optimizer.step()  # 进行一次梯度下降
            optimizer.zero_grad()  # 梯度清零

            # 输出当前效果
            predict_ans = torch.argmax(predict_label, dim=1).cpu().numpy()
            train_acc = metrics.accuracy_score(label.data.cpu(), predict_ans)
            dev_acc, dev_loss = test(model, dev_dataloader, device)

            print(f"[Train] acc = {train_acc}, loss = {loss.item()}")
            print(f"[Dev]   acc = {dev_acc}, loss = {dev_loss}")

            writer.add_scalar("train loss", loss.item())
            writer.add_scalar("train acc", train_acc)
            writer.add_scalar("dev loss", dev_loss)
            writer.add_scalar("dev acc", dev_acc)

    writer.close()


def test(model: torch.nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    loss_func = torch.nn.CrossEntropyLoss()
    loss_total = 0

    predict_labels = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            predict_label = model(data)

            loss = loss_func(predict_label, label)
            loss_total += loss

            predict_ans = torch.argmax(predict_label, dim=1).cpu().numpy()
            labels.extend(label.data.cpu())
            predict_labels.extend(predict_ans)

    acc = metrics.accuracy_score(labels, predict_labels)
    loss = loss_total / len(dataloader)

    return acc, loss
