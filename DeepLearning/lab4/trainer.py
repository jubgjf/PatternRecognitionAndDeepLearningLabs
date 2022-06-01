import torch
from sklearn import metrics
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def train(model: torch.nn.Module, train_dataloader: DataLoader, dev_dataloader: DataLoader, log_path: str,
          model_save_path: str, device: torch.device, epochs: int):
    loss_func = torch.nn.CrossEntropyLoss()  # 损失函数选用交叉熵
    optimizer = torch.optim.Adam(model.parameters(), lr=3 * 1e-5)

    writer = SummaryWriter(log_path)

    dev_acc_best = -1  # dev acc 的最大值，用于实时保存最好的模型

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")

        iter_count = 0
        for data, label in train_dataloader:
            data = data.to(device)
            label = label.to(device)

            predict_label = model.forward(data)

            loss = loss_func(predict_label, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 每 10 轮输出一次当前效果
            if iter_count % 10 == 0:
                ground_truth = label.data.cpu()
                predict_ans = torch.max(predict_label.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(ground_truth, predict_ans)
                dev_acc, dev_loss = test(model, dev_dataloader, device)

                if dev_acc > dev_acc_best:
                    torch.save(model, model_save_path)

                print(f"[Train] acc = {train_acc:6.4f}, loss = {loss.item():6.4f}  ||  "
                      f"[Dev] acc = {dev_acc:6.4f}, loss = {dev_loss:6.4f}")

                writer.add_scalar("loss/train", loss.item())
                writer.add_scalar("loss/dev", dev_loss)
                writer.add_scalar("acc/train", train_acc)
                writer.add_scalar("acc/dev", dev_acc)
            iter_count += 1

    writer.close()


def test(model: torch.nn.Module, dataloader: DataLoader, device: torch.device):
    loss_func = torch.nn.CrossEntropyLoss()
    loss_total = 0

    predict_labels = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            label = label.to(device)

            predict_label = model.forward(data)

            loss = loss_func(predict_label, label)
            loss_total += loss

            predict_ans = torch.max(predict_label.cpu().data, 1)[1].numpy()
            labels.extend(label.data.cpu())
            predict_labels.extend(predict_ans)

    acc = metrics.accuracy_score(labels, predict_labels)

    return acc, loss_total / len(dataloader)
