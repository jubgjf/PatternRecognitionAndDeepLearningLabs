import torch
from model import AlexNet
from dataloader import dataloader
from trainer import train, test

if __name__ == '__main__':
    load_model = True  # 是否加载已有模型
    model_path = './model/AlexNet.ckpt'  # 模型路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataloader, dev_dataloader, test_dataloader = dataloader(batch_size=20)

    if not load_model:
        model = AlexNet().to(device=device)
        train(model, train_dataloader, dev_dataloader, device=device, epochs=30)
        torch.save(model, model_path)
    else:
        model = torch.load(model_path).to(device=device)
        acc, loss = test(model, test_dataloader, device=device)
        print(f"[Test] acc = {acc}, loss = {loss}")
