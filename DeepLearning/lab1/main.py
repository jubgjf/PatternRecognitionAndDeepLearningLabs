from model import MLP
from trainer import train, test
import torch

if __name__ == '__main__':
    load_model = True  # 是否加载已有模型
    model_path = "./model/model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20

    if not load_model:
        model = MLP().to(device=device)
        epochs = 20

        for i in range(epochs):
            train(model, device, batch_size)
            test(model, device, batch_size, i)

        torch.save(model, model_path)
    else:
        model = torch.load(model_path)
        test(model, device, batch_size, -1)
