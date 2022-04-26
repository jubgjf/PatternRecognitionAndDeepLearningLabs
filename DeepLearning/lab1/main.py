from model import MLP
from trainer import train, test
import matplotlib.pyplot as plt
import torch

if __name__ == '__main__':
    load_model = True  # 是否加载已有模型
    model_path = "./model/model<@>.pt"  # 模型路径，按照隐藏层特征数量进行命名
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 20

    if not load_model:
        hidden_features_list = [5, 7, 10, 15, 20, 50, 100, 500]  # 隐藏层特征数量，多个特征数量用于对比效果
        acc_list: list[list[float]] = []  # 在各个隐藏层特征数下，每一次迭代后的正确率
        acc_final_list: list[float] = []  # 在各个隐藏层特征数下，完成所有 epochs 迭代后的正确率
        for hidden_features in hidden_features_list:
            model = MLP(hidden_features=hidden_features).to(device=device)
            epochs = 20

            acc_list.append([])
            for i in range(epochs):
                train(model, device, batch_size)
                acc = test(model, device, batch_size, i)
                acc_list[-1].append(acc.to(torch.device("cpu")))
            acc_final_list.append(acc_list[-1][-1])
            plt.plot(acc_list[-1], label=f"feat = {hidden_features}")

            torch.save(model, model_path.replace("<@>", str(hidden_features)))

        # ===== 绘制 迭代次数-准确率 曲线
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

        # ===== 绘制 特征数量-准确率 曲线
        plt.figure()
        plt.xlabel("Features")
        plt.ylabel("Accuracy")
        plt.plot(hidden_features_list, acc_final_list)

        plt.show()
    else:
        model = torch.load(model_path.replace("<@>", str(500)))  # 测试隐藏层特征数为 500 的模型
        test(model, device, batch_size, -1)
