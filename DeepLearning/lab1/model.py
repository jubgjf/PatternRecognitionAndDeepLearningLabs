import torch
import torch.nn.functional


class MLP(torch.nn.Module):
    """多层感知机模型"""

    def __init__(self):
        super(MLP, self).__init__()

        # 第一层：输入的是 28x28(= 784) 大小的图片，向隐藏层（设定为 100 个单元）输出
        self.linear1 = torch.nn.Linear(in_features=28 * 28, out_features=100)
        # 第二层：输入的是来自上一层的 100 个特征，输出是识别手写数字 0~9 这 10 个数字的概率
        self.linear2 = torch.nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        """
        模型进行前馈

        Args:
            x: 输入数据
        """

        out = x.view(-1, 28 * 28)  # 将一个多行的 Tensor 拼接成一行

        # 第一层：输入层
        out = self.linear1(out)
        out = torch.nn.functional.leaky_relu(out)  # 激活函数

        # 第二层：隐藏层
        out = self.linear2(out)
        out = torch.nn.functional.leaky_relu(out)  # 激活函数

        return out
