import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet 模型"""

    def __init__(self):
        super(AlexNet, self).__init__()

self.features = nn.Sequential(
    # 输入为 224x224x3 的图像

    # 第一个卷积层，使用 2 个大小为 11x11x3x48 的卷积核，步长 S = 4，零填充 P = 3
    nn.Conv2d(in_channels=3, out_channels=2 * 48, kernel_size=(11, 11), stride=4, padding=3),
    nn.ReLU(inplace=True),
    # 输出 2 个大小为 55x55x48 的特征映射组

    # 第一个汇聚层，使用大小为 3x3 的最大汇聚操作，步长 S = 2
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    # 输出 2 个 27x27x48 的特征映射组

    # 第二个卷积层，使用 2 个大小为 5x5x48x128 的卷积核，步长 S = 1，零填充 P = 2,
    nn.Conv2d(in_channels=2 * 48, out_channels=2 * 128, kernel_size=(5, 5), stride=1, padding=2),
    nn.ReLU(inplace=True),
    # 输出 2 个大小为 27x27x128 的特征映射组

    # 第二个汇聚层，使用大小为 3x3 的最大汇聚操作，步长 S = 2
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    # 输出 2 个大小为 13x13x128 的特征映射组

    # 第三个卷积层为两个路径的融合，使用 1 个大小为 3x3x256x384 的卷积核，步长 S = 1，零填充 P = 1
    nn.Conv2d(in_channels=2 * 128, out_channels=1 * 384, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(inplace=True),
    # 输出 2 个大小为 13x13x192 的特征映射组

    # 第四个卷积层，使用 2 个大小为 3x3x192x192 的卷积核，步长 S = 1，零填充 P = 1
    nn.Conv2d(in_channels=2 * 192, out_channels=2 * 192, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(inplace=True),
    # 输出 2 个大小为 13x13x192 的特征映射组

    # 第五个卷积层，使用 2 个大小为 3x3x192x128 的卷积核，步长 S = 1，零填充 P = 1
    nn.Conv2d(in_channels=2 * 192, out_channels=2 * 128, kernel_size=(3, 3), stride=1, padding=1),
    nn.ReLU(inplace=True),
    # 输出 2 个大小为 13x13x128 的特征映射组

    # 第三个汇聚层，使用大小为 3x3 的最大汇聚操作，步长 S = 2
    nn.MaxPool2d(kernel_size=3, stride=2),
    # 输出 2 个大小为 6x6x128 的特征映射组
)

# 三个全连接层，神经元数量分别为 4096/4096/101
self.Flatten = nn.Flatten()
self.Linear1 = nn.Sequential(
    nn.Linear(in_features=2 * 128 * 6 * 6, out_features=4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
)
self.Linear2 = nn.Sequential(
    nn.Linear(in_features=4096, out_features=4096),
    nn.ReLU(inplace=True),
    nn.Dropout(p=0.5),
)
self.Linear3 = nn.Linear(in_features=4096, out_features=101)

    def forward(self, x):
        """
        模型进行前馈

        Args:
            x: 输入数据
        """

        out = self.features(x)

        out = self.Flatten(out)
        out = self.Linear1(out)
        out = self.Linear2(out)
        out = self.Linear3(out)

        return out
