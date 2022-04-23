from torchvision import datasets
import torchvision.transforms as transforms
import torch.utils.data


def dataloader(data_type: str, batch_size: int = 20) -> torch.utils.data.DataLoader | None:
    """
    加载 MNIST 数据集，能够自动下载缺失的数据到 ./data 中

    Args:
        data_type: 数据集类型，只能从 "train" 或 "test" 中取值
        batch_size: 每次读取的数据量

    Returns:
        object: 返回 dataloader 数据，格式为 [(x, y), (x, y), (x, y), ...]
                其中 x 为数据，y 为标签
    """

    match data_type:
        case "train":
            data = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=0)
            return loader
        case "test":
            data = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
            loader = torch.utils.data.DataLoader(data, batch_size=batch_size, num_workers=0)
            return loader
        case _:
            print("No such data type. Choices: `train` or `test`")
            return None
