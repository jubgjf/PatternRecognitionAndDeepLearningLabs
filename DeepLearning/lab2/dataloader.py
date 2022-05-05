import os
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data.dataloader import DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split


def dataloader(batch_size: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    加载数据

    Args:
        batch_size: 一批的数据量

    Returns:
        返回 (训练集的 dataloader, 开发集的 dataloader, 测试集的 dataloader)
    """

    data = datasets.Caltech101(root='./data', download=True, transform=transforms.ToTensor())
    categories_paths = ['./data/caltech101/101_ObjectCategories/' + c for c in data.categories]

    # *_path_label 中，str 为图片的路径，int 为图片的标签号
    train_path_label: list[tuple[str, int]] = []
    dev_path_label: list[tuple[str, int]] = []
    test_path_label: list[tuple[str, int]] = []
    for category_path in categories_paths:
        # 当前标签下所有图片的路径
        image_paths = [os.path.join(category_path, file) for file in os.listdir(category_path)]

        # 分割训练/开发/测试集
        train, test = train_test_split(image_paths, train_size=0.8, random_state=2)
        dev, test = train_test_split(test, test_size=0.5, random_state=2)

        # 标签的序号
        label: int = data.categories.index(category_path.split("/")[-1])

        train_path_label.extend([(trainItem, label) for trainItem in train])
        dev_path_label.extend([(devItem, label) for devItem in dev])
        test_path_label.extend([(testItem, label) for testItem in test])

    # 预处理图片
    preprocess = transforms.Compose([
        transforms.Resize(256),  # 缩放到 256x256
        transforms.CenterCrop(224),  # 中心裁剪到 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 转换为 DataLoader
    train_dataloader = DataLoader(
        dataset=[(preprocess(Image.open(data[0]).convert('RGB')), data[1]) for data in train_path_label],
        batch_size=batch_size,
        shuffle=True
    )
    dev_dataloader = DataLoader(
        dataset=[(preprocess(Image.open(data[0]).convert('RGB')), data[1]) for data in dev_path_label],
        batch_size=batch_size,
        shuffle=False
    )
    test_dataloader = DataLoader(
        dataset=[(preprocess(Image.open(data[0]).convert('RGB')), data[1]) for data in test_path_label],
        batch_size=batch_size,
        shuffle=False
    )

    return train_dataloader, dev_dataloader, test_dataloader
