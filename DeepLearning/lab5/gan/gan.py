import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.Tanh(),

            nn.Linear(512, 512),
            nn.Tanh(),

            nn.Linear(512, 512),
            nn.Tanh(),

            nn.Linear(512, 2)
        )

    def forward(self, z):
        data = self.model(z)
        data = data.view(data.size(0), -1)
        return data


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(2, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        data_flat = data.view(data.size(0), -1)
        validity = self.model(data_flat)

        return validity


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    points_path = './data/points.mat'
    data = scio.loadmat(points_path)
    xx = data['xx']

    ground_truth = []
    for point in xx:
        ground_truth.append(point)

    batch_size = 8192

    loss_fn = nn.BCELoss()

    D = Discriminator().to(device)
    G = Generator().to(device)

    d_lr = 5e-4
    g_lr = 1e-4

    d_optimizer = torch.optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))

    points = torch.empty(len(ground_truth), 2).to(device)
    for i in range(len(ground_truth)):
        points[i][0] = ground_truth[i][0]
        points[i][1] = ground_truth[i][1]

    data_loader = DataLoader(dataset=points, batch_size=batch_size, shuffle=False)

    epochs = 3000
    for epoch in range(epochs):
        all_outputs = []
        for point in data_loader:
            ones = torch.normal(mean=1, std=0.05, size=(batch_size, 1), device=device)
            zeros = torch.normal(mean=0, std=0.05, size=(batch_size, 1), device=device)

            real_points = point

            # ===== 生成器 =====
            gk = 5
            for kk in range(gk):
                g_optimizer.zero_grad()

                gen_points = G(torch.randn(batch_size, 2, device=device))

                g_loss = loss_fn(D(gen_points), ones)
                g_loss.backward()

                g_optimizer.step()

            # ===== 判别器 =====
            dk = 5
            for kk in range(dk):
                d_optimizer.zero_grad()

                real_loss = loss_fn(D(real_points), ones)
                fake_loss = loss_fn(D(gen_points.detach()), zeros)
                d_loss = (real_loss + fake_loss) / 2
                d_loss.backward()

                d_optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}] D loss = {d_loss.item()} G loss = {g_loss.item()}")

            all_outputs.extend(gen_points.cpu().data)

        if (epoch + 1) % 100 == 0:
            plt.title(f"epoch = {epoch + 1}, d_k = {dk}, g_k = {gk}, d_lr = {d_lr}, g_lr = {g_lr}")

            plt.scatter([xy[0] for xy in ground_truth], [xy[1] for xy in ground_truth])
            plt.scatter([xy[0] for xy in all_outputs], [xy[1] for xy in all_outputs])

            x_min, x_max = min([xy[0] for xy in ground_truth]), max([xy[0] for xy in ground_truth])
            y_min, y_max = min([xy[1] for xy in ground_truth]), max([xy[1] for xy in ground_truth])

            # 绘制网格
            h = 0.01
            xx, yy = np.meshgrid(np.arange(1.2 * x_min, 1.2 * x_max, h), np.arange(1.2 * y_min, 1.2 * y_max, h))

            # 生成与网格上所有点对应的分类结果
            grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).to(torch.float).to(device)
            z = D(grid_points).reshape(xx.shape).cpu().data

            # 绘制contour
            plt.contour(xx, yy, z, levels=[0.5], colors=['blue'])

            plt.savefig(f"./images/GAN/GAN_{epoch + 1:04d}.png")
            plt.figure()
            # plt.show()

    torch.save(D, "./model/GAN/D.pt")
    torch.save(G, "./model/GAN/G.pt")
