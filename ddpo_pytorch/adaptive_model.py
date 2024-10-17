import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, image_size, encoder_dim=256, channels=(64, 128, 256)):  # (batch * steps) * 4 * 64 * 64
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            nn.Conv2d(4, channels[0], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(channels[0], channels[1], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(channels[1], channels[2], kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=3),

            nn.Conv2d(channels[2], encoder_dim, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        compressed_size = image_size // 2 // 2 // 3
        self.fc = nn.Linear(encoder_dim * compressed_size * compressed_size, 1)

    def forward(self, x):
        x = self.model(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        disoutput = torch.sigmoid(x)
        return disoutput

    def loss(self, latents):  # batch * steps * 4 * 64 * 64
        criterion = nn.BCELoss()
        disnet_batchsize, steps = latents.shape[:2]
        labels = torch.cat([torch.zeros(disnet_batchsize, steps - 5), torch.ones(disnet_batchsize, 5)], axis=1).to(latents.device)
        outputs = self.forward(latents.reshape(-1, *latents.shape[2:])).reshape(*latents.shape[:2])
        total_loss = criterion(outputs, labels)
        return total_loss, outputs[:, :-5].mean(), outputs[:, -5:].mean()
