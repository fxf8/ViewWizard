from dataclasses import dataclass
import torch
from torch import nn
import torchvision


@dataclass
class ThumbnailStatisticsTrainingBatch:
    video_ids: list[str]  # length (batch_size)
    image: torch.Tensor  # shape (batch_size, 3, image_input_width, image_input_height)
    view_count: torch.Tensor  # shape (batch_size)


class ThumbnailStatisticsModel(nn.Module):
    def __init__(self):
        super().__init__()

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(),
        )
        """

        self.encoder = torchvision.models.resnet18(weights="DEFAULT")

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.regression = nn.Linear(in_features=1000, out_features=1)  # (view count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of shape (batch_size, 3, image_input_width, image_input_height)
        and returns a tensor of shape (batch_size, 2)
        """

        encoded_image: torch.Tensor = self.encoder(image)

        return self.regression(encoded_image)


def train_model_batch(
    model: ThumbnailStatisticsModel,
    sample: ThumbnailStatisticsTrainingBatch,
    optimizer: torch.optim.Optimizer,
):
    optimizer.zero_grad()

    predicted_view_count = model(sample.image)

    view_count_loss = torch.nn.functional.l1_loss(
        predicted_view_count, sample.view_count
    )

    view_count_loss.backward()

    optimizer.step()
