from dataclasses import dataclass, field
import torch
import torchvision


@dataclass
class ThumbnailStatisticsTrainingBatch:
    video_ids: list[str]  # length (batch_size)
    image: torch.Tensor  # shape (batch_size, 3, image_input_width, image_input_height)
    view_count: torch.Tensor  # shape (batch_size)


class ThumbnailStatisticsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torchvision.models.mobilenet_v2(weights="DEFAULT").features

        for parameter in self.encoder.parameters():
            parameter.requires_grad = False

        self.global_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        self.regression = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=1280, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=256, out_features=1),
        )  # (view count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of shape (batch_size, 3, image_input_width, image_input_height)
        and returns a tensor of shape (batch_size, 1)
        """

        image_resized: torch.Tensor = torch.nn.functional.interpolate(
            image,
            size=(244, 244),
            mode="bilinear",
            align_corners=False,
        )

        encoded_image: torch.Tensor = self.encoder(image_resized)

        pooled_image: torch.Tensor = self.global_pool(encoded_image)

        return self.regression(pooled_image)


@dataclass
class ModelTrainingHistory:
    losses: list[tuple[int, float]] = field(default_factory=list)
    iteration_count: int = 0

    def add_loss(self, loss: float):
        self.losses.append((self.iteration_count, loss))
        self.iteration_count += 1

    def extend(self, losses: list[float]):
        for loss in losses:
            self.add_loss(loss)

    def merge(self, other: "ModelTrainingHistory"):
        for loss in other.losses:
            self.add_loss(loss[1])


smooth_loss_function = torch.nn.SmoothL1Loss()


def train_model_batch(
    model: ThumbnailStatisticsModel,
    sample: ThumbnailStatisticsTrainingBatch,
    optimizer: torch.optim.Optimizer,
    target_mean: float,
    target_std: float,
    clip_norm: float = 1.0,
) -> ModelTrainingHistory:
    training_history = ModelTrainingHistory()

    optimizer.zero_grad()

    predicted_view_count = model(sample.image).squeeze(-1)
    target: torch.Tensor = (torch.log1p(sample.view_count) - target_mean) / target_std

    view_count_loss = smooth_loss_function(predicted_view_count, target)

    training_history.add_loss(view_count_loss.item())

    view_count_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()

    return training_history


def optimize_image_batch(
    model: ThumbnailStatisticsModel,
    image_batch: torch.Tensor,
    iterations: int,
    total_variation_loss_weight: float = 1e-2,
) -> torch.Tensor:
    optimized_image: torch.Tensor = image_batch.clone().detach().requires_grad_(True)

    optimizer: torch.optim.Optimizer = torch.optim.Adam([optimized_image], lr=1e-2)

    for _ in range(iterations):
        optimizer.zero_grad()

        predicted_view_counts = model(optimized_image).squeeze()
        total_variation_loss = torch.abs(
            optimized_image[:, :, 1:, :] - optimized_image[:, :, :-1, :]
        ).mean(dim=(1, 2, 3)) + torch.abs(
            optimized_image[:, :, :, 1:] - optimized_image[:, :, :, :-1]
        ).mean(dim=(1, 2, 3))

        loss = (
            -predicted_view_counts + total_variation_loss * total_variation_loss_weight
        ).mean()

        loss.backward()
        optimizer.step()

        optimized_image.clamp_(0, 1)

    return optimized_image
