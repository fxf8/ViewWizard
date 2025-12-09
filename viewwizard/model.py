from dataclasses import dataclass, field

import torch
import torchvision
import torchvision.transforms.functional


@dataclass
class ThumbnailStatisticsTrainingBatch:
    video_ids: list[str]  # length (batch_size)
    image_batch: (
        torch.Tensor
    )  # shape (batch_size, 3, image_input_width, image_input_height)
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
            torch.nn.Dropout(p=0.3),
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
image_normalizer = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


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

    normalized_image_batch: torch.Tensor = sample.image_batch

    predicted_view_count = model(normalized_image_batch).squeeze(-1)
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
    preservation_loss_weight: float = 1e-3,
    saturation_loss_weight: float = 1e-7,
    blurring_interval: int = 10,
) -> torch.Tensor:
    """
    Takes a tensor of shape (batch_size, 3 (channels), image_input_width, image_input_height) where channels have values in [0, 1]
    Returns a tensor of shape (batch_size, 3 (channels), image_input_width, image_input_height) where channels have values in [0, 1]
    """

    image_normalizer = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # Latent batch has shape [-inf, inf]. Latent **image** batch has shape [0, 1] (after sigmoid)
    latent_batch: torch.Tensor = torch.logit(
        image_batch.clamp(1e-4, 1 - 1e-4).clone().detach().requires_grad_(True)
    )

    optimizer: torch.optim.Optimizer = torch.optim.Adam([latent_batch], lr=1e-2)

    for iteration in range(iterations):
        optimizer.zero_grad()

        latent_image_batch: torch.Tensor = torch.sigmoid(latent_batch)
        normalized_latent_image_batch = image_normalizer(latent_image_batch)

        # Shape (batch_size)
        predicted_view_counts = model(normalized_latent_image_batch).squeeze()

        # Shape (batch_size)
        total_variation_loss = (
            latent_image_batch[:, :, 1:, :] - latent_image_batch[:, :, :-1, :]
        ).abs().mean(dim=(1, 2, 3)) + (
            latent_image_batch[:, :, :, 1:] - latent_image_batch[:, :, :, :-1]
        ).abs().mean(dim=(1, 2, 3))

        # Shape (batch_size)
        preservation_loss: torch.Tensor = (
            (latent_image_batch - image_batch).pow(2).mean(dim=(1, 2, 3))
        )

        # Shape (batch_size)
        saturation_loss: torch.Tensor = (
            latent_image_batch.abs().mean(dim=(1, 2, 3)) - 0.5
        ) ** 2

        loss = (
            -predicted_view_counts
            + total_variation_loss * total_variation_loss_weight
            + preservation_loss * preservation_loss_weight
            + saturation_loss * saturation_loss_weight
        ).mean()

        loss.backward()
        optimizer.step()

        if iteration % blurring_interval == 0:
            with torch.no_grad():
                blurred_latent_image_batch: torch.Tensor = (
                    torchvision.transforms.functional.gaussian_blur(
                        torch.sigmoid(latent_batch.clone()), kernel_size=[5, 5]
                    )
                )

                latent_batch.copy_(
                    torch.logit(blurred_latent_image_batch.clamp(1e-4, 1 - 1e-4))
                )

    latent_image_batch = torch.sigmoid(latent_batch)

    with torch.no_grad():
        latent_image_batch.clamp_(0, 1)

    return latent_image_batch
