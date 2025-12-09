from dataclasses import dataclass, field

import pyiqa
import lpips
import torch
import torchvision
import torchvision.transforms.functional
import tqdm


@dataclass
class ThumbnailStatisticsTrainingBatch:
    video_ids: list[str]  # length (batch_size)
    image_batch: (
        torch.Tensor
    )  # shape (batch_size, 3, image_input_width, image_input_height)
    view_count: torch.Tensor  # shape (batch_size)

    def add_noise_samples(self):
        next_sample_image_batch: torch.Tensor = (
            torchvision.transforms.functional.gaussian_blur(
                (self.image_batch.detach() + torch.rand_like(self.image_batch) - 0.5),
                kernel_size=[5, 5],
            ).clamp(0, 1)
        )

        next_view_count: torch.Tensor = self.view_count.detach() / 5e1

        self.image_batch = torch.cat((self.image_batch, next_sample_image_batch))
        self.view_count = torch.cat((self.view_count, next_view_count))


class ThumbnailStatisticsModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = torchvision.models.mobilenet_v2(weights="DEFAULT").features

        for layer_index, layer in enumerate(self.encoder.parameters()):
            if layer_index >= 16:  # last few blocks
                layer.requires_grad = True

            else:
                layer.requires_grad = False

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


def train_model_batch(
    model: ThumbnailStatisticsModel,
    sample: ThumbnailStatisticsTrainingBatch,
    optimizer: torch.optim.Optimizer,
    target_mean: float,
    target_std: float,
    clip_norm: float = 1.0,
) -> ModelTrainingHistory:
    training_history = ModelTrainingHistory()
    smooth_loss_function = torch.nn.SmoothL1Loss()
    image_normalizer = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    sample.add_noise_samples()

    optimizer.zero_grad()

    normalized_image_batch: torch.Tensor = image_normalizer(sample.image_batch)

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
    learning_rate: float = 3e-2,
    total_variation_loss_weight: float = 1e-1,
    preservation_loss_weight: float = 5e-2,
    perceptual_loss_weight: float = 5e-1,
    edge_loss_weight: float = 1e-1,
    blurring_interval: int = 12,
) -> torch.Tensor:
    """
    Takes a tensor of shape (batch_size, 3 (channels), image_input_width, image_input_height) where channels have values in [0, 1]
    Returns a tensor of shape (batch_size, 3 (channels), image_input_width, image_input_height) where channels have values in [0, 1]
    """

    image_normalizer = torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    lpips_loss_function = lpips.LPIPS(net="alex").to(image_batch.device)
    nima_model = pyiqa.create_metric("nima")

    with torch.no_grad():
        latent_init: torch.Tensor = torch.logit(image_batch.clamp(1e-4, 1 - 1e-4))

    # Latent batch has shape [-inf, inf]. Latent **image** batch has shape [0, 1] (after sigmoid)
    latent_batch: torch.Tensor = (
        latent_init.clone()
        .detach()
        .requires_grad_(True)
        .to(image_batch.device)
        .to(torch.float32)
    )

    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        [latent_batch], lr=learning_rate
    )

    for iteration in tqdm.tqdm(range(iterations)):
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

        dx = latent_image_batch[:, :, 1:, :] - latent_image_batch[:, :, :-1, :]
        dy = latent_image_batch[:, :, :, 1:] - latent_image_batch[:, :, :, :-1]
        edge_loss = (dx.abs() + dy.abs()).mean()

        lpips_loss = lpips_loss_function(latent_image_batch, image_batch)

        nima_score = nima_model(image_batch, latent_image_batch).mean()

        loss = (
            -predicted_view_counts
            + total_variation_loss * total_variation_loss_weight
            + preservation_loss * preservation_loss_weight
            + perceptual_loss_weight * lpips_loss
            + edge_loss * edge_loss_weight
        ).mean()

        loss.backward()
        optimizer.step()

        if iteration % blurring_interval == 0 and iteration > 0:
            with torch.no_grad():
                blurred_latent_image_batch: torch.Tensor = (
                    torchvision.transforms.functional.gaussian_blur(
                        torch.sigmoid(latent_batch.detach()),
                        kernel_size=[3, 3],
                        sigma=[0.1, 0.1],
                    )
                )

                latent_batch.copy_(
                    torch.logit(blurred_latent_image_batch.clamp(1e-4, 1 - 1e-4))
                )

    latent_image_batch = torch.sigmoid(latent_batch)

    with torch.no_grad():
        latent_image_batch.clamp_(0, 1)

    return latent_image_batch
