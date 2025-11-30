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

        self.encoder = torchvision.models.squeezenet1_1(weights="DEFAULT")

        """
        for parameter in self.encoder.parameters():
            parameter.requires_grad = False
        """

        self.regression = torch.nn.Linear(
            in_features=1000, out_features=1
        )  # (view count)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Takes a tensor of shape (batch_size, 3, image_input_width, image_input_height)
        and returns a tensor of shape (batch_size, 2)
        """

        image_resized = torch.nn.functional.interpolate(
            image,
            size=(244, 244),
            mode="bilinear",
            align_corners=False,
        )

        encoded_image: torch.Tensor = self.encoder(image_resized)

        return self.regression(encoded_image)


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
) -> ModelTrainingHistory:
    training_history = ModelTrainingHistory()

    optimizer.zero_grad()

    predicted_view_count = model(sample.image)

    view_count_loss = torch.nn.functional.l1_loss(
        predicted_view_count, torch.log10(sample.view_count + 1)
    )

    training_history.add_loss(view_count_loss.item())

    view_count_loss.backward()
    optimizer.step()

    return training_history
