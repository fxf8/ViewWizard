import pathlib
import pickle
import random
from typing import cast
import uuid

import googleapiclient.discovery as discovery
import torch

import viewwizard.database as db
import viewwizard.model as vmodel
import viewwizard.schema as schema


class ProgramSession:
    session_id: uuid.UUID
    dataset: list[tuple[str, db.VideoDataset]]
    models: list[
        tuple[
            vmodel.ThumbnailStatisticsModel,
            torch.optim.Optimizer,
            vmodel.ModelTrainingHistory,
        ]
    ]
    client: discovery.Resource | None

    def __init__(self):
        self.session_id = uuid.uuid4()

        self.dataset = []
        self.models = []
        self.client = None

    def save_to_path(self, path: pathlib.Path):
        model_states: list[dict] = []
        optimizer_states: list[dict] = []
        optimizer_classes: list[type] = []
        training_histories: list[vmodel.ModelTrainingHistory] = []

        for model, optimizer, history in self.models:
            model.cpu()
            model_states.append(model.state_dict())

            optimizer_classes.append(type(optimizer))
            optimizer_states.append(optimizer.state_dict())

            training_histories.append(history)

        temporary_models = self.models
        temporary_client = self.client

        self.models = []
        self.client = None

        with path.open("wb") as file:
            pickle.dump(
                (
                    self.__dict__,
                    (
                        model_states,
                        optimizer_states,
                        optimizer_classes,
                        training_histories,
                    ),
                ),
                file,
            )

        self.models = temporary_models
        self.client = temporary_client

    @classmethod
    def load_from_path(cls, path: pathlib.Path) -> "ProgramSession":
        with path.open("rb") as f:
            session_dict, saved_states = pickle.load(f)

        (
            model_states,
            optimizer_states,
            optimizer_classes,
            training_histories,
        ) = saved_states

        session = cls()
        session.__dict__.update(session_dict)

        # Reconstruct everything
        session.models = []
        for model_state, opt_state, opt_class, history in zip(
            model_states, optimizer_states, optimizer_classes, training_histories
        ):
            model = vmodel.ThumbnailStatisticsModel()
            model.load_state_dict(model_state)

            optimizer = opt_class(model.parameters())
            optimizer.load_state_dict(opt_state)

            session.models.append((model, optimizer, history))

        session.client = None

        return session

    def ensure_client(self) -> discovery.Resource:
        resource = self.client

        if resource is None:
            resource = cast(
                discovery.Resource,
                discovery.build("youtube", "v3", cache_discovery=False),
            )

        self.client = resource

        return resource

    def create_dataset(self, name: str):
        self.dataset.append((name, db.VideoDataset()))

    def search_new_thumbnails(
        self, dataset_number: int, search_count: int
    ) -> int | None:
        if dataset_number >= len(self.dataset):
            return None

        client: discovery.Resource = self.ensure_client()

        _, dataset = self.dataset[dataset_number]

        new_ids: list[str] = sum(
            [
                db.sample_random_search_ids(
                    client, db.create_random_query(random.randint(2, 4))
                )
                for _ in range(search_count)
            ],
            [],
        )

        dataset.add_ids(new_ids)
        dataset.sync_id_video_data(client)

        return len(new_ids)
