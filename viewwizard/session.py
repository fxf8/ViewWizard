import asyncio
from collections.abc import Callable
from dataclasses import dataclass
import pathlib
import pickle
import random
from typing import cast
import uuid

import googleapiclient.discovery as discovery
import torch

import viewwizard.database as vdb
import viewwizard.model as vmodel
import viewwizard.actions as vactions


class ProgramSession:
    session_id: uuid.UUID
    datasets: list[tuple[str, vdb.VideoDataset]]
    models: list[
        tuple[
            str,
            vmodel.ThumbnailStatisticsModel,
            torch.optim.Optimizer,
            vmodel.ModelTrainingHistory,
        ]
    ]
    client: discovery.Resource | None

    def __init__(self):
        self.session_id = uuid.uuid4()

        self.datasets = []
        self.models = []
        self.client = None

    def save_to_path(self, path: pathlib.Path):
        model_names: list[str] = []
        model_states: list[dict] = []
        optimizer_states: list[dict] = []
        optimizer_classes: list[type] = []
        training_histories: list[vmodel.ModelTrainingHistory] = []

        for model_name, model, optimizer, history in self.models:
            model_names.append(model_name)

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
            model_names,
            model_states,
            optimizer_states,
            optimizer_classes,
            training_histories,
        ) = saved_states

        session = cls()
        session.__dict__.update(session_dict)

        # Reconstruct everything
        session.models = []
        for model_name, model_state, opt_state, opt_class, history in zip(
            model_names,
            model_states,
            optimizer_states,
            optimizer_classes,
            training_histories,
        ):
            model = vmodel.ThumbnailStatisticsModel()
            model.load_state_dict(model_state)

            optimizer = opt_class(model.parameters())
            optimizer.load_state_dict(opt_state)

            session.models.append((model_name, model, optimizer, history))

        session.client = None

        return session

    def ensure_client(self) -> discovery.Resource:
        resource: discovery.Resource | None = self.client

        if resource is None:
            resource = vdb.create_client()

        self.client = resource

        return resource

    def create_dataset(self, name: str):
        self.datasets.append((name, vdb.VideoDataset()))

    def search_new_thumbnails(
        self,
        dataset_number: int,
        search_count: int,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> int | None:
        if dataset_number >= len(self.datasets):
            return None

        client: discovery.Resource = self.ensure_client()

        _, dataset = self.datasets[dataset_number]

        new_ids: list[str] = []

        for index in range(search_count):
            new_ids.extend(
                vdb.sample_random_search_ids(
                    client, vdb.create_random_query(random.randint(2, 4))
                )
            )

            if progress_callback is not None:
                progress_callback(index, search_count)

        dataset.add_ids(new_ids)
        dataset.sync_id_video_data(client)

        return len(new_ids)

    def split_dataset(
        self,
        dataset_number: int,
        split_ratio: float,
        first_dataset_name: str,
        second_dataset_name: str,
    ) -> tuple[int, int] | None:
        if dataset_number >= len(self.datasets):
            return None

        _, dataset = self.datasets[dataset_number]

        first, second = dataset.split_dataset(split_ratio)

        self.datasets.append((first_dataset_name, first))
        self.datasets.append((second_dataset_name, second))

        return len(first.video_ids), len(second.video_ids)

    def delete_dataset(self, dataset_number: int):
        if dataset_number < len(self.datasets):
            del self.datasets[dataset_number]


@dataclass
class MenuOption:
    diolague: str

    suboptions: list["MenuOption"] | None = None
    callback: Callable[["MenuContext"], None] | None = None


DIOLAGUE_TREE: list[MenuOption] = [
    MenuOption(
        "Manage Session...",
        [
            MenuOption("Create New Session", callback=vactions.create_session),
            MenuOption("Save Session", callback=vactions.save_session),
            MenuOption("Load Session", callback=vactions.load_session),
        ],
    ),
    MenuOption(
        "Manage Datasets...",
        [
            MenuOption("List Datasets", callback=vactions.list_datasets),
            MenuOption("Create Dataset", callback=vactions.create_dataset),
            MenuOption("Delete Dataset", callback=vactions.delete_dataset),
            MenuOption("View Dataset Information", callback=vactions.view_dataset),
            MenuOption(
                "Search New Thumbnails", callback=vactions.search_new_thumbnails
            ),
            MenuOption("Shuffle Dataset", callback=vactions.shuffle_dataset),
            MenuOption("Merge Datasets", callback=vactions.merge_datasets),
            MenuOption("Split Dataset", callback=vactions.split_dataset),
        ],
    ),
    MenuOption(
        "Manage Models...",
        [
            MenuOption("List Models", callback=vactions.list_models),
            MenuOption(
                "Train Model",
                callback=lambda menu_context: asyncio.run(
                    vactions.train_model(menu_context)
                ),
            ),
        ],
    ),
    MenuOption("Exit"),
]


class MenuContext:
    session: ProgramSession
    options: list[MenuOption] = DIOLAGUE_TREE

    def __init__(
        self,
        session: ProgramSession,
        options: list[MenuOption] | None = None,
    ):
        self.session = session
        self.options = options or DIOLAGUE_TREE


def prompt(context: MenuContext) -> bool:
    print("\nSelect an option:")

    selection: str | None = None

    while (
        selection is None
        or not selection.isdigit()
        or not (0 <= int(selection) < len(context.options))
    ):
        for index, option in enumerate(context.options):
            print(f"{index}: {option.diolague}")

        selection = input("> ")

        if not selection.isdigit():
            print(f"Please enter a number. Your input `{selection}` is not a number.")

        if not (0 <= int(selection) < len(context.options)):
            print(
                f"Please enter a number between 0 and {len(context.options) - 1} (inclusive). Your input `{selection}` is out of range."
            )

    chosen_option: MenuOption = context.options[int(cast(str, selection))]

    if chosen_option.diolague == "Exit":
        return False

    if chosen_option.callback is not None:
        chosen_option.callback(context)

    if chosen_option.suboptions is not None:
        prompt(MenuContext(context.session, chosen_option.suboptions))

    return True
