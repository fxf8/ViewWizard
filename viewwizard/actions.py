from collections.abc import AsyncGenerator
import pathlib
import pickle
import tabulate
from typing import TYPE_CHECKING

import viewwizard.database as vdb
import viewwizard.model as vmodel

if TYPE_CHECKING:
    import viewwizard.session as vsession


def save_session(menu_context: "vsession.MenuContext"):
    save_path: pathlib.Path | None = None

    while save_path is None or not save_path.exists():
        save_path = pathlib.Path(input("Please enter a path to save the session to: "))

        if not save_path.exists():
            print(f"The path provided `{save_path}` does not exist. Please try again.")

        elif save_path.exists():
            print(
                f"Are you sure you want to overwrite the file at `{save_path}`? (y/n)"
            )

            response: str = input("> ")

            if response.lower() != "y":
                save_path = None

    menu_context.session.save_to_path(save_path)

    print(
        f"Session saved to `{save_path.absolute() if save_path.is_absolute() else save_path}`."
    )


def load_session(menu_context: "vsession.MenuContext"):
    import viewwizard.session as vsession

    load_path: pathlib.Path | None = None
    is_complete: bool = False
    is_session_loaded: bool = False

    while load_path is None or not load_path.exists() or not is_complete:
        load_path = pathlib.Path(
            input("Please enter a path to load the session from: ")
        )

        if not load_path.exists():
            print(f"The path provided `{load_path}` does not exist. Please try again.")

        try:
            menu_context.session = vsession.ProgramSession.load_from_path(load_path)
            is_complete = True
            is_session_loaded = True

        except pickle.UnpicklingError:
            print(
                f"Could not load session from `{load_path}`. Would you like to try again? (y/n)"
            )

            response: str = input("> ")

            if response.lower() != "y":
                is_complete = True

    if is_session_loaded:
        print(
            f"Session loaded from `{load_path.absolute() if load_path.is_absolute() else load_path}`."
        )


def list_datasets(menu_context: "vsession.MenuContext"):
    table = [
        [
            index,
            dataset_name,
            dataset.dataset_id,
            len(dataset.video_ids),
            len(dataset.videos),
        ]
        for index, (
            dataset_name,
            dataset,
        ) in enumerate(menu_context.session.datasets)
    ]

    print(
        tabulate.tabulate(
            table,
            headers=[
                "Dataset Index",
                "Dataset Name",
                "Dataset ID",
                "Video ID Count",
                "Video Data Count",
            ],
        )
    )


def create_dataset(menu_context: "vsession.MenuContext"):
    dataset_name: str = input("Please enter the name of the dataset: ")

    menu_context.session.create_dataset(dataset_name)


def delete_dataset(menu_context: "vsession.MenuContext"):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(input("Please enter the index of the dataset to delete: "))

    menu_context.session.delete_dataset(dataset_number)


def search_new_thumbnails(menu_context: "vsession.MenuContext"):
    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(input("Please enter the index of the dataset to search: "))

    search_count: int | None = None

    while search_count is None or search_count <= 0:
        search_count = int(input("Please enter the number of times to search: "))

    def progress_callback(current: int, total: int):
        print(f"Progress: {current}/{total}")

    menu_context.session.search_new_thumbnails(
        dataset_number, search_count, progress_callback
    )


def view_dataset(menu_context: "vsession.MenuContext"):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(input("Please enter the index of the dataset to view: "))

    dataset_name, dataset = menu_context.session.datasets[dataset_number]

    print(f"Dataset Name: {dataset_name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Dataset Video Count: {len(dataset.video_ids)}")
    print(f"Dataset Video IDs: {dataset.video_ids}")


def shuffle_dataset(menu_context: "vsession.MenuContext"):
    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(
            input("Please enter the index of the dataset to shuffle: ")
        )

    menu_context.session.datasets[dataset_number][1].shuffle_dataset()


def merge_datasets(menu_context: "vsession.MenuContext"):
    list_datasets(menu_context)

    dataset_number_1: int | None = None

    while (
        dataset_number_1 is None
        or dataset_number_1 >= len(menu_context.session.datasets)
        or dataset_number_1 < 0
    ):
        dataset_number_1 = int(
            input("Please enter the index of the first dataset to merge: ")
        )

    dataset_number_2: int | None = None

    while (
        dataset_number_2 is None
        or dataset_number_2 >= len(menu_context.session.datasets)
        or dataset_number_2 < 0
    ):
        dataset_number_2 = int(
            input("Please enter the index of the second dataset to merge: ")
        )

    default_merged_dataset_name: str = f"Merged {menu_context.session.datasets[dataset_number_1][0]}+{menu_context.session.datasets[dataset_number_2][0]}"

    merged_dataset_name: str = (
        input(
            f"Please enter the name of the merged dataset (default {default_merged_dataset_name}): "
        )
        or default_merged_dataset_name
    )

    merged_dataset: vdb.VideoDataset = vdb.VideoDataset.merge_datasets(
        [
            menu_context.session.datasets[dataset_number_1][1],
            menu_context.session.datasets[dataset_number_2][1],
        ]
    )

    menu_context.session.datasets.append((merged_dataset_name, merged_dataset))


def split_dataset(menu_context: "vsession.MenuContext"):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(input("Please enter the index of the dataset to split: "))

    split_ratio: float | None = None

    while split_ratio is None or split_ratio <= 0 or split_ratio >= 1:
        split_ratio = float(input("Please enter the split ratio: "))

    default_first_dataset_name: str = f"Split Beginning {split_ratio} {menu_context.session.datasets[dataset_number][0]}"
    default_second_dataset_name: str = (
        f"Split End {split_ratio} {menu_context.session.datasets[dataset_number][0]}"
    )

    first_dataset_name: str = (
        input(
            f"Please enter the name of the first dataset (default {default_first_dataset_name}): "
        )
        or default_first_dataset_name
    )
    second_dataset_name: str = (
        input(
            f"Please enter the name of the second dataset (default {default_second_dataset_name}): "
        )
        or default_second_dataset_name
    )

    menu_context.session.split_dataset(
        dataset_number, split_ratio, first_dataset_name, second_dataset_name
    )


def list_models(menu_context: "vsession.MenuContext"):
    print(
        tabulate.tabulate(
            [
                [
                    index,
                    model_name,
                    type(optimizer).__name__,
                    training_history.losses[-1][0],
                    training_history.losses[-1][1],
                ]
                for index, (
                    model_name,
                    _,
                    optimizer,
                    training_history,
                ) in enumerate(menu_context.session.models)
            ],
            headers=[
                "Index",
                "Model Name",
                "Optimizer Type",
                "Training Iterations",
                "Most Recent Loss",
            ],
        )
    )


async def train_model(menu_context: "vsession.MenuContext"):
    list_models(menu_context)

    model_index: int | None = None

    while (
        model_index is None
        or model_index >= len(menu_context.session.models)
        or model_index < 0
    ):
        try:
            model_index = int(input("Please enter the index of the model to train: "))

        except ValueError as error:
            print(error)

    list_datasets(menu_context)

    dataset_number: int | None = None

    while (
        dataset_number is None
        or dataset_number >= len(menu_context.session.datasets)
        or dataset_number < 0
    ):
        dataset_number = int(input("Please enter the index of the dataset to use: "))

    batch_size: int | None = None

    while batch_size is None or batch_size <= 0:
        try:
            batch_size = int(input("Please enter the batch size: "))

        except ValueError as error:
            print(error)

        if batch_size is not None and batch_size <= 0:
            print("Please enter a positive value for the batch size.")

    iterations: int | None = None

    while iterations is None or iterations <= 0:
        try:
            iterations = int(input("Please enter the number of training iterations: "))

        except ValueError as error:
            print(error)

    _, model, optimizer, training_history = menu_context.session.models[model_index]

    batches: AsyncGenerator[vmodel.ThumbnailStatisticsTrainingBatch, None] = (
        menu_context.session.datasets[dataset_number][1].get_training_batches(
            batch_size
        )
    )

    total_batches: int = len(menu_context.session.datasets[dataset_number][1].video_ids)
    current_batches_used: int = 0
    thresholds: list[int] = [int(total_batches * i / 10) for i in range(1, 11)]

    async for batch in batches:
        current_batches_used += 1

        training_record: vmodel.ModelTrainingHistory = vmodel.train_model_batch(
            model, batch, optimizer
        )

        training_history.merge(training_record)

        thresholds_passsed: list[int] = [
            threshold for threshold in thresholds if threshold <= current_batches_used
        ]

        if len(thresholds_passsed) > 0:
            highest_thresholds_passed = max(thresholds_passsed)
            thresholds = [
                threshold
                for threshold in thresholds
                if threshold > highest_thresholds_passed
            ]

            print(
                f"Progress: {current_batches_used}/{total_batches} ({100 * current_batches_used / total_batches:.2f}%)"
            )

    print(
        f"Training complete. Final loss: {training_history.losses[-1][0]}, Final accuracy: {training_history.losses[-1][1]}"
    )
