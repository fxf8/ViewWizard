import asyncio
import glob
import fnmatch
import pathlib
import pickle
from typing import TYPE_CHECKING
import uuid

import matplotlib


matplotlib.use("QtAgg")

import matplotlib.pyplot as plt

import aiohttp
import tabulate
import torch

import viewwizard.database as vdb
import viewwizard.model as vmodel

if TYPE_CHECKING:
    import viewwizard.session as vsession


def create_session(menu_context: "vsession.MenuContext"):
    should_create_new_session: bool = (
        input(
            "Are you sure you want to create a new session? The existing session will be lost unless it was already saved to a file (y/n)"
        )
        == "y"
    )

    if should_create_new_session:
        import viewwizard.session as vsession

        menu_context.session = vsession.ProgramSession()


def save_session(menu_context: "vsession.MenuContext"):
    save_path: pathlib.Path | None = None

    while save_path is None:
        user_input = input(
            "Please enter a path to save the session to (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        save_path = pathlib.Path(user_input)

        if save_path.exists():
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
    is_session_loaded: bool = False

    while True:
        user_input = input(
            "Please enter a path to load the session from (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        load_path = pathlib.Path(user_input)

        if not load_path.exists():
            print(f"The path provided `{load_path}` does not exist.")
            continue

        try:
            menu_context.session = vsession.ProgramSession.load_from_path(load_path)
            is_session_loaded = True
            break

        except pickle.UnpicklingError:
            print(
                f"Could not load session from `{load_path}`. Would you like to try again? (y/n)"
            )

            response: str = input("> ")

            if response.lower() != "y":
                break

    if is_session_loaded and load_path:
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
    while True:
        dataset_name = input(
            "Please enter the name of the dataset (or type 'e' to exit): "
        )
        if dataset_name.lower() == "e":
            return
        if dataset_name:
            menu_context.session.create_dataset(dataset_name)
            break
        else:
            print("Dataset name cannot be empty.")


def delete_dataset(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to delete.")
        return

    list_datasets(menu_context)

    dataset_number: int | None = None
    while True:
        user_input = input(
            "Please enter the index of the dataset to delete (or type 'e' to exit): "
        )
        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)
            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    menu_context.session.delete_dataset(dataset_number)


def search_new_thumbnails(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to search.")

        return

    list_datasets(menu_context)

    dataset_number: int | None = None

    print(
        "Note: In this action, each search costs 100 google api units. Yields exactly 0 to 50 videos per search."
    )

    while True:
        user_input = input(
            "Please enter the index of the dataset to search (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)

            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    search_count: int | None = None

    while True:
        user_input = input(
            "Please enter the number of times to search (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            search_count = int(user_input)

            if search_count <= 0:
                print("Please enter a positive number.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    def progress_callback(current: int, total: int):
        print(f"Progress: {current + 1}/{total}")

    menu_context.session.search_new_thumbnails(
        dataset_number, search_count, progress_callback
    )


def view_dataset(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to view.")

        return

    list_datasets(menu_context)

    dataset_number: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the dataset to view (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)

            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    dataset_name, dataset = menu_context.session.datasets[dataset_number]

    print(f"Dataset Name: {dataset_name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Dataset Video Count: {len(dataset.videos)}")
    print(f"Dataset Video ID Count: {len(dataset.video_ids)}")


def shuffle_dataset(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to shuffle.")
        return

    list_datasets(menu_context)

    dataset_number: int | None = None
    while True:
        user_input = input(
            "Please enter the index of the dataset to shuffle (or type 'e' to exit): "
        )
        if user_input.lower() == "e":
            return
        try:
            dataset_number = int(user_input)
            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    menu_context.session.datasets[dataset_number][1].shuffle_dataset()


def merge_datasets(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) < 2:
        print("There are not enough datasets to merge (there must be at least two).")

        return

    list_datasets(menu_context)

    dataset_number_1: int | None = None
    while True:
        user_input = input(
            "Please enter the index of the first dataset to merge (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number_1 = int(user_input)
            if not (0 <= dataset_number_1 < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue

            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    dataset_number_2: int | None = None
    while True:
        user_input = input(
            "Please enter the index of the second dataset to merge (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number_2 = int(user_input)

            if not (0 <= dataset_number_2 < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue

            if dataset_number_1 == dataset_number_2:
                print("Cannot merge a dataset with itself.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    default_merged_dataset_name: str = f"Merged {menu_context.session.datasets[dataset_number_1][0]}+{menu_context.session.datasets[dataset_number_2][0]}"

    while True:
        user_input = input(
            f"Please enter the name of the merged dataset (default {default_merged_dataset_name}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        merged_dataset_name = user_input or default_merged_dataset_name

        if merged_dataset_name:
            break

        else:
            print("Dataset name cannot be empty.")

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
    while True:
        user_input = input(
            "Please enter the index of the dataset to split (or type 'e' to exit): "
        )
        if user_input.lower() == "e":
            return
        try:
            dataset_number = int(user_input)
            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    split_ratio: float | None = None
    while True:
        user_input = input("Please enter the split ratio (or type 'e' to exit): ")
        if user_input.lower() == "e":
            return
        try:
            split_ratio = float(user_input)
            if not (0 < split_ratio < 1):
                print("Split ratio must be between 0 and 1.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")

    default_first_dataset_name: str = f"Split Beginning {split_ratio} {menu_context.session.datasets[dataset_number][0]}"
    default_second_dataset_name: str = (
        f"Split End {split_ratio} {menu_context.session.datasets[dataset_number][0]}"
    )

    while True:
        user_input = input(
            f"Please enter the name of the first dataset (default {default_first_dataset_name}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        first_dataset_name = user_input or default_first_dataset_name

        if first_dataset_name:
            break

        else:
            print("Dataset name cannot be empty.")

    while True:
        user_input = input(
            f"Please enter the name of the second dataset (default {default_second_dataset_name}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        second_dataset_name = user_input or default_second_dataset_name

        if second_dataset_name:
            break

        else:
            print("Dataset name cannot be empty.")

    menu_context.session.split_dataset(
        dataset_number, split_ratio, first_dataset_name, second_dataset_name
    )


def rename_dataset(menu_context: "vsession.MenuContext"):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the dataset to rename (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)

            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    default_dataset_name: str = menu_context.session.datasets[dataset_number][0]

    while True:
        user_input = input(
            f"Please enter the new name of the dataset (default {default_dataset_name}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        if user_input != "":
            break

    old_dataset_name: str = menu_context.session.datasets[dataset_number][0]

    menu_context.session.datasets[dataset_number] = (
        user_input,
        menu_context.session.datasets[dataset_number][1],
    )

    print(
        f"Dataset {old_dataset_name} renamed to {user_input} at index {dataset_number}."
    )


def list_models(menu_context: "vsession.MenuContext"):
    print(
        tabulate.tabulate(
            [
                [
                    index,
                    model_name,
                    type(optimizer).__name__,
                    training_history.losses[-1][0] if training_history.losses else None,
                    training_history.losses[-1][1] if training_history.losses else None,
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


def create_model(menu_context: "vsession.MenuContext"):
    default_model_name: str = f"New Model {len(menu_context.session.models) + 1}"
    model_name: str

    while True:
        user_input = input(
            f"Please enter the name of the model (default {default_model_name}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        if user_input == "":
            model_name = default_model_name

            break

        model_name = user_input or default_model_name

        if model_name:
            break

        else:
            print("Model name cannot be empty.")

    default_learning_rate: float = 0.001
    learning_rate: float

    while True:
        user_input = input(
            f"Please enter the learning rate (default {default_learning_rate}, or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        if user_input == "":
            learning_rate = default_learning_rate

            break

        try:
            learning_rate = float(user_input)

            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    model: vmodel.ThumbnailStatisticsModel = vmodel.ThumbnailStatisticsModel()
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=learning_rate)

    menu_context.session.models.append(
        (model_name, model, optimizer, vmodel.ModelTrainingHistory())
    )

    print(
        f"Successfully created model '{model_name}' with learning rate {learning_rate} and optimizer {type(optimizer).__name__}."
    )


def delete_model(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.models) == 0:
        print("There are no models to delete.")

        return

    list_models(menu_context)

    model_number: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the model to delete (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            model_number = int(user_input)

            if not (0 <= model_number < len(menu_context.session.models)):
                print(
                    f"Invalid index: {model_number}, input must be between 0 and {len(menu_context.session.models) - 1}"
                )

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    del menu_context.session.models[model_number]


async def train_model(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.models) == 0:
        print("There are no models to train.")

        return

    list_models(menu_context)

    print(
        "Note: This action may consume google api units since it may need to pull youtube video data. However, this data will be cached and therefore not require google api units for data reusage."
    )

    model_index: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the model to train (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            model_index = int(user_input)

            if not (0 <= model_index < len(menu_context.session.models)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    list_datasets(menu_context)

    dataset_number: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the dataset to use (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)

            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    batch_size: int | None = None

    while True:
        user_input = input("Please enter the batch size (or type 'e' to exit): ")

        if user_input.lower() == "e":
            return
        try:
            batch_size = int(user_input)

            if batch_size <= 0:
                print("Please enter a positive value for the batch size.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    iterations: int | None = None

    while True:
        user_input = input(
            "Please enter the number of training iterations (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            iterations = int(user_input)

            if iterations <= 0:
                print("Please enter a positive number.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    _, model, optimizer, training_history = menu_context.session.models[model_index]

    dataset = menu_context.session.datasets[dataset_number][1]

    batches: list[
        vmodel.ThumbnailStatisticsTrainingBatch
    ] = await dataset.get_training_batches(batch_size)

    target_mean, target_std = dataset.view_count_mean_stdev()

    thresholds: list[int] = [int(iterations * i / 10) for i in range(1, 11)]
    iteration = 0

    for batch in batches:
        iteration += 1

        if iteration > iterations:
            print("Finished training early due to finishing batches.")

            break

        training_record: vmodel.ModelTrainingHistory = vmodel.train_model_batch(
            model, batch, optimizer, target_mean, target_std
        )

        training_history.merge(training_record)

        thresholds_passsed: list[int] = [
            threshold for threshold in thresholds if threshold <= iteration
        ]

        if len(thresholds_passsed) > 0:
            highest_thresholds_passed = max(thresholds_passsed)
            thresholds = [
                threshold
                for threshold in thresholds
                if threshold > highest_thresholds_passed
            ]

            print(
                f"Progress: {iteration}/{iterations} ({100 * iteration / iterations:.2f}%)"
            )

    print(
        f"Training complete. Final iteration: {training_history.losses[-1][0]}, Final loss: {training_history.losses[-1][1]}"
    )


def view_training_history(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.models) == 0:
        print("There are no models to view.")

        return

    list_models(menu_context)

    model_index: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the model to view (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            model_index = int(user_input)

            if not (0 <= model_index < len(menu_context.session.models)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    model_name, _, optimizer, training_history = menu_context.session.models[
        model_index
    ]

    graph_title: str = f"{model_name} Training History (Learning Rate: {optimizer.param_groups[0]['lr']})"
    horizontal_label: str = "Training Iterations"
    vertical_label: str = "Loss (L1 Smooth)"

    iterations: list[int] = [loss[0] for loss in training_history.losses]
    losses: list[float] = [loss[1] for loss in training_history.losses]

    plt.plot(iterations, losses)
    plt.title(graph_title)
    plt.xlabel(horizontal_label)
    plt.ylabel(vertical_label)

    plt.show()


async def validate_model(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.models) == 0:
        print("There are no models to validate.")

        return

    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to validate.")

        return

    list_models(menu_context)

    model_index: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the model to validate (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            model_index = int(user_input)

            if not (0 <= model_index < len(menu_context.session.models)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    list_datasets(menu_context)

    dataset_number: int | None = None

    while True:
        user_input = input(
            "Please enter the index of the dataset to validate (or type 'e' to exit): "
        )

        if user_input.lower() == "e":
            return

        try:
            dataset_number = int(user_input)

            if not (0 <= dataset_number < len(menu_context.session.datasets)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Invalid input. Please enter a number.")

    model_name, model, optimizer, _ = menu_context.session.models[model_index]

    validation_dataset: vdb.VideoDataset = menu_context.session.datasets[
        dataset_number
    ][1]

    validation_views_map: dict[uuid.UUID, str] = {
        video_data.video_data_id: video_data.google_video_data["statistics"][
            "viewCount"
        ]
        for video_data in validation_dataset.videos.values()
    }

    validation_tensors: list[tuple[uuid.UUID, torch.Tensor]] = []

    async with aiohttp.ClientSession() as aiohttp_session:
        validation_tensors = await asyncio.gather(
            *(
                validation_tensor.image_tensor(aiohttp_session)
                for validation_tensor in validation_dataset.videos.values()
            )
        )

    validation_thumbnail_tensors_map: dict[uuid.UUID, torch.Tensor] = {
        video_data_id: image_tensor
        for video_data_id, image_tensor in validation_tensors
    }

    available_video_data_ids: list[uuid.UUID] = [
        *validation_thumbnail_tensors_map.keys()
    ]

    validation_tensor_batch: torch.Tensor = torch.stack(
        [
            vdb.image_rescaler(
                validation_thumbnail_tensors_map[video_data_id], (244, 244)
            )
            for video_data_id in available_video_data_ids
        ]
    )

    print(f"{validation_tensor_batch.shape = }")

    model_output_log: torch.Tensor = model(validation_tensor_batch)

    actual_views = torch.tensor(
        [
            int(validation_views_map[video_data_id])
            for video_data_id in available_video_data_ids
        ],
        dtype=torch.float32,
    )
    actual_views_log = torch.log10(actual_views + 1)

    # The model predicts log10(views + 1), so we reverse it
    predicted_views = torch.pow(10, model_output_log.squeeze()) - 1

    print(
        f"Validation Report for Model: '{model_name}' Optimizer Type: '{type(optimizer).__name__}' Learning Rate {optimizer.param_groups[0]['lr']}"
    )
    print(f"Log of Actual Views (sample): {actual_views_log[:5]}")
    print(f"Log of Predicted Views (sample): {model_output_log.squeeze()[:5]}")

    # Calculate difference in log space (which is what the model was trained on)
    log_difference = torch.abs(model_output_log.squeeze() - actual_views_log)
    print(f"Mean Absolute Error (log10 space): {torch.mean(log_difference)}")

    # Calculate difference and percent error in original view count space
    difference = torch.abs(predicted_views - actual_views)
    # Add a small epsilon to avoid division by zero for videos with 0 views
    percent_error = (difference / (actual_views + 1e-7)) * 100

    print(f"Mean Absolute Error (view count): {torch.mean(difference)}")
    print(f"Mean Percentage Error (view count): {torch.mean(percent_error)}%")


def list_imported_images(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.datasets) == 0:
        print("There are no datasets to view.")

        return

    filter_pattern: str = ""

    filter_pattern = input(
        "Please enter a filter pattern (supports glob) (enter nothing for no filter) (or type 'e' to exit): "
    )

    if filter_pattern.lower() == "e":
        return

    table = [
        [
            index,
            image.name,
            image.id,
            image.original_image_path,
            image.processed_image_iterations,
        ]
        for index, image in enumerate(menu_context.session.images)
        if filter_pattern == "" or fnmatch.fnmatch(image.name, filter_pattern)
    ]

    print(
        tabulate.tabulate(table, headers=["Index", "Name", "ID", "Path", "Iterations"])
    )


def import_images(menu_context: "vsession.MenuContext"):
    import viewwizard.session as vsession

    # Uses glob to import images

    import_image_glob: str = input(
        "Please enter a glob pattern (supports glob) (or type 'e' to exit): "
    )

    if import_image_glob.lower() == "e":
        return

    paths: list[pathlib.Path] = [*map(pathlib.Path, glob.glob(import_image_glob))]

    if len(paths) == 0:
        print("No images found.")

        return

    for path in paths:
        menu_context.session.images.append(
            vsession.ProcessedImage(path, name=path.name)
        )


def delete_images(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.images) == 0:
        print("There are no images to delete.")

        return

    image_glob: str = input(
        "Please enter a glob pattern (supports glob) (or type 'e' to exit): "
    )

    if image_glob.lower() == "e":
        return

    filtered_images: list[vsession.ProcessedImage] = [
        image
        for image in menu_context.session.images
        if not fnmatch.fnmatch(image.name, image_glob)
    ]

    menu_context.session.images = filtered_images


def optimize_images(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.images) == 0:
        print("There are no images to optimize.")

        return

    if len(menu_context.session.models) == 0:
        print("There are no models to optimize with.")

        return

    list_models(menu_context)

    model_choice_index: int = 0

    while True:
        try:
            model_choice_index = int(
                input("Please enter the index of the model to optimize with: ")
            )

            if not (0 <= model_choice_index < len(menu_context.session.models)):
                print("Invalid index.")

                continue
            break

        except ValueError:
            print("Please enter a number.")

    image_glob: str = input(
        "Please enter a glob pattern for the images to optimize (supports glob) (or type 'e' to exit): "
    )

    if image_glob.lower() == "e":
        return

    iterations: int = 0

    while True:
        try:
            iterations = int(
                input("Please enter the number of optimization iterations: ")
            )

            if iterations <= 0:
                print("Please enter a positive number.")

                continue
            break

        except ValueError:
            print("Please enter a number.")

    for image in menu_context.session.images:
        if not fnmatch.fnmatch(image.name, image_glob):
            continue

        image.processed_image = vmodel.optimize_image_batch(
            menu_context.session.models[model_choice_index][1],
            image.processed_image.unsqueeze(0),
            iterations,
        ).squeeze(0)

        image.processed_image_iterations += iterations

    print(f"Completed Optimization with {iterations} iterations.")


def view_images(menu_context: "vsession.MenuContext"):
    if len(menu_context.session.images) == 0:
        print("There are no images to view.")

        return

    image_glob: str = input(
        "Please enter a glob pattern for the images to view (supports glob) (or type 'e' to exit): "
    )

    if image_glob.lower() == "e":
        return

    images_to_display: list[vsession.ProcessedImage] = [
        image
        for image in menu_context.session.images
        if fnmatch.fnmatch(image.name, image_glob)
    ]

    print(f"Displaying {len(images_to_display)} image(s).")

    # Use matplotlib to display images in a grid

    show_original = (
        input("Would you like to view the original images as well? (y/n): ").lower()
        == "y"
    )

    # Prepare list of images and titles
    all_images = []
    titles = []

    for image in images_to_display:
        # Add processed image
        all_images.append(image.processed_image)
        titles.append(f"{image.name} (processed)")

        if show_original and image.original_image_tensor is not None:
            all_images.append(image.original_image_tensor)
            titles.append(f"{image.name} (original)")

    num_images = len(all_images)
    columns = 2  # you can change this if you want more columns
    rows = (num_images + columns - 1) // columns

    figure, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    axes = axes.flatten()  # flatten in case of multiple rows/cols

    for axis_index, axis in enumerate(axes):
        if axis_index < num_images:
            image = all_images[axis_index]
            # Convert tensor to numpy image for matplotlib
            if isinstance(image, torch.Tensor):
                image_numpy = image.permute(1, 2, 0).cpu().clone().detach().numpy()
            else:
                image_numpy = image

            axis.imshow(image_numpy)
            axis.set_title(titles[axis_index], fontsize=10)
            axis.axis("off")
        else:
            axis.axis("off")  # hide unused subplots

    plt.tight_layout()
    plt.show()


def export_images(menu_context: "vsession.MenuContext"): ...
