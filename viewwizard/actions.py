import pathlib
import pickle
import tabulate

import viewwizard.session as vsession
import viewwizard.database as vdb


def save_session(menu_context: vsession.MenuContext):
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


def load_session(menu_context: vsession.MenuContext):
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


def list_datasets(menu_context: vsession.MenuContext):
    table = [
        [index, dataset_name, dataset_id]
        for index, (dataset_name, dataset_id) in enumerate(
            menu_context.session.datasets
        )
    ]

    print(
        tabulate.tabulate(
            table, headers=["Dataset Index", "Dataset Name", "Dataset ID"]
        )
    )


def create_dataset(menu_context: vsession.MenuContext):
    dataset_name: str = input("Please enter the name of the dataset: ")

    menu_context.session.create_dataset(dataset_name)


def delete_dataset(menu_context: vsession.MenuContext):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while dataset_number is None or dataset_number >= len(
        menu_context.session.datasets
    ):
        dataset_number = int(input("Please enter the index of the dataset to delete: "))

    menu_context.session.delete_dataset(dataset_number)


def search_new_thumbnails(menu_context: vsession.MenuContext):
    dataset_number: int | None = None

    while dataset_number is None or dataset_number >= len(
        menu_context.session.datasets
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


def view_dataset(menu_context: vsession.MenuContext):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while dataset_number is None or dataset_number >= len(
        menu_context.session.datasets
    ):
        dataset_number = int(input("Please enter the index of the dataset to view: "))

    dataset_name, dataset = menu_context.session.datasets[dataset_number]

    print(f"Dataset Name: {dataset_name}")
    print(f"Dataset ID: {dataset.dataset_id}")
    print(f"Dataset Video Count: {len(dataset.video_ids)}")
    print(f"Dataset Video IDs: {dataset.video_ids}")


def shuffle_dataset(menu_context: vsession.MenuContext):
    dataset_number: int | None = None

    while dataset_number is None or dataset_number >= len(
        menu_context.session.datasets
    ):
        dataset_number = int(
            input("Please enter the index of the dataset to shuffle: ")
        )

    menu_context.session.datasets[dataset_number][1].shuffle_dataset()


def merge_datasets(menu_context: vsession.MenuContext):
    list_datasets(menu_context)

    dataset_number_1: int | None = None

    while dataset_number_1 is None or dataset_number_1 >= len(
        menu_context.session.datasets
    ):
        dataset_number_1 = int(
            input("Please enter the index of the first dataset to merge: ")
        )

    dataset_number_2: int | None = None

    while dataset_number_2 is None or dataset_number_2 >= len(
        menu_context.session.datasets
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


def split_dataset(menu_context: vsession.MenuContext):
    list_datasets(menu_context)

    dataset_number: int | None = None

    while dataset_number is None or dataset_number >= len(
        menu_context.session.datasets
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
