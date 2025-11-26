import pathlib
import pickle
import tabulate

import viewwizard.session as vsession


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


def view_dataset(menu_context: vsession.MenuContext):
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
