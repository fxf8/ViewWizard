import asyncio

import viewwizard.session as vsession
import viewwizard.actions as vactions
from viewwizard.session import MenuOption


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
            MenuOption("Rename Dataset", callback=vactions.rename_dataset),
        ],
    ),
    MenuOption(
        "Manage Models...",
        [
            MenuOption("List Models", callback=vactions.list_models),
            MenuOption("Create Model", callback=vactions.create_model),
            MenuOption("Delete Model", callback=vactions.delete_model),
            MenuOption(
                "Train Model",
                callback=lambda menu_context: asyncio.run(
                    vactions.train_model(menu_context)
                ),
            ),
            MenuOption(
                "View Model Training History Graph",
                callback=vactions.view_training_history,
            ),
            MenuOption(
                "Validate Model",
                callback=lambda menu_context: asyncio.run(
                    vactions.validate_model(menu_context)
                ),
            ),
        ],
    ),
    MenuOption(
        "Use Models...",
        [
            MenuOption("List Imported Images", callback=vactions.list_imported_images),
            MenuOption("Import Images", callback=vactions.import_images),
            MenuOption("Delete Images", callback=vactions.delete_images),
            MenuOption("Optimize Images", callback=vactions.optimize_images),
            MenuOption("View Optimized Images", callback=vactions.view_images),
            MenuOption("Export Images", callback=vactions.export_images),
        ],
    ),
]


def main():
    menu_context: vsession.MenuContext = vsession.MenuContext(
        vsession.ProgramSession(), DIOLAGUE_TREE
    )
    is_running: bool = True

    while is_running:
        is_running = vsession.prompt(menu_context)


if __name__ == "__main__":
    main()
