import pathlib
import pickle
import uuid

import viewwizard.database as db
import viewwizard.model as vmodel
import viewwizard.schema as schema


class ProgramSession:
    session_id: uuid.UUID
    dataset: list[tuple[str, db.VideoDataset]]
    models: list[vmodel.ThumbnailStatisticsModel]

    def __init__(self):
        self.session_id = uuid.uuid4()

        self.dataset = []
        self.models = []

    def save_to_path(self, path: pathlib.Path):
        model_states = [model.state_dict() for model in self.models]

        temporary_models = self.models
        self.models = []

        with path.open("wb") as file:
            pickle.dump((self.__dict__, model_states), file)

        self.models = temporary_models

    @classmethod
    def load_from_path(cls, path: pathlib.Path) -> "ProgramSession":
        with path.open("rb") as file:
            session_dict, model_states = pickle.load(file)

        session = cls()
        session.__dict__.update(session_dict)

        for model_state in model_states:
            session.models.append(vmodel.ThumbnailStatisticsModel())
            session.models[-1].load_state_dict(model_state)

        return session
