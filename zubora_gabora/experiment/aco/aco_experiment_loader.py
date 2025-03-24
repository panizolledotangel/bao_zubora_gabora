import pandas as pd
from os import listdir
from os.path import isfile, join


class ACOExperimentLoader:

    def __init__(self, experiment_folder: str):
        self.experiment_folder = experiment_folder
        self.experiment_data = self._load_data()

    def _load_data(self) -> pd.DataFrame:
        """
        Loads all experiments from the experiment folder.
        """
        files = [f for f in listdir(self.experiment_folder) if isfile(join(self.experiment_folder, f))]
        data = []
        for f in files:
            actual_data = pd.read_csv(f"{self.experiment_folder}/{f}")
            data.append(actual_data)
        return pd.concat(data)    