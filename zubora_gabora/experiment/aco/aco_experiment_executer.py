import pandas as pd
import shutil
from os import makedirs
from os.path import isdir
from typing import Dict, Tuple

from zubora_gabora.aco.aco_zubora_gabora import ACOZuboraGabora


class ACOExperimentExecuter:
    
    def __init__(self, data_path="data/datset.csv"):
        self.dataset = pd.read_csv(data_path)

    def run_single_experiment(self, experiment_id: int, n_ants=10, alpha=0.5, beta=1.0, rho=0.75, n_cicles_no_improve=50) -> pd.DataFrame:
        """
        Runs an experiment with the given ACO algorithm and dataset.
        """
        blades, times = self._read_data(experiment_id)
        aco = ACOZuboraGabora(
            n_ants=n_ants,      
            n_blades=blades,
            times=times,
            alpha=alpha,
            beta=beta,
            rho=rho,
            n_cicles_no_improve=n_cicles_no_improve
        )
        aco.optimize()
        return aco
    
    def run_repeated_experiment(self, experiment_id: int, n_repeat=31, n_ants=10, alpha=0.5, beta=1.0, rho=0.75, n_cicles_no_improve=50) -> pd.DataFrame:
        """
        Runs an experiment with the given ACO algorithm and dataset n_repeat times.
        Returns a DataFrame with the fitness and number of cicles of each run.
        """ 
        results = [] 
        for i in range(n_repeat):
            print(f"Running experiment: dataset {experiment_id} - run {i+1}/{n_repeat}")
            aco = self.run_single_experiment(experiment_id, n_ants, alpha, beta, rho, n_cicles_no_improve)
            actual_result = {
                "run": i,
                "fitness": 1.0/aco.best_fitness,
                "n_evaluations": aco.n_evaluations
            }
            results.append(actual_result)

        return pd.DataFrame(results)
    
    def run_all_experiments(self, experiment_folder: str, overwrite=False, n_repeat=31):
        """
        Runs all experiments in the dataset n_repeat times.
        Returns a DataFrame with the fitness and number of cicles of each run.
        """ 

        if isdir(experiment_folder):
            if overwrite:
                shutil.rmtree(experiment_folder)
            else:
                raise ValueError(f"Folder {experiment_folder} already exists. Set overwrite=True to overwrite the folder.")
        
        makedirs(experiment_folder)

        for i in self.dataset["id"].to_list():
            params = self._calculate_params(i)

            print(f"Running experiment {i} (n_blades: {self.dataset.query(f'id == {i}')['N'].iloc[0]}) with params: {params}")
            actual_result = self.run_repeated_experiment(
                experiment_id=i, 
                n_repeat=n_repeat,
                n_ants=params["n_ants"],
                alpha=params["alpha"],
                beta=params["beta"],
                rho=params["rho"],
                n_cicles_no_improve=params["n_cicles_no_improve"]
            )
            
            actual_result["experiment_id"] = i
            actual_result.to_csv(f"{experiment_folder}/experiment_{i}.csv", index=False)
            print(f"Experiment {i} finished and saved.")

    def _read_data(self, experiment_id: int) -> Tuple[int, Dict[str, Dict[str, float]]]:
        
        data = self.dataset.query(f"id == {experiment_id}").iloc[0]

        if len(data) == 0:
            raise ValueError(f"Experiment with id {experiment_id} not found.")

        n_blades = data["N"]
        times = {
            "Zu": {
                "F": data["forging_zubora"],
                "G": data["grinding_zubora"]
            },
            "Ga": {
                "F": data["forging_gabora"],
                "G": data["grinding_gabora"]
            }
        }
        return n_blades, times
    
    def _calculate_params(self, experiment_id: int) -> Dict[str, float]:
        """
        Calculates the parameters for the ACO algorithm based on the dataset.
        """
        fixed = {
            "alpha": 0.5,
            "beta": 1.0,
            "rho": 0.75,
            "n_cicles_no_improve": 50
        }

        data = self.dataset.query(f"id == {experiment_id}").iloc[0]
        if data["N"] < 100:
            fixed["n_ants"] = 10
        elif data["N"] < 1000:
            fixed["n_ants"] = 50
        else:
            fixed["n_ants"] = 100

        return fixed


        




