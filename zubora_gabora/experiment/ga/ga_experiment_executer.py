import pandas as pd
import shutil
from os import makedirs
from os.path import isdir
from typing import Dict, Tuple
from inspyred import ec

from zubora_gabora.ga.ga_zubora_gabora import GAZuboraGabora


class GAExperimentExecuter:
    
    def __init__(self, data_path="data/datset.csv"):
        self.dataset = pd.read_csv(data_path)

    def run_single_experiment(self, experiment_id: int, **kwargs) -> pd.DataFrame:
        """
        Runs an experiment with the given GA algorithm and dataset.
        """
        swords, zubora, gabora = self._read_data(experiment_id)
        ga = GAZuboraGabora(swords,zubora,gabora,**kwargs)
        candidate, fitness = ga.run()
        return ga, candidate, fitness
    
    def run_repeated_experiment(self, experiment_id: int, n_repeat=31, **kwargs) -> pd.DataFrame:
        """
        Runs an experiment with the given GA algorithm and dataset n_repeat times.
        Returns a DataFrame with the fitness and number of generations of each run.
        """ 
        results = [] 
        for i in range(n_repeat):
            print(f"Running experiment: dataset {experiment_id} - run {i+1}/{n_repeat}")
            ga, candidate, fitness = self.run_single_experiment(experiment_id, **kwargs)
            actual_result = {
                "run": i,
                "fitness": fitness,
                "n_evaluations": ga.num_evaluations
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
                **params
            )
            
            actual_result["experiment_id"] = i
            actual_result.to_csv(f"{experiment_folder}/experiment_{i}.csv", index=False)
            print(f"Experiment {i} finished and saved.")

    def _read_data(self, experiment_id: int) -> Tuple[int, Dict[str, Dict[str, float]]]:
        
        data = self.dataset.query(f"id == {experiment_id}").iloc[0]

        if len(data) == 0:
            raise ValueError(f"Experiment with id {experiment_id} not found.")

        n_blades = data["N"]
        zubora = (data["forging_zubora"], data["grinding_zubora"])
        gabora = (data["forging_gabora"], data["grinding_gabora"])
        return n_blades, zubora, gabora
        
    def _calculate_params(self, experiment_id: int) -> Dict[str, float]:
        """
        Calculates the parameters for the GA algorithm based on the dataset.
        """
        fixed = {
            "selection": ec.selectors.tournament_selection,
            "tournament_size": 2,
            "replacer": ec.replacers.generational_replacement,
            "num_elites": 1,
            "terminator": ec.terminators.no_improvement_termination,
            "max_generations": 20,
            "crossover": GAZuboraGabora.zg_crossover,
            "bin_crossover_rate": 1,
            "per_crossover_rate": 1,
            "mutation": GAZuboraGabora.zg_mutation,
            "bin_mutation_rate": 0.1,
            "per_mutation_rate": 0.3
        }
        data = self.dataset.query(f"id == {experiment_id}").iloc[0]
        if data["N"] < 50:
            fixed["pop_size"] = 20
        elif data["N"] < 100:
            fixed["pop_size"] = 50
        else:
            fixed["pop_size"] = 100

        return fixed


