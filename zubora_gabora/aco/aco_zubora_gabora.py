import numpy as np
from copy import deepcopy
from typing import Tuple, Dict, List

class ACOZuboraGabora:
    
    @classmethod
    def decode_solution(cls, solution: Tuple[List[int], List[int]]) -> Tuple[List[Dict[str, str]], List[Tuple[str, int]]]:
      """
      Returns a solution in a human readable format.
      """  
      return cls._decode_who(solution[0]), cls._decode_order(solution[1])

    @classmethod
    def _decode_who(cls, who: List[int]) -> List[Dict[str, str]]:
      """
      Returns the worker assignment part of the solution in a human readable format.
      """
      decoded = []
      for i in range(0,len(who),2):
        decoded.append({
            "F": "Zu" if who[i] == 0 else "Ga",
            "G": "Zu" if who[i+1] == 0 else "Ga"
        })
      return decoded
    
    @classmethod
    def _decode_order(cls, order: List[int]) -> List[Tuple[str, int]]:
        """
        Returns the forging order part of the solution in a human readable format.
        """
        decoded = []
        for taskid in order:
          swid = taskid // 2
          op = "F" if taskid % 2 == 0 else "G"
          decoded.append((op, swid+1))
        return decoded

    def __init__(
          self, 
          n_blades:int, 
          times: Dict[str, Dict[str, float]], 
          n_ants: int = 10, 
          alpha: float = 1, 
          beta: float = 5, 
          rho: float = 0.8, 
          n_cicles_no_improve = 5
    ):
        self.times = times
        self.n_blades = n_blades

        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        
        self.n_cicles_no_improve = n_cicles_no_improve

        self.pheromone_who = np.ones((self.n_blades*2,2))
        self.heuristic_who = self._make_who_heuristic()
        self.pheromone_order = np.ones((self.n_blades*2,self.n_blades*2))
        self.best_solution = None
        self.best_fitness = 0.0

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def optimize(self):
        self._initialize()

        while not self.stop_condition():
            trails = []
            for _ in range(self.n_ants):
                solution = self._construct_solution()
                fitness = self._evaluate(solution)
                trails.append((solution, fitness))

                if fitness > self.best_fitness:
                    self.best_solution = solution
                    self.best_fitness = fitness

            self._update_pheromone(trails, self.best_fitness)

            self.trails_history.append(deepcopy(trails))
            self.best_fitness_history.append(self.best_fitness)

        return self.best_solution

    def _initialize(self):
        self.pheromone_who = np.ones((self.n_blades*2,2))
        self.pheromone_order = np.ones((self.n_blades*2,self.n_blades*2))
        self.best_solution = None
        self.best_fitness = 0.0

        self.pheromone_history = []
        self.trails_history = []
        self.best_fitness_history = []

    def stop_condition(self) -> bool:
        """
        Check if if the N last iterations did not improve the best solution
        """
        if len(self.best_fitness_history) < self.n_cicles_no_improve:
          return False
        
        
        stop_condition = np.all(np.isclose(self.best_fitness_history[-self.n_cicles_no_improve:], self.best_fitness))
  
        return stop_condition

    def _evaluate(self, solution: Tuple[List[int], List[int]]) -> float:
        """
        Calculates the inverse end time of the sword that is forged last
        """
        who, order = solution
        _, times = self._simulate_forging(who, order)
        end_time = max([sword["G"][1] for sword in times])
        return 1.0/float(end_time)

    def _simulate_forging(self, who: List[int], order: List[int]) -> Tuple[Dict[str, int], List[Dict[str, int]]]:
        """
        Calculates the start and end times of each operation for each sword
        """
        free = {
            'Zu': 1e-6,
            'Ga': 1e-6
        }

        times = []
        for _ in range(self.n_blades):
          times.append({"F": [1e-6,1e-6], "G": [float('inf'),float('inf')]})

        for i in order:
          op = "F" if i % 2 == 0 else "G"
          swid = i // 2

          worker = "Zu" if who[i] == 0 else "Ga"
          op_time = self.times[worker][op]

          start_time = free[worker]
          if op == "G":
            start_time = max(times[swid]["F"][1], start_time)

          times[swid][op][0] = start_time
          times[swid][op][1] = start_time + op_time
          free[worker] = start_time + op_time

        return free, times

    def _construct_who(self) -> List[int]:
        """
        Constructs the worker assignment part of the solution
        """
        pheromone = self.pheromone_who**self.alpha
        heuristic = self.heuristic_who**self.beta

        probabilities = pheromone * heuristic
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)

        random_values = np.random.rand(probabilities.shape[0])
        choices = (random_values < probabilities[:, 1]).astype(int)

        return choices.tolist()

    def _construct_order(self, who: List[int]) -> List[int]:
      """
      Constructs the forging order part of the solution
      """
      pending = list(range(self.n_blades*2))
      order = []

      for i in range(self.n_blades*2):
        candidates = self._candidates(order, pending)

        pheromone = self.pheromone_order[i,:]
        pheromone = pheromone**self.alpha

        heuristic = self._make_order_heuristic(order, candidates, who)
        heuristic = heuristic**self.beta

        probabilities = pheromone * heuristic
        mask = np.array([False]*self.n_blades*2)
        mask[candidates] = True
        probabilities[~mask] = 0
        probabilities = probabilities / np.sum(probabilities)

        choice = np.random.choice(range(self.n_blades*2), p=probabilities)
        pending.remove(choice)

        order.append(int(choice))

      return order

    def _construct_solution(self) -> Tuple[List[int], List[int]]:
        """
        Constructs a solution by combining the worker assignment and forging order parts
        """
        who = self._construct_who()
        order = self._construct_order(who)
        return who, order

    def _candidates(self, order: List[int], pending: List[int]):
      """
      Retruns a list of tasks that are pending to be completed.
      The grinding operation are only considered if the forging operation is already in the order.
      """
      candidates = []

      for i in pending:
        if i % 2 == 0:
          candidates.append(i)
        else:
          if (i-1) in order:
            candidates.append(i)

      return candidates

    def _make_who_heuristic(self) -> np.ndarray:
        """
        Returns a matrix of size (n_blades*2, 2) where each row contains the inverse time of each worker needs to complete the operation.
        """
        heuristic = np.ones((self.n_blades*2,2))

        for i in range(self.n_blades*2):
          op = "F" if i % 2 == 0 else "G"
          heuristic[i][0] = 1.0/self.times["Zu"][op]
          heuristic[i][1] = 1.0/self.times["Ga"][op]

        return heuristic

    def _make_order_heuristic(self, order: List[int], candidates: List[int], who: List[int]) -> np.ndarray:
        """
        For each candidate task, returns the inverse time that task will finish if it is selected next.
        """
        free, times = self._simulate_forging(who, order)

        heuristic = np.ones(self.n_blades*2) * 1e-6
        for i in candidates:
          op = "F" if i % 2 == 0 else "G"
          swid = i // 2

          worker = "Zu" if who[i] == 0 else "Ga"
          op_time = self.times[worker][op]

          start_time = free[worker]
          if op == "G":
            start_time = max(times[swid]["F"][1], start_time)

          heuristic[i] = 1.0/(start_time + op_time)

        return heuristic

    def _update_pheromone(self, trails: List[Tuple[Tuple[List[int], List[int]], float]], best_fitness):
        self.pheromone_history.append((self.pheromone_who.copy(), self.pheromone_order.copy()))

        evaporation = 1 - self.rho
        self.pheromone_who *= evaporation
        self.pheromone_order *= evaporation

        for solution, fitness in trails:
          delta_fitness = 1.0/(1.0 + (best_fitness - fitness) / best_fitness)

          if delta_fitness < 0:
            print(best_fitness, fitness)

          who, order = solution
          self.pheromone_who[np.arange(len(who)), np.array(who)] += delta_fitness
          self.pheromone_order[np.arange(len(order)), np.array(order)] += delta_fitness