import math
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from random import Random
from time import time
from inspyred import ec, benchmarks

class ZuboraGabora(benchmarks.Benchmark):
    """Defines the Zubora Gabora benchmark problem.

    This class defines the Zubora Gabora problem: given a set of swords,
    which need forging and grinding, and given Zubora and Gabora times
    that they take to perform each task (in hours), we want to find the
    minimum makespan (finishing time) to finish every sword. It is
    necessary to assure that forging of each sword is always performed
    beforee grinding, and that Zubora and Gabora can only perform one
    task at a time. The problem is naturally defined as a minimization
    problem. We represent the problem with two parts: a binary array,
    which will indicate if the task is performed by Zubora (0) or
    Gabora (1), and a permutation indicating the order in which tasks
    are performed. To evaluate the individuals, two lists are created
    with the tasks performed by Zubora and Gabora, and ordered by the
    order of the permutation. Then start and end times are calculated
    for each task, and the makespan is calculated as the maximum end
    time of all tasks. Crossover are mutation are performed by using
    one-point crossover and bit-flip mutation for the binary part, and
    pmx and inversion mutation for the permutation part.

    Public Attributes:

    - *swords* -- the number of swords to produce
    - *zubora* -- a tuple indicating the time Zubora takes to forge and
      grind a sword
    - *gabora* -- a tuple indicating the time Gabora takes to forge and
      grind a sword

    """
    def __init__(self, swords, zubora, gabora):
        length = 2 * swords
        benchmarks.Benchmark.__init__(self, 2*length) # binary + permutation
        self.swords = swords
        self.tasks = length
        self.zubora = zubora
        self.gabora = gabora
        self.bounder = ec.DiscreteBounder([0, 1])
        self.permutation_bounder = ec.DiscreteBounder([i for i in range(length)])
        self.maximize = False
        self.max_time = 2 * swords * sum(max(z,b) for z, b in zip(zubora, gabora))

    def generator(self, random, args):
        """Return a candidate solution consisting of two parts: binary and permutation."""
        schedule = [random.choice([0, 1]) for _ in range(self.tasks)]
        order = [i for i in range(self.tasks)]
        random.shuffle(order)
        return schedule + order
    
    def calculate_times(self, schedule, order):
        task_order = sorted(range(self.tasks), key=lambda x: order[x])

        # Separate the tasks performed by Zubora and Gabora
        zubora_tasks = [i for i in range(self.tasks) if schedule[i] == 0]
        gabora_tasks = [i for i in range(self.tasks) if schedule[i] == 1]

        # Order the tasks by the permutation
        zubora_tasks = [i for i in order if i in zubora_tasks]
        gabora_tasks = [i for i in order if i in gabora_tasks]

        # Calculate the start and end times for each task
        start_times = [0] * self.tasks
        end_times = [0] * self.tasks
        zubora_start = [0] * len(zubora_tasks)
        zubora_end = [0] * len(zubora_tasks)
        gabora_start = [0] * len(gabora_tasks)
        gabora_end = [0] * len(gabora_tasks)
        next_zubora = 0
        next_gabora = 0
        # Go task by task following the order
        for i in range(self.tasks):
            next_task = task_order[i]
            # Zubora task
            if next_task in zubora_tasks:
                next_zubora_start = zubora_end[next_zubora-1] if next_zubora > 0 else 0
                # If it is a grinding task, it must wait for the forging task
                if next_task % 2 == 1:
                    zubora_start[next_zubora] = max(next_zubora_start,end_times[next_task-1])
                # If it is a forging task, it can start right after the previous task
                else:
                    zubora_start[next_zubora] = next_zubora_start
                zubora_end[next_zubora] = zubora_start[next_zubora] + self.zubora[next_task%2]
                start_times[next_task] = zubora_start[next_zubora]
                end_times[next_task] = zubora_end[next_zubora]
                next_zubora += 1
            # Gabora task
            else:
                next_gabora_start = gabora_end[next_gabora-1] if next_gabora > 0 else 0
                # If it is a grinding task, it must wait for the forging task
                if next_task % 2 == 1:
                    gabora_start[next_gabora] = max(next_gabora_start,end_times[next_task-1])
                # If it is a forging task, it can start right after the previous task
                else:
                    gabora_start[next_gabora] = next_gabora_start
                gabora_end[next_gabora] = gabora_start[next_gabora] + self.gabora[next_task%2]
                start_times[next_task] = gabora_start[next_gabora]
                end_times[next_task] = gabora_end[next_gabora]
                next_gabora += 1

        return start_times, end_times

    def evaluator(self, candidates, args):
        """Return the fitness values for the given candidates."""
        fitness = []
        for candidate in candidates:
            schedule = candidate[:self.tasks]
            order = candidate[self.tasks:]
            
            # Check if some grinding is performed before forging
            num_inv_const = 0
            for i in range(self.swords):
                if order[2*i] > order[2*i+1]:
                    num_inv_const += 1

            if num_inv_const > 0:
                fitness.append(num_inv_const*self.max_time)
            else:
                _, end_times = self.calculate_times(schedule, order)
                # Calculate the makespan
                makespan = max(end_times)
                fitness.append(makespan)
        return fitness


class GAZuboraGabora:

    @classmethod
    def zg_crossover(cls, random, candidates, args):
        """Return the crossover of the given candidates."""
        tasks = args['num_tasks']
        schedules = [cand[:tasks] for cand in candidates]
        orders = [cand[tasks:] for cand in candidates]

        args_schedule = args.copy()
        args_schedule['crossover_rate'] = args['bin_crossover_rate']
        # Perform one-point crossover for the binary part
        schedule_offspring = ec.variators.n_point_crossover(random, schedules, args_schedule)

        args_order = args.copy()
        args_order['crossover_rate'] = args['per_crossover_rate']
        # Perform pmx crossover for the permutation part
        order_offspring = ec.variators.partially_matched_crossover(random, orders, args_order)

        offspring = [schedule + order for schedule, order in zip(schedule_offspring, order_offspring)]
        return offspring

    @classmethod
    def zg_mutation(cls, random, candidates, args):
        """Return the mutation of the given candidate."""
        tasks = args['num_tasks']
        schedules = [cand[:tasks] for cand in candidates]
        orders = [cand[tasks:] for cand in candidates]
        
        args_schedule = args.copy()
        args_schedule['mutation_rate'] = args['bin_mutation_rate']
        # Perform bit-flip mutation for the binary part
        schedule_offspring = ec.variators.bit_flip_mutation(random, schedules, args)
        
        args_order = args.copy()
        args_order['mutation_rate'] = args['per_mutation_rate']
        # Perform inversion mutation for the permutation part
        order_offspring = ec.variators.inversion_mutation(random, orders, args)

        offspring = [schedule + order for schedule, order in zip(schedule_offspring, order_offspring)]
        return offspring

    def __init__(self, swords, zubora, gabora, **kwargs):
        self.problem = ZuboraGabora(swords, zubora, gabora)
        self.pop_size = kwargs.get('pop_size', 10)
        self.selector = kwargs.get('selector', ec.selectors.tournament_selection)
        self.tournament_size = kwargs.get('tournament_size', 2)
        self.replacer = kwargs.get('replacer', ec.replacers.generational_replacement)
        self.num_elites = kwargs.get('num_elites', 1)
        self.terminator = kwargs.get('terminator', ec.terminators.no_improvement_termination)
        self.max_generations = kwargs.get('max_generations', 5)
        self.crossover = kwargs.get('crossover', GAZuboraGabora.zg_crossover)
        self.bin_crossover_rate = kwargs.get('bin_crossover_rate', 1)
        self.per_crossover_rate = kwargs.get('per_crossover_rate', 1)
        self.mutation = kwargs.get('mutation', GAZuboraGabora.zg_mutation)
        self.bin_mutation_rate = kwargs.get('bin_mutation_rate', 0.1)
        self.per_mutation_rate = kwargs.get('per_mutation_rate', 0.3)
        self.best_fitness_history = []
        self.solutions_history = []
        self.num_evaluations = 0
        self.num_generations = 0
    
    def _initialize(self):
        self.best_fitness_history = []
        self.solutions_history = []
        self.num_evaluations = 0
        self.num_generations = 0

    def zg_decoder(self, candidate):
        """Return the decoded candidate."""
        tasks = self.problem.tasks
        schedule = candidate[:tasks]
        order = candidate[tasks:]
        start_times, end_times = self.problem.calculate_times(schedule, order)
        assignments = []
        times = []
        for i in range(0,tasks,2):
            assignments.append({
                "F": "Zu" if schedule[i] == 0 else "Ga",
                "G": "Zu" if schedule[i+1] == 0 else "Ga"
            })
            times.append({"F": [start_times[i],end_times[i]], "G": [start_times[i+1],end_times[i+1]]})
        
        return assignments, times

    def history_observer(self, population, num_generations, num_evaluations, args):
        """Observer to track best fitness and diversity."""
        best = max(population).fitness
        self.best_fitness_history.append(best)
        self.solutions_history.append([(deepcopy(i.candidate),i.fitness) for i in population])

    def run(self, seed=None):
        rand = Random()
        if seed is not None:
            rand.seed(seed)
        self._initialize()

        ga = ec.GA(rand)
        ga.terminator = self.terminator
        ga.observer = self.history_observer
        ga.selector = self.selector
        ga.replacer = self.replacer
        ga.variator = [self.crossover, self.mutation]
        final_pop = ga.evolve(generator=self.problem.generator,
                              evaluator=self.problem.evaluator,
                              pop_size=self.pop_size,
                              maximize=self.problem.maximize,
                              bounder=self.problem.bounder,
                              max_generations=self.max_generations,
                              num_elites=self.num_elites,
                              num_selected=self.pop_size,
                              tournament_size=self.tournament_size,
                              num_tasks=self.problem.tasks,
                              bin_crossover_rate=self.bin_crossover_rate,
                              per_crossover_rate=self.per_crossover_rate,
                              bin_mutation_rate=self.bin_mutation_rate,
                              per_mutation_rate=self.per_mutation_rate)
        self.num_generations = ga.num_generations
        self.num_evaluations = ga.num_evaluations
        best = max(final_pop)
        return best.candidate, best.fitness