import random
from typing import List, Tuple, Callable, TypeVar, Generic, Optional

import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

# Define a generic type variable for the individual's type
IndividualType = TypeVar("IndividualType")
ExtraArgType = TypeVar("ExtraArgType")

# Update the function types to use the generic type
GenerateIndividualType = Callable[[], Tuple[IndividualType, float]]
FitnessFunctionType = Callable[[IndividualType, Optional[ExtraArgType]], float]
MutateFunctionType = Callable[[IndividualType, Optional[ExtraArgType]], IndividualType]
# Crossover should take two parents and an optional extra argument and return two children
CrossoverFunctionType = Callable[
    [IndividualType, IndividualType, Optional[ExtraArgType]],
    Tuple[IndividualType, IndividualType],
]


class GeneticAlgorithm(Generic[IndividualType, ExtraArgType]):
    def __init__(
        self,
        fitness_function: FitnessFunctionType[IndividualType, Optional[ExtraArgType]],
        mutate_function: MutateFunctionType[IndividualType, Optional[ExtraArgType]],
        crossover_function: CrossoverFunctionType[
            IndividualType, Optional[ExtraArgType]
        ],
        generate_individual: Optional[GenerateIndividualType[IndividualType]] = None,
        initial_population: Optional[List[Tuple[IndividualType, float]]] = None,
        population_size: int = 1000,
        max_generations: int = 1000,
        early_stop_generations: int = 100,
        extra_arg: Optional[ExtraArgType] = None,
    ):
        if generate_individual is None and initial_population is None:
            raise ValueError(
                "Either generate_individual or initial_population must be provided."
            )
        if generate_individual is not None and initial_population is not None:
            raise ValueError(
                "Only one of generate_individual or initial_population can be provided."
            )
        if (
            initial_population is not None
            and len(initial_population) != population_size
        ):
            raise ValueError(
                "The length of initial_population must be equal to population_size."
            )

        self.evaluate_fitness = fitness_function
        self.mutate = mutate_function
        self.crossover = crossover_function
        self.population_size = population_size
        self.max_generations = max_generations
        self.early_stop_generations = early_stop_generations
        self.extra_arg = extra_arg
        self.fitness_history: List[float] = []
        self.best_fitness_stagnant_counter = 0
        self.best_individual: Optional[Tuple[IndividualType, float]] = None

        if generate_individual is not None:
            self.population: List[Tuple[IndividualType, float]] = [
                generate_individual() for _ in range(self.population_size)
            ]
        else:
            self.population = initial_population

    @staticmethod
    def selection(
        population: List[Tuple[IndividualType, float]],
        tournament_size: int = 5,
        num_to_select: Optional[int] = None,
    ) -> List[Tuple[IndividualType, float]]:
        if num_to_select is None:
            num_to_select = len(population)

        selected = []
        for _ in range(num_to_select):
            contenders = random.sample(population, tournament_size)
            selected.append(min(contenders, key=lambda individual: individual[1]))

        return selected

    def evolve(self, elite_size: int = 1) -> None:
        sorted_population = sorted(
            self.population, key=lambda individual: individual[1]
        )
        elites = sorted_population[:elite_size]
        selected = self.selection(sorted_population[elite_size:])
        new_population = elites[:]

        while len(new_population) < self.population_size:
            parent1, _ = random.choice(selected)
            parent2, _ = random.choice(selected)
            child1, child2 = self.crossover(parent1, parent2, self.extra_arg)
            child1 = self.mutate(child1, self.extra_arg)
            child2 = self.mutate(child2, self.extra_arg)
            new_population.append(
                (child1, self.evaluate_fitness(child1, self.extra_arg))
            )
            if len(new_population) < self.population_size:
                new_population.append(
                    (child2, self.evaluate_fitness(child2, self.extra_arg))
                )

        self.population = new_population

    def run(self):
        for generation in range(self.max_generations):
            best_individual = min(self.population, key=lambda individual: individual[1])
            self.fitness_history.append(best_individual[1])
            if (
                self.best_individual is None
                or best_individual[1] < self.best_individual[1]
            ):
                self.best_individual = best_individual

            logger.info(f"Generation {generation}, Best Fitness: {best_individual[1]}")

            if generation > 0 and self.fitness_history[-1] == self.fitness_history[-2]:
                self.best_fitness_stagnant_counter += 1
            else:
                self.best_fitness_stagnant_counter = 0

            if self.best_fitness_stagnant_counter >= self.early_stop_generations:
                logger.info(
                    f"Early stopping triggered after {generation} generations due to stagnant fitness."
                )
                break

            self.evolve()

        return self.best_individual

    def plot_fitness_history(self):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history, label="Best Fitness")
        plt.title("Fitness History Over Generations")
        plt.xlabel("Generation")
        plt.ylabel("Best Fitness")
        plt.legend()
        plt.show()
