import random
from typing import Optional

import pytest

from ga import GeneticAlgorithm  # Adjust the import as necessary

TARGET = "HELLO"


def generate_individual():
    import random

    return "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=len(TARGET))), 0


def evaluate_fitness(individual: str, target: Optional[str]):
    if target is None:
        raise ValueError("Target must be provided.")
    return sum(1 for a, b in zip(individual, target) if a != b)


def mutate(individual: str, target: Optional[str]):
    pos = random.randint(0, len(individual) - 1)
    char = random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return individual[:pos] + char + individual[pos + 1 :]


def crossover(parent1: str, parent2: str, target: Optional[str]):
    pos = random.randint(1, len(parent1) - 1)
    return parent1[:pos] + parent2[pos:], parent2[:pos] + parent1[pos:]


@pytest.fixture
def ga():
    return GeneticAlgorithm(
        generate_individual=generate_individual,
        fitness_function=evaluate_fitness,
        mutate_function=mutate,
        crossover_function=crossover,
        population_size=10,
        max_generations=100,
        early_stop_generations=10,
        extra_arg=TARGET,
    )


def test_initial_population(ga):
    assert len(ga.population) == 10, "Population size should be 10."


def test_fitness_evaluation():
    individual = "HELLO"
    fitness = evaluate_fitness(individual, TARGET)
    assert fitness == 0, "Fitness of the target should be 0."


def test_mutation():
    individual = "HELLO"
    mutated = mutate(individual, TARGET)
    assert individual != mutated, "Mutated individual should be different."


def test_crossover():
    parent1 = "AAAAA"
    parent2 = "BBBBB"
    child1, child2 = crossover(parent1, parent2, TARGET)
    assert child1 != parent1, "Child should be different from parent1."
    assert child2 != parent2, "Child should be different from parent2."


def test_early_stopping(ga):
    ga.run()
    assert (
        ga.best_fitness_stagnant_counter == ga.early_stop_generations
        or evaluate_fitness(ga.population[0][0], TARGET) == 0
    ), "Algorithm should stop early due to stagnant fitness or finding a solution."
