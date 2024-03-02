import random
from typing import Tuple, Optional

from loguru import logger

from ga import GeneticAlgorithm

# noinspection SpellCheckingInspection
CHARSET: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ, !"
TARGET: str = "Hello, World!"

ADD_CHAR_PROB = 0.02
REMOVE_CHAR_PROB = 0.02
MUTATE_CHAR_PROB = 0.1


def generate_individual_string() -> Tuple[str, float]:
    length = random.randint(1, len(TARGET) * 2)
    genotype = "".join(random.choice(CHARSET) for _ in range(length))
    fitness = evaluate_fitness_string(genotype, TARGET)
    return genotype, fitness


def evaluate_fitness_string(genotype: str, target: Optional[str]) -> float:
    if target is None:
        raise ValueError("Target must be provided.")

    fitness = abs(len(target) - len(genotype))
    fitness += sum(
        1 for expected, actual in zip(target, genotype) if expected != actual
    )
    return fitness


def mutate_string(genotype: str, target: Optional[str]) -> str:
    if target is not None:
        if random.random() < ADD_CHAR_PROB and len(genotype) < len(target) * 2:
            pos = random.randint(0, len(genotype))
            genotype = genotype[:pos] + random.choice(CHARSET) + genotype[pos:]

        if random.random() < REMOVE_CHAR_PROB and len(genotype) > 1:
            pos = random.randint(0, len(genotype) - 1)
            genotype = genotype[:pos] + genotype[pos + 1 :]

    genotype = list(genotype)
    for i in range(len(genotype)):
        if random.random() < MUTATE_CHAR_PROB:
            genotype[i] = random.choice(CHARSET)
    return "".join(genotype)


def crossover_string(
    parent1: str, parent2: str, target: Optional[str]
) -> Tuple[str, str]:
    return parent1, parent2


def main():
    ga = GeneticAlgorithm(
        evaluate_fitness_string,
        mutate_string,
        crossover_string,
        generate_individual=generate_individual_string,
        extra_arg=TARGET,
    )
    best_individual, best_fitness = ga.run()
    logger.info(f"Best individual: {best_individual}, fitness: {best_fitness}")
    ga.plot_fitness_history()


if __name__ == "__main__":
    main()
