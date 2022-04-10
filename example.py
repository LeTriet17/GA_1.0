import time

import numpy as np
import pandas as pd
import ga

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
start_time = time.time()
sol_per_pop = 10
num_parents_mating = 4
# Creating the initial population.
population = ga.createParent(sol_per_pop)
pop_size = population.shape

best_outputs = []
num_generations = 10
mutation_rate = 1
for generation in range(num_generations):
    print("Generation : ", generation)
    # Measuring the fitness of each chromosome in the population.'

    # ================================================================
    fitness = ga.cal_pop_fitness(population)
    # print(population)
    # break
    # ================================================================
    print("Fitness")
    print(fitness)

    best_outputs.append(np.max(fitness))
    # The best result in the current iteration.
    print("Best result : ", np.max(fitness))

    # Selecting the best parents in the population for mating.
    parents = ga.select_mating_pool(population,
                                    num_parents_mating)
    # print("Parents")
    # print(parents)
    # print('Fitness Parent')
    # print(ga.cal_pop_fitness(parents))
    # Generating next generation using crossover.
    # ==============================HP=============================

    # offspring_crossover = ga.crossover(parents,
    #                                    offspring_size= parents.shape[0])
    offspring_crossover = ga.crossover(parents)
    # print('Fitness Crossover')
    # print(ga.cal_pop_fitness(offspring_crossover))
    # ==============================HP=============================
    # print("Crossover")
    # print(offspring_crossover)

    # Adding some variations to the offspring using mutation.
    # ==============================HP=============================

    # offspring_mutation = ga.mutation(offspring_crossover, mutation_rate)
    offspring_mutation = ga.mutation(population, mutation_rate)
    # print('Fitness Mutation')
    # print(ga.cal_pop_fitness(offspring_mutation))
    # ==============================HP=============================
    # print("Mutation")
    # Creating the new population based on the parents and offspring.
    pop_and_child = np.concatenate((offspring_mutation, offspring_crossover, parents,population))

    pop_and_child_fitness = ga.cal_pop_fitness(pop_and_child)
    # get n-largest element from pop_and_child
    n_largest_index = pop_and_child_fitness.argsort()[-pop_size[0]:]
    population = pop_and_child[n_largest_index]


# Getting the best solution after iterating finishing all generations.
# At first, the fitness is calculated for each solution in the final generation.
fitness = ga.cal_pop_fitness(population)
# Then return the index of that solution corresponding to the best fitness.
best_match_idx = np.argmax(fitness == np.max(fitness))
best_result = population[best_match_idx, :]
for chromosome_index in range(best_result.shape[0]):
    chromosome = best_result[chromosome_index]
    chromosome = str(chromosome).split('-')
    shift = chromosome[-1][1]
    chromosome[-1] = ga.decode_datetime(chromosome[-1][1:])
    chromosome.append(shift)
    best_result[chromosome_index] = '-'.join(chromosome)
# print("Best solution : ", best_result)
print("Best solution fitness : ", fitness[best_match_idx])
print("--- %s seconds ---" % (time.time() - start_time))
import matplotlib.pyplot

matplotlib.pyplot.plot(best_outputs)
matplotlib.pyplot.xlabel("Iteration")
matplotlib.pyplot.ylabel("Fitness")
matplotlib.pyplot.show()
# Save result to out.csv

df = pd.DataFrame(best_result)
df = df[0].str.split('-', expand=True)
df.columns = ['wonum', 'targstartdate', 'targcompdate', 'schedstartdate', 'shift']
from pathlib import Path

filepath = Path('out.csv')
filepath.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(filepath)
