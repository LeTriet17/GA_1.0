import time

import ga

"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
start_time = time.time()
sol_per_pop = 1
num_parents_mating = 2
# Creating the initial population.
population = ga.createParent(sol_per_pop)
pop_size = population.shape

best_outputs = []
num_generations = 1
mutation_rate = 1
fitness = ga.cal_pop_fitness(population)
print(fitness)