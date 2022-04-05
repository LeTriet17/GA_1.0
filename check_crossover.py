import ga
import numpy as np
import time
"""
Genetic algorithm parameters:
    Mating pool size
    Population size
"""
start_time = time.time()
sol_per_pop = 2
num_parents_mating = 2
# Creating the initial population.
population = ga.createParent(sol_per_pop)
pop_size = population.shape

best_outputs = []
num_generations = 1
mutation_rate = 1
ga.cal_pop_fitness(population)