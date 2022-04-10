import copy
import random

import numpy as np


def multipoint_cross_over(parent1,parent2):
    
    candidate1 = copy.deepcopy(parent1)
    candidate2 = copy.deepcopy(parent2)
    
    # candidate1[begin : end],candidate2[begin : end] = candidate2[begin : end],candidate1[begin : end]
    i = 0
    for task1,task2 in zip(parent1,parent2):
        index = task1.rfind('-') + 1
        N = len(task1)
        begin = np.random.randint(index,(N + index)//2)
        end = np.random.randint((N + index)//2,N)
        
        task_candidate1 = copy.deepcopy(task1[:begin])
        task_candidate1 += copy.deepcopy(task2[begin:end])
        task_candidate1 += copy.deepcopy(task1[end:])
        
        task_candidate2 = copy.deepcopy(task2[:begin])
        task_candidate2 += copy.deepcopy(task1[begin:end])
        task_candidate2 += copy.deepcopy(task2[end:])
        
        candidate1[i] = task_candidate1
        candidate2[i] = task_candidate2
        i += 1
    return candidate1,candidate2

# def halfgen(parent1,parent2):
#     candidate1 = copy.deepcopy(parent1)
#     candidate2 = copy.deepcopy(parent2)
#     print(candidate1, candidate2)
#     i = 0
#     for task1,task2 in zip(parent1,parent2):
#         print (task1, task2)
#         index = task1.rfind('-') + 1
#         task_candidate1 = ""
#         task_candidate2 = ""
#         half_index = (len(task1) + index)//2
        
#         task_candidate1 = copy.deepcopy(task1[:index])
#         task_candidate1 += copy.deepcopy(task1[index:half_index])
#         task_candidate1 += copy.deepcopy(task2[half_index:])
        
#         task_candidate2 = copy.deepcopy(task1[:index])
#         task_candidate2 += copy.deepcopy(task2[index:half_index])
#         task_candidate2 += copy.deepcopy(task1[half_index:])
        
#         candidate1[i] = task_candidate1
#         candidate2[i] = task_candidate2
#         i += 1
#     return candidate1,candidate2

def crossover_parent_level(task1, task2):
    # vhaflgen = np.vectorize(haflgen)
        # Index of the second parent to mate.
    
    # parent2_idx = (k + 1) % parents.shape[0]
    # parent1 = parents[parent1_idx]
    # parent2 = parents[parent2_idx]
        # rand_option = np.random.randint(0,2)
        # if rand_option == 0:
        #     candidate1,candidate2 = multipoint_cross_over(parent1,parent2)
        # else:
        #     candidate1,candidate2 = halfgen(parent1,parent2)
    # candidate1,candidate2 = halfgen(parent1,parent2)
    index = task1.rfind('-') + 1
    task_candidate1 = ""
    task_candidate2 = ""
    half_index = random.randint(index, len(task1))
    # half_index = (len(task1) + index)//2
    task_candidate1 = copy.deepcopy(task1[:index])
    task_candidate1 += copy.deepcopy(task1[index:half_index])
    task_candidate1 += copy.deepcopy(task2[half_index:])
    task_candidate2 = copy.deepcopy(task1[:index])
    task_candidate2 += copy.deepcopy(task2[index:half_index])
    task_candidate2 += copy.deepcopy(task1[half_index:])
    # candidate1[i] = task_candidate1
    # candidate2[i] = task_candidate2
    # offspring.append(candidate1)
    # offspring.append(candidate2)
    return task_candidate1, task_candidate2