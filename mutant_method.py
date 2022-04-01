import copy
import numpy as np
import random
#import string_utils

def random_resetting(task):
    index = random.randrange(task.rfind('-') + 1, len(task))
    newGene, alternate = random.sample(['0', '1'], 2)
    mutate_gene = alternate \
        if newGene == task[index] \
        else newGene
    task = task[:index] + mutate_gene + task[index + 1:]
    return task
    
def swap_mutation(task):
    N = len(task)
    index = task.rfind('-') + 1

    begin = np.random.randint(index,(N + index)//2)
    end = np.random.randint((N + index)//2,N)
    
    task_new = ""
    for i,ele in enumerate(task):
        if i == begin:
            task_new += task[end]
        elif i == end:
            task_new += task[begin]
        else:
            task_new += ele
    return task_new
    
def inversion_mutation(task):
    N = len(task)
    index = task.rfind('-') + 1

    begin = np.random.randint(index,(N + index)//2)
    end = np.random.randint((N + index)//2,N)
    
    task_new = copy.deepcopy(task[:index])
    reverse_str = task[begin : end][::-1]

    #==============================
    task_new += task[index : begin]
    task_new += reverse_str
    task_new += task[end:]
    #==============================
    
    return task_new

def scramble_mutation(task):
    N = len(task)
    index = task.rfind('-') + 1

    begin = np.random.randint(index,(N + index)//2)
    end = np.random.randint((N + index)//2,N)   
    
    #string_utils.shuffle(task[begin : end])

    return task
    
    