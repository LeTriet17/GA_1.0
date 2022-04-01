from ctypes.wintypes import HDC
import numpy as np
import pandas as pd
import random
import datetime
import time
from cross_over_method import *
from mutant_method import *

# Get resource
resource_path = "resource.csv"
resource_data = pd.read_csv(resource_path)
# resource_data.columns = resource_data.columns.str.lower()
# resource_data = resource_data[['date', 'manday_ht', 'manday_mt', 'bdpocdiscipline']]
# resource_data = resource_data.rename(columns={"manday_ht": "HT", "manday_mt": "MT"})
# resource_data.date = resource_data.date.apply(lambda row: row[:-4] + "000" + row[-1])
resource_data = resource_data.loc[resource_data['bdpocdiscipline'] == 'PROD']
# print(resource_data)
date_unique = np.unique(resource_data.date.to_list()).astype(list)


def get_resource(team, date, site):  # return side resource
    if date not in date_unique:
        return -1
    return resource_data[(resource_data['bdpocdiscipline'] == team) & (resource_data['date'] == date)][site].item()


# Preprocessing WONUM_data
data_path = "data.csv"
data = pd.read_csv(data_path)

data = data.drop('Unnamed: 0', axis=1)
data = data[data.site != 'Not Defined']
data = data.loc[data['bdpocdiscipline'] == 'PROD']
data = data.reset_index()
data = data.drop('index', axis=1)
dict_wonum = {x: y for x, y in zip(data.wonum, data.index)}


# Generate random date
def _str_time_prop(start, end, time_format, prop):
    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)

    ptime = stime + prop * (etime - stime)
    ptime = ptime.strftime("%d/%m/%Y")
    return ptime


def _random_date(start, end, prop):  # 0001 = current year, 0002 = next year
    # generate date in current data
    sched_start = _str_time_prop(start, end, "%d/%m/%Y", prop)
    if (int(sched_start[:2]) != 0):
        date_sched_start = format(int(sched_start[:2]), '05b')
    else:
        date_sched_start = format(1, '05b')
    month_sched_start = format(int(sched_start[3:5]), '04b')
    year_sched_start = format(int(sched_start[6:]), '02b')
    sched_start = ''.join([date_sched_start, month_sched_start, year_sched_start])
    return sched_start


def _generate_parent():
    genes = []
    df = data
    for wonum, tarsd, tared in zip(df.wonum, df.targstartdate, df.targcompdate):
        rand_date = _random_date(tarsd, tared, random.random())
        chromosome = '-'.join([wonum, tarsd, tared, rand_date])
        genes.append(chromosome)
    return genes


# Create population function
def createParent(sol_per_pop):
    new_population = []
    for i in range(sol_per_pop):
        new_invidual = _generate_parent()
        new_population.append(new_invidual)
    new_population = np.asarray(new_population)
    return new_population


# helper function for fitness function
def decode_datetime(bit):
    date = int(bit[:5], 2)
    month = int(bit[5:-2], 2)
    year = int(bit[-2:], 2)

    return f"{date}/{month}/000{year}"


def access_row_by_wonum(wonum):
    return data.iloc[dict_wonum[wonum]]


def point_duration(duration):
    # point = 0
    # if duration > 2:  # target_start_date > 2 + start_date
    #     point += 10000
    # elif duration > 1:
    #     point += 500
    # elif duration > 0:
    #     point += 100
    # return point
    if duration > 0:
        return 1
    return 0


def convert_datetime_to_string(dt):
    # return dt.strftime("%d/%m/%Y")[:-1] + '000' + dt.strftime("%d/%m/%Y")[-1:]
    return dt.strftime("%d/%m/%Y")[:-1] + dt.strftime("%d/%m/%Y")[-1:]


# def compute_violate_child():

def manday_chromosome(chromosome):  # fitness function
    MANDAY = dict()
    HC_score = 0
    # SC_score = 0
    # violate_child = dict()

    for child in chromosome:
        # print(chromosome)
        # take bit and convert to component
        component = child.split(
            '-')  # convert: H13807098-01/12/0001-09/03/0002-10110000110 to ['H13807098', '01/12/0001', '09/03/0002', '10110000110']
        # take component
        wonum = component[0]
        target_date_begin = component[1]
        end_date_begin = component[2]
        bit_date = component[3]
        # print (target_date_begin)
        # print (end_date_begin)
        # print (bit_date)
        date_begin = decode_datetime(bit_date).split('/')
        date = int(date_begin[0])
        month = int(date_begin[1])
        year = int(date_begin[2])
        # # penalty on month and year
        # if month > 12 or month < 1 or year > 2:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue
        # # penalty on date
        # if (
        #         month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12) and date > 31:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue
        # if month == 2 and date > 28:
        #     HC_score += 1
        #     continue
        # if (month == 4 or month == 6 or month == 9 or month == 11) and date > 30:
        #     # SC_score += 1
        #     HC_score += 1
        #     continue

        date_begin = decode_datetime(bit_date)
        # access from dataframe
        est_dur = access_row_by_wonum(wonum)['r_estdur']
        site = access_row_by_wonum(wonum)['site']
        team = access_row_by_wonum(wonum)['bdpocdiscipline']

        # convert to datetime type
        try:
            dt_begin = datetime.datetime.strptime(date_begin, '%d/%m/%Y')
            dt_end = datetime.datetime.strptime(date_begin, '%d/%m/%Y') + datetime.timedelta(days=int(est_dur))
            std_begin = datetime.datetime.strptime(target_date_begin, '%d/%m/%Y')  # start target_day datetime
            etd_end = datetime.datetime.strptime(end_date_begin, '%d/%m/%Y')  # end target_day datetime

            duration_start = (std_begin - dt_begin).days
            duration_end = (dt_end - etd_end).days

            # compute violate point in every element
            if point_duration(duration_start):
                HC_score += 1
                continue
            if point_duration(duration_end):
                HC_score += 1
                continue
            # violate_child[wonum] = point

            tup = (team, convert_datetime_to_string(dt_begin), site)

            MANDAY[tup] = MANDAY.get(tup, 0) + 1
            # compute manday resource
            for i in range(1, est_dur):
                run_date = dt_begin + datetime.timedelta(days=i)
                tup_temp = (team, convert_datetime_to_string(run_date), site)

                MANDAY[tup_temp] = MANDAY.get(tup_temp, 0) + 1
        except Exception:
            # invalid days
            HC_score += 1
        # =========================ERORR==========================
        # tup = (team, convert_datetime_to_string(dt_end), site)
        # MANDAY[tup] = MANDAY.get(tup, 0) + 1
        # =========================ERORR==========================
    # print violate_child

    for key, value in MANDAY.items():
        team, date, site = key
        data_resource_value = get_resource(team, date, site)
        if data_resource_value == -1:  # gen date with date not in resouce
            HC_score += 1
        elif data_resource_value < value:
            HC_score += 1

    return HC_score
    # ,SC_score


# fitness function for caculate score for every chromosome
def cal_pop_fitness(pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = []
    for chromosome in pop:
        HC = manday_chromosome(chromosome)
        fitness.append(1 / (10 * HC + 1))
    fitness = np.array(fitness)

    return fitness


def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    np.delete(pop, index)
    # print (mating_pool)
    return random_individual


def select_mating_pool_distinct(pop, num_parents_mating):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next
    # generation.
    # first n-largest fitness
    # fitness_index = fitness.argsort()[-num_parents:]
    # tournament
    mating_pool = np.copy(pop[-num_parents_mating:])
    current_member = 1
    rand_select = 3
    visited = [False] * pop.shape[0]
    while current_member <= num_parents_mating:
        choose = True
        while choose:
            index = np.random.choice(pop.shape[0], rand_select, replace=False)
            flag = True
            for ele in index:
                if visited[ele] == True:
                    flag = False
                    break
            if flag == True:
                for ele in visited:
                    visited[ele] = True
                choose = False

        random_individual = pop[index]
        fitness = cal_pop_fitness(random_individual)
        largest_fitness_index = fitness.argsort()[-1]
        mating_pool[current_member - 1] = random_individual[largest_fitness_index]
        current_member += 1
    return mating_pool


def cross_over_on_two_chromosome(parent1, parent2):
    return


def crossover(parents):
    mating_pool = np.copy(parents)
    offsprings = []
    while mating_pool.size > 0:
        mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=False)
        mating_parents = mating_pool[mating_idx]
        parent_1 = mating_parents[0]
        parent_2 = mating_parents[1]
        swap_task_pos = random.randrange(parent_1.shape[0])
        crossover_point = random.randrange(parent_1[0].rfind('-') + 1, len(parent_1[0]) - 4)
        offspring_1 = parent_1[swap_task_pos][0:crossover_point] + parent_2[swap_task_pos][crossover_point:]
        offspring_2 = parent_2[swap_task_pos][0:crossover_point] + parent_1[swap_task_pos][crossover_point:]
        parent_1[swap_task_pos] = offspring_1
        parent_2[swap_task_pos] = offspring_2
        offsprings.append(parent_1)
        offsprings.append(parent_2)
        mating_pool = np.delete(mating_pool, list(mating_idx), axis=0)

    return np.array(offsprings)


# =============================HOANG PHU CODE==================================
def cross_over_HP(parents):
    # print ("off sprint : " + str(offspring))
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    # print ("parents : " + str(parents))
    # select random parents 
    print("Parents")
    np.random.shuffle(parents)
    parents_1 = parents[:int(parents.shape[0] / 2), :]
    parents_2 = parents[int(parents.shape[0] / 2):, :]
    print(parents_1.shape)
    print(parents_2.shape)
    vcrossover_parent_level = np.vectorize(crossover_parent_level)
    offspring1, offspring2 = vcrossover_parent_level(parents_1, parents_2)
    offspring = np.concatenate((offspring1, offspring2))
    return np.array(offspring)


def mutation_HP(offspring_crossover, random_rate):
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for offspring in offspring_crossover:
        for task in offspring:
            # print(task)
            rate = random.uniform(0, 1)
            if rate < random_rate:
                rand_option = np.random.randint(0, 8)
                if rand_option in range(0, 4):
                    task = random_resetting(task)
                elif rand_option == 5:
                    task = swap_mutation(task)
                elif rand_option == 6:
                    task = inversion_mutation(task)
                else:
                    task = scramble_mutation(task)
    return offspring_crossover


# =============================HOANG PHU CODE==================================

def mutation(population, random_rate):
    geneSet = ['0', '1']
    pop = np.copy(population)
    mutation_offspring = []
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for individual in pop:
        mutate_flag = 0
        for task in individual:
            # print(task)
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind('-') + 1, len(task))
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate \
                    if newGene == task[index] \
                    else newGene
                task = task[:index] + mutate_gene + task[index + 1:]
                if mutate_flag == 0:
                    mutate_flag = 1
        if mutate_flag:
            mutation_offspring.append(individual)

    return np.asarray(mutation_offspring)
