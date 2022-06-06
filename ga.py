import datetime
import copy
import pandas as pd

from cross_over_method import *
from mutant_method import *
from chromosome import *

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


def _cal_end_date(date_begin, shift, est_dur):
    after_shift = shift == 1 or int(est_dur) != est_dur
    if shift == 1 and int(est_dur) != est_dur:
        after_shift = 0
        est_dur += 1
    dt_end = datetime.datetime.strptime(date_begin, '%d/%m/%Y') + datetime.timedelta(days=int(est_dur))

    return dt_end, after_shift


# Preprocessing WONUM_data
data_path = "data.csv"
data = pd.read_csv(data_path)

data = data.drop('Unnamed: 0', axis=1)
data = data[data.site != 'Not Defined']
data = data.loc[data['bdpocdiscipline'] == 'PROD']
data = data.reset_index()
data = data.drop('index', axis=1)
dict_wonum = {x: y for x, y in zip(data.wonum, data.index)}


# Create population function
def createPop(sol_per_pop):
    new_population = []
    for i in range(sol_per_pop):
        new_individual = CHROMOSOME(data)
        new_population.append(new_individual)
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
    if duration > 0:
        return 1
    return 0


def convert_datetime_to_string(dt):
    # return dt.strftime("%d/%m/%Y")[:-1] + '000' + dt.strftime("%d/%m/%Y")[-1:]
    return dt.strftime("%d/%m/%Y")[:-1] + dt.strftime("%d/%m/%Y")[-1:]


# def compute_violate_child():

def manday_chromosome(chromosome, error_output=False):  # fitness function
    MANDAY = dict()
    HC_time = 0
    HC_resource = 0
    # SC_score = 0
    # violate_child = dict()

    for task in chromosome.chromosome:

        component = task.split(
            '-')  # convert: H13807098-01/12/0001-09/03/0002-101100001101 to ['H13807098', '01/12/0001', '09/03/0002', '101100001101']
        # take component
        wonum = component[0]
        target_date_begin = component[1]
        end_date_begin = component[2]
        bit_date = component[3]
        # shift = int(bit_date[1])
        shift = int(bit_date[0])
        date_begin = decode_datetime(bit_date[1:])
        # access from dataframe
        est_dur = access_row_by_wonum(wonum)['r_estdur']
        site = access_row_by_wonum(wonum)['site']
        team = access_row_by_wonum(wonum)['bdpocdiscipline']

        # convert to datetime type
        try:
            dt_begin = datetime.datetime.strptime(date_begin, '%d/%m/%Y')
            dt_end, shift_end = _cal_end_date(date_begin, shift, est_dur)
            std_begin = datetime.datetime.strptime(target_date_begin, '%d/%m/%Y')  # start target_day datetime
            etd_end = datetime.datetime.strptime(end_date_begin, '%d/%m/%Y')  # end target_day datetime
            duration_start = (std_begin - dt_begin).days
            duration_end = (dt_end - etd_end).days
            # compute violate point in every element
            if point_duration(duration_start) or point_duration(duration_end):
                if wonum not in chromosome.HC_time:
                    HC_time += 1
                if error_output:
                    chromosome.HC_time.append(wonum)
                continue

            tup = (team, convert_datetime_to_string(dt_begin), shift, site)

            MANDAY[tup] = MANDAY.get(tup, 0) + 1

            # compute manday resource
            for i in np.arange(0, est_dur, 0.5):
                run_date, run_shift = _cal_end_date(date_begin, shift, i)
                tup_temp = (team, convert_datetime_to_string(run_date), shift, site)
                MANDAY[tup_temp] = MANDAY.get(tup_temp, 0) + 1
        except Exception:
            # invalid days
            if wonum not in chromosome.HC_time:
                HC_time += 1
            if error_output:
                chromosome.HC_time.append(wonum)

    for key, value in MANDAY.items():
        team, date, shift, site = key
        date = date[:len(date) - 1] + '000' + date[-1]
        data_resource_value = get_resource(team, date, site)

        if data_resource_value == -1 or data_resource_value < value:
            if date not in chromosome.HC_resource:  # gen date with date not in resouce
                HC_resource += 1
            if error_output:
                chromosome.HC_resource.append(date)

    return HC_time, HC_resource
    # ,SC_score


# fitness function for caculate score for every chromosome
def cal_pop_fitness(pop):
    # Calculating the fitness value of each solution in the current population.
    # The fitness function calulates the sum of products between each input and its corresponding weight.
    fitness = []
    for chromosome in pop:
        HC_time, HC_rs = manday_chromosome(chromosome)
        fitness.append(1 / (10 * (HC_time + HC_rs) + 1))
    fitness = np.array(fitness)

    return fitness


def select_mating_pool(pop, num_parents_mating):
    # shuffling the pop then select top of pops
    index = np.random.choice(pop.shape[0], num_parents_mating, replace=False)
    random_individual = pop[index]
    pop = np.delete(pop, index)  # split current pop into remain_pop and mating_pool
    # print (mating_pool)
    return random_individual


def crossover(parents):
    mating_pool = copy.deepcopy(parents)
    offsprings = []
    while mating_pool.size > 0:
        mating_idx = np.random.choice(mating_pool.shape[0], 2, replace=False)
        mating_parents = mating_pool[mating_idx]
        parent_1 = mating_parents[0]
        parent_2 = mating_parents[1]
        swap_task_pos = random.randrange(parent_1.chromosome.shape[0])
        crossover_point = random.sample(range(parent_1.chromosome[0].rfind('-') + 1, len(parent_1.chromosome[0]) - 4),
                                        2)
        crossover_point.sort()
        offspring_1 = parent_1.chromosome[swap_task_pos][0:crossover_point[0]] + parent_2.chromosome[swap_task_pos][
                                                                                 crossover_point[0]:crossover_point[
                                                                                     1]] + parent_1.chromosome[
                                                                                               swap_task_pos][
                                                                                           crossover_point[
                                                                                               1]:]
        offspring_2 = parent_2.chromosome[swap_task_pos][0:crossover_point[0]] + parent_1.chromosome[swap_task_pos][
                                                                                 crossover_point[0]:crossover_point[
                                                                                     1]] + parent_2.chromosome[
                                                                                               swap_task_pos][
                                                                                           crossover_point[
                                                                                               1]:]
        parent_1.chromosome[swap_task_pos] = offspring_1
        parent_2.chromosome[swap_task_pos] = offspring_2
        parent_1.HC_time = []
        parent_1.HC_resource = []
        parent_2.HC_time = []
        parent_2.HC_resource = []
        offsprings.append(parent_1)
        offsprings.append(parent_2)
        mating_pool = np.delete(mating_pool, list(mating_idx), axis=0)

    return np.array(offsprings)


def mutation(population, random_rate):
    geneSet = ['0', '1']
    pop = copy.deepcopy(population)

    mutation_offspring = []
    # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
    for chromosome in pop:
        mutate_flag = 0
        for task in chromosome.chromosome:
            # print(task)
            rate = random.uniform(0, 1)
            if rate < random_rate:
                index = random.randrange(task.rfind('-') + 1, len(task) - 4)
                newGene, alternate = random.sample(geneSet, 2)
                mutate_gene = alternate \
                    if newGene == task[index] \
                    else newGene
                task = task[:index] + mutate_gene + task[index + 1:]
                if mutate_flag == 0:
                    mutate_flag = 1
        if mutate_flag:
            chromosome.HC_time = []
            chromosome.HC_resource = []
            mutation_offspring.append(chromosome)

    return np.asarray(mutation_offspring)
