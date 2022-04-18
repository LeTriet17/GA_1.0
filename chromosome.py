import datetime
import numpy as np
import random


def _str_time_prop(start, end, time_format, prop):
    stime = datetime.datetime.strptime(start, time_format)
    etime = datetime.datetime.strptime(end, time_format)
    ptime = stime + prop * (etime - stime)
    ptime = ptime.strftime("%d/%m/%Y")
    return ptime


def _random_date(start, end, prop):  # 0001 = current year, 0002 = next year
    # generate date in current data
    sched_start = _str_time_prop(start, end, "%d/%m/%Y", prop)
    if int(sched_start[:2]) != 0:
        date_sched_start = format(int(sched_start[:2]), '05b')
    else:
        date_sched_start = format(1, '05b')
    month_sched_start = format(int(sched_start[3:5]), '04b')
    year_sched_start = format(int(sched_start[6:]), '02b')
    sched_start = ''.join([date_sched_start, month_sched_start, year_sched_start])
    return sched_start


class CHROMOSOME:
    def __init__(self, df):
        self.HC_resource = []
        self.HC_time = []
        self.df = df
        self.chromosome = self._generate_parent()
    # Generate random date

    def _generate_parent(self):
        genes = []
        for wonum, tarsd, tared in zip(self.df.wonum, self.df.targstartdate, self.df.targcompdate):
            rand_date = _random_date(tarsd, tared, random.random())
            shift = random.choice([0, 1])
            rand_date = ''.join([str(shift), rand_date])
            chromosome = '-'.join([wonum, tarsd, tared, rand_date])
            genes.append(chromosome)
        return np.asarray(genes)
