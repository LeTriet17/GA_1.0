from pathlib import Path

import numpy as np
import pandas as pd


# folder_name = "population"

# pathlist = Path(folder_name).rglob('*.csv')
# filename = []
# for path in pathlist:
#     path_in_str = str(path)
#     filename.append(path_in_str)

# print(filename)
class RETRAIN_MODULE(object):
    def __init__(self,population_folder):
        self.population_folder = population_folder
        self.iterative_folder()
    
    def iterative_folder(self):
        pathlist = Path(self.population_folder).rglob('*.csv')
        filename = []
        for path in pathlist:
            path_in_str = str(path)
            filename.append(path_in_str)
        self.filename = filename
        
    def add_chromosome(self,chromosome : list) -> None:
        df = pd.DataFrame(chromosome)
        df = df[0].str.split('-', expand=True)
        df.columns = ['wonum', 'targstartdate', 'targcompdate', 'schedstartdate']
        filepath = self.population_folder + '\\' + 'chromosome_{}.csv'.format(len(self.filename))
        self.filename.append(filepath)
        df.to_csv(filepath)
    
    #for population
    def load_population(self,filepath : str):
        pop = []
        for f_name in self.filename:
            pop.append(self.load_chromosome(f_name))
        self.pop = pop
        return pop
    
    def delete_wonum_population(self,wonum):
        for f_name in self.filename:
            self.delete_chromosome(f_name,wonum)
    
    def update_wonum_population(self,wonum,wonum_info):
        for f_name in self.filename:
            self.update_chromosome(f_name,wonum,wonum,wonum_info)
    
    #for one chromosome
    def load_chromosome(self,filepath : str):
        df = pd.read_csv(filepath)
        df = df.drop('Unnamed: 0', 1)
        
        chromosome = []
        
        for i,line in df.iterrows():
            line = dict(line)
            temp_list = list(line.values())
            
            str_line = '-'.join(temp_list)
            chromosome.append(str_line)
        
        chromosome = np.array(chromosome)
        return chromosome
    
    def delete_chromosome(self,filepath,wonum):
        if filepath in self.filename:
            df = pd.read_csv(filepath)
            df = df.drop('Unnamed: 0', 1)
            df = df.set_index('wonum')
            df = df.drop(wonum)
            df = df.reset_index()
            df.to_csv(filepath)
        else:
            print("file {} not exist".format(filepath))
    
    def update_chromosome(self,filepath,wonum,wonum_info : list):
        if filepath in self.filename:
            #wonum_info = [wonum, targstartdate, targcompdate, schedstartdate]
            df = pd.read_csv(filepath)
            df = df.drop('Unnamed: 0', 1)
            df = df.set_index('wonum')
            
            cols = ['targstartdate', 'targcompdate', 'schedstartdate']
            
            for col,info in zip(cols,wonum_info[1:]):
                df.at[wonum,col] = info
            
            print(df.head())
            df = df.reset_index()
            df.to_csv(filepath)
        else:
            print("file {} not exist".format(filepath))
        
if __name__ == "__main__":
    a = RETRAIN_MODULE("population")
    wonum_info = ['H13825208','01/03/0003','30/04/0003','00001010010']
    a.update_chromosome("population\\chromosome_0.csv",'H13825208',wonum_info)
    
        