import pandas as pd
import numpy as np
import ortools
from ortools.linear_solver import pywraplp


solver = solver = pywraplp.Solver.CreateSolver('SCIP')

df = pd.read_csv("./data/data_df.csv")
df.drop("Unnamed: 0", axis=1, inplace=True)
data_df = df.copy()
data_df = data_df.astype({"WDT_LEN" : "str", "WGT_WGT" : "str", "IND_CD" : "str", "OUD_LEN" : "str", "STL_CD" : "str"})

coil_number = data_df["COIL_NO"].tolist()
ann_number = data_df["PNSPRC_CD"].tolist()
cycle = data_df["cycle"].tolist()
coil_heights = data_df["WDT_LEN"].tolist()
coil_weights = data_df["WGT_WGT"].tolist()
coil_inner = data_df["IND_CD"].tolist()
coil_outer = data_df["OUD_LEN"].tolist()
coil_emergency = data_df["EMG_CD"].tolist()


data = {}

assert len(coil_number) == len(ann_number) == len(cycle) == len(coil_heights) == len(coil_weights) == len(coil_inner) == len(coil_outer) == len(coil_emergency) 
data['coil_number'] = coil_number
data['ann_number'] = ann_number
data['cycle'] = cycle
data['coil_heights'] = coil_heights
data['coil_weights'] = coil_weights
data['coil_inner'] = coil_inner
data['coil_outer'] = coil_outer
data['coil_emergency'] = coil_emergency


data['coils'] = list(range(len(coil_weights)))
data['num_coils'] = len(coil_number)

number_bags = 5 #All have the same capacity of 50 pounds

data['bag_capacities'] = [50, 50, 50, 50, 50] #pounds
data['bag_volume'] = [50,50,50,50,50] #while this equals bag_capacities, I made it its own variable in case
data['rad_capacities'] = [5,5,5,5,5]
#I wanted to change the values at a later data
data['bags'] = list(range(number_bags))
assert len(data['bag_capacities']) == number_bags
assert len(data['bag_capacities']) == len(data['bag_volume']) == len(data['rad_capacities'])

print("coil_number: ",*data['coil_number'])
print('coil_heights:',*data['coil_heights'])
print('coil_weights:',*data['coil_weights'])
print('coil_inner:', *data['coil_inner'])
print('coil_outer:', *data['coil_outer'])
print("Number of coils:", data['num_coils'])
print("Number of Knapsacks:" , number_bags)
print('Knapsack Capacities: 50 Pounds, 50 cubic inches, 5 Levels of Radiation')