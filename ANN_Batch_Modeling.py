import pandas as pd
import numpy as np
import ortools
from ortools.linear_solver import pywraplp


solver = solver = pywraplp.Solver.CreateSolver('SCIP')

data_df = pd.read_csv("./data/data_df.csv")
data_df.drop("Unnamed: 0", axis=1, inplace=True)
coil_df = data_df.copy()
#data_df = data_df.astype({"WDT_LEN" : "str", "WGT_WGT" : "str", "IND_CD" : "str", "OUD_LEN" : "str", "STL_CD" : "str"})

coil_number = coil_df["COIL_NO"].tolist()
ann_number = coil_df["PNSPRC_CD"].tolist()
cycle = coil_df["cycle"].tolist()
coil_heights = coil_df["WDT_LEN"].tolist()
coil_weights = coil_df["WGT_WGT"].tolist()
coil_inner = coil_df["IND_CD"].tolist()
coil_outer = coil_df["OUD_LEN"].tolist()
coil_emergency = coil_df["EMG_CD"].tolist()


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

# Base
base_info = pd.read_csv("./data/base_capacity_information.csv")
base_df = base_info.copy()

base_maker = base_df['Maker'].tolist()
base_number = base_df['Base_number'].tolist()
base_weights = base_df['Weight(Ton)'].tolist()
base_heights = base_df['Height(mm)'].tolist()
base_outer_max = base_df['Outer_max'].tolist()
base_outer_min = base_df['Outer_min'].tolist()
base_inner = base_df['Inner'].tolist()


data['base_maker'] = base_maker
data['base_number'] = base_number
data['base_weights'] = base_weights
data['base_heights'] = base_heights
data['base_outer_max'] = base_outer_max
data['base_outer_min'] = base_outer_min
data['base_inner'] = base_inner

number_base = len(base_number)

#I wanted to change the values at a later data
data['bases'] = list(range(number_base))

assert len(data['base_maker']) == len(data['base_number']) == len(data['base_weight']) == len(data['base_height']) == len(data['base_outer_max']) == len(data['base_outer_min']) == len(data['base_inner'])

print("coil_number: ",*data['coil_number'][:5])
print('coil_heights:',*data['coil_heights'][:5])
print('coil_weights:',*data['coil_weights'][:5])
print('coil_inner:', *data['coil_inner'][:5])
print('coil_outer:', *data['coil_outer'][:5])
print("Number of coils:", data['num_coils'][:5])
print("Number of Bases:" , number_base)
print('Knapsack Capacities: 50 Pounds, 50 cubic inches, 5 Levels of Radiation')


x = {}
for i in data['coils']:
    for j in data['bases']:
        x[(i,j)] = solver.IntVar(0,1,'x_%i_%i' % (i, j))


#Constraints
for i in data['coils']:
    solver.Add(sum(x[i,j] for j in data['bases'])<=1)

for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_weights'][i] 
                  for i in data['coils']) <= data['base_weights'][j])


for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_heights'][i] 
                  for i in data['coils']) <= data['base_heights'][j])

for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_outer'][i] 
                  for i in data['coils']) <= data['base_outer_max'][j])

for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_outer'][i] 
                  for i in data['coils']) >= data['base_outer_min'][j])