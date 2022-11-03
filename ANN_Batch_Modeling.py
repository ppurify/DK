import pandas as pd
import numpy as np
import ortools
from ortools.linear_solver import pywraplp


# data_df = pd.read_csv("./data/data_df.csv")
# data_df.drop("Unnamed: 0", axis=1, inplace=True)
# coil_df = data_df.copy()
#data_df = data_df.astype({"WDT_LEN" : "str", "WGT_WGT" : "str", "IND_CD" : "str", "OUD_LEN" : "str", "STL_CD" : "str"})

# coil_number = first_coil_group["COIL_NO"].tolist()
# ann_number = first_coil_group["PNSPRC_CD"].tolist()
# cycle = first_coil_group["cycle"].tolist()
# coil_heights = first_coil_group["WDT_LEN"].tolist()
# coil_weights = first_coil_group["WGT_WGT"].tolist()
# coil_inner = first_coil_group["IND_CD"].tolist()
# coil_outer = first_coil_group["OUD_LEN"].tolist()
# coil_emergency = first_coil_group["EMG_CD"].tolist()
# Coil Data



data = {}

# Coil
coil_information = pd.read_csv("./data/coil_information.csv")
coil_information_df = coil_information.copy()
coil_information_df.drop("Unnamed: 0", axis=1, inplace=True)

# print("coil_information_df dtypes is ", coil_information_df.dtypes)

# 첫번째 코일 그룹
# cycle의 type : object
separated_df = coil_information_df.loc[(coil_information_df['PNSPRC_CD'] == 'AN11') & (coil_information_df['cycle'] == '620')]
first_coil_group = separated_df.loc[separated_df['IND_CD'] == 508]

coil_number = first_coil_group["COIL_NO"].tolist()
ann_number = first_coil_group["PNSPRC_CD"].tolist()
cycle = first_coil_group["cycle"].tolist()
coil_heights = first_coil_group["WDT_LEN"].tolist()
coil_weights = first_coil_group["WGT_WGT"].tolist()
coil_inner = first_coil_group["IND_CD"].tolist()
coil_outer = first_coil_group["OUD_LEN"].tolist()
coil_emergency = first_coil_group["EMG_CD"].tolist()


# Base Data
base_capacity_information = pd.read_csv("./data/base_capacity_information.csv")
first_base_group = base_capacity_information.loc[(base_capacity_information['Maker'] == 'EBNER') & (base_capacity_information['Base_number'] <= 6)]


# 첫번째 베이스 그룹
base_weight_capa = first_base_group["Weight(Ton)"].tolist()
base_height_capa = first_base_group["Height(mm)"].tolist()
base_min_od_capa = first_base_group["Outer_max(mm)"].tolist()
base_max_od_capa = first_base_group["Outer_min(mm)"].tolist()
base_id_capa = first_base_group["Inner(mm)"].tolist()

kind_of_ann = coil_information_df["PNSPRC_CD"].unique().tolist()
kind_of_cycle = coil_information_df["cycle"].unique().tolist()
solver = solver = pywraplp.Solver.CreateSolver('SCIP')


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

# base_maker = base_df['Maker'].tolist()
# base_number = base_df['Base_number'].tolist()
base_weights = first_base_group['Weight(Ton)'].tolist()
base_heights = first_base_group['Height(mm)'].tolist()
base_outer_max = first_base_group['Outer_max(mm)'].tolist()
base_outer_min = first_base_group['Outer_min(mm)'].tolist()
base_inner = first_base_group['Inner(mm)'].tolist()


# data['base_maker'] = base_maker
# data['base_number'] = base_number
data['base_weights'] = base_weights
data['base_heights'] = base_heights
data['base_outer_max'] = base_outer_max
data['base_outer_min'] = base_outer_min
data['base_inner'] = base_inner

number_base = len(base_weights)

#I wanted to change the values at a later data
data['bases'] = list(range(number_base))

assert len(data['base_weights']) == number_base
assert len(data['base_weights']) == len(data['base_heights']) == len(data['base_outer_max']) == len(data['base_outer_min']) == len(data['base_inner'])

# print("coil_number: ",data['coil_number'])
# print('coil_heights:',data['coil_heights'][:5])
# print('coil_weights:',data['coil_weights'][:5])
# print('coil_inner:', data['coil_inner'][:5])
# print('coil_outer:', data['coil_outer'][:5])
# print("Number of coils:", len(data['coil_number']))
# print("Number of Bases:" , number_base)
# print('Knapsack Capacities: 50 Pounds, 50 cubic inches, 5 Levels of Radiation')


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

# TODO : 외경조건은 sum이 아니므로 그 문법에 맞게 코드 수정
for j in data['bases']:
    for i in data['coils']:
        solver.Add(x[(i,j)]*data['coil_outer'][i] <= data['base_outer_max'][j])


for j in data['bases']:
    for i in data['coils']:
        solver.Add(x[(i,j)]*data['coil_outer'][i] >= data['base_outer_min'][j])



# objective function
objective = solver.Objective()
for i in data['coils']:
    for j in data['bases']:
        objective.SetCoefficient(x[(i,j)], data['coil_heights'][i])
objective.SetMaximization()


# Solve
solv = solver.Solve()
if solv == pywraplp.Solver.OPTIMAL:
    print('Total Batched Heights:', objective.Value())
    total_weight = 0
    for j in data['bases']:
        base_weights = 0
        base_heights = 0
        base_outer_max= 0
        base_outer_min = 0
        print('\n','Base', j+1 , '\n')
        for i in data['coils']:
            if x[i,j].solution_value()>0:
                print('coils:', i , 
                      'coil_heights',data['coil_heights'][i],
                      'coil_weights', data['coil_weights'][i],
                    #   'coil_inner', data['coil_inner'][i],
                      'coil_outer', data['coil_outer'][i],
                    #   'coil_emergency',data['coil_emergency'][i]
                     )
                base_weights += data['base_weights'][i]
                base_heights += data['base_heights'][i]
        print('Packed Knapsack Value: ', base_weights)
        print('Packed Knapsack Weight: ', base_heights)
else:
    print("There is no optimal solution")