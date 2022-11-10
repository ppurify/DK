import pandas as pd
import numpy as np
import ortools
from ortools.linear_solver import pywraplp



# create model
solver = solver = pywraplp.Solver.CreateSolver('SCIP')

# set Parameter
bigM = 10000
data = {}

# Coil Data
coil_information = pd.read_csv("./data/coil_information.csv")
coil_information_df = coil_information.copy()
coil_information_df.drop("Unnamed: 0", axis=1, inplace=True)
print(coil_information_df.groupby(['PNSPRC_CD','cycle'])['IND_CD'].value_counts())


# 첫번째(임시) 코일 그룹
# cycle의 type : object
first_coil_group = coil_information_df.loc[(coil_information_df['PNSPRC_CD'] == 'AN11') & (coil_information_df['cycle'] == '725')]

coil_number = first_coil_group["COIL_NO"].tolist()
ann_number = first_coil_group["PNSPRC_CD"].tolist()
cycle = first_coil_group["cycle"].tolist()
coil_heights = first_coil_group["WDT_LEN"].tolist()
coil_weights = first_coil_group["WGT_WGT"].tolist()
coil_inner = first_coil_group["IND_CD"].tolist()
coil_outer = first_coil_group["OUD_LEN"].tolist()
coil_emergency = first_coil_group["EMG_CD"].tolist()

# kind_of_ann = coil_information_df["PNSPRC_CD"].unique().tolist()
# kind_of_cycle = coil_information_df["cycle"].unique().tolist()

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
print("Total coil count is ", data['num_coils'])


# Base Data
base_capacity_information = pd.read_csv("./data/base_capacity_information.csv")


# 첫번째(임시) 베이스 그룹
first_base_group = base_capacity_information.loc[(base_capacity_information['Maker'] == 'EBNER') & (base_capacity_information['Base_number'] >= 27) &(base_capacity_information['Base_number'] <=29)]

base_weights = first_base_group['Weight(Ton)'].tolist()
base_heights = first_base_group['Height(mm)'].tolist()
base_outer_max = first_base_group['Outer_max(mm)'].tolist()
base_outer_min = first_base_group['Outer_min(mm)'].tolist()
base_inner = first_base_group['Inner(mm)'].tolist()
# print("First Base Gruop Inner Unique : ", first_base_group['Inner(mm)'].unique())

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




# set Decision Variables
x = {}
for i in data['coils']:
    for j in data['bases']:
        x[(i,j)] = solver.IntVar(0,1,'x_%i_%i' % (i, j))

y = {}
for j in data['bases']:
    y[(j)] = solver.IntVar(0,1,'x_%i_%i' % (i, j))





# Constraints
for i in data['coils']:
    solver.Add(sum(x[i,j] for j in data['bases'])<=1)


# Weight Constraint
for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_weights'][i] 
                  for i in data['coils']) <= data['base_weights'][j])


# Height Constraint
for j in data['bases']:
    solver.Add(sum(x[(i,j)]*data['coil_heights'][i] 
                  for i in data['coils']) <= data['base_heights'][j])

# Outer Constraint - 1
for j in data['bases']:
    for i in data['coils']:
        solver.Add(x[(i,j)]*data['coil_outer'][i] <= x[(i,j)]*data['base_outer_max'][j])

# Outer Constraint - 2
for j in data['bases']:
    for i in data['coils']:
        solver.Add(x[(i,j)]*data['coil_outer'][i] >= x[(i,j)]*data['base_outer_min'][j])                  

# Inner Constraint
for j in data['bases']:
    for i in data['coils']:
        solver.Add(x[(i,j)]*data['coil_inner'][i] == x[(i,j)]*data['base_inner'][j])




for j in data['bases']:
    solver.Add(y[j] <= sum(x[(i,j)] for i in data['coils']))

for j in data['bases']:
    solver.Add(sum(x[(i,j)] for i in data['coils']) <= bigM*y[j])

for j in data['bases']:
    solver.Add(data['base_heights'][j]*0.9*y[j] <= sum(x[(i,j)]*data['coil_heights'][i] 
                  for i in data['coils']))





# print(solver.ExportModelAsLpFormat(False).replace('\\', '').replace(',_', ','), sep='\n')

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
    used_coils_count = 0
    for j in data['bases']:
        base_weights = 0
        base_heights = 0
        base_outer_max= 0
        base_outer_min = 0
        print('\n','Base', j+1 , '\n')
        for i in data['coils']:
            if x[i,j].solution_value()>0:
                print('coils : ', i , 
                      'coil_heights : ',data['coil_heights'][i],
                      'coil_weights : ', data['coil_weights'][i],
                      'coil_outer : ', data['coil_outer'][i],
                      'coil_inner : ', data['coil_inner'][i]
                    #   'coil_emergency',data['coil_emergency'][i]
                     )
                base_weights += data['coil_weights'][i]
                base_heights += data['coil_heights'][i]
                used_coils_count = used_coils_count + 1

        print('Packed base height: ', base_heights)
        print('Packed base Weight: ',base_weights)
        print('Base', j+1, 's height', data['base_heights'][j])
    
    print('Used_coils_count : ', used_coils_count)
else:
    print("There is no optimal solution")