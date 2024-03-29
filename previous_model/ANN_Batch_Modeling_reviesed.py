from datetime import datetime, timedelta

import os
import argparse
    

import numpy as np
import pandas as pd
import ortools
from ortools.linear_solver import pywraplp


# 준비물
# 1. 코일 재공 데이터.csv : coil_information,
# 2. 적재 MAST.csv 

def print_status(batched_base_heights, batched_base_weights, base_heights, base_weights, base_inner, base_outer_min, base_outer_max, batched_coils_count):
    print('')
    print('')
    print('Filling rate of',data['base_number'][j], ' : ', 100*batched_base_heights/data['base_heights'][j], '%')
    
    if batched_base_heights / data['base_heights'][j] >= threshold:
        batch_complete_base.append(data['base_number'][j])

    print('')
    print('')
    print('Batched base height : ', batched_base_heights)
    print('Batched base Weight : ',batched_base_weights)
    print('')
    print('Base capacity height : ', data['base_heights'][j])
    print('Base capacity weight : ', data['base_weights'][j])
    print('Base capacity inner : ', data['base_inner'][j])
    print('Base outer range : ', data['base_outer_min'][j],' ~ ',data['base_outer_max'][j])

    print('')
    print('----> Batched_coils_count : ', batched_coils_count)


def multi_dimensional_multiple_knapsack(coil_data, base_data, threshold, bigM):
    # create model
    solver = solver = pywraplp.Solver.CreateSolver('SCIP')

    # set Parameter input params
    batch_complete_base = []
    data = {}

    # Load Coil Data
    coil_number = coil_data["COIL_NO"].tolist()
    ann_number = coil_data["PNSPRC_CD"].tolist()
    cycle = coil_data["cycle"].tolist()
    coil_heights = coil_data["WDT_LEN"].tolist()
    coil_weights = coil_data["WGT_WGT"].tolist()
    coil_inner = coil_data["IND_CD"].tolist()
    coil_outer = coil_data["OUD_LEN"].tolist()
    coil_emergency = coil_data["EMG_CD"].tolist()

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


    # Loda Base data
    base_number = base_data['BAS_NM'].tolist()
    base_weights = base_data['Weight(Ton)'].tolist()
    base_heights = base_data['Height(mm)'].tolist()
    base_outer_max = base_data['Outer_max(mm)'].tolist()
    base_outer_min = base_data['Outer_min(mm)'].tolist()
    base_inner = base_data['Inner(mm)'].tolist()

    data['base_number'] = base_number
    data['base_weights'] = base_weights
    data['base_heights'] = base_heights
    data['base_outer_max'] = base_outer_max
    data['base_outer_min'] = base_outer_min
    data['base_inner'] = base_inner

    number_base = len(base_number)
    data['bases'] = list(range(number_base))

    assert len(data['base_weights']) == number_base
    assert len(data['base_weights']) == len(data['base_heights']) == len(data['base_outer_max']) == len(data['base_outer_min']) == len(data['base_inner'])


    # set Decision Variables
    x = {}
    y = {}

    for i in data['coils']:
        for j in data['bases']:
            x[(i,j)] = solver.IntVar(0,1,'x_%i_%i' % (i, j))

    for j in data['bases']:
        y[(j)] = solver.IntVar(0,1,'x_%i_%i' % (i, j))

    # set Constraints
    for i in data['coils']:
        solver.Add(sum(x[i,j] for j in data['bases']) <= 1)

    # Weight Constraint
    # for j in data['bases']:
        

    # Height Constraint
    # for j in data['bases']:
        
    # Outer Constraint - 1
    for j in data['bases']:
        solver.Add(sum(x[(i,j)]*data['coil_weights'][i] 
                    for i in data['coils']) <= data['base_weights'][j])

        solver.Add(sum(x[(i,j)]*data['coil_heights'][i] 
                    for i in data['coils']) <= data['base_heights'][j])

        solver.Add(x[(i,j)]*data['coil_outer'][i] >= x[(i,j)]*data['base_outer_min'][j])                  
        solver.Add(x[(i,j)]*data['coil_inner'][i] == x[(i,j)]*data['base_inner'][j])

        solver.Add(y[j] <= sum(x[(i,j)] for i in data['coils']))
        solver.Add(sum(x[(i,j)] for i in data['coils']) <= bigM*y[j])

        solver.Add(data['base_heights'][j]*threshold*y[j] <= sum(x[(i,j)]*data['coil_heights'][i] 
                    for i in data['coils']))

        for i in data['coils']:
            solver.Add(x[(i,j)]*data['coil_outer'][i] <= x[(i,j)]*data['base_outer_max'][j])

    # # Outer Constraint - 2
    # for j in data['bases']:
    #     for i in data['coils']:
            
    # Inner Constraint
    # for j in data['bases']:
    #     for i in data['coils']:
            
    # Threshold for Filling rate Constraint
    # for j in data['bases']:
        # 
    # for j in data['bases']:


    # for j in data['bases']:
        

    # objective function
    objective = solver.Objective()
    for i in data['coils']:
        for j in data['bases']:
            objective.SetCoefficient(x[(i,j)], data['coil_heights'][i])
    objective.SetMaximization()


    # Solve
    start_time = datetime.now()
    
    solv = solver.Solve()

    end_time = datetime.now()

    elapsed_time = (end_time - start_time)
    print(elapsed_time)

    if solv == pywraplp.Solver.OPTIMAL:
        # print('Total Batched Heights:', objective.Value())
        # total_weight = 0
        batched_coils_count = 0
        for j in data['bases']:
            batched_base_weights = 0
            batched_base_heights = 0
            # base_outer_max= 0
            # base_outer_min = 0
            print_status()
            # print('\n', '-------------------------------', data['base_number'][j], '-------------------------------' , '\n')
            # for i in data['coils']:
            #     if x[i,j].solution_value()>0:
            #         print('coils : ', i , ' ',
            #             'coil_heights : ',data['coil_heights'][i], ' ',
            #             'coil_weights : ', data['coil_weights'][i], ' ',
            #             'coil_outer : ', data['coil_outer'][i],' ',
            #             'coil_inner : ', data['coil_inner'][i]
            #             #   'coil_emergency',data['coil_emergency'][i]
            #             )
            #         batched_base_weights += data['coil_weights'][i]
            #         batched_base_heights += data['coil_heights'][i]
            #         batched_coils_count = batched_coils_count + 1
            # print('')
            # print('')
            # print('Filling rate of',data['base_number'][j], ' : ', 100*batched_base_heights/data['base_heights'][j], '%')
            
            if batched_base_heights / data['base_heights'][j] >= threshold:
                batch_complete_base.append(data['base_number'][j])

            # print('')
            # print('')
            # print('Batched base height : ', batched_base_heights)
            # print('Batched base Weight : ',batched_base_weights)
            # print('')
            # print('Base capacity height : ', data['base_heights'][j])
            # print('Base capacity weight : ', data['base_weights'][j])
            # print('Base capacity inner : ', data['base_inner'][j])
            # print('Base outer range : ', data['base_outer_min'][j],' ~ ',data['base_outer_max'][j])
        
        # print('')
        # print('----> Batched_coils_count : ', batched_coils_count)

        #TODO : 재귀적으로 코딩하면 더 좋을듯
        possible_base_data = possible_base_data[~possible_base_data['BAS_NM'].isin(batch_complete_base)]
        # for i in range(len(batch_complete_base)):
        #     possible_base_data.drop(possible_base_data.loc[possible_base_data['BAS_NM']==batch_complete_base[i]].index)
        #     print(possible_base_data)

    else:
        print("There is no optimal solution")

    return possible_base_data


if __name__ == "__main__":
    # Coil Data
    # PEP8
    BIG_M = 10000
    DATA_DIR = './data/'

    
    # python ANN_batch_modeling.py --date 20221118 --time 080000 --interval 8 --threshold 0.9
    
    parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does')

    parser.add_argument('-d', '--date', required=True)
    parser.add_argument('-t', '--time', default='080000')
    parser.add_argument('--interval', default=8, type=int)
    parser.add_argument('--threshold', default=0.9, type=float)
 
    # namespace
    args = vars(parser.parse_args())

    
    target_time = args.time
    target_date = args.date # 2022-09-05-08:00:00
    interval = int(args.interval)
    threshold = float(args.threshold)

    # pickle => data dtype
    coil_information_df = pd.read_csv(os.path.join(DATA_DIR, "coil_information.csv"))
    coil_information_df = coil_information_df.drop("Unnamed: 0", axis=1)

    coil_groups = coil_information_df.groupby(['PNSPRC_CD','cycle'], as_index=False).agg(counts=('IND_CD', 'count'))
    coil_groups = coil_groups.sort_values('counts', ascending=False)

    print("")
    print("")
    print("[[ ANN차수, cycle, 내경에 따라 구분된 coil groups ]]")
    print("")
    print(coil_groups)
    print("")

    # Base Data
    batch_master_df = pd.read_csv(os.path.join(DATA_DIR, "ANN 적재 MAST.csv"), dtype={'COL_DT': 'str', 'COLWRK_DUR': 'int'})

    batch_master_df['COL_DT'] = pd.to_datetime(batch_master_df['COL_DT'], format="%Y-%m-%d")
    batch_master_df['COL_FIN_DT'] = batch_master_df['COL_DT'] + pd.to_timedelta(batch_master_df['COLWRK_DUR'], unit='m')

    batch_master_df = batch_master_df.dropna(subset=['COL_DT', 'COLWRK_DUR'], how='any')
    batch_master_df = batch_master_df.loc[:, ['BAS_NM','COL_DT','COLWRK_DUR','COL_FIN_DT']]


    base_capacity_info = pd.read_csv(os.path.join(DATA_DIR, "base_capacity_information.csv"))

    base_enable_info = pd.merge(batch_master_df, base_capacity_info, left_on='BAS_NM', right_on='Base_name', how='inner').drop('Base_name', axis=1)
    # base_enable_info = base_merge_info.copy()
    # Time : now, future

    now = datetime.strptime(f"{target_date} {target_time}", '%Y%m%d %H:%M:%S')

    # now = datetime(2022, 9, 5, hour=8, minute=0, second=0, microsecond=0, tzinfo = None, fold=0) # ex. 20220905080000
    reschedule_interval = timedelta(hours=interval)
    future = now + reschedule_interval
    # 대상이 될 Base 추려내기
    possible_base_data = base_enable_info[(base_enable_info['COL_FIN_DT'] >= now) & (base_enable_info['COL_FIN_DT'] <= future)]
    print(possible_base_data)




    for (pnsprc_cd, cycle, ind_cd), coil_group in coil_information_df.groupby(['PNSPRC_CD', 'cycle', 'IND_CD']):
        # coil_group = dataframe
    # for i in range(len(coil_groups.index)):
    # TODO : threshold만 넘으면 문제는 풀릴텐데 더 나은 코일배치가 뒷순서에 나올 때는 max 등으로 개선해야할까?

        # coil_group = coil_information_df.loc[
        #     (coil_information_df['PNSPRC_CD'] == coil_groups.index[0][0]) & \
        #     (coil_information_df['cycle'] == coil_groups.index[0][1]) & \
        #     (coil_information_df['IND_CD'] == coil_groups.index[0][2])]
        print(f"적재할 코일그룹 : ({len(coil_group)} 개, / PNSPRC_CD : {pnsprc_cd} / cycle : {cycle}, IND_CD : {ind_cd}")

        # print('적재할 코일그룹 : ', '  (', len(coil_group),'개)', pnsprc_cd, cycle, ind_cd)
        print('적재가능한 베이스 : ', '  (', len(possible_base_data['BAS_NM'].tolist()),'개)', possible_base_data['BAS_NM'].tolist())
        
        possible_base_data = multi_dimensional_multiple_knapsack(coil_group, possible_base_data, threshold, bigM=BIG_M)