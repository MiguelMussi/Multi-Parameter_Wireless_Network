#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
# !/usr/bin/env python3
import sys
import pandas as pd
import random
import time

# The maximum number of iterations, you can adjust this value to improve your scores
# but please be careful not to exceed the time limit, it is recommended that this value be less than 50.
max_loop = 50

random.seed(time.time())

def safe_print(*args):
    try:
        print(*args)
        sys.stdout.flush()
    except Exception:
        pass


def calc_cell_params_next(case_info):
    indices=[]
    for ant in range(56):
        indices.append(ant * 20 + random.randint(0,19))
    return case_info.iloc[indices].values.tolist()



def print_cell_params(cell_params):
    safe_print(len(cell_params))
    for cell_param in cell_params:
        safe_print(" ".join(str(c) for c in cell_param))


def get_start_from_server():
    return int(sys.stdin.readline().strip().split(' ')[0])


def get_result_from_server():
    k1 = sys.stdin.readline().strip().split(': ')[1].split('%')[0]
    k2 = sys.stdin.readline().strip().split(': ')[1].split('%')[0]
    k3 = sys.stdin.readline().strip().split(': ')[1].split('%')[0]
    u1 = sys.stdin.readline().strip().split(': ')[1].split('%')[0]
    u2 = sys.stdin.readline().strip().split(': ')[1].split('%')[0]
    score = sys.stdin.readline().strip().split(': ')[1]
    return map(float, [k1, k2, k3, u1, u2, score])


def get_cell_param_from_server():
    cell_param = sys.stdin.readline().strip()
    all_eng = pd.DataFrame(
        eval(cell_param), columns=['Site Name', 'Cell ID', 'Azimuth', 'Mechanical Downtilt', 'Antenna']
    )
    return all_eng


def main():
    # step1-input: wait for server's start msg
    case = get_start_from_server()

    # step1-output: send 'start'
    safe_print('start')
    # step2-input: read cell's params from case_x.csv
    all_cell_param = get_cell_param_from_server()
    # loop 50 times
    for loop in range(0, max_loop):
        # calculate cell's params use your optimization algorithm
    
        cell_params = calc_cell_params_next(all_cell_param)
        
        # step2-output: send to server by send to stdout
        print_cell_params(cell_params)

        # step3-input: get result from server by get from stdin
        try:
            r1, r2, r3, u1, u2, score = get_result_from_server()
        except:
            sys.exit(0)

    safe_print('end')


if __name__ == '__main__':
    # preventing bugs in submitted code requires exception protection
    try:
        main()
    except Exception as e:
        safe_print(e)
        sys.exit(0)