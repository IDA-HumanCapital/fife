"""
This script fabricates unbalanced panel data with a simple data generating process with multiple exit types.
"""

import numpy as np
import pandas as pd


def make_person(i, N_PERIODS, k=0, exit_prob=.5, exit_type_prob=None):
    df = []
    date = np.random.randint(N_PERIODS)
    x1 = np.random.uniform()  # trending variable
    x2 = np.random.choice(['A', 'B', 'C'])  # categorical variable
    x3 = np.random.normal()  # and a stationary continuous variable
    # initialize any other characteristics here
    X = [np.random.normal() for i in range(k) if k > 0]
    cols = ['ID', 'period', 'exit_type', 'X1', 'X2', 'X3'] + [''.join('X' + str(i)) for i in range(4, k+4) if k > 0]
    exit_type = 'No_exit'
    exit_prob_base = exit_prob
    while (date <= N_PERIODS):
        # print([i, date, x1, x2])
        df.append([i, date, exit_type, x1, x2, x3] + X)
        if exit_type != 'No_exit':
            break
        ### could make this part, below, its own function, or put the whole while loop in its own function
        # this part handles characteristics that change over time
        x1 += 0.1
        x3 = np.random.normal()
        if x2 == 'A':
            exit_prob = exit_prob_base + .2
            exit_type_prob = [.4, .3, .1, .1, .1]
        if x2 == 'B':
            exit_prob += .1
        else:
            x1 += 0.1
        ###
        exit_type = make_exit_type(exit_prob=exit_prob, p=exit_type_prob)
        date += 1
    df = pd.DataFrame(df, columns=cols)
    return df


def make_exit_type(df=None, exit_prob=.5, p=None):
    if p is not None:
        assert round(sum(p)) == 1
    if exit_prob > 1:
        exit_prob = 1
    elif exit_prob < 0:
        exit_prob = 0
    exit_flag = np.random.binomial(1, exit_prob)
    if exit_flag == 1:
        exit_type = np.random.choice(['A', 'B', 'C', 'D', 'E'], p=p)
    else:
        exit_type = 'No_exit'
    return exit_type


def fabricate_data(N_PERSONS = 100, N_PERIODS = 10, SEED = None, k=0, exit_prob=.5, exit_type_prob=None):
    if SEED is not None:
        np.random.seed(SEED)
    df = make_person(0, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob)
    for i in np.arange(1, N_PERSONS):
        temp = make_person(i, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob)
        df = df.append(temp)
    return df


if __name__ == '__main__':
    df = fabricate_data(N_PERSONS=10, N_PERIODS=4, SEED=1234)
    print(df['exit_type'].value_counts())
    df.to_csv('simulated_exit_data.csv', index=False)