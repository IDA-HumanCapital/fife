"""
This script fabricates unbalanced panel data with a simple data generating process with multiple exit types.
"""

import numpy as np
import pandas as pd


def update_person(x1, x2, x3, exit_prob=None, exit_type_prob=None, dgp=1):
    x2 = np.random.normal()
    if dgp == 1:
        # x3 += 0.1
        if x1 == "A":
            # exit_prob = exit_prob_base + 0.2
            exit_type_prob = [0.7, 0.2, 0.1]
        if x1 == "B":
            # exit_prob += 0.1
            exit_type_prob = [0.2, 0.7, 0.1]
        if x1 == "C":
            exit_type_prob = [0.1, 0.2, 0.7]
            # x3 += 0.1
    elif dgp == 2:
        if x1 == "A":
            exit_type_prob = [0.7, 0.2, 0.1]
    else:
        raise NameError('Invalid DGP')
    return x1, x2, x3, exit_prob, exit_type_prob


def make_person(i, N_PERIODS, k=0, exit_prob=0.5, exit_type_prob=None, dgp=1):
    df = []
    date = np.random.randint(N_PERIODS)
    x1 = np.random.choice(["A", "B", "C"])  # categorical variable
    x2 = np.random.normal()  # and a stationary continuous variable
    x3 = np.random.uniform()  # numerical variable fixed per i
    # initialize any other characteristics here
    Z = [np.random.normal() for i in range(k) if k > 0]
    cols = ["ID", "period", "X1", "X2", "X3"] + [
        "".join("X" + str(i)) for i in range(4, k + 4) if k > 0
    ]
    exit_type = "No_exit"
    exit_prob_base = exit_prob
    while date <= N_PERIODS:
        # print([i, date, x3, x1])
        df.append([i, date, x1, x2, x3] + Z)
        if exit_type != "No_exit":
            break
        # this part handles characteristics that change over time
        x1, x2, x3, exit_prob, exit_type_prob = update_person(x1, x2, x3, exit_prob=exit_prob,
                                                              exit_type_prob=exit_type_prob, dgp=dgp)
        exit_type = make_exit_type(exit_prob=exit_prob, p=exit_type_prob)
        date += 1
    df = pd.DataFrame(df, columns=cols)
    df["exit_type"] = exit_type
    return df


def make_exit_type(exit_prob=0.5, p=None):
    if p is not None:
        assert round(sum(p)) == 1
    if exit_prob > 1:
        exit_prob = 1
    elif exit_prob < 0:
        exit_prob = 0
    exit_flag = np.random.binomial(1, exit_prob)
    if exit_flag == 1:
        exit_type = np.random.choice(["X", "Y", "Z"], p=p)
    else:
        exit_type = "No_exit"
    return exit_type


def fabricate_data(
        N_PERSONS=100, N_PERIODS=10, SEED=None, k=0, exit_prob=0.5, exit_type_prob=None, dgp=1
):
    if SEED is not None:
        np.random.seed(SEED)
    df = make_person(
        0, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob, dgp=dgp
    )
    for i in np.arange(1, N_PERSONS):
        temp = make_person(
            i, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob, dgp=dgp
        )
        df = df.append(temp)
    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    df1 = fabricate_data(N_PERSONS=1000, N_PERIODS=4, SEED=1234, exit_prob=0.3, dgp=1)
    df2 = fabricate_data(N_PERSONS=1000, N_PERIODS=4, SEED=1234, exit_prob=0.3, dgp=2)
    print(df1.groupby(["X1", "exit_type"]).size())
    print(df2.groupby(["X1", "exit_type"]).size())
    # df.to_csv("simulated_exit_data.csv", index=False)
