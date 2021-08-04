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
    elif dgp == 3:
        if x1 == "A":
            exit_type_prob = [0.95, 0.025, 0.025]  # [0.7, 0.2, 0.1]
        if x1 == "B":
            exit_type_prob = [0.025, 0.95, 0.025]  # [0.2, 0.7, 0.1]
        if x1 == "C":
            exit_type_prob = [0.025, 0.025, 0.95]  # [0.1, 0.2, 0.7]
    else:
        raise NameError('Invalid DGP')
    return x1, x2, x3, exit_prob, exit_type_prob



def make_person(i, N_PERIODS, k=0, exit_prob=0.5, exit_type_prob=None, dgp=1, beta_dict = None):
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
    if beta_dict is not None:
        if x1 == "A":
            beta_x1 = beta_dict['beta_xa']
        elif x1 == "B":
            beta_x1 = beta_dict['beta_xb']
        else:
            beta_x1 = beta_dict['beta_xc']

        # use Weibull distribution for survival curves
        logh = beta_dict['beta_0'] + beta_dict['beta_t'] * np.log(beta_dict['t'] * (10 / N_PERIODS)) + beta_x1 + beta_dict['beta_x2'] * x2 + beta_dict['beta_x3'] * (x3 - 0.5)
        p = 1 - np.exp(-(np.exp(logh)))
        survival_prob = [1 - p[0]]
        for tp in range(1,len(p)):
            survival_prob.append(survival_prob[tp-1] * (1 - p[tp]))
        time = 0
        while date < N_PERIODS:
            exit_prob = p[min(time, N_PERIODS - 1)]
            # print([i, date, x3, x1])
            df.append([i, date, x1, x2, x3] + Z)
            if exit_type != "No_exit":
                break
            # this part handles characteristics that change over time
            x1, x2_temp, x3, exit_prob, exit_type_prob = update_person(x1, x2, x3, exit_prob=exit_prob,
                                                                  exit_type_prob=exit_type_prob, dgp=dgp)
            exit_type = make_exit_type(exit_prob=exit_prob, p=exit_type_prob)
            date += 1
            time += 1

    else:
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

    if beta_dict is not None:
        df["exit_prob"] = p[0:len(df)]


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
        N_PERSONS=100, N_PERIODS=10, SEED=None, k=0, exit_prob=0.5, exit_type_prob=None, dgp=1,
        covariates_affect_outcome= False
):
    if SEED is not None:
        np.random.seed(SEED)
    if covariates_affect_outcome:
        t = np.arange(1,N_PERIODS + 1)
        # beta_t = np.random.uniform(low = 0, high = 1.2)
        beta_t = 0.6
        beta_0 = np.log((2 ** beta_t) * ((8) ** (-beta_t)) * np.log(1 / (1 - exit_prob)))
        # beta_xa = np.random.normal(scale = 0.5)
        # beta_xb = np.random.normal(scale = 0.5)
        # beta_xc = np.random.normal(scale = 0.5)
        # beta_x2 = np.random.normal(loc = -0.2, scale = 0.7)
        # beta_x3 = np.random.normal(scale = 0.7)
        beta_xa = 0.5
        beta_xb = -0.5
        beta_xc = 0.7
        beta_x2 = -0.7
        beta_x3 = 0.2
        beta_dict = {
            "beta_0": beta_0,
            "beta_t": beta_t,
            "beta_xa": beta_xa,
            "beta_xb": beta_xb,
            "beta_xc": beta_xc,
            "beta_x2": beta_x2,
            "beta_x3": beta_x3,
            "t": t
        }

    else:
        beta_dict = None

    df = make_person(
        0, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob, dgp=dgp,
        beta_dict = beta_dict
    )
    for i in np.arange(1, N_PERSONS):
        temp = make_person(
            i, N_PERIODS, k=k, exit_prob=exit_prob, exit_type_prob=exit_type_prob, dgp=dgp,
            beta_dict = beta_dict
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
