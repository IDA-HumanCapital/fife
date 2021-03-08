
###
# creating a hold out sample
from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler
from tests_performance.Data_Fabrication import fabricate_data

N_PERIODS = 6

d0 = fabricate_data(N_PERSONS=1000, N_PERIODS=N_PERIODS, SEED=1234, exit_prob=0.3)
d1 = d0.copy()
for i in d1["ID"]:
    if N_PERIODS in d1[d1["ID"] == i]["period"].values:
        if (d1[(d1["ID"] == i) & (d1["period"] == N_PERIODS)]["exit_type"] != "No_exit").bool():
            d1.loc[d1["ID"] == i, "exit_type"] = "No_exit"

d1.drop(d0[d0["period"] == N_PERIODS].index, inplace=True)
d1 = d1.reset_index(drop=True)

dp1 = PanelDataProcessor(
    config={"CATEGORICAL_SUFFIXES": ["X1"], "NUMERIC_SUFFIXES": ["X2", "X3"]}, data=d1
)
dp1.build_processed_data()

m1 = LGBExitModeler(data=dp1.data, exit_col="exit_type")

m1.build_model(parallelize=False)

f1 = m1.forecast()

f1 = f1.reset_index()
f1["period"] = N_PERIODS
df = d0.merge(f1, how="right", on=["ID", "period"])
df2 = df.copy()

fet = df["Future exit_type"].unique()
df2.drop(df2[[i not in fet for i in df["exit_type"].values]]["exit_type"].index, inplace=True)




####
# Chi square power calculation
from scipy.stats import chi2
import numpy as np
import pandas as pd

def chi_square_power(n=100, alpha=0.05, dof = 2, J=40, p0=[.7, .2, .1]):
    p = [j for j in range(J)]
    b = [j for j in range(J)]
    for j in range(J):
        delta = j * .01
        d = np.array([-delta, delta/2, delta/2])
        p0 = np.array(p0)
        temp = p0 + d
        if all(temp <= 1) & all(temp >= 0):
            p1 = temp
        mu = n * sum( (p0 - p1)**2 / p0 )
        cv = chi2.ppf(1-alpha, df=dof)
        power = 1 - chi2.cdf(cv, df=dof, loc=mu)
        p[j] = power
        b[j] = delta
    a = pd.DataFrame({'delta': b, 'power':p})
    return a

a = chi_square_power(J=20, p0=[.3334, .3333, .3333])
chi_square_power(J=10, alpha=.1)