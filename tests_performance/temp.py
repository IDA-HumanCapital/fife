from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler
from tests_performance.Data_Fabrication import fabricate_data

N_PERIODS = 6

d0 = fabricate_data(N_PERSONS=10000, N_PERIODS=N_PERIODS, SEED=1234, exit_prob=0.2)
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
