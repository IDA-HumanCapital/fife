from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler
from tests_performance.Data_Fabrication import fabricate_data

N_PERIODS = 5

d0 = fabricate_data(N_PERSONS=100, N_PERIODS=N_PERIODS, SEED=1234, exit_prob=0.2)

dp0 = PanelDataProcessor(
    config={"CATEGORICAL_SUFFIXES": ["X1"], "NUMERIC_SUFFIXES": ["X2", "X3"]}, data=d0
)
dp0.build_processed_data()

m0 = LGBExitModeler(data=dp0.data, exit_col="exit_type")

m0.build_model(parallelize=False)

f0 = m0.forecast()
f0 = f0.reset_index()

f0["period"] = N_PERIODS+1
df = d0.merge(f0, how="right", on=["ID"])
df