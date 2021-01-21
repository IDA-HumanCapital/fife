from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler
from Data_Fabrication import fabricate_data

d0 = fabricate_data(N_PERSONS=100, N_PERIODS=5, SEED=1234, exit_prob=0.2)

dp0 = PanelDataProcessor(
    config={"CATEGORICAL_SUFFIXES": ["X1"], "NUMERIC_SUFFIXES": ["X2", "X3"]}, data=d0
)
dp0.build_processed_data()

m0 = LGBExitModeler(data=dp0.data, exit_col="exit_type")

m0.build_model(parallelize=False)

f0 = m0.forecast()
