"""
This script tests the functionality of FIFE's exit modeler on a known dgp.
"""
# Notes
# dask, shap not picked up when installing requirements for fife

from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler
from Data_Fabrication import fabricate_data


data = fabricate_data(N_PERSONS=1000, N_PERIODS=20, SEED=1234)

data_processor = PanelDataProcessor(data=data)
data_processor.build_processed_data()

modeler = LGBExitModeler(data=data_processor.data, exit_col='exit_type')
modeler.build_model(parallelize=False)

forecasts = modeler.forecast()

forecasts.to_csv('simulated_exit_data_forecasts.csv')



