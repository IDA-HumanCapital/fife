from fife.processors import PanelDataProcessor
from fife.tf_modelers import TFSurvivalModeler
# from tests_performance.Data_Fabrication import fabricate_data
import pandas as pd
from tensorflow.python.keras.backend import eager_learning_phase_scope


def TF_model_and_forecasts(df):
    dp = PanelDataProcessor(data=df)
    dp.build_processed_data()
    m = TFSurvivalModeler(data=dp.data)
    m.build_model()
    f = m.forecast()
    f = f.reset_index()
    return m, f


def TF_uncertainty(m):
    p = m.compute_model_uncertainty(subset=m.data._predict_obs)
    return p



if __name__ == "__main__":
    # d0 = fabricate_data(N_PERSONS=10000, N_PERIODS=20, SEED=1234, exit_prob=.3)
    # d0.to_csv('temp_data.csv', index=False)
    d0 = pd.read_csv('./PIs/temp_data.csv')
    m, f = TF_model_and_forecasts(d0)
    with eager_learning_phase_scope(value=1):
        p = TF_uncertainty(m)

