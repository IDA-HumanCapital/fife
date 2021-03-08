
from fife.processors import PanelDataProcessor
from fife.lgb_modelers import LGBExitModeler, LGBSurvivalModeler, LGBStateModeler
from tests_performance.Data_Fabrication import fabricate_data
import numpy as np


def get_forecast(df, modeler="LGBSurvivalModeler", exit_col='exit_type'):
    dp = PanelDataProcessor(data=df)
    dp.build_processed_data()
    if modeler == "LGBSurvivalModeler":
        m = LGBSurvivalModeler(data=dp.data)
    elif modeler == "LGBExitModeler":
        m = LGBExitModeler(data=dp.data, exit_col=exit_col)
    else:
        raise NameError('Invalid modeler')
    m.build_model(parallelize=False)
    f = m.forecast()
    f = f.reset_index()
    return f


if __name__ == "__main__":
    # Trying to reproduce Mike's bug report on our simulated data
    d0 = fabricate_data(N_PERSONS=5000, N_PERIODS=20, SEED=1234, exit_prob=.3)
    d0 = d0.rename(columns={"exit_type": "outcome"})
    exit_modeler_forecasts = get_forecast(d0, modeler="LGBExitModeler", exit_col="outcome")

    # Do the outcomes always sum to 1?
    summary1a = exit_modeler_forecasts.groupby(["ID"]).sum().reset_index()
    summary1a = summary1a.drop(["ID"], axis=1)
    # pd.DataFrame(index = exit_modeler_forecasts.index.unique(), columns = exit_modeler_forecasts.columns[:-1])
    # for i in exit_modeler_forecasts.index.unique():
    #     summary1a.loc[i,:] = exit_modeler_forecasts.loc[i, exit_modeler_forecasts.columns != 'Future outcome'].sum(axis = 0)
    (summary1a!=1).sum().sum()/(summary1a.shape[0]*summary1a.shape[1]) # Should be 0
    abs(1-summary1a).sum().sum()/(summary1a.shape[0]*summary1a.shape[1]) # should be 0

    # it is effectively within floating point precision, so I'd call this a non-issue.

    # However, they are "close"
    tmp = summary1a.to_numpy(dtype='float64')
    np.isclose(tmp,np.full(tmp.shape,1)).sum()/(tmp.shape[0] * tmp.shape[1])

