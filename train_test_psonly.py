import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
import plot_results as pltr
import Scaling

class Args:
    def __init__(self, runmode='', telescope='uGMRT', t_int=500):
        self.runmode = runmode
        self.telescope = telescope
        self.t_int = t_int
        self.scale_y = True

args = Args() 
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

# Load training data using numpy
train_data = np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
X_train = train_data[:,:4]
y_train = train_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded training data (ps only): {X_train.shape} {y_train.shape}")

# Load test data using numpy
test_data = np.loadtxt('saved_output/bispectrum_data_20k/all_test_data.csv', delimiter=',')
X_test = test_data[:,:4]
y_test = test_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded test data (ps only): {X_test.shape} {y_test.shape}")

# Create and fit the XGBRegressor
model = XGBRegressor(n_estimators= 132, max_depth= 3,
                    learning_rate= 0.010025545726491792,
                    subsample= 0.5844341031772149,
                    colsample_bytree= 0.7071988176515248,
                    min_child_weight= 10,
                    gamma= 0.15770877256219745,
                    reg_lambda= 1.0056385373224926e-05,
                    reg_alpha= 7.99951836479736e-07
                    )
model.fit(X_train, y_train)
logger.info(f"Model fitted")

types = ['weight','gain', 'cover', 'total_gain', 'total_cover']
imp = np.zeros((X_train.shape[1], len(types)))  
for j,imp_type in enumerate(types):
    imp[:,j] = list(model.get_booster().get_score(importance_type=imp_type).values())

logger.info(f"Importances:")
logger.info(types)
logger.info(imp)

# Make predictions
predictions = model.predict(X_test)

# Calculate R2 score and RMS error
r2 = r2_score(y_test, predictions)
rms_error = np.sqrt(mean_squared_error(y_test, predictions))

logger.info(f'R2 Score: {r2}')
logger.info(f'RMS Error: {rms_error}')

# Save predictions and y_test to a CSV file
test_results = np.column_stack((y_test, predictions))
np.savetxt(f"{output_dir}/test_results.csv", test_results, delimiter=",", header="y_test_0,y_test_1,pred_0,pred_1", comments='')

predictions = Scaling.Scaler(args).unscale_y(predictions)
y_test = Scaling.Scaler(args).unscale_y(y_test)
pltr.summarize_test_1000(predictions, y_test, output_dir=output_dir, showplots=True, saveplots=True)
"""
"""