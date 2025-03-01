import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
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
ind_descriptions = np.array(["PS 1st bin",
                        "PS 2nd bin",
                        "PS 3rd bin",
                        "PS 4th bin",
                        "Total mean",
                        "Total std",
                        "Skewness",
                        "Skewness - std",
                        "Skew2",
                        "Min skew",
                        "Bispectrum 1st bin",
                        "Bispectrum 2nd bin",
                        "Bispectrum 3rd bin",
                        "Bispectrum 4th bin",])
del_ind = [1,2,3,5,9,11,12] #[4,5,6,7,8,9,10,11,12,13] #[9] # # 6,13
logger.info(f"Removing features with indices {ind_descriptions[del_ind]}")
remaining_indices = [i for i in range(len(ind_descriptions)) if i not in del_ind]
logger.info(f"Using features with indices {ind_descriptions[remaining_indices]}")

# Load training data using numpy
#train_data = np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
#train_data = np.loadtxt('saved_output/f21_predict_bispec_train_test_uGMRT_t500.0_20250115140235_complete/all_training_data.csv', delimiter=',')
train_data = np.loadtxt('saved_output/bispectrum_data_complete/all_training_data.csv', delimiter=',')
X_train = np.delete(train_data[:,:-2], del_ind, axis=1) 
y_train = train_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape} (skipped indices {del_ind})")

# Load test data using numpy
#test_data = np.loadtxt('saved_output/bispectrum_data_20k/all_test_data.csv', delimiter=',')
#test_data = np.loadtxt('saved_output/f21_predict_bispec_train_test_uGMRT_t500.0_20250115140235_complete/all_test_data.csv', delimiter=',')
test_data = np.loadtxt('saved_output/bispectrum_data_complete/all_test_data.csv', delimiter=',')
X_test = np.delete(test_data[:,:-2], del_ind, axis=1)  
y_test = test_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape} (skipped indices {del_ind})")

# Create and fit the XGBRegressor
model = XGBRegressor(n_estimators=121, max_depth=9, learning_rate=0.02034585149604065, 
                     subsample=0.7951818164938813, colsample_bytree=0.8724517169927652, 
                     min_child_weight=8, gamma=7.718075964724362e-06, 
                     reg_lambda=1.089312256998268e-07, reg_alpha=0.0003573596701472528)
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
base.calc_squared_error(predictions, y_test)
base.summarize_test_1000(predictions, y_test, output_dir=output_dir, showplots=True, saveplots=True, label="")
"""
"""
