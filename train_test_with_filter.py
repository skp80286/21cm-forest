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
del_ind = [9] #[1,2,3,5,9,11,12] # 6,13
logger.info(f"Removing features with indices {del_ind}")
# Load training data using numpy
train_data = np.loadtxt('saved_output/bispectrum_data_20k/all_training_data.csv', delimiter=',')
train_data = train_data[train_data[:, -1] >= 0.5]  # Keep rows where log_fX is <= 0.5
X_train = np.delete(train_data[:,:-2], del_ind, axis=1) 
y_train = train_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape} (skipped indices {del_ind})")

# Load test data using numpy
test_data = np.loadtxt('saved_output/bispectrum_data_20k/all_test_data.csv', delimiter=',')
test_data = test_data[test_data[:, -1] <= 0.5]  # Keep rows where log_fX is <= 0.5
X_test = np.delete(test_data[:,:-2], del_ind, axis=1)  
y_test = test_data[:, -2:]    # Last two columns as output
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape} (skipped indices {del_ind})")

# Create and fit the XGBRegressor
model = XGBRegressor()
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
# Create a 2D scatter plot
base.initplt()
plt.scatter(y_test[:, 0], y_test[:, 1], alpha=0.5, label='True Values', marker='*')  # True values as stars

# Calculate mean predictions for each unique test point
unique_test_points = np.unique(y_test, axis=0)
mean_predictions = []

for point in unique_test_points:
    # Get indices of all predictions corresponding to this test point
    indices = np.all(y_test == point, axis=1)
    logger.info(f"{point}: found {len(indices)} predictions")
    mean_pred = np.mean(predictions[indices], axis=0)
    mean_predictions.append((point[0], point[1], mean_pred[0], mean_pred[1]))

mean_predictions = np.array(mean_predictions)

# Plot mean predictions
plt.scatter(mean_predictions[:, 0], mean_predictions[:, 1], color='red', marker='X', s=100, label='Mean Predictions')  # Mean predictions as X
plt.xlabel('xHI')
plt.ylabel('log_fX')
plt.title('True Values vs Predictions')
plt.legend()
plt.show()
"""