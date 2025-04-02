import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import f21_predict_base as base
import plot_results as pltr
import Scaling


def test_multiple(grouped_test_data, regression_model, reps=10000, size=10):
    num_test_points = len(grouped_test_data)
    logger.info(f"Test_multiple started. {reps} reps x {size} points will be tested for {num_test_points} parameter combinations")
    all_y_test = np.zeros((num_test_points*reps, 2))
    all_y_pred = np.zeros((num_test_points*reps, 2))
    squared_error = 0
    for i, (key, value) in enumerate(grouped_test_data.items()):
        X = np.array(value['X'])
        y = np.array(value['y'])
        logger.info(f"Working on param combination #{i+1}: {key}")

        y_pred_for_test_point = []
        for j in range(reps):
            #pick 10 samples
            rdm = np.random.randint(len(X), size=size)
            X_set = X[rdm]

            #print(f"latent_features_set.shape={latent_features_set.shape}")
            X_mean = np.mean(X_set, axis=0, keepdims=True)
            #print(f"latent_features_mean.shape={latent_features_mean.shape}")
            y_pred = regression_model.predict(X_mean)  # Predict using the trained regressor
        
            y_pred_for_test_point.append(y_pred)
            all_y_pred[i*reps+j,:] = y_pred
            all_y_test[i*reps+j,:] = y[0]
            
        pred_mean_for_test_point = np.mean(y_pred_for_test_point, axis=0)    
        squared_error_for_test_point = np.sum((pred_mean_for_test_point - y[0])**2)
        logger.info(f"Test_multiple: param combination:{y[0]} predicted mean:{pred_mean_for_test_point}, squared_error={squared_error_for_test_point} ")
        squared_error += squared_error_for_test_point

    logger.info(f"### Test_multiple completed. actual shape {all_y_test.shape} predicted shape {all_y_pred.shape}, squared_error={squared_error} ")
    
    r2_means = pltr.summarize_test_1000(all_y_pred, all_y_test, output_dir, showplots=args.interactive, saveplots=True, label="_1000")
    r2 = np.mean(r2_means)
    base.save_test_results(all_y_pred, all_y_test, output_dir)

    return r2

"""
y_test has two two float columns. create a string key using the 2 values upto 2 decimals. 
Use this key group the X_test and y_test records . Return a structure with key associated
 with list of X_test and y_test. Print the count of items for each key.
"""
def group_by_params(X_test, y_test):
    grouped_data = {}
    
    for i in range(len(y_test)):
        # Create a key from the two float values in y_test, rounded to 2 decimals
        key = f"{y_test[i][0]:.2f}_{y_test[i][1]:.2f}"
        
        # Initialize the key in the dictionary if it doesn't exist
        if key not in grouped_data:
            grouped_data[key] = {'X': [], 'y': []}
        
        # Append the corresponding X_test and y_test to the lists
        grouped_data[key]['X'].append(X_test[i])
        grouped_data[key]['y'].append(y_test[i])
    
    # Print the count of items for each key
    for key, value in grouped_data.items():
        logger.info(f"Key: {key}, Count: {len(value['X'])}")
    
    return grouped_data    

# main code starts here
#torch.manual_seed(42)
np.random.seed(42)
#torch.backends.cudnn.determinisitc=True
#torch.backends.cudnn.benchmark=False

parser = base.setup_args_parser()
parser.add_argument('--train_test_filepath', type=str, default='output/noise_sub_ps_load_train_test_uGMRT_t500.0_20250225192715/', help='filepath')
parser.add_argument('--without_noise', action='store_true', help='Use PS without noise subtraction')

args = parser.parse_args()

if not args.train_test_filepath.endswith('/'):
    args.train_test_filepath += '/'
output_dir = base.create_output_dir(args)
logger = base.setup_logging(output_dir)

# Load training data using numpy
if args.without_noise:
    logger.info("Using PS training data without noise")
    X_train = np.loadtxt(f'{args.train_test_filepath}ps_train.csv', delimiter=',')
else:
    X_train = np.loadtxt(f'{args.train_test_filepath}X_train.csv', delimiter=',')
y_train = np.loadtxt(f'{args.train_test_filepath}y_train.csv', delimiter=',')
logger.info(f"Loaded training data: {X_train.shape} {y_train.shape}")

# Load test data using numpy
# Create and fit the XGBRegressor
model = XGBRegressor()
model.fit(X_train, y_train)
logger.info(f"Model fitted")

types = ['weight','gain', 'cover', 'total_gain', 'total_cover']
imp = np.zeros((X_train.shape[1]-3, len(types)))  
for j,imp_type in enumerate(types):
    imp[:,j] = list(model.get_booster().get_score(importance_type=imp_type).values())

logger.info(f"Importances:")
logger.info(types)
logger.info(imp)

logger.info("Starting testing...")
if args.without_noise:
    logger.info("Using PS test data without noise")
    X_test = np.loadtxt(f'{args.train_test_filepath}ps_test.csv', delimiter=',')
else:
    X_test = np.loadtxt(f'{args.train_test_filepath}X_test.csv', delimiter=',')
y_test = np.loadtxt(f'{args.train_test_filepath}y_test.csv', delimiter=',')

logger.info(f"Loaded training data: {X_train.shape} {y_train.shape}")
logger.info(f"Loaded test data: {X_test.shape} {y_test.shape}")
grouped_test_data = group_by_params(X_test, y_test)
test_multiple(grouped_test_data, model, size=args.psbatchsize)
# Make predictions
#predictions = model.predict(X_test)

# Calculate R2 score and RMS error
#r2 = r2_score(y_test, predictions)
#rms_error = np.sqrt(mean_squared_error(y_test, predictions))

#logger.info(f'R2 Score: {r2}')
#logger.info(f'RMS Error: {rms_error}')

# Save predictions and y_test to a CSV file
#test_results = np.column_stack((y_test, predictions))
#np.savetxt(f"{output_dir}/test_results.csv", test_results, delimiter=",", header="y_test_0,y_test_1,pred_0,pred_1", comments='')

#pltr.summarize_test_1000(predictions, y_test, output_dir=output_dir, showplots=False, saveplots=True)
