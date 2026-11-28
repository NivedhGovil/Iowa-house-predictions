
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# 1. Load the training data (make sure train.csv is in the same folder as this script)
iowa_file_path = 'train.csv'
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# 2. Select features (cleaned up with duplicates removed)
features = [
    'MSSubClass', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF',
    'LowQualFinSF', 'GrLivArea', 'FullBath', 'HalfBath',
    'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
    'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',
    'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold'
]
X = home_data[features]

# 3. Split into validation and training data to check model accuracy
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# 4. Train a new Random Forest model on ALL training data
rf_model_on_full_data = RandomForestRegressor(random_state=1, max_leaf_nodes= 10)
rf_model_on_full_data.fit(X, y)

# 5. Load test data and generate predictions (make sure test.csv is in the same folder)
test_data_path = 'test.csv'
test_data = pd.read_csv(test_data_path)
test_X = test_data[features]
test_preds = rf_model_on_full_data.predict(test_X)


# 6. Save predictions to submission.csv
output = pd.DataFrame({
    'Id': test_data.Id,
    'SalePrice': test_preds
})
output.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv!")