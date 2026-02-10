import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Generate sample regression data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)