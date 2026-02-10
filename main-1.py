import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split

#Ethan: used google ai mode


# def print_performance_metrics(y_test, y_pred):
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     r2 = r2_score(y_test, y_pred)
#     print ("R^2 is {r2}\n RMSE is {rmse}")

# def plot_rediduals(y_test, y_pred):
#     residuals = y_test - y_pred
#     plt.scatter(y_pred, residuals)
#     plt.axhline(y=0, color='r', linestyle='--')
#     plt.xlabel('Predicted Values')
#     plt.ylabel('Residuals')
#     plt.title('Residual Plot')
#     plt.show()


#format data
gdf = gpd.read_file("Study Area.shp")


X  = gdf.drop(columns=['target', 'geometry'])
y = gdf['target']

#initiallize models
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,)
rf_model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)


#train using five fold cross val
rf_r2 = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
xgb_r2 = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')