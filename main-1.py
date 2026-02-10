import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def shapefile_to_gdf (file):
    """formats to shapefiles to geodata frame"""
    pass

def gdf_to_train_test_data(gdf):
    """splits geodataframe to training and testing features"""
    pass

def train_model(model, X_train y_train ):
    """
    
    """
    pass

#format data
gdf = shapefile_to_gdf(".\\Shapefiles\\Study Area.shp")
print(gdf.head())
print(gdf.info())

X_FB = gdf.iloc[:,1:2].values
y_FB = gdf.iloc[:,2].values

label_encoder = LabelEncoder()
x_categorical = gdf.select_dtypes(include=['object']).apply(label_encoder.fit_transform)
x_numerical = gdf.select_dtypes(exclude=['object']).values
x = pd.concat([pd.DataFrame(x_numerical), x_categorical], axis=1).values

regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

regressor.fit(x, y_FB)

oob_score = regressor.oob_score_
print(f'Out-of-Bag Score: {oob_score}')

predictions = regressor.predict(x)

mse = mean_squared_error(y_FB, predictions)
print(f'Mean Squared Error: {mse}')

r2 = r2_score(y_FB, predictions)
print(f'R-squared: {r2}')
#split to training and test data
#X_train, X_test, y_train, y_test = gdf_to_train_test_data(gdf)

#initiallize models
#xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,)
#rf_model = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)


#train using five fold cross val
rf_r2 = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
xgb_r2 = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')