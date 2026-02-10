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
gdf = shapefile_to_gdf("Study Area.shp")
#split to training and test data
X_train, X_test, y_train, y_test = gdf_to_train_test_data(gdf)

#initiallize models
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100,)
regressor = RandomForestRegressor(n_estimators=10, random_state=0, oob_score=True)

#train model

