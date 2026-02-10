import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     train_test_split)

warnings.filterwarnings('ignore')


def train_validate_and_predict(model, X, y):
    model.fit()

    #cross validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Scores: for {model=}{cv_scores}")
    print(f"Mean: {cv_scores.mean():.2f}")

    #cross val predictions
    y_pred = cross_val_predict(model, X, y, cv=5)
    print(f"\n--- XGBoost Results ---")
    
    #validation performance
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"\n--- {model=} Results ---")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")


    #residual graphing
    print("close graph to advance program")

    res = y - y_pred

    fig, ax1= plt.subplots(figsize=(15, 6), sharey=True)

    sns.scatterplot(x=y_pred, y=res, ax=ax1, alpha=0.5, color='steelblue')
    ax1.axhline(0, color='red', linestyle='--')

    #get model name
    model_name = type(model).__name__
    
    ax1.set_title(f"{model_name} Residuals")
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')

    plt.tight_layout()


    plt.savefig(f".\\Figures\\{model_name}_residuals.png")
    plt.show()



#format data
gdf = gpd.read_file(".\\Shapefiles\\Study Area.shp")
# print(gdf.head())
# print(gdf.info())

#1) Bring the GIS data into Python
X = gdf.iloc[:,1:2].values
y = gdf.iloc[:,2].values



#2) Train and validate models on “Study Area”

#initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

train_validate_and_predict(rf_model,X,y)
train_validate_and_predict(xgb_model,X,y)

#3) Predict for “All Census Tracts” and evaluate testing performance