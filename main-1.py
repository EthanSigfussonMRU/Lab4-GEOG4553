import warnings

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
import xgboost as xgb
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.model_selection import (cross_val_predict, cross_val_score,
                                     train_test_split)
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')


def train_validate_and_predict(model, X, y):
    #cross validation scores
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"Scores: for {model=}{cv_scores}")
    print(f"Mean: {cv_scores.mean():.2f}")

    #cross val predictions
    y_pred = cross_val_predict(model, X, y, cv=5)
    print(f"\n--- XGBoost Results ---")
    
    #score validation
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


X = gdf.iloc[:,1:2].values
y = gdf.iloc[:,2].values

#initialize models
rf_model = RandomForestRegressor(n_estimators=100, random_state=0)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)

train_validate_and_predict(rf_model,X,y)
train_validate_and_predict(xgb_model,X,y)


# #cross validation scores
# rf_cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring='r2')
# xgb_cv_scores = cross_val_score(xgb_model, X, y, cv=5, scoring='r2')


# print(f"Scores: {rf_cv_scores}")
# print(f"Mean: {rf_cv_scores.mean():.2f}")

# print(f"Scores: {xgb_cv_scores}")
# print(f"Mean: {xgb_cv_scores.mean():.2f}")

# #cross val predictions
# y_rf_pred = cross_val_predict(rf_model, X, y, cv=5)
# y_xgb_pred = cross_val_predict(xgb_model, X, y, cv=5)

# #score validation
# mse_xgb = mean_squared_error(y, y_xgb_pred)
# r2_xgb = r2_score(y, y_xgb_pred)

# print(f"\n--- XGBoost Results ---")
# print(f"MSE: {mse_xgb:.4f}")
# print(f"R2 Score: {r2_xgb:.4f}")

# #residual graphing
# res_rf = y - y_rf_pred
# res_xgb = y - y_xgb_pred


# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
# # Random Forest Plot
# sns.scatterplot(x=y_rf_pred, y=res_rf, ax=ax1, alpha=0.5, color='steelblue')
# ax1.axhline(0, color='red', linestyle='--')
# ax1.set_title('Random Forest Residuals')
# ax1.set_xlabel('Predicted Values')
# ax1.set_ylabel('Residuals')

# # XGBoost Plot
# sns.scatterplot(x=y_xgb_pred, y=res_xgb, ax=ax2, alpha=0.5, color='darkorange')
# ax2.axhline(0, color='red', linestyle='--')
# ax2.set_title('XGBoost Residuals')
# ax2.set_xlabel('Predicted Values')

# plt.tight_layout()
# plt.show()


# gdf_all_tracts = gpd.read_file(".\\Shapefiles\\All Census Tracts.shp")