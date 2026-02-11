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
    """
    trains a model using 5 fold cross validation
    reports validation performance to terminal.
    And, genrates residual graph.
    """

    #get model name
    model_name = type(model).__name__

    #split to test and train
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # #cross validation scores
    # cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
    # print(f"Cross validation scores: for {model_name}\n{cv_scores}")
    # print(f"Mean: {cv_scores.mean():.2f}")

    #cross val predictions
    y_pred = cross_val_predict(model, X_train, y_train, cv=5)

    #validation performance
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    print(f"\n--- {model_name} validation performance ---")
    print(f"MSE: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")



    #holdout test
    model.fit(X_train, y_train)
    model_score = model.score(X_test, y_test)
    print(f"holdout score for {model_name} is {model_score}")

    #final fit
    model.fit(X,y)


    #residual graphing
    print("Graph saved in Figures, close graph to advance program\n")

    res = y_train - y_pred

    fig, ax1= plt.subplots(figsize=(15, 6), sharey=True)

    sns.scatterplot(x=y_pred, y=res, ax=ax1, alpha=0.5, color='steelblue')
    ax1.axhline(0, color='red', linestyle='--')

    
    ax1.set_title(f"{model_name} Residuals")
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Residuals')

    plt.tight_layout()


    plt.savefig(f".\\Figures\\{model_name}_residuals.png")
    plt.show()



def predict_BUILD_LOSS_RATE(model, X):
    """
    Predicts the unknown build Loss rate of Census Tracts
    and produces a histogram predicted loss rates
    """
    #get model name
    model_name = type(model).__name__
    pred_y = model.predict(X)

    plt.hist(pred_y, bins=30, color="#FF00DD", edgecolor='black')

    plt.title(f"BUILD_LOSS_RATE(Predicted) dist by {model_name}")
    plt.xlabel("BUILD_LOSS_RATE(predicted)")
    plt.ylabel("Count")

    plt.savefig(f".\\Figures\\BUILD_LOSS_RATE(Predicted) dist by {model_name}.png")
    print("Graph saved in Figures, close graph to advance program\n")
    
    plt.show()




#1) Bring the GIS data into Python
print("\n--------------------------------")


gdf = gpd.read_file(".\\Shapefiles\\Study Area.shp")
# print(gdf.head())
# print(gdf.info())
# input()

features = [
    'AREA', 'BUILDVALUE','POPULATION',
    'ERQK_AFREQ', 'LNDS_AFREQ', 'SWND_AFREQ', 'WFIR_AFREQ']
X = gdf[features].values
y = gdf['BUILD_LOSS'].values

#2) Train and validate models on “Study Area”

#initialize models
rf_model = RandomForestRegressor(n_estimators=1000, random_state=0)
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.1)

train_validate_and_predict(rf_model,X,y)
train_validate_and_predict(xgb_model,X,y)

#3) Predict for “All Census Tracts” and evaluate testing performance


#format data
gdf_all = gpd.read_file(".\\Shapefiles\\All Census Tracts.shp")
X_all = gdf_all[features].values

predict_for_unknown_X(rf_model, X_all)
predict_for_unknown_X(xgb_model, X_all)
