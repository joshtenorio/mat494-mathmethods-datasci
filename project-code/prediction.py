import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.ensemble import GradientBoostingRegressor



def cv_score(model, X, y, groups):
    # Perform the cross validation
    scores = cross_val_score(model, X, y,
                             cv=groups,
                             scoring='neg_mean_squared_error')
    # Taking the expe
    # of the numbers we are getting
    corrected_score = [np.sqrt(-x) for x in scores]
    return corrected_score

def gb_score(X, y):
    # Model assignment
    model = GradientBoostingRegressor
    # Perform Grid-Search
    gsc = GridSearchCV(
        estimator=model(),
        param_grid={
            'learning_rate': [0.01, 0.1, 0.2, 0.3],
            'n_estimators': [100, 300, 500, 1000],
        },
        cv=5,
        scoring='neg_mean_squared_error',
        verbose=0,
        n_jobs=-1)
    # Finding the best parameter
    grid_result = gsc.fit(X, y)
    best_params = grid_result.best_params_
    # Random forest regressor with specification
    gb = model(learning_rate=best_params["learning_rate"],
               n_estimators=best_params["n_estimators"],
               random_state=False,
               verbose=False)
    # Apply cross validiation on the model
    gb.fit(X, y)
    score = cv_score(gb, X, y, 5)
    # Return information
    return score, best_params, gb

# import data
raw = pd.read_csv("data/processed/final-final.csv")
y = raw["Price"]
final_X = raw.drop(columns=["Price", "address", "Local_area", "url", "w-latitude", "w-longitude", "Unnamed: 0", "latitude", "longitude", "Unnamed: 0.1"])

# split dataset
X_train, X_test, y_train, y_test = train_test_split(final_X, y,
                                                    test_size=0.15,
                                                    random_state=28)


# get prediction
score, best_params, gb = gb_score(X_train, y_train)

# mean absolute percent error
gb_predictions = gb.predict(X_test)
mape = (abs(gb_predictions - y_test)/y_test) * 100
print(mape.describe())

feats = gb.feature_importances_
feats *= 100

plt.bar(gb.feature_names_in_, feats)
plt.title("Feature Importance")
plt.xlabel("Feature")
plt.ylabel("Relative Importance")
plt.show()

print(gb.feature_names_in_)