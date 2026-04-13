# grid_search.py

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from model import build_rf_model


def run_rf_grid_search(X, y, groups, scoring="f1_macro", n_splits=5, random_state=42):

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    rf = build_rf_model(
        random_state=random_state,
        n_jobs=-1
    )

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf": [1, 4],
        "max_features": ["sqrt"],
        "class_weight": ["balanced"],
    }

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid.fit(X, y, groups=groups)

    print("Best params:", grid.best_params_)
    print("Best score:", grid.best_score_)

    return grid