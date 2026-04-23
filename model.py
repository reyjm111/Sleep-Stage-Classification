from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def build_rf_model(
    n_estimators=1000,
    max_depth=11,
    min_samples_split=2,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced_subsample", 
    bootstrap=True, 
    max_samples=0.75, 
    random_state=42,
    n_jobs=-1,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight, 
        bootstrap=bootstrap, 
        max_samples=max_samples, 
        random_state=random_state,
        n_jobs=n_jobs,
    )

def build_xgb_model(
    n_estimators=700,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    reg_alpha=1.0,
    gamma=1.0,
    min_child_weight=5,
    random_state=42,
    n_jobs=-1,
):
    
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        gamma=gamma,
        min_child_weight=min_child_weight,
        objective="multi:softprob", 
        eval_metric="mlogloss",
        random_state=random_state,
        n_jobs=n_jobs,
        tree_method="hist"
    )