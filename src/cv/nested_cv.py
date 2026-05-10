import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    matthews_corrcoef, roc_auc_score, balanced_accuracy_score, 
    f1_score, recall_score, precision_score
)

class RepeatedNestedCV:
    def __init__(self, n_rounds=10, n_outer=5, n_inner=3, random_state=42):
        self.n_rounds = n_rounds     
        self.n_outer = n_outer       
        self.n_inner = n_inner       
        self.random_state = random_state
        self.results = []

    def _get_pipeline(self, estimator):
        return Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('clf', estimator)
        ])

    def run(self, X, y, estimators_dict, param_grids):
        all_model_results = {}

        for name, clf in estimators_dict.items():
            print(f"Processing algorithm: {name}")
            algorithm_metrics = []
            
            # Use StratifiedKFold because the dataset may have a class imbalance
            for r in range(self.n_rounds):
                outer_cv = StratifiedKFold(
                    n_splits=self.n_outer, 
                    shuffle=True, 
                    random_state=self.random_state + r
                )

                for train_idx, test_idx in outer_cv.split(X, y):
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                    # Setup the Pipeline 
                    pipeline = self._get_pipeline(clf)
                    
                    # Inner Loop
                    # Using GridSearchCV
                    inner_cv = StratifiedKFold(
                        n_splits=self.n_inner, 
                        shuffle=True, 
                        random_state=self.random_state
                    )
                    
                    grid_search = GridSearchCV(
                        estimator=pipeline,
                        param_grid=param_grids[name],
                        cv=inner_cv,
                        scoring='roc_auc',
                        n_jobs=-1
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    # Outer Loop
                    best_model = grid_search.best_estimator_
                    y_pred = best_model.predict(X_test)
                    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else y_pred

                    # Calculate required clinical metrics
                    metrics = {
                        'MCC': matthews_corrcoef(y_test, y_pred),
                        'AUC': roc_auc_score(y_test, y_proba),
                        'BA': balanced_accuracy_score(y_test, y_pred),
                        'F1': f1_score(y_test, y_pred),
                        'Recall': recall_score(y_test, y_pred),
                        'Precision': precision_score(y_test, y_pred)
                    }
                    algorithm_metrics.append(metrics)

            all_model_results[name] = pd.DataFrame(algorithm_metrics)
            
        return all_model_results