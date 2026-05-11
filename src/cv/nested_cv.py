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
    

from sklearn.feature_selection import SelectFromModel
from collections import Counter
from sklearn.linear_model import LogisticRegression

class FeatureSelectingRNCV(RepeatedNestedCV):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_frequencies = Counter()

    def run_with_selection(self, X, y, estimator, param_grid):
        algorithm_metrics = []
        feature_names = X.columns
        
        for r in range(self.n_rounds):
            outer_cv = StratifiedKFold(
                n_splits=self.n_outer, 
                shuffle=True, 
                random_state=self.random_state + r
            )

            for train_idx, test_idx in outer_cv.split(X, y):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Preprocessing (Internal to loop)
                imputer = SimpleImputer(strategy='median')
                scaler = StandardScaler()
                
                X_train_proc = scaler.fit_transform(imputer.fit_transform(X_train))
                X_test_proc = scaler.transform(imputer.transform(X_test))

                # Feature Selection (using Lasso to identify important features)
                selector = SelectFromModel(
                    LogisticRegression(solver='liblinear', C=0.1, random_state=42)
                )
                selector.fit(X_train_proc, y_train)
                
                # Track stable features 
                selected_mask = selector.get_support()
                selected_features = feature_names[selected_mask]
                self.feature_frequencies.update(selected_features)

                # Filter datasets
                X_train_sel = selector.transform(X_train_proc)
                X_test_sel = selector.transform(X_test_proc)

                # Inner Loop: Tuning on selected features
                inner_cv = StratifiedKFold(n_splits=self.n_inner, shuffle=True, random_state=self.random_state)

                cleaned_param_grid = {k.replace('clf__', ''): v for k, v in param_grid.items()}
                
                grid_search = GridSearchCV(
                    estimator=estimator,
                    param_grid=cleaned_param_grid,
                    cv=inner_cv,
                    scoring='roc_auc',
                    n_jobs=-1
                )
                grid_search.fit(X_train_sel, y_train)

                # Evaluation
                best_model = grid_search.best_estimator_
                y_pred = best_model.predict(X_test_sel)
                y_proba = best_model.predict_proba(X_test_sel)[:, 1]

                algorithm_metrics.append({
                    'MCC': matthews_corrcoef(y_test, y_pred),
                    'AUC': roc_auc_score(y_test, y_proba),
                    'BA': balanced_accuracy_score(y_test, y_pred)
                })

        return pd.DataFrame(algorithm_metrics)