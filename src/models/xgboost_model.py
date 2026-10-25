import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
import joblib
from typing import Dict, Any
from src.models.base import BaseModel
from sklearn.metrics import roc_auc_score

# Silence Optuna logger to prevent cluttering stdout unless requested
optuna.logging.set_verbosity(optuna.logging.WARNING)

class XGBoostModel(BaseModel):
    """XGBoost wrapper with optional Optuna hyperparameter tuning."""

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        tune: bool = False,
        n_trials: int = 50
    ) -> None:
        """
        If tune=False: trains with self.config["params"].
        If tune=True: runs Optuna study for n_trials, uses best params, then
                      trains final model on full train+val with best params.
        """
        # Calculate scale_pos_weight for class imbalance correction
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

        if tune:
            print(f"Running XGBoost Optuna tuning with {n_trials} trials...")
            
            def objective(trial):
                # Sample parameters
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 2000),
                    "max_depth": trial.suggest_int("max_depth", 3, 9),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "tree_method": self.config.get("params", {}).get("tree_method", "hist"),
                    "device": self.config.get("params", {}).get("device", "cpu"),
                    "random_state": 42,
                    "eval_metric": "auc",
                    "scale_pos_weight": scale_pos_weight,
                    "n_jobs": -1
                }
                
                # Fit classifier on trial parameters with early stopping
                clf = xgb.XGBClassifier(**params)
                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False
                )
                
                # Evaluate on validation split
                preds = clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            print(f"Best XGBoost Trial AUC: {study.best_value:.5f}")
            print(f"Best XGBoost Params: {study.best_params}")
            
            # Combine train and val for final fit
            X_comb = pd.concat([X_train, X_val], axis=0)
            y_comb = pd.concat([y_train, y_val], axis=0)
            
            best_params = study.best_params
            best_params["tree_method"] = self.config.get("params", {}).get("tree_method", "hist")
            best_params["device"] = self.config.get("params", {}).get("device", "cpu")
            best_params["random_state"] = 42
            best_params["eval_metric"] = "auc"
            # Recalculate scale_pos_weight on combined dataset
            n_neg_comb = (y_comb == 0).sum()
            n_pos_comb = (y_comb == 1).sum()
            best_params["scale_pos_weight"] = n_neg_comb / n_pos_comb if n_pos_comb > 0 else 1.0
            best_params["n_jobs"] = -1

            self.model = xgb.XGBClassifier(**best_params)
            self.model.fit(X_comb, y_comb, verbose=False)
            
            # Evaluate final AUC on training data as validation proxy
            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Final retrained XGBoost Val AUC: {val_auc:.5f}")
        else:
            # Train model using default configs
            params = self.config.get("params", {}).copy()
            params["scale_pos_weight"] = scale_pos_weight
            params["n_jobs"] = -1
            
            self.model = xgb.XGBClassifier(**params)
            print(f"Training XGBoost with device={params.get('device', 'cpu')}...")
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"XGBoost Val AUC: {val_auc:.5f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet. Cannot save.")
        joblib.dump(self.model, path)
        print(f"Saved XGBoost model to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        model_obj = joblib.load(path)
        inst = cls(config={})
        inst.model = model_obj
        return inst
