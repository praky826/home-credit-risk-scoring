import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
import joblib
from typing import Dict, Any
from src.models.base import BaseModel
from sklearn.metrics import roc_auc_score

# Silence LightGBM logs and Optuna logger
optuna.logging.set_verbosity(optuna.logging.WARNING)

class LGBMModel(BaseModel):
    """LightGBM wrapper with optional Optuna tuning."""

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
        Same contract as XGBoostModel.fit.
        Uses lgb.LGBMClassifier.
        Early stopping: callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)].
        is_unbalance=True (instead of scale_pos_weight).
        """
        if tune:
            print(f"Running LightGBM Optuna tuning with {n_trials} trials...")
            
            def objective(trial):
                params = {
                    "n_estimators": trial.suggest_int("n_estimators", 300, 3000),
                    "num_leaves": trial.suggest_int("num_leaves", 20, 300),
                    "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
                    "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                    "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
                    "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                    "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                    "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                    "is_unbalance": True,
                    "metric": "auc",
                    "random_state": 42,
                    "verbose": -1,
                    "n_jobs": -1
                }
                
                clf = lgb.LGBMClassifier(**params)
                clf.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(100, verbose=False)]
                )
                
                preds = clf.predict_proba(X_val)[:, 1]
                return roc_auc_score(y_val, preds)

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=n_trials)
            
            print(f"Best LightGBM Trial AUC: {study.best_value:.5f}")
            print(f"Best LightGBM Params: {study.best_params}")
            
            # Combine train and val for final fit
            X_comb = pd.concat([X_train, X_val], axis=0)
            y_comb = pd.concat([y_train, y_val], axis=0)
            
            best_params = study.best_params
            best_params["is_unbalance"] = True
            best_params["metric"] = "auc"
            best_params["random_state"] = 42
            best_params["verbose"] = -1
            best_params["n_jobs"] = -1

            self.model = lgb.LGBMClassifier(**best_params)
            self.model.fit(X_comb, y_comb)
            
            val_preds = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"Final retrained LightGBM Val AUC: {val_auc:.5f}")
        else:
            params = self.config.get("params", {}).copy()
            params["is_unbalance"] = True
            params["n_jobs"] = -1
            
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100, verbose=False)]
            )
            
            val_preds = self.predict_proba(X_val)
            val_auc = roc_auc_score(y_val, val_preds)
            print(f"LightGBM Val AUC: {val_auc:.5f}")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model is not trained yet. Call fit() first.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Model is not trained yet. Cannot save.")
        joblib.dump(self.model, path)
        print(f"Saved LightGBM model to {path}")

    @classmethod
    def load(cls, path: str) -> "LGBMModel":
        model_obj = joblib.load(path)
        inst = cls(config={})
        inst.model = model_obj
        return inst
