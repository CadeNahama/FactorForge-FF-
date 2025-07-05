#!/usr/bin/env python3
"""
Ensemble Machine Learning System for Quantitative Trading
Advanced ML techniques for institutional-grade predictions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import warnings
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
warnings.filterwarnings('ignore')

class EnsembleMLSystem:
    """
    Ensemble machine learning system for quantitative trading
    """
    
    def __init__(self, n_splits: int = 5, horizon: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.horizon = horizon
        self.random_state = random_state
        
        # Initialize models
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.model_weights = {}
        self.performance_metrics = {}
        
        # Model configurations
        self.model_configs = {
            'xgboost': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state
            },
            'lightgbm': {
                'n_estimators': 200,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': random_state
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': random_state
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'random_state': random_state
            },
            'ridge': {
                'alpha': 1.0,
                'random_state': random_state
            },
            'lasso': {
                'alpha': 0.01,
                'random_state': random_state
            },
            'svr': {
                'kernel': 'rbf',
                'C': 1.0,
                'epsilon': 0.1
            }
        }
    
    def prepare_data(self, features: pd.DataFrame, prices: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for ML training"""
        
        # Align features and prices
        features = features.sort_index()
        prices = prices.reindex(features.index)
        
        # Calculate forward returns
        forward_returns = prices.pct_change(periods=self.horizon).shift(-self.horizon)
        
        # Remove NaN values
        valid_mask = ~(features.isnull().any(axis=1) | forward_returns.isnull())
        X = features[valid_mask]
        y = forward_returns[valid_mask]
        
        return X, y
    
    def train_individual_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, object]:
        """Train individual models"""
        
        trained_models = {}
        
        # XGBoost
        try:
            xgb_model = xgb.XGBRegressor(**self.model_configs['xgboost'])
            xgb_model.fit(X, y)
            trained_models['xgboost'] = xgb_model
            print("✓ XGBoost trained")
        except Exception as e:
            print(f"✗ XGBoost training failed: {e}")
        
        # LightGBM
        try:
            lgb_model = lgb.LGBMRegressor(**self.model_configs['lightgbm'])
            lgb_model.fit(X, y)
            trained_models['lightgbm'] = lgb_model
            print("✓ LightGBM trained")
        except Exception as e:
            print(f"✗ LightGBM training failed: {e}")
        
        # Random Forest
        try:
            rf_model = RandomForestRegressor(**self.model_configs['random_forest'])
            rf_model.fit(X, y)
            trained_models['random_forest'] = rf_model
            print("✓ Random Forest trained")
        except Exception as e:
            print(f"✗ Random Forest training failed: {e}")
        
        # Gradient Boosting
        try:
            gb_model = GradientBoostingRegressor(**self.model_configs['gradient_boosting'])
            gb_model.fit(X, y)
            trained_models['gradient_boosting'] = gb_model
            print("✓ Gradient Boosting trained")
        except Exception as e:
            print(f"✗ Gradient Boosting training failed: {e}")
        
        # Ridge Regression
        try:
            ridge_model = Ridge(**self.model_configs['ridge'])
            ridge_model.fit(X, y)
            trained_models['ridge'] = ridge_model
            print("✓ Ridge Regression trained")
        except Exception as e:
            print(f"✗ Ridge Regression training failed: {e}")
        
        # Lasso Regression
        try:
            lasso_model = Lasso(**self.model_configs['lasso'])
            lasso_model.fit(X, y)
            trained_models['lasso'] = lasso_model
            print("✓ Lasso Regression trained")
        except Exception as e:
            print(f"✗ Lasso Regression training failed: {e}")
        
        # Support Vector Regression
        try:
            svr_model = SVR(**self.model_configs['svr'])
            svr_model.fit(X, y)
            trained_models['svr'] = svr_model
            print("✓ SVR trained")
        except Exception as e:
            print(f"✗ SVR training failed: {e}")
        
        return trained_models
    
    def evaluate_models(self, models: Dict[str, object], X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict]:
        """Evaluate individual models using walk-forward validation"""
        
        evaluation_results = {}
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            
            scores = {
                'mse': [],
                'mae': [],
                'r2': [],
                'predictions': [],
                'actuals': []
            }
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train and predict
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                scores['mse'].append(mean_squared_error(y_test, y_pred))
                scores['mae'].append(mean_absolute_error(y_test, y_pred))
                scores['r2'].append(r2_score(y_test, y_pred))
                scores['predictions'].extend(y_pred)
                scores['actuals'].extend(y_test)
            
            # Aggregate results
            evaluation_results[model_name] = {
                'mean_mse': np.mean(scores['mse']),
                'std_mse': np.std(scores['mse']),
                'mean_mae': np.mean(scores['mae']),
                'std_mae': np.std(scores['mae']),
                'mean_r2': np.mean(scores['r2']),
                'std_r2': np.std(scores['r2']),
                'predictions': scores['predictions'],
                'actuals': scores['actuals']
            }
        
        return evaluation_results
    
    def calculate_model_weights(self, evaluation_results: Dict[str, Dict], method: str = 'inverse_mse') -> Dict[str, float]:
        """Calculate ensemble weights based on model performance"""
        
        if method == 'inverse_mse':
            # Weight inversely proportional to MSE
            mse_scores = {name: results['mean_mse'] for name, results in evaluation_results.items()}
            total_inverse_mse = sum(1 / mse for mse in mse_scores.values())
            weights = {name: (1 / mse) / total_inverse_mse for name, mse in mse_scores.items()}
            
        elif method == 'r2_based':
            # Weight based on R-squared scores
            r2_scores = {name: max(0, results['mean_r2']) for name, results in evaluation_results.items()}
            total_r2 = sum(r2_scores.values())
            weights = {name: r2 / total_r2 for name, r2 in r2_scores.items()}
            
        elif method == 'equal':
            # Equal weights
            n_models = len(evaluation_results)
            weights = {name: 1.0 / n_models for name in evaluation_results.keys()}
            
        else:
            weights = {name: 1.0 / len(evaluation_results) for name in evaluation_results.keys()}
        
        return weights
    
    def train_ensemble(self, X: pd.DataFrame, y: pd.Series, weight_method: str = 'inverse_mse') -> Dict:
        """Train the complete ensemble system"""
        
        print("=" * 60)
        print("ENSEMBLE ML SYSTEM TRAINING")
        print("=" * 60)
        
        # Train individual models
        print("\nTraining individual models...")
        self.models = self.train_individual_models(X, y)
        
        # Evaluate models
        print("\nEvaluating models...")
        evaluation_results = self.evaluate_models(self.models, X, y)
        
        # Calculate weights
        print("\nCalculating ensemble weights...")
        self.model_weights = self.calculate_model_weights(evaluation_results, weight_method)
        
        # Store results
        self.performance_metrics = evaluation_results
        
        # Print results
        self.print_ensemble_results()
        
        return {
            'models': self.models,
            'weights': self.model_weights,
            'performance': evaluation_results
        }
    
    def predict_ensemble(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make ensemble predictions with confidence intervals"""
        
        if not self.models or not self.model_weights:
            raise ValueError("Models not trained. Run train_ensemble() first.")
        
        predictions = {}
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            if model_name in self.model_weights:
                pred = model.predict(X_scaled)
                predictions[model_name] = pred
        
        # Calculate weighted ensemble prediction
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            weight = self.model_weights[model_name]
            ensemble_pred += weight * pred
        
        # Calculate prediction confidence (standard deviation across models)
        pred_array = np.array(list(predictions.values()))
        confidence = np.std(pred_array, axis=0)
        
        return ensemble_pred, confidence
    
    def get_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """Get feature importance from all models"""
        
        importance_dict = {}
        
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                elif hasattr(model, 'coef_'):
                    importance = np.abs(model.coef_)
                else:
                    continue
                
                importance_df = pd.DataFrame({
                    'feature': range(len(importance)),
                    'importance': importance
                }).sort_values('importance', ascending=False)
                
                importance_dict[model_name] = importance_df
                
            except Exception as e:
                print(f"Could not get feature importance for {model_name}: {e}")
        
        return importance_dict
    
    def print_ensemble_results(self):
        """Print ensemble training results"""
        
        print("\n" + "=" * 60)
        print("ENSEMBLE TRAINING RESULTS")
        print("=" * 60)
        
        print("\n📊 MODEL PERFORMANCE:")
        for model_name, results in self.performance_metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  MSE: {results['mean_mse']:.6f} ± {results['std_mse']:.6f}")
            print(f"  MAE: {results['mean_mae']:.6f} ± {results['std_mae']:.6f}")
            print(f"  R²:  {results['mean_r2']:.4f} ± {results['std_r2']:.4f}")
        
        print(f"\n⚖️  ENSEMBLE WEIGHTS:")
        for model_name, weight in self.model_weights.items():
            print(f"  {model_name}: {weight:.4f}")
        
        # Calculate ensemble performance
        ensemble_predictions = []
        ensemble_actuals = []
        
        for model_name, results in self.performance_metrics.items():
            weight = self.model_weights[model_name]
            ensemble_predictions.extend(np.array(results['predictions']) * weight)
            ensemble_actuals.extend(results['actuals'])
        
        # Average predictions for ensemble
        ensemble_pred_array = np.array(ensemble_predictions).reshape(len(self.models), -1)
        ensemble_pred_final = np.mean(ensemble_pred_array, axis=0)
        ensemble_actual_final = np.array(ensemble_actuals)[:len(ensemble_pred_final)]
        
        ensemble_mse = mean_squared_error(ensemble_actual_final, ensemble_pred_final)
        ensemble_mae = mean_absolute_error(ensemble_actual_final, ensemble_pred_final)
        ensemble_r2 = r2_score(ensemble_actual_final, ensemble_pred_final)
        
        print(f"\n🎯 ENSEMBLE PERFORMANCE:")
        print(f"  MSE: {ensemble_mse:.6f}")
        print(f"  MAE: {ensemble_mae:.6f}")
        print(f"  R²:  {ensemble_r2:.4f}")
        
        print("=" * 60)
    
    def save_ensemble(self, filepath: str):
        """Save ensemble model"""
        import pickle
        
        ensemble_data = {
            'models': self.models,
            'weights': self.model_weights,
            'performance': self.performance_metrics,
            'configs': self.model_configs,
            'timestamp': datetime.now()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        print(f"Ensemble saved to {filepath}")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble model"""
        import pickle
        
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        self.models = ensemble_data['models']
        self.model_weights = ensemble_data['weights']
        self.performance_metrics = ensemble_data['performance']
        self.model_configs = ensemble_data['configs']
        
        print(f"Ensemble loaded from {filepath}")
    
    def generate_trading_signals(self, features: pd.DataFrame, confidence_threshold: float = 0.1) -> Dict[str, float]:
        """Generate trading signals from ensemble predictions"""
        
        if not self.models:
            raise ValueError("Models not trained. Run train_ensemble() first.")
        
        # Get predictions and confidence
        predictions, confidence = self.predict_ensemble(features)
        
        # Generate signals based on predictions and confidence
        signals = {}
        
        for i, ticker in enumerate(features.index):
            pred = predictions[i]
            conf = confidence[i]
            
            # Only generate signal if confidence is above threshold
            if conf < confidence_threshold:
                # Strong signal
                if pred > 0.02:  # 2% expected return
                    signals[ticker] = min(0.05, pred * 10)  # Scale prediction to position size
                elif pred < -0.02:
                    signals[ticker] = max(-0.05, pred * 10)
                else:
                    signals[ticker] = 0.0
            else:
                # Weak signal or no signal
                signals[ticker] = 0.0
        
        return signals

def main():
    """Test the ensemble ML system"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic features
    features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    # Generate synthetic target (forward returns)
    target = pd.Series(
        np.random.normal(0.001, 0.02, n_samples),  # Small positive drift
        index=features.index
    )
    
    # Add some signal to target
    target += 0.1 * features['feature_0'] + 0.05 * features['feature_1']
    
    # Initialize and train ensemble
    ensemble = EnsembleMLSystem(n_splits=3, horizon=5)
    results = ensemble.train_ensemble(features, target)
    
    # Generate trading signals
    signals = ensemble.generate_trading_signals(features.tail(10))
    print(f"\nGenerated signals for {len(signals)} tickers")
    
    # Save ensemble
    ensemble.save_ensemble('ensemble_model.pkl')

if __name__ == "__main__":
    main() 