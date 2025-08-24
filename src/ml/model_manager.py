import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import joblib
import asyncio
from pathlib import Path
import logging
from datetime import datetime, timedelta
import json

# ML Libraries
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import xgboost as xgb
import lightgbm as lgb

# Deep Learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Deep learning models disabled.")

logger = logging.getLogger(__name__)

class LSTMPredictor(nn.Module):
    """LSTM model for price prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, 
                 dropout: float = 0.2, output_size: int = 1):
        super(LSTMPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last output
        out = self.fc(out)
        return out

class TransformerPredictor(nn.Module):
    """Transformer model for price prediction."""
    
    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 8, 
                 num_layers: int = 3, dropout: float = 0.1, output_size: int = 1):
        super(TransformerPredictor, self).__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = self._generate_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        
    def _generate_positional_encoding(self, max_len: int, d_model: int):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x):
        seq_len = x.size(1)
        x = self.input_projection(x)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(1):
            x += self.positional_encoding[:, :seq_len, :].to(x.device)
        
        x = self.transformer(x)
        x = self.dropout(x[:, -1, :])  # Take last output
        x = self.output_projection(x)
        return x

class ModelManager:
    """
    Manages multiple ML models for price prediction and signal generation.
    Supports ensemble methods, online learning, and model performance tracking.
    """
    
    def __init__(self, model_dir: str = "models", lookback_window: int = 50):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.lookback_window = lookback_window
        
        # Model registry
        self.models = {}
        self.model_performance = {}
        self.ensemble_weights = {}
        
        # Training configuration
        self.retrain_frequency = 3600  # seconds
        self.min_training_samples = 1000
        self.validation_split = 0.2
        
        # Performance tracking
        self.prediction_history = []
        self.actual_history = []
        
        logger.info(f"ModelManager initialized with model directory: {self.model_dir}")
    
    def initialize_models(self, feature_columns: List[str]):
        """Initialize all available models."""
        n_features = len(feature_columns)
        
        # Traditional ML models
        self.models['random_forest'] = RandomForestRegressor(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        
        self.models['gradient_boosting'] = GradientBoostingRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
        )
        
        self.models['xgboost'] = xgb.XGBRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1, 
            random_state=42, n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            random_state=42, n_jobs=-1, verbose=-1
        )
        
        self.models['ridge'] = Ridge(alpha=1.0)
        self.models['lasso'] = Lasso(alpha=0.1)
        
        # Deep learning models (if PyTorch is available)
        if TORCH_AVAILABLE:
            self.models['lstm'] = LSTMPredictor(
                input_size=n_features, hidden_size=64, num_layers=2
            )
            self.models['transformer'] = TransformerPredictor(
                input_size=n_features, d_model=64, nhead=8, num_layers=3
            )
        
        # Initialize performance tracking
        for model_name in self.models.keys():
            self.model_performance[model_name] = {
                'mse': [],
                'mae': [],
                'r2': [],
                'sharpe': [],
                'last_updated': None
            }
            self.ensemble_weights[model_name] = 1.0 / len(self.models)
        
        logger.info(f"Initialized {len(self.models)} models")
    
    async def train_models(self, X: pd.DataFrame, y: pd.Series, 
                          validation_data: Tuple[pd.DataFrame, pd.Series] = None):
        """Train all models with the provided data."""
        if len(X) < self.min_training_samples:
            logger.warning(f"Insufficient training samples: {len(X)} < {self.min_training_samples}")
            return
        
        # Prepare data
        X_clean, y_clean = self._clean_training_data(X, y)
        
        if validation_data:
            X_val, y_val = validation_data
            X_val_clean, y_val_clean = self._clean_training_data(X_val, y_val)
        else:
            # Split data for validation
            split_idx = int(len(X_clean) * (1 - self.validation_split))
            X_train, X_val_clean = X_clean[:split_idx], X_clean[split_idx:]
            y_train, y_val_clean = y_clean[:split_idx], y_clean[split_idx:]
            X_clean = X_train
            y_clean = y_train
        
        # Train each model
        training_tasks = []
        for model_name, model in self.models.items():
            task = self._train_single_model(
                model_name, model, X_clean, y_clean, X_val_clean, y_val_clean
            )
            training_tasks.append(task)
        
        # Execute training in parallel
        results = await asyncio.gather(*training_tasks, return_exceptions=True)
        
        # Update ensemble weights based on validation performance
        self._update_ensemble_weights()
        
        logger.info("Model training completed")
    
    async def _train_single_model(self, model_name: str, model: Any, 
                                 X_train: pd.DataFrame, y_train: pd.Series,
                                 X_val: pd.DataFrame, y_val: pd.Series):
        """Train a single model asynchronously."""
        try:
            if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                await self._train_deep_model(model_name, model, X_train, y_train, X_val, y_val)
            else:
                await self._train_sklearn_model(model_name, model, X_train, y_train, X_val, y_val)
            
            logger.info(f"Successfully trained {model_name}")
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
    
    async def _train_sklearn_model(self, model_name: str, model: Any,
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series):
        """Train scikit-learn compatible models."""
        # Run training in executor to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, model.fit, X_train, y_train)
        
        # Validate model
        y_pred = await loop.run_in_executor(None, model.predict, X_val)
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Update performance tracking
        self.model_performance[model_name]['mse'].append(mse)
        self.model_performance[model_name]['mae'].append(mae)
        self.model_performance[model_name]['r2'].append(r2)
        self.model_performance[model_name]['last_updated'] = datetime.now()
        
        # Save model
        model_path = self.model_dir / f"{model_name}.joblib"
        await loop.run_in_executor(None, joblib.dump, model, model_path)
    
    async def _train_deep_model(self, model_name: str, model: nn.Module,
                               X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series):
        """Train PyTorch deep learning models."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Prepare data for sequence models
        if model_name in ['lstm', 'transformer']:
            X_train_seq, y_train_seq = self._prepare_sequence_data(X_train, y_train)
            X_val_seq, y_val_seq = self._prepare_sequence_data(X_val, y_val)
        else:
            X_train_seq, y_train_seq = X_train.values, y_train.values
            X_val_seq, y_val_seq = X_val.values, y_val.values
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_seq).to(device)
        y_train_tensor = torch.FloatTensor(y_train_seq).to(device)
        X_val_tensor = torch.FloatTensor(X_val_seq).to(device)
        y_val_tensor = torch.FloatTensor(y_val_seq).to(device)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        batch_size = 32
        epochs = 100
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size]
                batch_y = y_train_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs.squeeze(), y_val_tensor)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_dir / f"{model_name}.pth")
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break
        
        # Calculate final metrics
        model.eval()
        with torch.no_grad():
            y_pred = model(X_val_tensor).squeeze().cpu().numpy()
            y_val_np = y_val_tensor.cpu().numpy()
        
        mse = mean_squared_error(y_val_np, y_pred)
        mae = mean_absolute_error(y_val_np, y_pred)
        r2 = r2_score(y_val_np, y_pred)
        
        # Update performance tracking
        self.model_performance[model_name]['mse'].append(mse)
        self.model_performance[model_name]['mae'].append(mae)
        self.model_performance[model_name]['r2'].append(r2)
        self.model_performance[model_name]['last_updated'] = datetime.now()
    
    def _prepare_sequence_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for sequence models (LSTM, Transformer)."""
        sequences = []
        targets = []
        
        for i in range(self.lookback_window, len(X)):
            sequences.append(X.iloc[i-self.lookback_window:i].values)
            targets.append(y.iloc[i])
        
        return np.array(sequences), np.array(targets)
    
    def _clean_training_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """Clean training data by removing NaN values and outliers."""
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask].copy()
        y_clean = y[mask].copy()
        
        # Remove extreme outliers in target variable
        q1, q3 = y_clean.quantile([0.01, 0.99])
        outlier_mask = (y_clean >= q1) & (y_clean <= q3)
        X_clean = X_clean[outlier_mask]
        y_clean = y_clean[outlier_mask]
        
        return X_clean, y_clean
    
    async def predict(self, X: pd.DataFrame, use_ensemble: bool = True) -> Dict[str, float]:
        """Generate predictions from all models."""
        predictions = {}
        
        # Get predictions from each model
        for model_name, model in self.models.items():
            try:
                if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                    pred = await self._predict_deep_model(model_name, model, X)
                else:
                    pred = await self._predict_sklearn_model(model_name, model, X)
                
                predictions[model_name] = pred
                
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = 0.0
        
        # Ensemble prediction
        if use_ensemble and len(predictions) > 1:
            ensemble_pred = sum(
                pred * self.ensemble_weights.get(model_name, 0)
                for model_name, pred in predictions.items()
            )
            predictions['ensemble'] = ensemble_pred
        
        return predictions
    
    async def _predict_sklearn_model(self, model_name: str, model: Any, X: pd.DataFrame) -> float:
        """Generate prediction from scikit-learn model."""
        loop = asyncio.get_event_loop()
        
        # Take the last row for prediction
        X_last = X.iloc[-1:].fillna(0)
        prediction = await loop.run_in_executor(None, model.predict, X_last)
        
        return float(prediction[0])
    
    async def _predict_deep_model(self, model_name: str, model: nn.Module, X: pd.DataFrame) -> float:
        """Generate prediction from PyTorch model."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        # Prepare sequence data
        if len(X) >= self.lookback_window:
            X_seq = X.iloc[-self.lookback_window:].fillna(0).values
            X_seq = X_seq.reshape(1, self.lookback_window, -1)
        else:
            # Pad with zeros if insufficient data
            X_padded = np.zeros((self.lookback_window, X.shape[1]))
            X_available = X.fillna(0).values
            X_padded[-len(X_available):] = X_available
            X_seq = X_padded.reshape(1, self.lookback_window, -1)
        
        X_tensor = torch.FloatTensor(X_seq).to(device)
        
        with torch.no_grad():
            prediction = model(X_tensor)
            return float(prediction.squeeze().cpu().numpy())
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance."""
        # Calculate weights based on inverse of recent MSE
        recent_performance = {}
        
        for model_name, perf in self.model_performance.items():
            if perf['mse']:
                # Use recent performance (last 5 evaluations)
                recent_mse = np.mean(perf['mse'][-5:])
                recent_performance[model_name] = 1.0 / (recent_mse + 1e-8)
            else:
                recent_performance[model_name] = 1.0
        
        # Normalize weights
        total_weight = sum(recent_performance.values())
        if total_weight > 0:
            for model_name in self.ensemble_weights:
                self.ensemble_weights[model_name] = recent_performance[model_name] / total_weight
        
        logger.info(f"Updated ensemble weights: {self.ensemble_weights}")
    
    async def update_performance(self, predictions: Dict[str, float], actual: float):
        """Update model performance with new actual value."""
        self.actual_history.append(actual)
        
        for model_name, pred in predictions.items():
            if model_name != 'ensemble':  # Don't track ensemble separately
                error = abs(pred - actual)
                
                # Update rolling performance metrics
                if model_name not in self.model_performance:
                    self.model_performance[model_name] = {'errors': []}
                
                if 'errors' not in self.model_performance[model_name]:
                    self.model_performance[model_name]['errors'] = []
                
                self.model_performance[model_name]['errors'].append(error)
                
                # Keep only recent errors (last 1000)
                if len(self.model_performance[model_name]['errors']) > 1000:
                    self.model_performance[model_name]['errors'] = \
                        self.model_performance[model_name]['errors'][-1000:]
        
        # Update ensemble weights periodically
        if len(self.actual_history) % 100 == 0:
            self._update_ensemble_weights()
    
    def get_model_performance_summary(self) -> Dict:
        """Get summary of model performance."""
        summary = {}
        
        for model_name, perf in self.model_performance.items():
            model_summary = {}
            
            if 'errors' in perf and perf['errors']:
                errors = np.array(perf['errors'])
                model_summary['mean_error'] = np.mean(errors)
                model_summary['std_error'] = np.std(errors)
                model_summary['median_error'] = np.median(errors)
            
            if perf['mse']:
                model_summary['latest_mse'] = perf['mse'][-1]
                model_summary['latest_r2'] = perf['r2'][-1] if perf['r2'] else None
            
            model_summary['last_updated'] = perf['last_updated']
            model_summary['ensemble_weight'] = self.ensemble_weights.get(model_name, 0)
            
            summary[model_name] = model_summary
        
        return summary
    
    async def save_models(self):
        """Save all trained models to disk."""
        for model_name, model in self.models.items():
            try:
                if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                    model_path = self.model_dir / f"{model_name}.pth"
                    torch.save(model.state_dict(), model_path)
                else:
                    model_path = self.model_dir / f"{model_name}.joblib"
                    joblib.dump(model, model_path)
                
                logger.info(f"Saved {model_name} to {model_path}")
                
            except Exception as e:
                logger.error(f"Error saving {model_name}: {e}")
        
        # Save performance history
        performance_path = self.model_dir / "performance_history.json"
        with open(performance_path, 'w') as f:
            # Convert datetime objects to strings for JSON serialization
            serializable_perf = {}
            for model_name, perf in self.model_performance.items():
                serializable_perf[model_name] = {}
                for key, value in perf.items():
                    if key == 'last_updated' and value:
                        serializable_perf[model_name][key] = value.isoformat()
                    else:
                        serializable_perf[model_name][key] = value
            
            json.dump(serializable_perf, f, indent=2)
    
    async def load_models(self, feature_columns: List[str]):
        """Load previously trained models from disk."""
        self.initialize_models(feature_columns)
        
        for model_name in self.models.keys():
            try:
                if model_name in ['lstm', 'transformer'] and TORCH_AVAILABLE:
                    model_path = self.model_dir / f"{model_name}.pth"
                    if model_path.exists():
                        self.models[model_name].load_state_dict(torch.load(model_path))
                        logger.info(f"Loaded {model_name} from {model_path}")
                else:
                    model_path = self.model_dir / f"{model_name}.joblib"
                    if model_path.exists():
                        self.models[model_name] = joblib.load(model_path)
                        logger.info(f"Loaded {model_name} from {model_path}")
                        
            except Exception as e:
                logger.error(f"Error loading {model_name}: {e}")
        
        # Load performance history
        performance_path = self.model_dir / "performance_history.json"
        if performance_path.exists():
            try:
                with open(performance_path, 'r') as f:
                    loaded_perf = json.load(f)
                    
                for model_name, perf in loaded_perf.items():
                    if model_name in self.model_performance:
                        for key, value in perf.items():
                            if key == 'last_updated' and value:
                                self.model_performance[model_name][key] = datetime.fromisoformat(value)
                            else:
                                self.model_performance[model_name][key] = value
                
                logger.info("Loaded performance history")
                
            except Exception as e:
                logger.error(f"Error loading performance history: {e}")
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained."""
        # Check if enough time has passed
        now = datetime.now()
        
        for model_name, perf in self.model_performance.items():
            last_updated = perf.get('last_updated')
            if last_updated is None:
                return True  # Never trained
            
            time_since_update = (now - last_updated).total_seconds()
            if time_since_update > self.retrain_frequency:
                return True
        
        # Check if performance has degraded
        for model_name, perf in self.model_performance.items():
            if 'errors' in perf and len(perf['errors']) >= 100:
                recent_errors = perf['errors'][-50:]  # Last 50 predictions
                older_errors = perf['errors'][-100:-50]  # Previous 50 predictions
                
                if np.mean(recent_errors) > np.mean(older_errors) * 1.2:  # 20% degradation
                    logger.info(f"Performance degradation detected for {model_name}")
                    return True
        
        return False