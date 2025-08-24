import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from collections import deque
import json

from .feature_engine import FeatureEngine
from .model_manager import ModelManager

logger = logging.getLogger(__name__)

class PricePredictor:
    """
    High-level interface for price prediction combining feature engineering
    and machine learning models. Optimized for real-time trading applications.
    """
    
    def __init__(self, lookback_window: int = 50, prediction_horizons: List[int] = [1, 5, 15, 60]):
        self.lookback_window = lookback_window
        self.prediction_horizons = prediction_horizons  # seconds
        
        # Initialize components
        self.feature_engine = FeatureEngine(lookback_period=lookback_window)
        self.model_manager = ModelManager(lookback_window=lookback_window)
        
        # Data storage
        self.price_history = deque(maxlen=1000)
        self.feature_history = deque(maxlen=1000)
        self.prediction_cache = {}
        
        # Performance tracking
        self.prediction_accuracy = {horizon: deque(maxlen=1000) for horizon in prediction_horizons}
        self.confidence_scores = deque(maxlen=1000)
        
        # Configuration
        self.min_confidence_threshold = 0.6
        self.ensemble_threshold = 0.7
        self.retrain_interval = 3600  # seconds
        self.last_retrain = None
        
        logger.info(f"PricePredictor initialized with horizons: {prediction_horizons}")
    
    async def initialize(self, historical_data: pd.DataFrame):
        """Initialize the predictor with historical data."""
        logger.info("Initializing PricePredictor with historical data...")
        
        # Generate features from historical data
        features_df = self.feature_engine.generate_technical_features(historical_data)
        features_df = self.feature_engine.generate_statistical_features(features_df)
        
        # Prepare training data for each prediction horizon
        training_data = {}
        for horizon in self.prediction_horizons:
            X, y = self._prepare_training_data(features_df, horizon)
            if len(X) > 100:  # Minimum samples for training
                training_data[horizon] = (X, y)
        
        if not training_data:
            logger.warning("Insufficient data for model training")
            return
        
        # Initialize and train models for the primary horizon
        primary_horizon = self.prediction_horizons[0]
        if primary_horizon in training_data:
            X, y = training_data[primary_horizon]
            
            # Feature selection and scaling
            X_selected = self.feature_engine.select_features(X, y)
            self.feature_engine.fit_scalers(X_selected)
            X_scaled = self.feature_engine.transform_features(X_selected)
            
            # Initialize and train models
            feature_columns = X_scaled.columns.tolist()
            self.model_manager.initialize_models(feature_columns)
            await self.model_manager.train_models(X_scaled, y)
            
            logger.info(f"Models trained with {len(X_scaled)} samples and {len(feature_columns)} features")
        
        self.last_retrain = datetime.now()
    
    def _prepare_training_data(self, features_df: pd.DataFrame, horizon_seconds: int) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for a specific prediction horizon."""
        # Calculate future returns as target
        horizon_periods = max(1, horizon_seconds // 1)  # Assuming 1-second data
        future_returns = features_df['close'].pct_change(horizon_periods).shift(-horizon_periods)
        
        # Remove rows with NaN targets
        valid_mask = ~future_returns.isna()
        
        # Select feature columns (exclude price columns to avoid lookahead bias)
        feature_columns = [col for col in features_df.columns 
                          if col not in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
        
        X = features_df[feature_columns][valid_mask]
        y = future_returns[valid_mask]
        
        return X, y
    
    async def predict_price_movement(self, current_data: Dict, orderbook: Dict = None, 
                                   recent_trades: List[Dict] = None) -> Dict:
        """
        Generate price movement predictions for all configured horizons.
        
        Args:
            current_data: Current market data (price, volume, etc.)
            orderbook: Current order book data
            recent_trades: Recent trade data
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            # Extract current price and volume
            current_price = current_data.get('price', 0)
            current_volume = current_data.get('volume', 0)
            
            # Update price history
            self.price_history.append({
                'timestamp': datetime.now(),
                'price': current_price,
                'volume': current_volume
            })
            
            # Generate real-time features
            features = await self.feature_engine.process_realtime_features(
                current_price, current_volume, orderbook, {'recent_trades': recent_trades}
            )
            
            # Convert to DataFrame for model input
            feature_df = pd.DataFrame([features])
            
            # Apply feature scaling
            if hasattr(self.feature_engine, 'scalers') and self.feature_engine.scalers:
                feature_df = self.feature_engine.transform_features(feature_df)
            
            # Generate predictions from all models
            model_predictions = await self.model_manager.predict(feature_df, use_ensemble=True)
            
            # Calculate confidence scores
            confidence = self._calculate_prediction_confidence(model_predictions)
            
            # Prepare final predictions
            predictions = {}
            for horizon in self.prediction_horizons:
                # For now, use the same model prediction for all horizons
                # In production, you'd have separate models for each horizon
                base_prediction = model_predictions.get('ensemble', 0)
                
                # Adjust prediction based on horizon (longer horizons = more uncertainty)
                horizon_factor = np.sqrt(horizon / self.prediction_horizons[0])
                adjusted_prediction = base_prediction / horizon_factor
                
                predictions[f'{horizon}s'] = {
                    'direction': 1 if adjusted_prediction > 0 else -1,
                    'magnitude': abs(adjusted_prediction),
                    'confidence': confidence * (1 / horizon_factor),
                    'raw_prediction': adjusted_prediction
                }
            
            # Add overall market assessment
            predictions['market_assessment'] = self._assess_market_conditions(features, model_predictions)
            
            # Cache predictions
            self.prediction_cache = {
                'timestamp': datetime.now(),
                'predictions': predictions,
                'features': features
            }
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in predict_price_movement: {e}")
            return self._get_default_predictions()
    
    def _calculate_prediction_confidence(self, model_predictions: Dict[str, float]) -> float:
        """Calculate confidence score based on model agreement."""
        if len(model_predictions) < 2:
            return 0.5
        
        # Remove ensemble prediction for agreement calculation
        individual_predictions = {k: v for k, v in model_predictions.items() if k != 'ensemble'}
        
        if len(individual_predictions) < 2:
            return 0.5
        
        predictions = list(individual_predictions.values())
        
        # Calculate agreement based on standard deviation
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)
        
        # Normalize confidence (lower std = higher confidence)
        if abs(pred_mean) > 0:
            confidence = 1.0 / (1.0 + pred_std / abs(pred_mean))
        else:
            confidence = 1.0 / (1.0 + pred_std)
        
        # Ensure confidence is between 0 and 1
        confidence = max(0.1, min(0.95, confidence))
        
        return confidence
    
    def _assess_market_conditions(self, features: Dict, predictions: Dict) -> Dict:
        """Assess overall market conditions based on features and predictions."""
        assessment = {
            'volatility': 'normal',
            'trend': 'neutral',
            'momentum': 'neutral',
            'liquidity': 'normal',
            'risk_level': 'medium'
        }
        
        # Volatility assessment
        if 'volatility_5' in features:
            vol = features['volatility_5']
            if vol > 0.02:  # 2% volatility
                assessment['volatility'] = 'high'
            elif vol < 0.005:  # 0.5% volatility
                assessment['volatility'] = 'low'
        
        # Trend assessment
        if 'price_to_sma_20' in features:
            price_to_sma = features['price_to_sma_20']
            if price_to_sma > 0.02:  # 2% above SMA
                assessment['trend'] = 'bullish'
            elif price_to_sma < -0.02:  # 2% below SMA
                assessment['trend'] = 'bearish'
        
        # Momentum assessment
        if 'rsi' in features:
            rsi = features['rsi']
            if rsi > 70:
                assessment['momentum'] = 'overbought'
            elif rsi < 30:
                assessment['momentum'] = 'oversold'
            elif rsi > 55:
                assessment['momentum'] = 'bullish'
            elif rsi < 45:
                assessment['momentum'] = 'bearish'
        
        # Liquidity assessment
        if 'volume_ratio' in features:
            vol_ratio = features['volume_ratio']
            if vol_ratio > 2.0:
                assessment['liquidity'] = 'high'
            elif vol_ratio < 0.5:
                assessment['liquidity'] = 'low'
        
        # Risk level assessment
        risk_factors = 0
        if assessment['volatility'] == 'high':
            risk_factors += 1
        if assessment['liquidity'] == 'low':
            risk_factors += 1
        if 'spread_bps' in features and features['spread_bps'] > 10:  # Wide spread
            risk_factors += 1
        
        if risk_factors >= 2:
            assessment['risk_level'] = 'high'
        elif risk_factors == 0:
            assessment['risk_level'] = 'low'
        
        return assessment
    
    def _get_default_predictions(self) -> Dict:
        """Return default predictions when prediction fails."""
        default_predictions = {}
        
        for horizon in self.prediction_horizons:
            default_predictions[f'{horizon}s'] = {
                'direction': 0,
                'magnitude': 0,
                'confidence': 0,
                'raw_prediction': 0
            }
        
        default_predictions['market_assessment'] = {
            'volatility': 'unknown',
            'trend': 'neutral',
            'momentum': 'neutral',
            'liquidity': 'unknown',
            'risk_level': 'high'
        }
        
        return default_predictions
    
    async def update_with_actual(self, actual_price: float, timestamp: datetime = None):
        """Update predictor with actual price for performance tracking."""
        if timestamp is None:
            timestamp = datetime.now()
        
        # Find corresponding predictions
        if self.prediction_cache and 'predictions' in self.prediction_cache:
            cache_time = self.prediction_cache['timestamp']
            
            for horizon in self.prediction_horizons:
                horizon_key = f'{horizon}s'
                if horizon_key in self.prediction_cache['predictions']:
                    # Check if enough time has passed for this horizon
                    time_diff = (timestamp - cache_time).total_seconds()
                    
                    if abs(time_diff - horizon) <= 5:  # Within 5 seconds of target horizon
                        predicted = self.prediction_cache['predictions'][horizon_key]['raw_prediction']
                        
                        # Calculate actual return
                        if len(self.price_history) > 0:
                            previous_price = self.price_history[-1]['price']
                            actual_return = (actual_price - previous_price) / previous_price
                            
                            # Calculate prediction accuracy
                            error = abs(predicted - actual_return)
                            accuracy = max(0, 1 - error)  # Simple accuracy metric
                            
                            self.prediction_accuracy[horizon].append(accuracy)
                            
                            # Update model performance
                            await self.model_manager.update_performance(
                                {'prediction': predicted}, actual_return
                            )
        
        # Check if retraining is needed
        if self.should_retrain():
            logger.info("Retraining models due to performance degradation or time interval")
            await self.retrain_models()
    
    def should_retrain(self) -> bool:
        """Determine if models should be retrained."""
        # Time-based retraining
        if self.last_retrain is None:
            return True
        
        time_since_retrain = (datetime.now() - self.last_retrain).total_seconds()
        if time_since_retrain > self.retrain_interval:
            return True
        
        # Performance-based retraining
        for horizon, accuracies in self.prediction_accuracy.items():
            if len(accuracies) >= 100:
                recent_accuracy = np.mean(list(accuracies)[-50:])
                if recent_accuracy < 0.4:  # Less than 40% accuracy
                    return True
        
        return self.model_manager.should_retrain()
    
    async def retrain_models(self):
        """Retrain models with recent data."""
        if len(self.price_history) < 200:
            logger.warning("Insufficient data for retraining")
            return
        
        try:
            # Convert price history to DataFrame
            df = pd.DataFrame(list(self.price_history))
            df['close'] = df['price']
            df['volume'] = df['volume']
            
            # Generate features
            features_df = self.feature_engine.generate_technical_features(df)
            features_df = self.feature_engine.generate_statistical_features(features_df)
            
            # Prepare training data
            primary_horizon = self.prediction_horizons[0]
            X, y = self._prepare_training_data(features_df, primary_horizon)
            
            if len(X) > 50:  # Minimum samples for retraining
                # Feature selection and scaling
                X_selected = self.feature_engine.select_features(X, y)
                X_scaled = self.feature_engine.transform_features(X_selected)
                
                # Retrain models
                await self.model_manager.train_models(X_scaled, y)
                
                self.last_retrain = datetime.now()
                logger.info(f"Models retrained with {len(X_scaled)} samples")
            
        except Exception as e:
            logger.error(f"Error during model retraining: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary of the predictor."""
        summary = {
            'prediction_accuracy': {},
            'model_performance': self.model_manager.get_model_performance_summary(),
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'total_predictions': len(self.confidence_scores)
        }
        
        # Calculate accuracy for each horizon
        for horizon, accuracies in self.prediction_accuracy.items():
            if accuracies:
                summary['prediction_accuracy'][f'{horizon}s'] = {
                    'mean_accuracy': np.mean(accuracies),
                    'recent_accuracy': np.mean(list(accuracies)[-50:]) if len(accuracies) >= 50 else np.mean(accuracies),
                    'total_predictions': len(accuracies)
                }
        
        return summary
    
    def get_feature_importance(self) -> Dict:
        """Get feature importance from the feature engine."""
        return self.feature_engine.feature_importance
    
    async def save_state(self, filepath: str):
        """Save predictor state to file."""
        state = {
            'lookback_window': self.lookback_window,
            'prediction_horizons': self.prediction_horizons,
            'last_retrain': self.last_retrain.isoformat() if self.last_retrain else None,
            'performance_summary': self.get_performance_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save models
        await self.model_manager.save_models()
        
        logger.info(f"Predictor state saved to {filepath}")
    
    async def load_state(self, filepath: str, feature_columns: List[str]):
        """Load predictor state from file."""
        try:
            with open(filepath, 'r') as f:
                state = json.load(f)
            
            self.lookback_window = state.get('lookback_window', self.lookback_window)
            self.prediction_horizons = state.get('prediction_horizons', self.prediction_horizons)
            
            if state.get('last_retrain'):
                self.last_retrain = datetime.fromisoformat(state['last_retrain'])
            
            # Load models
            await self.model_manager.load_models(feature_columns)
            
            logger.info(f"Predictor state loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading predictor state: {e}")