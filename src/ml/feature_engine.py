import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import talib
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from numba import jit, njit
import logging
from collections import deque
import asyncio

logger = logging.getLogger(__name__)

class FeatureEngine:
    """
    Advanced feature engineering for cryptocurrency trading.
    Generates technical indicators, market microstructure features,
    and statistical features optimized for high-frequency trading.
    """
    
    def __init__(self, lookback_period: int = 100, max_features: int = 50):
        self.lookback_period = lookback_period
        self.max_features = max_features
        self.scalers = {}
        self.feature_selector = None
        self.feature_importance = {}
        self.price_buffer = deque(maxlen=lookback_period)
        self.volume_buffer = deque(maxlen=lookback_period)
        self.feature_cache = {}
        
    @njit
    def _fast_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Optimized Simple Moving Average calculation using Numba."""
        n = len(prices)
        sma = np.empty(n)
        sma[:window-1] = np.nan
        
        for i in range(window-1, n):
            sma[i] = np.mean(prices[i-window+1:i+1])
        return sma
    
    @njit
    def _fast_ema(self, prices: np.ndarray, window: int) -> np.ndarray:
        """Optimized Exponential Moving Average calculation using Numba."""
        alpha = 2.0 / (window + 1.0)
        ema = np.empty_like(prices)
        ema[0] = prices[0]
        
        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        return ema
    
    @njit
    def _fast_rsi(self, prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Optimized RSI calculation using Numba."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        
        avg_gains = np.empty_like(prices)
        avg_losses = np.empty_like(prices)
        avg_gains[:window] = np.nan
        avg_losses[:window] = np.nan
        
        if len(gains) >= window:
            avg_gains[window] = np.mean(gains[:window])
            avg_losses[window] = np.mean(losses[:window])
            
            for i in range(window + 1, len(prices)):
                avg_gains[i] = (avg_gains[i-1] * (window - 1) + gains[i-1]) / window
                avg_losses[i] = (avg_losses[i-1] * (window - 1) + losses[i-1]) / window
        
        rs = avg_gains / avg_losses
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi
    
    def generate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive technical analysis features optimized for speed."""
        features = df.copy()
        close_prices = df['close'].values
        high_prices = df['high'].values
        low_prices = df['low'].values
        volume = df['volume'].values
        
        # Price-based features
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Multi-timeframe momentum
        for period in [3, 5, 10, 20, 50]:
            features[f'momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
            features[f'price_change_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Volatility features
        for window in [5, 10, 20]:
            features[f'volatility_{window}'] = features['returns'].rolling(window).std()
            features[f'volatility_rank_{window}'] = features[f'volatility_{window}'].rolling(100).rank(pct=True)
        
        # Volume features
        features['volume_sma_10'] = self._fast_sma(volume, 10)
        features['volume_sma_20'] = self._fast_sma(volume, 20)
        features['volume_ratio'] = volume / features['volume_sma_10']
        features['volume_momentum'] = df['volume'] / df['volume'].shift(5) - 1
        features['volume_price_trend'] = features['returns'] * features['volume_ratio']
        
        # Technical indicators using optimized functions
        features['rsi_14'] = self._fast_rsi(close_prices, 14)
        features['rsi_7'] = self._fast_rsi(close_prices, 7)
        features['rsi_21'] = self._fast_rsi(close_prices, 21)
        
        # Moving averages
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = self._fast_sma(close_prices, period)
            features[f'ema_{period}'] = self._fast_ema(close_prices, period)
            features[f'price_to_sma_{period}'] = df['close'] / features[f'sma_{period}'] - 1
            features[f'price_to_ema_{period}'] = df['close'] / features[f'ema_{period}'] - 1
        
        # Bollinger Bands
        features['bb_middle'] = features['sma_20']
        bb_std = df['close'].rolling(20).std()
        features['bb_upper'] = features['bb_middle'] + (bb_std * 2)
        features['bb_lower'] = features['bb_middle'] - (bb_std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / features['bb_middle']
        features['bb_position'] = (df['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        features['bb_squeeze'] = (features['bb_width'] < features['bb_width'].rolling(20).quantile(0.1)).astype(int)
        
        # MACD
        ema_12 = self._fast_ema(close_prices, 12)
        ema_26 = self._fast_ema(close_prices, 26)
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = self._fast_ema(features['macd'].values, 9)
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        features['macd_cross'] = np.where(features['macd'] > features['macd_signal'], 1, -1)
        
        # Stochastic Oscillator
        lowest_low = df['low'].rolling(14).min()
        highest_high = df['high'].rolling(14).max()
        features['stoch_k'] = 100 * (df['close'] - lowest_low) / (highest_high - lowest_low)
        features['stoch_d'] = features['stoch_k'].rolling(3).mean()
        
        # Average True Range
        tr1 = df['high'] - df['low']
        tr2 = abs(df['high'] - df['close'].shift(1))
        tr3 = abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        features['atr'] = true_range.rolling(14).mean()
        features['atr_ratio'] = true_range / features['atr']
        
        # Support and Resistance levels
        features['resistance'] = df['high'].rolling(20).max()
        features['support'] = df['low'].rolling(20).min()
        features['distance_to_resistance'] = (features['resistance'] - df['close']) / df['close']
        features['distance_to_support'] = (df['close'] - features['support']) / df['close']
        
        return features
    
    def generate_microstructure_features(self, orderbook_data: Dict, trade_data: Dict = None) -> Dict:
        """Generate market microstructure features from order book and trade data."""
        features = {}
        
        if 'bids' in orderbook_data and 'asks' in orderbook_data:
            bids = np.array(orderbook_data['bids'])
            asks = np.array(orderbook_data['asks'])
            
            if len(bids) > 0 and len(asks) > 0:
                # Basic spread features
                best_bid = bids[0][0]
                best_ask = asks[0][0]
                mid_price = (best_bid + best_ask) / 2
                
                features['spread'] = best_ask - best_bid
                features['spread_bps'] = (features['spread'] / mid_price) * 10000
                features['mid_price'] = mid_price
                
                # Order book imbalance at multiple levels
                for levels in [1, 5, 10]:
                    bid_volume = sum([bid[1] for bid in bids[:levels]])
                    ask_volume = sum([ask[1] for ask in asks[:levels]])
                    total_volume = bid_volume + ask_volume
                    
                    if total_volume > 0:
                        features[f'imbalance_{levels}'] = (bid_volume - ask_volume) / total_volume
                        features[f'bid_ratio_{levels}'] = bid_volume / total_volume
                    else:
                        features[f'imbalance_{levels}'] = 0
                        features[f'bid_ratio_{levels}'] = 0.5
                
                # Order book depth and liquidity
                features['bid_depth_5'] = sum([bid[0] * bid[1] for bid in bids[:5]])
                features['ask_depth_5'] = sum([ask[0] * ask[1] for ask in asks[:5]])
                features['total_depth_5'] = features['bid_depth_5'] + features['ask_depth_5']
                
                if features['ask_depth_5'] > 0:
                    features['depth_ratio'] = features['bid_depth_5'] / features['ask_depth_5']
                else:
                    features['depth_ratio'] = 1.0
                
                # Price impact estimation for different trade sizes
                for trade_size in [1000, 5000, 10000]:  # USD equivalent
                    features[f'bid_impact_{trade_size}'] = self._calculate_price_impact(bids, 'bid', trade_size)
                    features[f'ask_impact_{trade_size}'] = self._calculate_price_impact(asks, 'ask', trade_size)
                
                # Order book slope (price elasticity)
                features['bid_slope'] = self._calculate_orderbook_slope(bids[:10])
                features['ask_slope'] = self._calculate_orderbook_slope(asks[:10])
                
                # Weighted mid price
                if bid_volume + ask_volume > 0:
                    features['weighted_mid'] = (best_bid * ask_volume + best_ask * bid_volume) / (bid_volume + ask_volume)
                else:
                    features['weighted_mid'] = mid_price
        
        # Trade flow features
        if trade_data:
            features.update(self._generate_trade_flow_features(trade_data))
        
        return features
    
    def _calculate_price_impact(self, orders: np.ndarray, side: str, trade_size: float) -> float:
        """Calculate estimated price impact for a given trade size."""
        if len(orders) == 0:
            return 0
            
        cumulative_volume = 0
        weighted_price = 0
        reference_price = orders[0][0]
        
        for price, volume in orders:
            volume_usd = price * volume
            if cumulative_volume >= trade_size:
                break
            
            volume_to_use = min(volume_usd, trade_size - cumulative_volume)
            weighted_price += price * (volume_to_use / price)  # Convert back to base currency
            cumulative_volume += volume_to_use
        
        if cumulative_volume == 0:
            return 0
            
        avg_price = weighted_price / (cumulative_volume / reference_price)
        
        if side == 'bid':
            return (reference_price - avg_price) / reference_price
        else:
            return (avg_price - reference_price) / reference_price
    
    def _calculate_orderbook_slope(self, orders: np.ndarray) -> float:
        """Calculate the slope of the order book (price elasticity)."""
        if len(orders) < 2:
            return 0
        
        prices = orders[:, 0]
        volumes = orders[:, 1]
        cumulative_volume = np.cumsum(volumes)
        
        if len(prices) > 1:
            # Linear regression to find slope
            price_changes = np.diff(prices)
            volume_changes = np.diff(cumulative_volume)
            
            if np.std(volume_changes) > 0:
                correlation = np.corrcoef(price_changes, volume_changes)[0, 1]
                slope = correlation * (np.std(price_changes) / np.std(volume_changes))
                return slope
        
        return 0
    
    def _generate_trade_flow_features(self, trade_data: Dict) -> Dict:
        """Generate features from recent trade flow."""
        features = {}
        
        if 'recent_trades' in trade_data:
            trades = trade_data['recent_trades']
            
            # Trade size distribution
            sizes = [trade['size'] for trade in trades]
            if sizes:
                features['avg_trade_size'] = np.mean(sizes)
                features['trade_size_std'] = np.std(sizes)
                features['large_trade_ratio'] = sum(1 for size in sizes if size > np.percentile(sizes, 90)) / len(sizes)
            
            # Buy/sell pressure
            buy_volume = sum(trade['size'] for trade in trades if trade['side'] == 'buy')
            sell_volume = sum(trade['size'] for trade in trades if trade['side'] == 'sell')
            total_volume = buy_volume + sell_volume
            
            if total_volume > 0:
                features['buy_pressure'] = buy_volume / total_volume
                features['sell_pressure'] = sell_volume / total_volume
                features['trade_imbalance'] = (buy_volume - sell_volume) / total_volume
            
            # Trade frequency
            if len(trades) > 1:
                timestamps = [trade['timestamp'] for trade in trades]
                time_diffs = np.diff(sorted(timestamps))
                features['avg_trade_interval'] = np.mean(time_diffs)
                features['trade_frequency'] = len(trades) / (max(timestamps) - min(timestamps)) if max(timestamps) > min(timestamps) else 0
        
        return features
    
    def generate_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate statistical and time-series features."""
        features = df.copy()
        returns = features['returns'].dropna()
        
        # Rolling statistical moments
        for window in [10, 20, 50]:
            features[f'skewness_{window}'] = returns.rolling(window).skew()
            features[f'kurtosis_{window}'] = returns.rolling(window).kurt()
            features[f'jarque_bera_{window}'] = self._rolling_jarque_bera(returns, window)
        
        # Autocorrelation features
        for lag in [1, 2, 5, 10]:
            features[f'autocorr_{lag}'] = returns.rolling(50).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else 0, raw=False
            )
        
        # Regime detection
        features['volatility_regime'] = self._detect_volatility_regime(returns)
        features['trend_regime'] = self._detect_trend_regime(df['close'])
        features['volume_regime'] = self._detect_volume_regime(df['volume'])
        
        # Fractal dimension
        features['fractal_dimension'] = self._calculate_fractal_dimension(df['close'])
        
        # Hurst exponent
        features['hurst_exponent'] = self._calculate_hurst_exponent(returns)
        
        return features
    
    def _rolling_jarque_bera(self, series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling Jarque-Bera test statistic."""
        def jarque_bera_stat(x):
            if len(x) < 4:
                return np.nan
            n = len(x)
            skew = x.skew()
            kurt = x.kurt()
            jb = (n / 6) * (skew**2 + (kurt**2) / 4)
            return jb
        
        return series.rolling(window).apply(jarque_bera_stat, raw=False)
    
    def _detect_volatility_regime(self, returns: pd.Series, window: int = 20) -> pd.Series:
        """Detect volatility regime using quantile-based classification."""
        vol = returns.rolling(window).std()
        vol_ma = vol.rolling(100).mean()
        vol_std = vol.rolling(100).std()
        
        # Standardize volatility
        vol_z = (vol - vol_ma) / vol_std
        
        # Classify into regimes
        regime = pd.Series(index=returns.index, dtype=int)
        regime[vol_z <= -0.5] = 0  # Low volatility
        regime[(vol_z > -0.5) & (vol_z <= 0.5)] = 1  # Normal volatility
        regime[vol_z > 0.5] = 2  # High volatility
        
        return regime
    
    def _detect_trend_regime(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Detect trend regime using moving average slopes."""
        ma = prices.rolling(window).mean()
        slope = ma.diff(5) / ma.shift(5)  # 5-period slope
        
        regime = pd.Series(index=prices.index, dtype=int)
        regime[slope <= -0.001] = 0  # Downtrend
        regime[(slope > -0.001) & (slope <= 0.001)] = 1  # Sideways
        regime[slope > 0.001] = 2  # Uptrend
        
        return regime
    
    def _detect_volume_regime(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Detect volume regime."""
        vol_ma = volume.rolling(window).mean()
        vol_ratio = volume / vol_ma
        
        regime = pd.Series(index=volume.index, dtype=int)
        regime[vol_ratio <= 0.7] = 0  # Low volume
        regime[(vol_ratio > 0.7) & (vol_ratio <= 1.3)] = 1  # Normal volume
        regime[vol_ratio > 1.3] = 2  # High volume
        
        return regime
    
    def _calculate_fractal_dimension(self, prices: pd.Series, window: int = 50) -> pd.Series:
        """Calculate rolling fractal dimension using Higuchi's method."""
        def higuchi_fd(x, k_max=10):
            if len(x) < k_max:
                return np.nan
            
            n = len(x)
            lk = []
            
            for k in range(1, k_max + 1):
                lm = []
                for m in range(k):
                    ll = 0
                    for i in range(1, int((n - m) / k)):
                        ll += abs(x[m + i * k] - x[m + (i - 1) * k])
                    ll = ll * (n - 1) / (((n - m) / k) * k) / k
                    lm.append(ll)
                lk.append(np.mean(lm))
            
            lk = np.array(lk)
            k_range = np.arange(1, k_max + 1)
            
            # Linear regression in log-log space
            log_k = np.log(k_range)
            log_lk = np.log(lk)
            
            if np.std(log_k) > 0:
                slope = np.corrcoef(log_k, log_lk)[0, 1] * (np.std(log_lk) / np.std(log_k))
                return -slope
            return np.nan
        
        return prices.rolling(window).apply(higuchi_fd, raw=True)
    
    def _calculate_hurst_exponent(self, returns: pd.Series, window: int = 100) -> pd.Series:
        """Calculate rolling Hurst exponent."""
        def hurst_exp(x):
            if len(x) < 10:
                return np.nan
            
            x = np.array(x)
            n = len(x)
            
            # Calculate cumulative deviations
            y = np.cumsum(x - np.mean(x))
            
            # Calculate R/S for different lags
            lags = range(2, min(n//2, 20))
            rs = []
            
            for lag in lags:
                # Split into non-overlapping windows
                num_windows = n // lag
                if num_windows == 0:
                    continue
                    
                rs_values = []
                for i in range(num_windows):
                    start_idx = i * lag
                    end_idx = start_idx + lag
                    window_y = y[start_idx:end_idx]
                    
                    if len(window_y) > 1:
                        r = np.max(window_y) - np.min(window_y)
                        s = np.std(x[start_idx:end_idx])
                        if s > 0:
                            rs_values.append(r / s)
                
                if rs_values:
                    rs.append(np.mean(rs_values))
            
            if len(rs) < 3:
                return np.nan
            
            # Linear regression in log space
            log_lags = np.log(lags[:len(rs)])
            log_rs = np.log(rs)
            
            if np.std(log_lags) > 0:
                hurst = np.corrcoef(log_lags, log_rs)[0, 1] * (np.std(log_rs) / np.std(log_lags))
                return hurst
            return np.nan
        
        return returns.rolling(window).apply(hurst_exp, raw=False)
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info') -> pd.DataFrame:
        """Select the most informative features."""
        if method == 'mutual_info':
            # Remove NaN values for feature selection
            mask = ~(X.isna().any(axis=1) | y.isna())
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) == 0:
                return X
            
            # Calculate mutual information
            mi_scores = mutual_info_regression(X_clean, y_clean, random_state=42)
            
            # Select top features
            selector = SelectKBest(score_func=mutual_info_regression, k=min(self.max_features, len(X.columns)))
            selector.fit(X_clean, y_clean)
            
            selected_features = X.columns[selector.get_support()]
            self.feature_importance = dict(zip(X.columns, mi_scores))
            
            return X[selected_features]
        
        return X
    
    def fit_scalers(self, features: pd.DataFrame):
        """Fit scalers for feature normalization."""
        for column in features.select_dtypes(include=[np.number]).columns:
            if features[column].std() > 0:  # Only scale if there's variation
                self.scalers[column] = RobustScaler()
                valid_data = features[column].dropna()
                if len(valid_data) > 0:
                    self.scalers[column].fit(valid_data.values.reshape(-1, 1))
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Apply fitted scalers to features."""
        scaled_features = features.copy()
        
        for column in features.select_dtypes(include=[np.number]).columns:
            if column in self.scalers:
                valid_mask = ~features[column].isna()
                if valid_mask.any():
                    scaled_values = self.scalers[column].transform(
                        features.loc[valid_mask, column].values.reshape(-1, 1)
                    ).flatten()
                    scaled_features.loc[valid_mask, column] = scaled_values
        
        return scaled_features
    
    async def process_realtime_features(self, price: float, volume: float, 
                                      orderbook: Dict = None, trades: Dict = None) -> Dict:
        """Process features in real-time with minimal latency."""
        # Update buffers
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        features = {}
        
        if len(self.price_buffer) >= 5:
            prices = np.array(self.price_buffer)
            volumes = np.array(self.volume_buffer)
            
            # Fast technical indicators
            features['rsi'] = self._fast_rsi(prices)[-1] if len(prices) >= 14 else 50
            features['sma_5'] = np.mean(prices[-5:])
            features['sma_20'] = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
            features['price_to_sma_5'] = price / features['sma_5'] - 1
            
            # Momentum
            if len(prices) >= 5:
                features['momentum_5'] = prices[-1] / prices[-5] - 1
            
            # Volume features
            features['volume_ratio'] = volume / np.mean(volumes[-10:]) if len(volumes) >= 10 else 1.0
        
        # Microstructure features
        if orderbook:
            micro_features = self.generate_microstructure_features(orderbook, trades)
            features.update(micro_features)
        
        return features