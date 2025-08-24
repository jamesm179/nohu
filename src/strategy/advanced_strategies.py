import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass
from enum import Enum

from src.models import Order, OrderSide
from src.ml.predictor import PricePredictor

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4

@dataclass
class TradingSignal:
    """Represents a trading signal with metadata."""
    product_id: str
    side: OrderSide
    strength: SignalStrength
    confidence: float
    price: float
    size: float
    strategy: str
    timestamp: datetime
    metadata: Dict = None

class AdvancedStrategyEngine:
    """
    Advanced trading strategies for cryptocurrency scalping.
    Implements multiple sophisticated strategies with ML integration.
    """
    
    def __init__(self, risk_engine, execution_handler):
        self.risk_engine = risk_engine
        self.execution_handler = execution_handler
        
        # Strategy components
        self.ml_predictor = PricePredictor()
        self.market_maker = MarketMakingStrategy()
        self.arbitrage_detector = ArbitrageDetector()
        self.momentum_strategy = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.volume_profile = VolumeProfileStrategy()
        
        # Signal aggregation
        self.signal_buffer = deque(maxlen=1000)
        self.active_positions = {}
        
        # Performance tracking
        self.strategy_performance = {}
        
        logger.info("AdvancedStrategyEngine initialized")
    
    async def process_market_data(self, trade_data: Dict, orderbook_data: Dict = None):
        """Process incoming market data and generate trading signals."""
        product_id = trade_data['product_id']
        
        # Generate signals from all strategies
        signals = []
        
        # ML-based prediction signal
        ml_signal = await self._generate_ml_signal(trade_data, orderbook_data)
        if ml_signal:
            signals.append(ml_signal)
        
        # Market making signals
        mm_signals = await self.market_maker.generate_signals(trade_data, orderbook_data)
        signals.extend(mm_signals)
        
        # Arbitrage signals
        arb_signals = await self.arbitrage_detector.detect_opportunities(trade_data)
        signals.extend(arb_signals)
        
        # Momentum signals
        momentum_signals = await self.momentum_strategy.analyze(trade_data)
        signals.extend(momentum_signals)
        
        # Mean reversion signals
        reversion_signals = await self.mean_reversion.analyze(trade_data)
        signals.extend(reversion_signals)
        
        # Volume profile signals
        volume_signals = await self.volume_profile.analyze(trade_data, orderbook_data)
        signals.extend(volume_signals)
        
        # Aggregate and filter signals
        final_signals = self._aggregate_signals(signals)
        
        # Execute approved signals
        for signal in final_signals:
            await self._execute_signal(signal)
    
    async def _generate_ml_signal(self, trade_data: Dict, orderbook_data: Dict) -> Optional[TradingSignal]:
        """Generate trading signal using ML predictions."""
        try:
            # Get ML predictions
            predictions = await self.ml_predictor.predict_price_movement(
                trade_data, orderbook_data
            )
            
            # Extract primary prediction (1-second horizon)
            primary_pred = predictions.get('1s', {})
            direction = primary_pred.get('direction', 0)
            confidence = primary_pred.get('confidence', 0)
            magnitude = primary_pred.get('magnitude', 0)
            
            # Only generate signal if confidence is high enough
            if confidence < 0.7 or magnitude < 0.001:  # 0.1% minimum movement
                return None
            
            # Determine signal strength based on confidence and magnitude
            if confidence > 0.9 and magnitude > 0.005:
                strength = SignalStrength.VERY_STRONG
            elif confidence > 0.8 and magnitude > 0.003:
                strength = SignalStrength.STRONG
            elif confidence > 0.7 and magnitude > 0.002:
                strength = SignalStrength.MODERATE
            else:
                strength = SignalStrength.WEAK
            
            side = OrderSide.BUY if direction > 0 else OrderSide.SELL
            
            # Calculate position size based on confidence and magnitude
            base_size = 0.01  # Base position size
            size_multiplier = min(confidence * magnitude * 10, 3.0)  # Max 3x base size
            position_size = base_size * size_multiplier
            
            return TradingSignal(
                product_id=trade_data['product_id'],
                side=side,
                strength=strength,
                confidence=confidence,
                price=trade_data['price'],
                size=position_size,
                strategy='ml_prediction',
                timestamp=datetime.now(),
                metadata={
                    'magnitude': magnitude,
                    'predictions': predictions
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating ML signal: {e}")
            return None
    
    def _aggregate_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Aggregate multiple signals for the same product."""
        if not signals:
            return []
        
        # Group signals by product_id
        signal_groups = {}
        for signal in signals:
            if signal.product_id not in signal_groups:
                signal_groups[signal.product_id] = []
            signal_groups[signal.product_id].append(signal)
        
        aggregated_signals = []
        
        for product_id, product_signals in signal_groups.items():
            # Calculate weighted average of signals
            buy_weight = 0
            sell_weight = 0
            total_confidence = 0
            
            for signal in product_signals:
                weight = signal.strength.value * signal.confidence
                total_confidence += signal.confidence
                
                if signal.side == OrderSide.BUY:
                    buy_weight += weight
                else:
                    sell_weight += weight
            
            # Determine final signal
            if buy_weight > sell_weight * 1.2:  # 20% threshold for signal
                final_side = OrderSide.BUY
                final_confidence = buy_weight / (buy_weight + sell_weight)
            elif sell_weight > buy_weight * 1.2:
                final_side = OrderSide.SELL
                final_confidence = sell_weight / (buy_weight + sell_weight)
            else:
                continue  # No clear signal
            
            # Use the most recent price
            latest_signal = max(product_signals, key=lambda s: s.timestamp)
            
            # Calculate aggregated position size
            total_size = sum(s.size for s in product_signals if s.side == final_side)
            
            aggregated_signal = TradingSignal(
                product_id=product_id,
                side=final_side,
                strength=SignalStrength.MODERATE,  # Conservative aggregation
                confidence=min(final_confidence, 0.95),
                price=latest_signal.price,
                size=min(total_size, 0.1),  # Cap at 10% of typical position
                strategy='aggregated',
                timestamp=datetime.now(),
                metadata={
                    'component_signals': len(product_signals),
                    'strategies': [s.strategy for s in product_signals]
                }
            )
            
            aggregated_signals.append(aggregated_signal)
        
        return aggregated_signals
    
    async def _execute_signal(self, signal: TradingSignal):
        """Execute a trading signal after risk checks."""
        try:
            # Create order
            order = Order(
                product_id=signal.product_id,
                side=signal.side,
                size=signal.size,
                price=signal.price
            )
            
            # Risk check
            if self.risk_engine.check_pre_trade_risk(order):
                # Execute order
                success = await self.execution_handler.execute_order(order)
                
                if success:
                    logger.info(f"Executed {signal.strategy} signal: {order}")
                    
                    # Track performance
                    self._track_signal_execution(signal, order)
                else:
                    logger.warning(f"Failed to execute signal: {signal.strategy}")
            else:
                logger.warning(f"Signal rejected by risk engine: {signal.strategy}")
                
        except Exception as e:
            logger.error(f"Error executing signal: {e}")
    
    def _track_signal_execution(self, signal: TradingSignal, order: Order):
        """Track signal execution for performance analysis."""
        if signal.strategy not in self.strategy_performance:
            self.strategy_performance[signal.strategy] = {
                'signals_generated': 0,
                'signals_executed': 0,
                'total_pnl': 0.0,
                'win_rate': 0.0
            }
        
        self.strategy_performance[signal.strategy]['signals_executed'] += 1


class MarketMakingStrategy:
    """Market making strategy for providing liquidity."""
    
    def __init__(self):
        self.target_spread = 0.001  # 0.1%
        self.inventory_target = 0.0
        self.max_position_ratio = 0.3
        self.quote_refresh_rate = 0.1  # seconds
        
        self.active_quotes = {}
        self.inventory = {}
        
    async def generate_signals(self, trade_data: Dict, orderbook_data: Dict) -> List[TradingSignal]:
        """Generate market making signals."""
        if not orderbook_data or 'bids' not in orderbook_data or 'asks' not in orderbook_data:
            return []
        
        signals = []
        product_id = trade_data['product_id']
        
        try:
            bids = orderbook_data['bids']
            asks = orderbook_data['asks']
            
            if not bids or not asks:
                return []
            
            best_bid = bids[0][0]
            best_ask = asks[0][0]
            mid_price = (best_bid + best_ask) / 2
            current_spread = (best_ask - best_bid) / mid_price
            
            # Only make markets when spread is wide enough
            if current_spread > self.target_spread * 2:
                # Calculate optimal bid/ask prices
                half_spread = self.target_spread / 2
                our_bid = mid_price * (1 - half_spread)
                our_ask = mid_price * (1 + half_spread)
                
                # Adjust for inventory
                current_inventory = self.inventory.get(product_id, 0)
                inventory_adjustment = current_inventory * 0.0001  # Small adjustment
                
                our_bid -= inventory_adjustment
                our_ask -= inventory_adjustment
                
                # Generate bid signal
                if our_bid > best_bid:
                    bid_signal = TradingSignal(
                        product_id=product_id,
                        side=OrderSide.BUY,
                        strength=SignalStrength.MODERATE,
                        confidence=0.8,
                        price=our_bid,
                        size=0.01,
                        strategy='market_making_bid',
                        timestamp=datetime.now()
                    )
                    signals.append(bid_signal)
                
                # Generate ask signal
                if our_ask < best_ask:
                    ask_signal = TradingSignal(
                        product_id=product_id,
                        side=OrderSide.SELL,
                        strength=SignalStrength.MODERATE,
                        confidence=0.8,
                        price=our_ask,
                        size=0.01,
                        strategy='market_making_ask',
                        timestamp=datetime.now()
                    )
                    signals.append(ask_signal)
            
        except Exception as e:
            logger.error(f"Error in market making strategy: {e}")
        
        return signals


class ArbitrageDetector:
    """Detects arbitrage opportunities across exchanges."""
    
    def __init__(self):
        self.exchange_prices = {}
        self.min_profit_threshold = 0.002  # 0.2%
        self.max_execution_time = 5  # seconds
        
    async def detect_opportunities(self, trade_data: Dict) -> List[TradingSignal]:
        """Detect arbitrage opportunities."""
        signals = []
        product_id = trade_data['product_id']
        exchange = trade_data['exchange']
        price = trade_data['price']
        
        # Update price for this exchange
        if product_id not in self.exchange_prices:
            self.exchange_prices[product_id] = {}
        
        self.exchange_prices[product_id][exchange] = {
            'price': price,
            'timestamp': datetime.now()
        }
        
        # Check for arbitrage opportunities
        if len(self.exchange_prices[product_id]) > 1:
            prices = []
            for ex, data in self.exchange_prices[product_id].items():
                # Only consider recent prices (within 5 seconds)
                if (datetime.now() - data['timestamp']).total_seconds() < 5:
                    prices.append((ex, data['price']))
            
            if len(prices) > 1:
                prices.sort(key=lambda x: x[1])  # Sort by price
                lowest_exchange, lowest_price = prices[0]
                highest_exchange, highest_price = prices[-1]
                
                profit_ratio = (highest_price - lowest_price) / lowest_price
                
                if profit_ratio > self.min_profit_threshold:
                    # Buy on lowest exchange, sell on highest
                    buy_signal = TradingSignal(
                        product_id=product_id,
                        side=OrderSide.BUY,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        price=lowest_price,
                        size=0.05,
                        strategy='arbitrage_buy',
                        timestamp=datetime.now(),
                        metadata={
                            'target_exchange': lowest_exchange,
                            'profit_ratio': profit_ratio
                        }
                    )
                    
                    sell_signal = TradingSignal(
                        product_id=product_id,
                        side=OrderSide.SELL,
                        strength=SignalStrength.STRONG,
                        confidence=0.9,
                        price=highest_price,
                        size=0.05,
                        strategy='arbitrage_sell',
                        timestamp=datetime.now(),
                        metadata={
                            'target_exchange': highest_exchange,
                            'profit_ratio': profit_ratio
                        }
                    )
                    
                    signals.extend([buy_signal, sell_signal])
        
        return signals


class MomentumStrategy:
    """Momentum-based trading strategy."""
    
    def __init__(self):
        self.price_history = {}
        self.volume_history = {}
        self.lookback_periods = [5, 10, 20, 50]
        
    async def analyze(self, trade_data: Dict) -> List[TradingSignal]:
        """Analyze momentum and generate signals."""
        signals = []
        product_id = trade_data['product_id']
        price = trade_data['price']
        volume = trade_data['size']
        
        # Update history
        if product_id not in self.price_history:
            self.price_history[product_id] = deque(maxlen=100)
            self.volume_history[product_id] = deque(maxlen=100)
        
        self.price_history[product_id].append(price)
        self.volume_history[product_id].append(volume)
        
        if len(self.price_history[product_id]) < 50:
            return signals
        
        try:
            prices = np.array(self.price_history[product_id])
            volumes = np.array(self.volume_history[product_id])
            
            # Calculate momentum indicators
            momentum_signals = []
            
            for period in self.lookback_periods:
                if len(prices) >= period:
                    # Price momentum
                    price_momentum = (prices[-1] - prices[-period]) / prices[-period]
                    
                    # Volume momentum
                    recent_volume = np.mean(volumes[-5:])
                    historical_volume = np.mean(volumes[-period:-5])
                    volume_momentum = (recent_volume - historical_volume) / historical_volume if historical_volume > 0 else 0
                    
                    # Combined momentum score
                    momentum_score = price_momentum + (volume_momentum * 0.3)
                    momentum_signals.append(momentum_score)
            
            # Aggregate momentum
            avg_momentum = np.mean(momentum_signals)
            momentum_strength = abs(avg_momentum)
            
            # Generate signal if momentum is strong enough
            if momentum_strength > 0.005:  # 0.5% threshold
                side = OrderSide.BUY if avg_momentum > 0 else OrderSide.SELL
                
                # Determine strength
                if momentum_strength > 0.02:
                    strength = SignalStrength.VERY_STRONG
                elif momentum_strength > 0.015:
                    strength = SignalStrength.STRONG
                elif momentum_strength > 0.01:
                    strength = SignalStrength.MODERATE
                else:
                    strength = SignalStrength.WEAK
                
                signal = TradingSignal(
                    product_id=product_id,
                    side=side,
                    strength=strength,
                    confidence=min(momentum_strength * 20, 0.95),
                    price=price,
                    size=0.02 * momentum_strength * 10,  # Size based on momentum
                    strategy='momentum',
                    timestamp=datetime.now(),
                    metadata={
                        'momentum_score': avg_momentum,
                        'momentum_strength': momentum_strength
                    }
                )
                signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in momentum strategy: {e}")
        
        return signals


class MeanReversionStrategy:
    """Mean reversion trading strategy."""
    
    def __init__(self):
        self.price_history = {}
        self.bollinger_period = 20
        self.bollinger_std = 2.0
        self.rsi_period = 14
        
    async def analyze(self, trade_data: Dict) -> List[TradingSignal]:
        """Analyze mean reversion opportunities."""
        signals = []
        product_id = trade_data['product_id']
        price = trade_data['price']
        
        # Update price history
        if product_id not in self.price_history:
            self.price_history[product_id] = deque(maxlen=100)
        
        self.price_history[product_id].append(price)
        
        if len(self.price_history[product_id]) < self.bollinger_period:
            return signals
        
        try:
            prices = np.array(self.price_history[product_id])
            
            # Calculate Bollinger Bands
            sma = np.mean(prices[-self.bollinger_period:])
            std = np.std(prices[-self.bollinger_period:])
            upper_band = sma + (self.bollinger_std * std)
            lower_band = sma - (self.bollinger_std * std)
            
            # Calculate RSI
            rsi = self._calculate_rsi(prices, self.rsi_period)
            
            # Mean reversion signals
            bb_position = (price - lower_band) / (upper_band - lower_band)
            
            # Oversold condition (buy signal)
            if bb_position < 0.1 and rsi < 30:
                signal = TradingSignal(
                    product_id=product_id,
                    side=OrderSide.BUY,
                    strength=SignalStrength.STRONG,
                    confidence=0.85,
                    price=price,
                    size=0.03,
                    strategy='mean_reversion_buy',
                    timestamp=datetime.now(),
                    metadata={
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'distance_from_mean': (price - sma) / sma
                    }
                )
                signals.append(signal)
            
            # Overbought condition (sell signal)
            elif bb_position > 0.9 and rsi > 70:
                signal = TradingSignal(
                    product_id=product_id,
                    side=OrderSide.SELL,
                    strength=SignalStrength.STRONG,
                    confidence=0.85,
                    price=price,
                    size=0.03,
                    strategy='mean_reversion_sell',
                    timestamp=datetime.now(),
                    metadata={
                        'bb_position': bb_position,
                        'rsi': rsi,
                        'distance_from_mean': (price - sma) / sma
                    }
                )
                signals.append(signal)
        
        except Exception as e:
            logger.error(f"Error in mean reversion strategy: {e}")
        
        return signals
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50  # Neutral RSI
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi


class VolumeProfileStrategy:
    """Volume profile analysis for support/resistance levels."""
    
    def __init__(self):
        self.volume_profiles = {}
        self.price_levels = 100  # Number of price levels to track
        
    async def analyze(self, trade_data: Dict, orderbook_data: Dict) -> List[TradingSignal]:
        """Analyze volume profile for trading opportunities."""
        signals = []
        product_id = trade_data['product_id']
        price = trade_data['price']
        volume = trade_data['size']
        
        # Update volume profile
        if product_id not in self.volume_profiles:
            self.volume_profiles[product_id] = {}
        
        # Round price to create levels
        price_level = round(price, 2)  # 2 decimal places
        
        if price_level not in self.volume_profiles[product_id]:
            self.volume_profiles[product_id][price_level] = 0
        
        self.volume_profiles[product_id][price_level] += volume
        
        # Analyze volume profile
        if len(self.volume_profiles[product_id]) > 20:
            try:
                # Find high volume nodes (support/resistance)
                volume_data = self.volume_profiles[product_id]
                sorted_levels = sorted(volume_data.items(), key=lambda x: x[1], reverse=True)
                
                # Top 10% of volume levels are considered significant
                significant_levels = sorted_levels[:max(1, len(sorted_levels) // 10)]
                
                # Check if current price is near a significant level
                for level_price, level_volume in significant_levels:
                    distance = abs(price - level_price) / price
                    
                    if distance < 0.002:  # Within 0.2% of significant level
                        # Determine if this is support or resistance
                        if price > level_price:
                            # Price above level = support
                            signal = TradingSignal(
                                product_id=product_id,
                                side=OrderSide.BUY,
                                strength=SignalStrength.MODERATE,
                                confidence=0.75,
                                price=price,
                                size=0.02,
                                strategy='volume_profile_support',
                                timestamp=datetime.now(),
                                metadata={
                                    'support_level': level_price,
                                    'level_volume': level_volume,
                                    'distance': distance
                                }
                            )
                            signals.append(signal)
                        else:
                            # Price below level = resistance
                            signal = TradingSignal(
                                product_id=product_id,
                                side=OrderSide.SELL,
                                strength=SignalStrength.MODERATE,
                                confidence=0.75,
                                price=price,
                                size=0.02,
                                strategy='volume_profile_resistance',
                                timestamp=datetime.now(),
                                metadata={
                                    'resistance_level': level_price,
                                    'level_volume': level_volume,
                                    'distance': distance
                                }
                            )
                            signals.append(signal)
                        
                        break  # Only one signal per analysis
            
            except Exception as e:
                logger.error(f"Error in volume profile analysis: {e}")
        
        return signals