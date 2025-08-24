import asyncio
import json
import os
import websockets
from kafka import KafkaProducer
import redis
import aiohttp
import time
from typing import Dict, List, Optional, Callable
from collections import deque
import logging
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class MarketDataPipeline:
    """
    Production-grade market data pipeline supporting multiple exchanges.
    Features:
    - Multi-exchange WebSocket connections with failover
    - Real-time order book reconstruction
    - Trade aggregation and VWAP calculations
    - Data quality monitoring and anomaly detection
    - High-performance data distribution via Kafka
    - Redis caching for low-latency access
    """
    
    def __init__(self, product_ids: List[str], exchanges: List[str] = ['coinbase']):
        logger.info(f"Initializing MarketDataPipeline for exchanges: {exchanges}")
        self.product_ids = product_ids
        self.exchanges = exchanges
        self.kafka_topic = "market_data.trades"
        self.orderbook_topic = "market_data.orderbook"
        
        # Exchange configurations
        self.exchange_configs = {
            'coinbase': {
                'ws_url': 'wss://advanced-trade-ws.coinbase.com',
                'rest_url': 'https://api.exchange.coinbase.com',
                'channels': ['market_trades', 'level2_batch']
            },
            'binance': {
                'ws_url': 'wss://stream.binance.com:9443/ws',
                'rest_url': 'https://api.binance.com',
                'channels': ['trade', 'depth']
            }
        }
        
        # Connection management
        self.connections = {}
        self.connection_tasks = {}
        self.reconnect_delays = {}
        
        # Data storage and processing
        self.order_books = {pid: {'bids': {}, 'asks': {}} for pid in product_ids}
        self.trade_buffers = {pid: deque(maxlen=1000) for pid in product_ids}
        self.price_history = {pid: deque(maxlen=10000) for pid in product_ids}
        
        # Performance monitoring
        self.message_counts = {exchange: 0 for exchange in exchanges}
        self.latency_stats = {exchange: deque(maxlen=1000) for exchange in exchanges}
        self.error_counts = {exchange: 0 for exchange in exchanges}
        
        # Data quality monitoring
        self.anomaly_detector = AnomalyDetector()
        self.data_validators = []
        
        # Callbacks for real-time processing
        self.trade_callbacks = []
        self.orderbook_callbacks = []

        # Initialize connections

        self._initialize_kafka_producer()
        self._initialize_redis_client()
        
        logger.info(f"MarketDataPipeline initialized for {len(product_ids)} products")

    def _initialize_kafka_producer(self):
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                retries=5,
                acks='all',
                # Short timeout for environments where Kafka is not available
                bootstrap_servers_timeout_ms=5000
            )
            print("Kafka Producer initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Kafka Producer: {e}. Using mock producer.")
            self.kafka_producer = self._get_mock_kafka_producer()

    def _initialize_redis_client(self):
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'redis'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info("Redis client initialized and connected successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {e}. Using mock client.")
            self.redis_client = self._get_mock_redis_client()

    def _get_mock_kafka_producer(self):
        class MockKafkaProducer:
            def send(self, topic, value):
                logger.debug(f"[Mock Kafka] Topic '{topic}': {json.dumps(value)}")
            def close(self):
                logger.info("[Mock Kafka] Producer closed.")
        return MockKafkaProducer()

    def _get_mock_redis_client(self):
        class MockRedisClient:
            def set(self, key, value):
                logger.debug(f"[Mock Redis] SET {key} {value}")
            def hset(self, key, field, value):
                logger.debug(f"[Mock Redis] HSET {key} {field} {value}")
            def get(self, key):
                return None
        return MockRedisClient()

    def add_trade_callback(self, callback: Callable):
        """Add callback function for trade events."""
        self.trade_callbacks.append(callback)
    
    def add_orderbook_callback(self, callback: Callable):
        """Add callback function for order book events."""
        self.orderbook_callbacks.append(callback)
    
    async def _handle_coinbase_message(self, message: str, exchange: str):
        """Handle Coinbase WebSocket messages."""
        try:
            data = json.loads(message)
            receive_time = time.time()
            
            # Update message count
            self.message_counts[exchange] += 1
            
            # Handle different message types
            if data.get('channel') == 'market_trades' and data.get('events'):
                await self._process_coinbase_trades(data, receive_time)
            elif data.get('channel') == 'level2_batch' and data.get('events'):
                await self._process_coinbase_orderbook(data, receive_time)
                
        except Exception as e:
            logger.error(f"Error handling Coinbase message: {e}")
            self.error_counts[exchange] += 1
    
    async def _process_coinbase_trades(self, data: Dict, receive_time: float):
        """Process Coinbase trade messages."""
        for event in data['events']:
            for trade in event.get('trades', []):
                try:
                    # Calculate latency
                    trade_time = datetime.fromisoformat(trade['time'].replace('Z', '+00:00')).timestamp()
                    latency = (receive_time - trade_time) * 1000  # ms
                    self.latency_stats['coinbase'].append(latency)
                    
                    # Normalize trade data
                    normalized_trade = {
                        'timestamp': trade['time'],
                        'product_id': trade['product_id'],
                        'price': float(trade['price']),
                        'size': float(trade['size']),
                        'side': trade['side'].lower(),
                        'exchange': 'coinbase',
                        'trade_id': trade.get('trade_id'),
                        'latency_ms': latency
                    }
                    
                    # Data quality checks
                    if self._validate_trade_data(normalized_trade):
                        # Update internal state
                        product_id = normalized_trade['product_id']
                        self.trade_buffers[product_id].append(normalized_trade)
                        self.price_history[product_id].append({
                            'timestamp': receive_time,
                            'price': normalized_trade['price'],
                            'volume': normalized_trade['size']
                        })
                        
                        # Send to Kafka
                        self.kafka_producer.send(self.kafka_topic, value=normalized_trade)
                        
                        # Update Redis cache
                        await self._update_redis_cache(normalized_trade)
                        
                        # Execute callbacks
                        for callback in self.trade_callbacks:
                            try:
                                await callback(normalized_trade)
                            except Exception as e:
                                logger.error(f"Error in trade callback: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing trade: {e}")
    
    async def _process_coinbase_orderbook(self, data: Dict, receive_time: float):
        """Process Coinbase order book messages."""
        for event in data['events']:
            product_id = event.get('product_id')
            if not product_id or product_id not in self.product_ids:
                continue
                
            try:
                # Update order book
                changes = event.get('updates', [])
                for change in changes:
                    side = change['side']
                    price = float(change['price_level'])
                    size = float(change['new_quantity'])
                    
                    if size == 0:
                        # Remove level
                        if price in self.order_books[product_id][side + 's']:
                            del self.order_books[product_id][side + 's'][price]
                    else:
                        # Update level
                        self.order_books[product_id][side + 's'][price] = size
                
                # Create normalized order book snapshot
                orderbook_data = {
                    'timestamp': datetime.now().isoformat(),
                    'product_id': product_id,
                    'exchange': 'coinbase',
                    'bids': [[price, size] for price, size in 
                            sorted(self.order_books[product_id]['bids'].items(), reverse=True)[:20]],
                    'asks': [[price, size] for price, size in 
                            sorted(self.order_books[product_id]['asks'].items())[:20]]
                }
                
                # Send to Kafka
                self.kafka_producer.send(self.orderbook_topic, value=orderbook_data)
                
                # Update Redis
                redis_key = f"orderbook:{product_id}"
                self.redis_client.set(redis_key, json.dumps(orderbook_data))
                
                # Execute callbacks
                for callback in self.orderbook_callbacks:
                    try:
                        await callback(orderbook_data)
                    except Exception as e:
                        logger.error(f"Error in orderbook callback: {e}")
                        
            except Exception as e:
                logger.error(f"Error processing orderbook update: {e}")
    
    def _validate_trade_data(self, trade: Dict) -> bool:
        """Validate trade data for quality and anomalies."""
        try:
            # Basic validation
            if trade['price'] <= 0 or trade['size'] <= 0:
                return False
            
            # Price anomaly detection
            product_id = trade['product_id']
            if product_id in self.price_history and len(self.price_history[product_id]) > 10:
                recent_prices = [p['price'] for p in list(self.price_history[product_id])[-10:]]
                median_price = np.median(recent_prices)
                
                # Reject prices that are more than 10% away from recent median
                if abs(trade['price'] - median_price) / median_price > 0.1:
                    logger.warning(f"Price anomaly detected: {trade['price']} vs median {median_price}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating trade data: {e}")
            return False
    
    async def _update_redis_cache(self, trade: Dict):
        """Update Redis cache with latest trade data."""
        try:
            product_id = trade['product_id']
            
            # Latest price
            self.redis_client.set(f"latest_price:{product_id}", trade['price'])
            
            # OHLCV data (1-minute aggregation)
            minute_key = f"ohlcv:{product_id}:{int(time.time() // 60)}"
            current_data = self.redis_client.get(minute_key)
            
            if current_data:
                ohlcv = json.loads(current_data)
                ohlcv['high'] = max(ohlcv['high'], trade['price'])
                ohlcv['low'] = min(ohlcv['low'], trade['price'])
                ohlcv['close'] = trade['price']
                ohlcv['volume'] += trade['size']
            else:
                ohlcv = {
                    'open': trade['price'],
                    'high': trade['price'],
                    'low': trade['price'],
                    'close': trade['price'],
                    'volume': trade['size'],
                    'timestamp': int(time.time())
                }
            
            self.redis_client.set(minute_key, json.dumps(ohlcv), ex=3600)  # 1 hour expiry
            
        except Exception as e:
            logger.error(f"Error updating Redis cache: {e}")

    async def _run_exchange_consumer(self, exchange: str):
        """Run WebSocket consumer for a specific exchange."""
        config = self.exchange_configs[exchange]
        reconnect_delay = 1
        
        while True:
            try:
                logger.info(f"Connecting to {exchange} WebSocket...")
                
                async with websockets.connect(
                    config['ws_url'],
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=10
                ) as ws:
                    self.connections[exchange] = ws
                    
                    # Subscribe to channels
                    if exchange == 'coinbase':
                        await self._subscribe_coinbase(ws)
                    elif exchange == 'binance':
                        await self._subscribe_binance(ws)
                    
                    logger.info(f"Connected to {exchange} WebSocket")
                    reconnect_delay = 1  # Reset delay on successful connection
                    
                    # Message processing loop
                    async for message in ws:
                        if exchange == 'coinbase':
                            await self._handle_coinbase_message(message, exchange)
                        elif exchange == 'binance':
                            await self._handle_binance_message(message, exchange)
                            
            except (websockets.exceptions.ConnectionClosedError, OSError) as e:
                logger.warning(f"{exchange} connection error: {e}. Reconnecting in {reconnect_delay}s...")
                self.error_counts[exchange] += 1
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)  # Exponential backoff, max 60s
            except Exception as e:
                logger.error(f"Unexpected error in {exchange} consumer: {e}. Reconnecting in {reconnect_delay}s...")
                self.error_counts[exchange] += 1
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 60)
            finally:
                if exchange in self.connections:
                    del self.connections[exchange]
    
    async def _subscribe_coinbase(self, ws):
        """Subscribe to Coinbase channels."""
        # Subscribe to trades
        trade_message = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channel": "market_trades"
        }
        await ws.send(json.dumps(trade_message))
        
        # Subscribe to order book
        orderbook_message = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channel": "level2_batch"
        }
        await ws.send(json.dumps(orderbook_message))
        
        logger.info(f"Subscribed to Coinbase channels for: {', '.join(self.product_ids)}")
    
    async def _subscribe_binance(self, ws):
        """Subscribe to Binance channels."""
        # Convert product IDs to Binance format (e.g., BTC-USD -> btcusdt)
        binance_symbols = []
        for product_id in self.product_ids:
            symbol = product_id.replace('-', '').lower()
            binance_symbols.append(symbol)
        
        # Subscribe to trades and depth
        streams = []
        for symbol in binance_symbols:
            streams.extend([f"{symbol}@trade", f"{symbol}@depth20@100ms"])
        
        subscribe_message = {
            "method": "SUBSCRIBE",
            "params": streams,
            "id": 1
        }
        await ws.send(json.dumps(subscribe_message))
        
        logger.info(f"Subscribed to Binance streams for: {', '.join(binance_symbols)}")
    
    async def _handle_binance_message(self, message: str, exchange: str):
        """Handle Binance WebSocket messages."""
        try:
            data = json.loads(message)
            receive_time = time.time()
            
            self.message_counts[exchange] += 1
            
            # Handle trade messages
            if 'stream' in data and '@trade' in data['stream']:
                await self._process_binance_trade(data['data'], receive_time)
            # Handle depth messages
            elif 'stream' in data and '@depth' in data['stream']:
                await self._process_binance_depth(data['data'], receive_time)
                
        except Exception as e:
            logger.error(f"Error handling Binance message: {e}")
            self.error_counts[exchange] += 1
    
    async def _process_binance_trade(self, data: Dict, receive_time: float):
        """Process Binance trade messages."""
        try:
            # Convert Binance symbol back to our format
            symbol = data['s'].lower()
            if symbol.endswith('usdt'):
                product_id = f"{symbol[:-4].upper()}-USD"
            else:
                product_id = symbol.upper()
            
            if product_id not in self.product_ids:
                return
            
            # Calculate latency
            trade_time = data['T'] / 1000  # Convert from ms to seconds
            latency = (receive_time - trade_time) * 1000  # ms
            self.latency_stats['binance'].append(latency)
            
            # Normalize trade data
            normalized_trade = {
                'timestamp': datetime.fromtimestamp(trade_time).isoformat(),
                'product_id': product_id,
                'price': float(data['p']),
                'size': float(data['q']),
                'side': 'buy' if data['m'] else 'sell',  # m=true means buyer is market maker
                'exchange': 'binance',
                'trade_id': str(data['t']),
                'latency_ms': latency
            }
            
            # Process similar to Coinbase
            if self._validate_trade_data(normalized_trade):
                product_id = normalized_trade['product_id']
                self.trade_buffers[product_id].append(normalized_trade)
                self.price_history[product_id].append({
                    'timestamp': receive_time,
                    'price': normalized_trade['price'],
                    'volume': normalized_trade['size']
                })
                
                self.kafka_producer.send(self.kafka_topic, value=normalized_trade)
                await self._update_redis_cache(normalized_trade)
                
                for callback in self.trade_callbacks:
                    try:
                        await callback(normalized_trade)
                    except Exception as e:
                        logger.error(f"Error in trade callback: {e}")
                        
        except Exception as e:
            logger.error(f"Error processing Binance trade: {e}")
    
    async def _process_binance_depth(self, data: Dict, receive_time: float):
        """Process Binance depth messages."""
        try:
            # Convert symbol format
            symbol = data['s'].lower()
            if symbol.endswith('usdt'):
                product_id = f"{symbol[:-4].upper()}-USD"
            else:
                product_id = symbol.upper()
            
            if product_id not in self.product_ids:
                return
            
            # Update order book
            self.order_books[product_id]['bids'] = {
                float(bid[0]): float(bid[1]) for bid in data['bids'] if float(bid[1]) > 0
            }
            self.order_books[product_id]['asks'] = {
                float(ask[0]): float(ask[1]) for ask in data['asks'] if float(ask[1]) > 0
            }
            
            # Create normalized order book
            orderbook_data = {
                'timestamp': datetime.now().isoformat(),
                'product_id': product_id,
                'exchange': 'binance',
                'bids': [[price, size] for price, size in 
                        sorted(self.order_books[product_id]['bids'].items(), reverse=True)[:20]],
                'asks': [[price, size] for price, size in 
                        sorted(self.order_books[product_id]['asks'].items())[:20]]
            }
            
            self.kafka_producer.send(self.orderbook_topic, value=orderbook_data)
            
            redis_key = f"orderbook:{product_id}"
            self.redis_client.set(redis_key, json.dumps(orderbook_data))
            
            for callback in self.orderbook_callbacks:
                try:
                    await callback(orderbook_data)
                except Exception as e:
                    logger.error(f"Error in orderbook callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error processing Binance depth: {e}")
    
    def get_latest_price(self, product_id: str) -> Optional[float]:
        """Get latest price for a product."""
        try:
            price_str = self.redis_client.get(f"latest_price:{product_id}")
            return float(price_str) if price_str else None
        except Exception:
            return None
    
    def get_order_book(self, product_id: str) -> Optional[Dict]:
        """Get current order book for a product."""
        try:
            orderbook_str = self.redis_client.get(f"orderbook:{product_id}")
            return json.loads(orderbook_str) if orderbook_str else None
        except Exception:
            return None
    
    def get_recent_trades(self, product_id: str, limit: int = 100) -> List[Dict]:
        """Get recent trades for a product."""
        if product_id in self.trade_buffers:
            return list(self.trade_buffers[product_id])[-limit:]
        return []
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        stats = {
            'message_counts': self.message_counts.copy(),
            'error_counts': self.error_counts.copy(),
            'latency_stats': {}
        }
        
        for exchange, latencies in self.latency_stats.items():
            if latencies:
                stats['latency_stats'][exchange] = {
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'p95': np.percentile(latencies, 95),
                    'p99': np.percentile(latencies, 99)
                }
        
        return stats

    async def start(self):
        logger.info("MarketDataPipeline starting...")
        
        # Start consumers for each exchange
        for exchange in self.exchanges:
            task = asyncio.create_task(self._run_exchange_consumer(exchange))
            self.connection_tasks[exchange] = task
        
        logger.info(f"Market data consumers started for {len(self.exchanges)} exchanges")

    async def stop(self):
        logger.info("MarketDataPipeline stopping...")
        
        # Cancel all connection tasks
        for exchange, task in self.connection_tasks.items():
            task.cancel()
        
        # Wait for all tasks to complete
        if self.connection_tasks:
            await asyncio.gather(*self.connection_tasks.values(), return_exceptions=True)
        
        # Close connections
        for exchange, ws in self.connections.items():
            try:
                await ws.close()
            except Exception as e:
                logger.error(f"Error closing {exchange} connection: {e}")
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        logger.info("Market data pipeline stopped")


class AnomalyDetector:
    """Detects anomalies in market data streams."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.price_windows = {}
        self.volume_windows = {}
    
    def detect_price_anomaly(self, product_id: str, price: float) -> bool:
        """Detect if a price is anomalous."""
        if product_id not in self.price_windows:
            self.price_windows[product_id] = deque(maxlen=self.window_size)
        
        window = self.price_windows[product_id]
        
        if len(window) < 10:
            window.append(price)
            return False
        
        # Calculate z-score
        mean_price = np.mean(window)
        std_price = np.std(window)
        
        if std_price == 0:
            return False
        
        z_score = abs(price - mean_price) / std_price
        
        window.append(price)
        
        # Flag as anomaly if z-score > 3
        return z_score > 3
    
    def detect_volume_anomaly(self, product_id: str, volume: float) -> bool:
        """Detect if a volume is anomalous."""
        if product_id not in self.volume_windows:
            self.volume_windows[product_id] = deque(maxlen=self.window_size)
        
        window = self.volume_windows[product_id]
        
        if len(window) < 10:
            window.append(volume)
            return False
        
        # Use median and MAD for volume anomaly detection (more robust)
        median_volume = np.median(window)
        mad = np.median(np.abs(np.array(window) - median_volume))
        
        if mad == 0:
            return False
        
        modified_z_score = 0.6745 * (volume - median_volume) / mad
        
        window.append(volume)
        
        # Flag as anomaly if modified z-score > 3.5
        return abs(modified_z_score) > 3.5
