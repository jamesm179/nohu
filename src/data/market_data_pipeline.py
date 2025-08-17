import asyncio
import json
import os
import websockets
from kafka import KafkaProducer
import redis

class MarketDataPipeline:
    """
    Handles the ingestion, processing, and distribution of market data from Coinbase.
    - Connects to the Coinbase WebSocket feed.
    - Subscribes to the 'market_trades' channel for multiple products.
    - Normalizes trade data and streams it to Kafka.
    - Caches the latest price in Redis.
    - Falls back to mock clients if Kafka/Redis are unavailable.
    """
    def __init__(self, product_ids):
        print("Initializing MarketDataPipeline for Coinbase...")
        self.product_ids = product_ids
        self.ws_url = "wss://advanced-trade-ws.coinbase.com"
        self.kafka_topic = "market_data.trades"

        self._connection = None
        self._consumer_task = None

        self._initialize_kafka_producer()
        self._initialize_redis_client()

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
            print(f"Failed to initialize Kafka Producer: {e}. Using mock producer.")
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
            print("Redis client initialized and connected successfully.")
        except Exception as e:
            print(f"Failed to initialize Redis client: {e}. Using mock client.")
            self.redis_client = self._get_mock_redis_client()

    def _get_mock_kafka_producer(self):
        class MockKafkaProducer:
            def send(self, topic, value):
                print(f"[Mock Kafka] Topic '{topic}': {json.dumps(value)}")
            def close(self):
                print("[Mock Kafka] Producer closed.")
        return MockKafkaProducer()

    def _get_mock_redis_client(self):
        class MockRedisClient:
            def set(self, key, value):
                print(f"[Mock Redis] SET {key} {value}")
        return MockRedisClient()

    async def _message_handler(self, message):
        data = json.loads(message)
        if data.get('channel') == 'market_trades' and data.get('events'):
            for event in data['events']:
                for trade in event.get('trades', []):
                    # The 'time' field from the trade data is the timestamp.
                    normalized_trade = {
                        'timestamp': trade.get('time'),
                        'product_id': trade['product_id'],
                        'price': float(trade['price']),
                        'size': float(trade['size']),
                        'side': trade['side'],
                        'exchange': 'coinbase'
                    }
                    self.kafka_producer.send(self.kafka_topic, value=normalized_trade)
                    redis_key = f"latest_price:{trade['product_id']}"
                    self.redis_client.set(redis_key, normalized_trade['price'])
                    # Reduce log verbosity
                    # print(f"Processed trade for {trade['product_id']}")

    async def _run_consumer(self):
        while True:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    self._connection = ws
                    subscribe_message = {"type": "subscribe", "product_ids": self.product_ids, "channel": "market_trades"}
                    await ws.send(json.dumps(subscribe_message))
                    print(f"Subscribed to 'market_trades' for: {', '.join(self.product_ids)}")
                    while True:
                        message = await ws.recv()
                        await self._message_handler(message)
            except (websockets.exceptions.ConnectionClosedError, OSError) as e:
                print(f"Connection error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)
            except Exception as e:
                print(f"Unexpected error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def start(self):
        print("MarketDataPipeline starting...")
        self._consumer_task = asyncio.create_task(self._run_consumer())
        print("Market data consumer started.")

    async def stop(self):
        print("MarketDataPipeline stopping...")
        if self._consumer_task:
            self._consumer_task.cancel()
            await asyncio.gather(self._consumer_task, return_exceptions=True)
        if self.kafka_producer:
            self.kafka_producer.close()
        print("Market data consumer stopped.")
