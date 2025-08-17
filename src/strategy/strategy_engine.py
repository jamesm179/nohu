import asyncio
import json
import os
import collections
import numpy as np
from kafka import KafkaConsumer
from src.models import Order, OrderSide

class StrategyEngine:
    """
    Executes trading strategies based on real-time market data from Kafka.
    """
    def __init__(self, market_data_pipeline, risk_engine, short_window=5, long_window=12):
        print("Initializing StrategyEngine...")
        self.market_data_pipeline = market_data_pipeline
        self.risk_engine = risk_engine
        self.kafka_topic = "market_data.trades"
        self.short_window = short_window
        self.long_window = long_window
        self.product_strategies = {}

        self._consumer_task = None
        self._initialize_kafka_consumer()

    def _initialize_kafka_consumer(self):
        """Initializes the Kafka consumer with a fallback to a mock."""
        try:
            self.kafka_consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='strategy-engine-group-1',
                client_id='strategy-engine-1'
            )
            print("Kafka Consumer initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize Kafka Consumer: {e}. Using mock consumer.")
            self.kafka_consumer = self._get_mock_kafka_consumer()

    def _get_mock_kafka_consumer(self):
        """Returns a mock Kafka consumer that yields a predictable stream of trades."""
        class MockMessage:
            def __init__(self, value):
                self.value = value

        class MockKafkaConsumer:
            def __init__(self, topic, short_window, long_window):
                self._topic = topic
                self.short_window = short_window
                self.long_window = long_window
                self._messages = self._generate_test_data()
                print(f"[Mock Kafka] Consumer subscribed to topic '{topic}' with test data.")

            def _generate_test_data(self):
                """Generates a clean data sequence to test crossovers."""
                base_price = 100
                messages = []

                # Phase 1: Long period of low, stable prices
                prices = [float(base_price)] * self.long_window
                # Phase 2: Sharp rise to trigger a Golden Cross
                prices.extend([float(base_price + 10)] * self.short_window)
                # Phase 3: Sharp fall to trigger a Death Cross
                prices.extend([float(base_price - 10)] * self.short_window)

                for i, price in enumerate(prices):
                    trade = {
                        'timestamp': f'2025-01-01T12:00:{i:02d}Z',
                        'product_id': 'BTC-USD',
                        'price': price,
                        'size': 1.0,
                        'side': 'BUY'
                    }
                    messages.append(MockMessage(trade))
                return messages

            def __iter__(self): return iter(self._messages)
            def close(self): print("[Mock Kafka] Consumer closed.")

        return MockKafkaConsumer(self.kafka_topic, self.short_window, self.long_window)

    def _initialize_strategy_for_product(self, product_id):
        """Initializes the state for a new product if not already present."""
        if product_id not in self.product_strategies:
            print(f"Initializing SMA Crossover strategy for {product_id}...")
            self.product_strategies[product_id] = {
                'prices': collections.deque(maxlen=self.long_window),
                'prev_short_sma': None,
                'prev_long_sma': None,
            }

    def _process_trade_message(self, message):
        """Processes a single trade message to check for trading signals."""
        trade_data = message.value
        product_id = trade_data['product_id']
        price = trade_data['price']

        self._initialize_strategy_for_product(product_id)

        state = self.product_strategies[product_id]
        state['prices'].append(price)

        if len(state['prices']) < self.long_window:
            return

        prices_array = np.array(state['prices'])
        short_sma = np.mean(prices_array[-self.short_window:])
        long_sma = np.mean(prices_array)

        prev_short_sma = state.get('prev_short_sma')
        prev_long_sma = state.get('prev_long_sma')

        print(f"[{product_id}] Price: {price:7.2f} | Short SMA: {short_sma:7.2f} | Long SMA: {long_sma:7.2f}")

        if prev_short_sma is not None and prev_long_sma is not None:
            signal = None
            if short_sma > long_sma and prev_short_sma <= prev_long_sma:
                signal = OrderSide.BUY
            elif short_sma < long_sma and prev_short_sma >= prev_long_sma:
                signal = OrderSide.SELL

            if signal:
                # 1. Create an order object
                # For now, use a fixed size. A real system would use a position sizing model.
                order_size = 0.01
                order = Order(
                    product_id=product_id,
                    side=signal,
                    size=order_size,
                    price=price
                )

                # 2. Check the order with the risk engine
                is_approved = self.risk_engine.check_pre_trade_risk(order)

                # 3. Log the outcome of the risk check
                if is_approved:
                    print(f"[{product_id}] --- {signal.value} SIGNAL: Order APPROVED by Risk Engine ---")
                    # In a real system, this approved order would be sent to an execution handler.
                else:
                    print(f"[{product_id}] --- {signal.value} SIGNAL: Order REJECTED by Risk Engine ---")

        state['prev_short_sma'] = short_sma
        state['prev_long_sma'] = long_sma

    async def _run_kafka_consumer(self):
        """The main loop to consume messages from Kafka."""
        print("Starting Kafka consumer loop...")
        loop = asyncio.get_running_loop()
        try:
            for message in self.kafka_consumer:
                await loop.run_in_executor(None, self._process_trade_message, message)
            print("Finished processing all messages in mock consumer.")
        except asyncio.CancelledError:
            print("Kafka consumer loop cancelled.")
        finally:
            print("Kafka consumer loop stopped.")

    async def start(self):
        """Starts the strategy engine and its Kafka consumer."""
        print("StrategyEngine starting...")
        self._consumer_task = asyncio.create_task(self._run_kafka_consumer())

    async def stop(self):
        """Stops the strategy engine and its Kafka consumer."""
        print("StrategyEngine stopping...")
        if self._consumer_task:
            self._consumer_task.cancel()
            await asyncio.gather(self._consumer_task, return_exceptions=True)
        if self.kafka_consumer:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.kafka_consumer.close)
            print("Kafka consumer closed.")
        print("StrategyEngine stopped.")
