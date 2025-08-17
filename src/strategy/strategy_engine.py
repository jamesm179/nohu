import asyncio
import json
import os
import collections
import numpy as np
import pandas as pd
from kafka import KafkaConsumer
from src.models import Order, OrderSide

class StrategyEngine:
    """
    Executes multiple trading strategies based on real-time market data.
    """
    def __init__(self, market_data_pipeline, risk_engine, execution_handler):
        print("Initializing StrategyEngine...")
        self.market_data_pipeline = market_data_pipeline
        self.risk_engine = risk_engine
        self.execution_handler = execution_handler
        self.kafka_topic = "market_data.trades"

        # --- Strategy Configurations ---
        self.strategies = {
            'sma_crossover': {
                'short_window': 5,
                'long_window': 12,
            },
            'rsi': {
                'window': 14,
                'oversold_threshold': 30,
                'overbought_threshold': 70,
            }
        }
        self.product_states = {}

        self._consumer_task = None
        self._initialize_kafka_consumer()

    def _initialize_kafka_consumer(self):
        try:
            self.kafka_consumer = KafkaConsumer(
                self.kafka_topic,
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'),
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                auto_offset_reset='latest', group_id='strategy-engine-group-1'
            )
        except Exception as e:
            print(f"Failed to initialize Kafka Consumer: {e}. Using mock consumer.")
            self.kafka_consumer = self._get_mock_kafka_consumer()

    def _get_mock_kafka_consumer(self):
        class MockMessage:
            def __init__(self, value): self.value = value
        class MockKafkaConsumer:
            def __init__(self, topic, strategies):
                self._topic = topic
                self.strategies = strategies
                self._messages = self._generate_test_data()
            def _generate_test_data(self):
                messages = []
                # RSI test data: flat, then dip (BUY), then spike (SELL)
                prices = [100.0] * 20 + [95.0, 94.0, 93.0, 92.0, 91.0] + [105.0, 106.0, 107.0, 108.0, 109.0]
                for i, price in enumerate(prices):
                    messages.append(MockMessage({'product_id': 'BTC-USD', 'price': price, 'side': 'BUY'}))
                return messages
            def __iter__(self): return iter(self._messages)
            def close(self): pass
        return MockKafkaConsumer(self.kafka_topic, self.strategies)

    def _initialize_product_state(self, product_id):
        if product_id not in self.product_states:
            print(f"Initializing strategies for {product_id}...")
            self.product_states[product_id] = {
                'sma_crossover': {
                    'prices': collections.deque(maxlen=self.strategies['sma_crossover']['long_window']),
                    'prev_short_sma': None, 'prev_long_sma': None,
                },
                'rsi': {
                    'prices': collections.deque(maxlen=self.strategies['rsi']['window'] + 1),
                    'prev_rsi': None
                }
            }

    async def _process_trade_message(self, message):
        trade_data = message.value
        product_id = trade_data['product_id']
        price = trade_data['price']
        self._initialize_product_state(product_id)

        # Dispatch to all registered strategies
        await self._process_sma_crossover(product_id, price)
        await self._process_rsi_strategy(product_id, price)

    async def _process_sma_crossover(self, product_id, price):
        state = self.product_states[product_id]['sma_crossover']
        state['prices'].append(price)

        cfg = self.strategies['sma_crossover']
        if len(state['prices']) < cfg['long_window']: return

        prices_array = np.array(state['prices'])
        short_sma = np.mean(prices_array[-cfg['short_window']:])
        long_sma = np.mean(prices_array)

        signal = None
        if state['prev_short_sma'] is not None:
            if short_sma > long_sma and state['prev_short_sma'] <= state['prev_long_sma']:
                signal = OrderSide.BUY
            elif short_sma < long_sma and state['prev_short_sma'] >= state['prev_long_sma']:
                signal = OrderSide.SELL

        state['prev_short_sma'], state['prev_long_sma'] = short_sma, long_sma
        if signal: await self._create_and_execute_order(product_id, signal, price, "SMA Crossover")

    async def _process_rsi_strategy(self, product_id, price):
        state = self.product_states[product_id]['rsi']
        state['prices'].append(price)

        cfg = self.strategies['rsi']
        if len(state['prices']) < cfg['window'] + 1: return

        deltas = pd.Series(state['prices']).diff().dropna()
        gain = deltas.clip(lower=0).ewm(com=cfg['window'] - 1, min_periods=cfg['window']).mean().iloc[-1]
        loss = -deltas.clip(upper=0).ewm(com=cfg['window'] - 1, min_periods=cfg['window']).mean().iloc[-1]

        if loss == 0: rsi = 100
        else: rs = gain / loss; rsi = 100 - (100 / (1 + rs))

        print(f"[{product_id}] RSI: {rsi:.2f}")

        signal = None
        if state['prev_rsi'] is not None:
            if rsi < cfg['oversold_threshold'] and state['prev_rsi'] >= cfg['oversold_threshold']:
                signal = OrderSide.BUY # Oversold, signal to buy
            elif rsi > cfg['overbought_threshold'] and state['prev_rsi'] <= cfg['overbought_threshold']:
                signal = OrderSide.SELL # Overbought, signal to sell

        state['prev_rsi'] = rsi
        if signal: await self._create_and_execute_order(product_id, signal, price, "RSI")

    async def _create_and_execute_order(self, product_id, side, price, strategy_name):
        order_size = 0.01
        order = Order(product_id=product_id, side=side, size=order_size, price=price)

        if self.risk_engine.check_pre_trade_risk(order):
            print(f"[{product_id}] --- {side.value} SIGNAL ({strategy_name}): Order APPROVED. Sending to ExecutionHandler. ---")
            await self.execution_handler.execute_order(order)
        else:
            print(f"[{product_id}] --- {side.value} SIGNAL ({strategy_name}): Order REJECTED by Risk Engine. ---")

    async def _run_kafka_consumer(self):
        print("Starting Kafka consumer loop...")
        try:
            for message in self.kafka_consumer:
                await self._process_trade_message(message)
            print("Finished processing all messages in mock consumer.")
        except asyncio.CancelledError:
            print("Kafka consumer loop cancelled.")
        finally:
            print("Kafka consumer loop stopped.")

    async def start(self):
        print("StrategyEngine starting...")
        self._consumer_task = asyncio.create_task(self._run_kafka_consumer())

    async def stop(self):
        print("StrategyEngine stopping...")
        if self._consumer_task:
            self._consumer_task.cancel()
            await asyncio.gather(self._consumer_task, return_exceptions=True)
        if self.kafka_consumer:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self.kafka_consumer.close)
            print("Kafka consumer closed.")
        print("StrategyEngine stopped.")
