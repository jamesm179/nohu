import asyncio
import pytest
from unittest.mock import MagicMock, patch

# Since we are testing components in isolation, we need to adjust the path
# to allow for correct module imports.
from src.data.market_data_pipeline import MarketDataPipeline

@pytest.fixture
def mock_kafka_producer():
    """Fixture to mock the KafkaProducer."""
    # We use patch on the location where the object is *looked up*, not where it's defined.
    with patch('src.data.market_data_pipeline.KafkaProducer') as mock_producer_class:
        mock_instance = MagicMock()
        mock_producer_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_redis_client():
    """Fixture to mock the Redis client."""
    with patch('src.data.market_data_pipeline.redis.Redis') as mock_redis_class:
        mock_instance = MagicMock()
        # Mock the ping method to avoid connection errors on init
        mock_instance.ping.return_value = True
        mock_redis_class.return_value = mock_instance
        yield mock_instance

@pytest.mark.asyncio
async def test_message_handler_normalizes_and_sends_data(mock_kafka_producer, mock_redis_client):
    """
    Tests that the message handler correctly processes a raw message,
    normalizes it, and sends it to Kafka and Redis.
    """
    # 1. Arrange
    # Initialize the pipeline; the mocks will prevent real connections.
    pipeline = MarketDataPipeline(product_ids=["BTC-USD"])
    # Manually replace clients with mocks if initialization is complex
    pipeline.kafka_producer = mock_kafka_producer
    pipeline.redis_client = mock_redis_client

    raw_message = """
    {
        "channel": "market_trades",
        "client_id": "",
        "timestamp": "2025-01-01T12:00:00.123456Z",
        "sequence_num": 0,
        "events": [
            {
                "type": "snapshot",
                "trades": [
                    {
                        "trade_id": "123",
                        "product_id": "BTC-USD",
                        "price": "50000.00",
                        "size": "0.5",
                        "side": "BUY",
                        "time": "2025-01-01T12:00:00.123Z"
                    }
                ]
            }
        ]
    }
    """

    # 2. Act
    await pipeline._message_handler(raw_message)

    # 3. Assert
    # Assert Kafka producer was called correctly
    expected_kafka_payload = {
        'timestamp': '2025-01-01T12:00:00.123Z',
        'product_id': 'BTC-USD',
        'price': 50000.0,
        'size': 0.5,
        'side': 'BUY',
        'exchange': 'coinbase'
    }
    pipeline.kafka_producer.send.assert_called_once_with(
        "market_data.trades",
        value=expected_kafka_payload
    )

    # Assert Redis client was called correctly
    expected_redis_key = "latest_price:BTC-USD"
    expected_redis_value = 50000.0
    pipeline.redis_client.set.assert_called_once_with(
        expected_redis_key,
        expected_redis_value
    )
