import asyncio
from src.data import MarketDataPipeline
from src.strategy import StrategyEngine
from src.risk import RiskEngine

async def main():
    """
    Main function to initialize and run the trading bot.
    """
    print("Starting trading bot...")

    # Initialize the core components
    product_ids_to_trade = ["BTC-USD", "ETH-USD"]
    risk_engine = RiskEngine()
    market_data_pipeline = MarketDataPipeline(product_ids=product_ids_to_trade)
    strategy_engine = StrategyEngine(market_data_pipeline, risk_engine)

    # Start the components in the correct order
    await risk_engine.start()
    await market_data_pipeline.start()
    await strategy_engine.start()

    print("Trading bot is now running. Press Ctrl+C to stop.")

    try:
        # Keep the bot running indefinitely.
        # In a real application, this might be a more sophisticated event loop.
        while True:
            await asyncio.sleep(3600) # Sleep for an hour, or until interrupted
    except asyncio.CancelledError:
        print("Bot shutdown sequence initiated.")
    finally:
        # Gracefully stop the components in the reverse order of startup
        await strategy_engine.stop()
        await market_data_pipeline.stop()
        await risk_engine.stop()
        print("Trading bot has been shut down gracefully.")

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        print("Shutdown signal received. Cleaning up...")
        # Gather all tasks to cancel them
        tasks = asyncio.all_tasks(loop=loop)
        for task in tasks:
            task.cancel()
        # Wait for all tasks to be cancelled
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
        loop.close()
        print("Cleanup complete.")
