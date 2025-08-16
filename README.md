# Production-Grade Cryptocurrency Scalping Bot

This repository contains the source code for a high-frequency cryptocurrency scalping bot, built with an institutional-grade architecture. The system is designed for full autonomy, robustness, and profitability.

## Overview

The bot connects to multiple cryptocurrency exchanges to stream real-time market data, identify trading opportunities using sophisticated strategies, and execute trades while managing risk according to a predefined framework.

The core design principles are:
- **Low Latency**: Optimized for sub-millisecond execution.
- **Concurrency**: Built with Python's `asyncio` for handling thousands of concurrent operations.
- **Scalability**: Architected to scale horizontally across multiple assets and exchanges.
- **Robustness**: Includes comprehensive risk management, monitoring, and automated recovery.

## System Architecture

The system is built on a microservices-oriented architecture, with the following key components:

-   **Application Core (`app`)**: The main Python application that houses the trading logic.
    -   **`MarketDataPipeline`**: Ingests and processes real-time market data from exchanges via WebSockets.
    -   **`StrategyEngine`**: Generates trading signals and manages order execution.
    -   **`RiskEngine`**: Monitors portfolio exposure and enforces risk limits.
-   **PostgreSQL (`postgres`)**: The primary database for storing historical trade data, performance metrics, and configuration.
-   **Redis (`redis`)**: Used as a real-time cache for market data, order book state, and other low-latency data.
-   **Apache Kafka (`kafka`)**: Acts as a high-throughput, distributed event streaming platform for communication between components (e.g., market data events, trade signals).

All services are containerized using Docker for consistency and ease of deployment.

## Getting Started

### Prerequisites

-   [Docker](https://www.docker.com/get-started)
-   [Docker Compose](https://docs.docker.com/compose/install/)

### Setup & Configuration

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Environment Variables:**
    The application is configured using environment variables. While default values are provided in `docker-compose.yml` for development, you can create a `.env` file to override them.

    *Create a file named `.env` in the project root:*
    ```env
    # .env file for local development
    POSTGRES_USER=myuser
    POSTGRES_PASSWORD=mypassword
    POSTGRES_DB=mydb
    ```
    *The `docker-compose.yml` file will automatically pick up variables from a `.env` file.*

### Running the System

1.  **Build and start all services in detached mode:**
    ```bash
    docker-compose up --build -d
    ```

2.  **Check the status of the containers:**
    ```bash
    docker-compose ps
    ```
    You should see `trading_bot_app`, `trading_bot_postgres`, `trading_bot_redis`, `trading_bot_zookeeper`, and `trading_bot_kafka` running.

3.  **Accessing the application logs:**
    ```bash
    docker-compose logs -f app
    ```

4.  **Executing commands inside the application container:**
    To run tests or other commands, you can get a shell inside the `app` container:
    ```bash
    docker-compose exec app /bin/bash
    ```

5.  **Stopping the system:**
    ```bash
    docker-compose down
    ```

## Project Structure

```
├── .github/            # CI/CD workflows
├── config/             # Strategy and risk configuration files
├── docs/               # System documentation and diagrams
├── scripts/            # Deployment and operational scripts
├── src/                # Main application source code
│   ├── data/           # Market data pipeline
│   ├── strategy/       # Trading strategy engine
│   ├── risk/           # Risk management engine
│   └── utils/          # Shared utilities
├── tests/              # Unit and integration tests
├── Dockerfile          # Defines the application container
├── docker-compose.yml  # Orchestrates all services
├── main.py             # Application entry point
└── requirements.txt    # Python dependencies
```

## Next Steps

-   Implement WebSocket client for Binance in `MarketDataPipeline`.
-   Develop a basic RSI-based trading strategy in `StrategyEngine`.
-   Implement pre-trade risk checks in `RiskEngine`.
-   Set up unit tests with `pytest`.
