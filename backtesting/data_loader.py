class DataLoader:
    """
    Loads historical market data for backtesting.

    This component will be responsible for fetching data from various
    sources, such as a database (PostgreSQL) or flat files (CSV), and
    providing it to the BacktestingEngine in a clean, iterable format.
    """
    def __init__(self, source_type, path=None):
        self.source_type = source_type
        self.path = path
        print(f"Initializing DataLoader for source: {source_type}")

    def load_data(self):
        """
        Loads the data from the specified source.
        """
        print(f"Loading data from {self.path}...")
        # In a real implementation, this would connect to the DB or read a CSV.
        # It would yield data tick by tick.
        yield
