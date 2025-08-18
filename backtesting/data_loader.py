import pandas as pd

class DataLoader:
    """
    Loads historical market data for backtesting from a CSV file.
    """
    def __init__(self, csv_path: str):
        """
        Initializes the DataLoader.

        Args:
            csv_path (str): The path to the CSV file containing historical data.
        """
        self.csv_path = csv_path
        print(f"Initializing DataLoader for source: {self.csv_path}")

    def load_data(self):
        """
        Loads data from the CSV file and yields it row by row as a dictionary.

        This generator function mimics a real-time data stream.
        """
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} records from {self.csv_path}.")

            # Convert price and size to numeric types
            df['price'] = pd.to_numeric(df['price'])
            df['size'] = pd.to_numeric(df['size'])

            for row in df.to_dict(orient='records'):
                yield row
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.csv_path}")
            return
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return
