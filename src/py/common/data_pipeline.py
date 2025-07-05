from core.data import EnhancedDataLoader

class DataPipeline:
    """
    Base interface for loading and preprocessing market data.
    """
    def __init__(self):
        self.loader = EnhancedDataLoader()

    def load_data(self, start_date, end_date, symbols, bar_size):
        """
        Load and preprocess market data.
        Args:
            start_date: Start date for data loading.
            end_date: End date for data loading.
            symbols: List of symbols to load.
            bar_size: Granularity of data (e.g., 'daily', '1min').
        Returns:
            data: Loaded and preprocessed market data.
        """
        # For now, ignore bar_size (add support later if needed)
        return self.loader.download_data(symbols, start_date, end_date) 