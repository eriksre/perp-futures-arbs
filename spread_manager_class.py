import pandas as pd
from collections import deque, defaultdict
from webhook_class import Webhook
from datetime import datetime

class SpreadManager:
    def __init__(self) -> None:
        self.symbol_spread_counts = defaultdict(int)
        self.symbol_spread_history = defaultdict(lambda: deque(maxlen=15))
        self.webhook = Webhook(webhook_type='futures')  # Webhook is properly defined elsewhere

    def add_dataframe(self, df):
        self.update_spread_counts(df)

    def update_spread_counts(self, df):
        df = df.fillna(0)
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            adjusted_spread = row['adjusted_futures_spread_value']
            spread = row['futures_spread_value']
            meets_condition = (adjusted_spread > 0.02) or (spread > 0.015)

            history = self.symbol_spread_history[symbol]
            history.appendleft(meets_condition)
            self.symbol_spread_counts[symbol] = sum(history)

            if self.symbol_spread_counts[symbol] >= 4:
                self.send_alert(row)

    def send_alert(self, row):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        alert_message = (
            f"Symbol: {row['symbol']}\n"
            f"Spread: {row['futures_spread_value']}\n"
            f"Spread Instruments: {row['futures_spread_instruments']}\n"
            f"Adj Spread: {row['adjusted_futures_spread_value']}\n"
            f"Adj Spread Instruments: {row['adjusted_futures_spread_instruments']}\n"
            f"Time: {timestamp}"
        )
        self.webhook.send_message(alert_message)
