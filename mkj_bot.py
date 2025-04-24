from computebot import Compute
import polars as pl
import numpy as np
import asyncio
import time
import math
from utcxchangelib import xchange_client
class MKJBot(Compute):
    """Specialized bot for MKJ symbol"""
    
    def __init__(self, parent_client=None):
        super().__init__(parent_client)
        self.symbol = "MKJ"
        self.trade_count = 0
        self.fair_value = None
        
        # Order book tracking
        
        # Order book imbalance tracking with Polars
        self.imbalance_timeseries = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "imbalance": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64
        })
        
        
        # Volatility calculation
        self.volatility_window = 20  # Window for volatility calculation
        self.volatility = 0.0  # Current volatility
        
        # Regression parameters
        self.alpha = 0.0  # Spread coefficient
        self.beta = 0.0   # Order book imbalance coefficient
        self.regression_window = 5  # Window for regression
        
        
        self.spread = None 
        self.trade_count = 0
        self.trading_frequency = 50
        # Avellaneda–Stoikov parameters 
        self.T = 15 * 60 # 15 minute horizon 
        self.S0 = 1000
        self.deltaBid = None 
        self.deltaAsk = None 
        self.sigma = None
        self.A = 0.05 
        self.k = .4
        self.q_tilde = 10 
        self.gamma = 0.01 / self.q_tilde
        self.n_steps = int(self.T)
        self.n_paths = 500 
        
        self.rolling_count = 0
        
        self.fair_value_regression = None
        
        self.volatility_history = []
        
        self.open_orders = []
        
        # GARCH parameters random initialization
        self.omega_garch, self.alpha_garch, self.beta_garch = [0.1, 0.05, 0.94]
        
    
    def get_imbalances(self, regression_window):
        return self.imbalance_timeseries.select("imbalance").tail(regression_window-1).to_series().to_numpy()
    
    def get_regression_window(self):
        return self.regression_window
    
    def update_vol_params(self, omega_garch, alpha_garch, beta_garch, sigma):
        self.omega_garch = omega_garch
        self.alpha_garch = alpha_garch
        self.beta_garch = beta_garch
        self.sigma = sigma
        
    def get_vol_params(self):
        return self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma
    
    def calc_bid_ask_spread(self):
        """
        Calculate the bid-ask spread for MKJ
        
        Returns:
            tuple: (bid_price, ask_price)
        """
        # Get the current fair value
        fair_value = self.calc_fair_value()
        
        # Calculate a dynamic spread based on volatility
        # Higher volatility = wider spread
        spread = max(self.min_spread, min(self.max_spread, self.volatility * 2))
        
        # Set bid and ask prices around the fair value
        bid_price = max(0, fair_value - spread/2)
        ask_price = min(100, fair_value + spread/2)
        
        return bid_price, ask_price
        
    def calc_fair_value(self):
        """
        Calculate the fair value of MKJ based on order book dynamics
        
        Returns:
            float: Fair value of MKJ
        """
        self.fair_value_regression = super().calc_regression_fair_value(self.symbol, self.rolling_count)
        print("symbol: ", self.symbol)
        print("fair_value: ", self.fair_value_regression)
        print("rolling_count: ", self.rolling_count)
        if self.fair_value_regression is None:
            return None
        timestamp = int(time.time()*100)/100 - self.parent_client.start_time
        self._update_fair_value_timeseries(timestamp, self.fair_value_regression)
        return self.fair_value_regression
    
    def calc_volatility(self, omega_garch, alpha_garch, beta_garch):
        params = super().calc_volatility(omega_garch, alpha_garch, beta_garch, index= self.regression_window)
        if params is not None and not any(np.isnan(p) for p in params):
            self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma = params
            self.volatility_history.append(self.sigma)
            return params
        self.volatility_history.append(self.sigma)
        print("volatility history: ", self.volatility_history)
        return self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma

    def _update_fair_value_timeseries(self, timestamp, fair_value):
        """
        Update the fair value timeseries with a new snapshot
        """
        new_row = pl.DataFrame([{
            "timestamp": timestamp,
            "fair_value": int(fair_value),
        }])
        self.parent_client.fair_value_timeseries["MKJ"] = pl.concat([self.parent_client.fair_value_timeseries["MKJ"], new_row])
    
    def _calculate_order_book_imbalance(self):
        """
        Calculate the order book imbalance using the parent client's LOB_timeseries
        
        Returns:
            float: Order book imbalance between -1 and 1
        """
            
        # Get the LOB timeseries from the parent client
        lob_df = self.parent_client.stock_LOB_timeseries["MKJ"]
        
        # If the timeseries is empty, return 0
        if len(lob_df) == 0:
            return 0.0
            
        # Get the last row of the timeseries
        last_row = lob_df.tail(1)
        
        # Calculate total bid volume from all 4 price levels
        bid_volume = (
            last_row["best_bid_qt"].item() +  # Level 1
            last_row["2_bid_qt"].item()     # Level 2
        )
        
        # Calculate total ask volume from all 4 price levels
        ask_volume = (
            last_row["best_ask_qt"].item() +  # Level 1
            last_row["2_ask_qt"].item()     # Level 2
        )
        
        # Calculate total volume
        total_volume = bid_volume + ask_volume
        if total_volume == 0:
            return 0.0
        
        # Imbalance is (bid_volume - ask_volume) / (bid_volume + ask_volume)
        imbalance = (bid_volume - ask_volume) / total_volume
        timestamp = self.parent_client.stock_LOB_timeseries[self.symbol]["timestamp"].tail(1).item()
        #print("imbalance: ", imbalance)
        self._update_imbalance_timeseries(timestamp, imbalance, bid_volume, ask_volume)
        return imbalance
    
    def _update_imbalance_timeseries(self, timestamp, imbalance, bid_volume, ask_volume):
        """
        Update the imbalance timeseries with a new snapshot
        
        Args:
            timestamp: Current timestamp
            imbalance: Calculated order book imbalance
            bid_volume: Total bid volume
            ask_volume: Total ask volume
        """
        # Create a new row
        new_row = pl.DataFrame([{
            "timestamp": timestamp,
            "imbalance": imbalance,
            "bid_volume": bid_volume,
            "ask_volume": ask_volume
        }])
        
        # Append to the timeseries
        #print("new_row: ", new_row)
        self.imbalance_timeseries = pl.concat([self.imbalance_timeseries, new_row])
    
    def get_fair_value(self):
        return self.fair_value
    
    async def process_update(self, update):
        """
        Process updates from the parent client
        
        Args:
            update: The update to process
        """
        # This method would be called by the parent client's compute thread
        # It would process updates and calculate new fair values
        pass
    # async def handle_trade(self):
    #     latest_timestamp = int(time.time()) - self.parent_client.start_time
    #     if latest_timestamp is None:
    #         return 
    #     #print("type of latest_timestamp: ", type(latest_timestamp))
    #     bid_price, ask_price = self.calc_bid_ask_price(latest_timestamp)
    #     #print("========================================")
    #     #print("handle_trade for ", self.symbol)
    #     #print("fair_value: ", self.fair_value)
    #     #print("Adjusted Bid Price:", bid_price)
    #     #print("Adjusted Ask Price:", ask_price)
    #     #print("========================================")
    #     await self.parent_client.place_order(self.symbol, self.q_tilde, xchange_client.Side.BUY, bid_price)
    #     await self.parent_client.place_order(self.symbol, self.q_tilde, xchange_client.Side.SELL, ask_price)
    #     #print("my positions:", self.parent_client.positions)
    
    
    
    def increment_trade(self):
        """
        Increment the trade counter
        """
        # For MKJ, we want to only update our model based on a new LOB snapshot
        # do we want to perform a trade after a fixed number of trades or time interval?
        self.trade_count += 1
        # if self.trade_count % self.trading_frequency == 0:
        #     self.handle_trade("MKJ")
        
    def handle_snapshot(self):
        #print("handle_snapshot")
        #only want to update the fair value when we have a new LOB snapshot
        self.rolling_count += 1
        self.fair_value = self.calc_fair_value()
        
    def calc_bid_ask_price(self, symbol=None, t=None):
        return super().calc_bid_ask_price(self.symbol, t)
    
    def get_avellaneda_stoikov_params(self):
        return self.gamma, self.k, self.fair_value, self.T, self.sigma
    
    
    def unstructured_update(self, news_data):
        """
        Handle unstructured news updates
        
        Args:
            news_data: The news data
        """
        # For MKJ, we might want to update our model based on news
        pass
    
    def get_q_tilde(self):
        return self.q_tilde
    
    def handle_news_update(self):
        self.rolling_count = 0
        self.trade_count = self.trading_frequency/2
        
    def unstructured_update(self, news_data):
        self.handle_news_update()
    
    ##### closing positions #####
    def begin_closing_positions(self):
        self.closing_positions = True
        
    
    def cancel_all_orders(self):
        pass
    
    def handle_news_update(self):
        self.rolling_count = 0
        self.news_update_freeze = True
        self.trade_count = self.trading_frequency/2
        
    def unstructured_update(self, news_data):
        self.handle_news_update()
        
    
        
    def handle_snapshot(self):
        self.rolling_count += 1
        self.regression_fair_value = self.calc_regression_fair_value(self.symbol, self.rolling_count)
        self._calculate_order_book_imbalance()
    