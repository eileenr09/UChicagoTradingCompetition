from computebot import Compute
import polars as pl
import numpy as np
from scipy.stats import norm
import asyncio
import time
import math
from utcxchangelib import xchange_client

class DLRBot(Compute):
    """
    DLR Bot for calculating fair value based on signature process
    """
    
    def __init__(self, parent_client=None):
        super().__init__(parent_client)
        
        # Model parameters
        self.S0_sig = 5000  # Initial signature count
        self.received_news = False
        self.alpha = 1.0630449594499  # Growth factor
        self.log_alpha = np.log(self.alpha)  # Log growth factor
        self.sigma_sig = 0.006  # Volatility
        self.S_star = 100000  # Success threshold
        self.log_S_star = np.log(self.S_star)
        self.T_sig = 50  # Total time periods (deadline)
        self.update_counter = 0
        self.symbol = "DLR"
        
        self.open_orders = []
        
        self.imbalance_timeseries = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "imbalance": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64
        })
         
        # Current signature count
        self.current_signatures = self.S0_sig
        self.fair_value = None
        #set to 0 to start with only fundamental fair value for now
        self.regression_weight = 0.5
        self.alpha = 0.0  # Spread coefficient
        self.beta = 0.0   # Order book imbalance coefficient
        self.regression_window = 5  # Window for regression
        
        self.fair_value_regression = None
        
        # volatility parameters
        self.omega_garch, self.alpha_garch, self.beta_garch = [0.1, 0.05, 0.94]
        
        self.spread = None 
        self.trade_count = 0
        self.trading_frequency = 50
        # Avellaneda–Stoikov parameters 
        self.T = 15 * 60 # 15 minute horizon 
        self.S0 = 5000
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
        self.news_update_freeze = False
        
        # Track signature updates
        self.signature_history = []
        
    
    ######### regression functions #########
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
    
    def _calculate_order_book_imbalance(self):
        """
        Calculate the order book imbalance using the parent client's LOB_timeseries
        
        Returns:
            float: Order book imbalance between -1 and 1
        """
            
        # Get the LOB timeseries from the parent client
        lob_df = self.parent_client.stock_LOB_timeseries["DLR"]
        
        # If the timeseries is empty, return 0
        if len(lob_df) == 0:
            return 0.0
            
        # Get the last row of the timeseries
        last_row = lob_df.tail(1)
        
        # Calculate total bid volume from all 4 price levels
        bid_volume = (
            last_row["best_bid_qt"].item() +  # Level 1
            last_row["2_bid_qt"].item() +     # Level 2
            last_row["3_bid_qt"].item()    # Level 3
        )
        
        # Calculate total ask volume from all 4 price levels
        ask_volume = (
            last_row["best_ask_qt"].item() +  # Level 1
            last_row["2_ask_qt"].item() +     # Level 2
            last_row["3_ask_qt"].item()   # Level 3
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
    
    ######### fair value and market making functions #########
    
    def monte_carlo_vectorized(self, current_signatures, rounds_remaining, num_simulations):
        # Initialize an array for the signature counts in each simulation.
        signatures = np.full(num_simulations, current_signatures, dtype=float)
        
        for _ in range(rounds_remaining):
            # Compute parameters for the lognormal distribution for each simulation
            mu_vals = np.log(signatures) + self.log_alpha
            #print("mu_vals: ", mu_vals)
            sigma_vals = np.full(num_simulations, self.sigma_sig)
            
            # Draw new signatures for each simulation (vectorized)
            signatures = np.random.lognormal(mean=mu_vals, sigma=sigma_vals)
        # Calculate payoffs: $100 if signature count reaches 100,000, else $0.
        payoffs = np.where(signatures >= 100000, 100, 0)
        fair_value = np.mean(payoffs)
        
        return fair_value * 100
    
    def get_avellaneda_stoikov_params(self):
        return self.gamma, self.k, self.fair_value, self.T, self.sigma
        
    def calc_fair_value(self):
        """
        Calculate the fair value of DLR based on the lognormal signature process
        
        Returns:
            float: Fair value of DLR
        """
        
        #current_time = int(time.time()) - self.parent_client.start_time
        # Calculate time remaining
        
        #print("current time: ", current_time)
    
        fundamental_fair_value = self.monte_carlo_vectorized(self.current_signatures, self.T_sig - self.update_counter, 1000000)
        #print("monte_carlo_value: ", fair_value)
        print("update_counter: ", self.update_counter)
        print("T: ", self.T_sig)
        print("current_signatures: ", self.current_signatures)
        
        
        current_time = int(time.time()*100)/100 - self.parent_client.start_time
        
        if self.news_update_freeze == True:
            if self.rolling_count > 4: #our regression window is 5 so remove freeze for next update
                self.news_update_freeze = False
            self.fair_value_regression = None # in freeze dont use regression
        else:
            self.fair_value_regression = self.calc_regression_fair_value(self.symbol, self.rolling_count)
            print("regression_fair_value: ", self.fair_value_regression)
        
        print("symbol: ", self.symbol)
        print("regression_fair_value: ", self.fair_value_regression)
        print("fundamental_fair_value: ", fundamental_fair_value)
        
        if self.fair_value_regression is None:
            fair_value = fundamental_fair_value
        else:
            fair_value = self.regression_weight * self.fair_value_regression + (1 - self.regression_weight) * fundamental_fair_value
        
        if fair_value is None:
            return None
        self._update_fair_value_timeseries(current_time, fair_value)
        return fair_value
        
    
    def _update_fair_value_timeseries(self, timestamp, fair_value):
        """
        Update the fair value timeseries with a new snapshot
        """
        new_row = pl.DataFrame([{
            "timestamp": timestamp,
            "fair_value": int(fair_value)
        }])
        self.parent_client.fair_value_timeseries["DLR"] = pl.concat([self.parent_client.fair_value_timeseries["DLR"], new_row])
    
        
    def calc_bid_ask_price(self, symbol=None, t=None):
        return super().calc_bid_ask_price(self.symbol, t)
    
    
    async def process_update(self, update):
        """
        Process updates from the parent client
        
        Args:
            update: The update to process
        """
        # This method would be called by the parent client's compute thread
        # It would process updates and calculate new fair values
        pass
    
    async def bot_handle_trade_msg(self, symbol, price, qty):
        """
        Handle trade messages
        
        Args:
            symbol: The trading symbol
            price: The trade price
            qty: The trade quantity
        """
            
    def get_fair_value(self):
        return self.fair_value
    
    ######### news handling functions #########
    
    def increment_trade(self):
        self.trade_count += 1
        # if self.trade_count % self.trading_frequency == 0:
        #     self.handle_trade("DLR")
    
    def signature_update(self, new_signatures, cumulative):
        """
        Update the signature count
        
        Args:
            new_signatures: New signatures since last update
            cumulative: Cumulative signature count
        """
        # Update the current signature count
        self.received_news = True
        self.current_signatures = cumulative
        current_time = int(time.time()*100)/100 - self.parent_client.start_time
        # Add to history
        self.signature_history.append({
            "time": current_time,
            "signatures": cumulative
        })
        
        self.update_counter += 1
        
        # Calculate new fair value
        self.fair_value = self.calc_fair_value()
        #print("DLR fair value: ", self.fair_value)
        
    
    def unstructured_update(self, news_data):
        """
        Handle unstructured news updates
        
        Args:
            news_data: The news data
        """
        # For DLR, we might want to update our model based on news
        pass
    
    def get_q_tilde(self):
        return self.q_tilde
    
    def handle_news_update(self):
        self.rolling_count = 0
        self.news_update_freeze = True
        self.trade_count = self.trading_frequency/2
        
    def unstructured_update(self, news_data):
        self.handle_news_update()
        
    def update_rounds(self):
        num_days = (int(time.time()) - round(self.parent_client.start_time))/90 #90 seconds per day
        day_updates = round(num_days * 5)
        day_remainder = (int(time.time()*100)/100 - self.parent_client.start_time*num_days)
        
        day_remainder = round(day_remainder * 90)
        
        # 5 updates during day 15, 30, 45, 60, 75 seconds, too lazy to do this in a loop
        if day_remainder < 15:
            self.update_counter = 0
        elif day_remainder < 30:
            self.update_counter = 1
        elif day_remainder < 45:
            self.update_counter = 2
        elif day_remainder < 60:
            self.update_counter = 3
        elif day_remainder < 75:
            self.update_counter = 4
            
        self.update_counter += day_updates
        self.received_news = True
        print("update_counter: ", self.update_counter)

    def calc_volatility(self, omega_garch, alpha_garch, beta_garch):
        params = super().calc_volatility(omega_garch, alpha_garch, beta_garch, index= self.regression_window)
        if params is not None and not any(np.isnan(p) for p in params):
            self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma = params
            return params
        return self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma
    def handle_snapshot(self):
        if self.received_news == False:
            self.fair_value = self.calc_regression_fair_value(self.symbol, self.rolling_count)
        else:
            self.fair_value = self.calc_fair_value()
        self._calculate_order_book_imbalance()
        
    