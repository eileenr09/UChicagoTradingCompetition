from computebot import Compute
import polars as pl
import numpy as np
import math 
import time
import asyncio
from utcxchangelib import xchange_client

class APTBot(Compute):
    """Specialized bot for APT symbol"""
    
    def __init__(self, parent_client=None):
        
        super().__init__(parent_client)
        #print(parent_client)
        self.symbol = "APT"
        self.earnings = None
        self.fair_value = None
        self.spread = None 
        self.pe_ratio = 10 # for practice rounds
        self.trade_count = 0
        self.trading_frequency = 50
        # Avellaneda–Stoikov parameters 
        self.T = 15 * 60 # 15 minute horizon 
        self.S0 = None
        self.deltaBid = None 
        self.deltaAsk = None 
        self.A = 0.05 
        self.sigma = None
        self.k = .4
        self.q_tilde = 10 
        self.gamma = 0.1 / self.q_tilde
        self.n_steps = int(self.T)
        self.n_paths = 500 
        self.received_earnings = False
        
        self.open_orders = []
        
        # Volatility calculation
        self.volatility_window = 20  # Window for volatility calculation
        self.volatility = 0.0  # Current volatility
        
        self.rolling_count = 0
        
        self.fair_value_regression = None
        
        # Regression parameters
        self.alpha = 0.0  # Spread coefficient
        self.beta = 0.0   # Order book imbalance coefficient
        self.regression_window = 5  # Window for regression
        # Order book imbalance tracking with Polars
        self.imbalance_timeseries = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "imbalance": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64
        })
        
        self.news_update_freeze = False
        self.alpha = 0.0  # Spread coefficient
        self.beta = 0.0   # Order book imbalance coefficient
        self.regression_weight = .6 #weight for regression fair value, start with 0
        
        # GARCH parameters random initialization
        self.omega_garch, self.alpha_garch, self.beta_garch = [0.1, 0.05, 0.94]
        
        
    def calc_bid_ask_price(self, symbol=None, t=None):
        return super().calc_bid_ask_price(self.symbol, t)
    
    def get_q_tilde(self):
        return self.q_tilde
    
    def get_avellaneda_stoikov_params(self):
        return self.gamma, self.k, self.fair_value, self.T, self.sigma
    
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
        lob_df = self.parent_client.stock_LOB_timeseries["APT"]
        
        # If the timeseries is empty, return 0
        if len(lob_df) == 0:
            return 0.0
            
        # Get the last row of the timeseries
        last_row = lob_df.tail(1)
        
        # Calculate total bid volume from all 4 price levels
        bid_volume = (
            last_row["best_bid_qt"].item() +  # Level 1
            last_row["2_bid_qt"].item() +     # Level 2
            last_row["3_bid_qt"].item() +     # Level 3
            last_row["4_bid_qt"].item()       # Level 4
        )
        
        # Calculate total ask volume from all 4 price levels
        ask_volume = (
            last_row["best_ask_qt"].item() +  # Level 1
            last_row["2_ask_qt"].item() +     # Level 2
            last_row["3_ask_qt"].item() +     # Level 3
            last_row["4_ask_qt"].item()       # Level 4
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
        
        
    # def calc_bid_ask_spread(self):
        
    #     book = self.parent_client.order_books["APT"]

    #     sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
    #     sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
    #     #print("Sorted Bids:", sorted_bids)
    #     #print("Sorted Asks:", sorted_asks)
    
    #     best_bid,_ = max(sorted_bids) if len(sorted_bids) > 0 else None 
    #     #print("best bid: ", best_bid)
    #     best_ask,_ = min(sorted_asks) if len(sorted_asks) > 0 else None 
    #     #print("best ask: ", best_ask)

    #     if best_bid is None or best_ask is None: 
    #         return None 
        
    #     self.spread = (best_ask - best_bid) / 100
    #     return self.spread

    # def calc_bid_ask_price(self, t):
    #     """Override with APT-specific spread calculation"""

    #     positions = self.parent_client.positions.get("APT", 0)

    #     # print all variables 
    #     #print("gamma: ", self.gamma)
    #     #print("k: ", self.k)
    #     #print("T: ", self.T)
    #     #print("t:", t)
    #     #print("S0: ", self.S0)
    #     #print("A: ", self.A)
    #     #print("sigma: ", self.sigma)
    #     #print("q_tilde: ", self.q_tilde)
    #     #print("n_steps: ", self.n_steps)
    #     #print("n_paths: ", self.n_paths)
    #     #print("positions: ", self.parent_client.positions.get("APT", 0))

    #     constant_term = (1 / self.gamma) * math.log(1 + self.gamma / self.k) 

    #     #print("constant term: ", constant_term)

    #     reservation_price = self.calc_reservation_price(t)

    #     self.deltaBid = max(self.gamma * (self.sigma ** 2) * self.T * (positions + 0.5) + constant_term, 0)
    #     self.deltaAsk = max(-self.gamma * (self.sigma ** 2) * self.T * (positions - 0.5) + constant_term, 0)

    #     delta_bid_price = int(reservation_price - self.deltaBid ) * 100 
    #     #print("delta ver. bid price: ", delta_bid_price )
    #     delta_ask_price = int(reservation_price + self.deltaAsk) * 100
    #     #print("delta ver. ask price: ", delta_ask_price )
        
    #     # using spread instead 
    #     spread = self.calc_bid_ask_spread()
    #     #print("spread: ", spread)
    #     bid_price = int((reservation_price - spread / 2) * 100)
    #     #print("spead ver. bid price: ", bid_price )
    #     ask_price = int((reservation_price + spread / 2) * 100)
    #     #print("spead ver. ask price: ", ask_price )

    #     return delta_bid_price, delta_ask_price
        
    # def calc_reservation_price(self, t):
    #     """We use the avellinda stoikov model to calculate the reservation price for 
    #     interday trading p_{\ text{mm}} = s - q \cdot \gamma \sigma^2 (T - t)"""
    #     self.S0 = self.get_fair_value()
    #     self.sigma = self.S0 * 0.02 / math.sqrt(self.T)
    #     dt = self.T - t # t is the current time stamp 

    #     positions = self.parent_client.positions.get("APT", 0)

    #     reservation_price = self.S0 - positions * self.gamma * self.sigma**2 * dt
    #     #print("reservation price: ", reservation_price) 
    #     return reservation_price
    
    def calc_fair_value(self):
        # add regression to fair value and weight it according to volatility
        if self.news_update_freeze == True:
            if self.rolling_count > 4: #our regression window is 5 so remove freeze for next update
                self.news_update_freeze = False
            self.fair_value_regression = None # in freeze dont use regression
        else:
            self.fair_value_regression = self.calc_regression_fair_value(self.symbol, self.rolling_count)
            print("regression_fair_value: ", self.fair_value_regression)
        if self.earnings is None: 
            return 
        fundamental_fair_value = 100 * self.earnings / self.pe_ratio
        print("fundamental_fair_value: ", fundamental_fair_value)
        if self.fair_value_regression is None:
            self.fair_value = fundamental_fair_value
        else:   
            self.fair_value = self.regression_weight * self.fair_value_regression + (1 - self.regression_weight) * fundamental_fair_value
        
        current_time = int(time.time()*100)/100 - self.parent_client.start_time
        if self.fair_value is None:
            return None
        self._update_fair_value_timeseries(current_time, self.fair_value)

    
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
        
    def handle_earnings_update(self, earnings):  
        self.earnings = earnings 
        self.received_earnings = True
        self.fair_value = 100 * self.earnings / self.pe_ratio #dont want to use regression for new information
        current_time = int(time.time()*100)/100 - self.parent_client.start_time
        self._update_fair_value_timeseries(current_time, self.fair_value)
        #print("handling earnings: ", self.fair_value)
        
        self.parent_client.pnl_timeseries = self.parent_client.pnl_timeseries.with_columns(
            pl.when(pl.col("timestamp") == pl.col("timestamp").max())
                .then(2)
                .otherwise(pl.col("is_news_event"))
                .alias("is_news_event")
        )
        
    def get_fair_value(self): 

        self.S0 = self.fair_value
        current_time = int(time.time()*100)/100 - self.parent_client.start_time
        return self.fair_value 
    
    def _update_fair_value_timeseries(self, timestamp, fair_value):
        """
        Update the fair value timeseries with a new snapshot
        """
        new_row = pl.DataFrame([{
            "timestamp": timestamp,
            "fair_value": int(fair_value)
        }])
        self.parent_client.fair_value_timeseries["APT"] = pl.concat([self.parent_client.fair_value_timeseries["APT"], new_row])
        #print("self.parent_client.fair_value_timeseries['APT']: ", self.parent_client.fair_value_timeseries["APT"])
    
    # async def handle_trade(self):
    #     with self.parent_client._lock: 
    #         latest_timestamp = int(time.time()) - self.parent_client.start_time
    #         if latest_timestamp is None:
    #             return 
    #         #print("type of latest_timestamp: ", type(latest_timestamp))
    #         bid_price, ask_price = self.calc_bid_ask_price(latest_timestamp)
    #     #print("========================================")
    #     #print("Adjusted Bid Price:", bid_price)
    #     await self.parent_client.place_order("APT",self.q_tilde, xchange_client.Side.BUY, bid_price)
    #     #print("Adjusted Ask Price:", ask_price)
    #     await self.parent_client.place_order("APT",self.q_tilde, xchange_client.Side.SELL, ask_price)
    #     #print("my positions:", self.parent_client.positions)
        
    
    def increment_trade(self):
        try:
            self.trade_count += 1
            #dont trade apt while testing
            # if self.trade_count % self.trading_frequency == 0:
            #     try:
            #         self.handle_trade("APT")
            #     except Exception as e:
            #         print(f"Error in handle_trade for APT: {e}")
            #         import traceback
            #         traceback.print_exc()
        except Exception as e:
            print(f"Unexpected error in increment_trade for APT: {e}")
            import traceback
            traceback.print_exc()
            
            
    def unstructured_update(self, news_data):
        with self.parent_client._lock:
            self.parent_client.pnl_timeseries = self.parent_client.pnl_timeseries.with_columns(
                pl.when(pl.col("timestamp") == pl.col("timestamp").max())
                .then(1)
                .otherwise(pl.col("is_news_event"))
                .alias("is_news_event")
            )
    
    def calc_volatility(self, omega_garch, alpha_garch, beta_garch):
        params = super().calc_volatility(self.omega_garch, self.alpha_garch, self.beta_garch, index= self.regression_window)
        if params is not None and not any(np.isnan(p) for p in params):
            self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma = params
            return params
        return self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma
    
    
        #print("volatility: ", self.sigma)
        
    def handle_news_update(self):
        self.rolling_count = 0
        self.news_update_freeze = True
        self.trade_count = self.trading_frequency/2
        
    def unstructured_update(self, news_data):
        self.handle_news_update()
    
    def handle_snapshot(self):
        self.rolling_count += 1
        self._calculate_order_book_imbalance()
        if self.received_earnings == False:
            self.fair_value = self.calc_regression_fair_value(self.symbol, self.rolling_count)
            self.fair_value_regression = self.fair_value
        else:
            self.fair_value = self.calc_fair_value()
            self.fair_value_regression = self.fair_value
    ##### closing positions #####
    def begin_closing_positions(self):
        self.closing_positions = True
        
    
    def cancel_all_orders(self):
        pass