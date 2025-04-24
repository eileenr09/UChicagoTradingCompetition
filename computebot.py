# imports 
import polars as pl
import numpy as np
import asyncio
from scipy.optimize import minimize
import math
import time
from utcxchangelib import xchange_client
from arch import arch_model
# super class for all compute bots
class Compute: 

    def __init__(self, parent_client=None, symbol=None): 
        self.parent_client = parent_client
        self.symbol = symbol
        self.omega = None
        self.alpha = None
        self.beta = None
        self.sigma = None
        self.gamma = None
        self.k = None
        self.T = 15 * 60
        self.S0 = None
        self.deltaBid = None
        self.deltaAsk = None
        self.A = None
        
    

    def calc_fair_value(): 
        pass 
    
    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass
    
    def calc_best_bid_ask(self, symbol=None):
        
        book = self.parent_client.order_books[symbol]

        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        #print("Sorted Bids:", sorted_bids)
        #print("Sorted Asks:", sorted_asks)
        
    
        best_bid = max(sorted_bids) if len(sorted_bids) > 0 else None 
        #print("best bid: ", best_bid)
        best_ask = min(sorted_asks) if len(sorted_asks) > 0 else None 
        #print("best ask: ", best_ask)

        if best_bid is None or best_ask is None: 
            return None, None
        
        return best_bid[0], best_ask[0]

    def calc_bid_ask_price(self, symbol=None, t=None):

        positions = self.parent_client.positions.get(symbol, 0)

        # print all variables 
        #print("gamma: ", self.gamma)
        #print("k: ", self.k)
        #print("T: ", self.T)
        #print("t:", t)
        #print("S0: ", self.S0)
        #print("A: ", self.A)
        #print("sigma: ", self.sigma)
        #print("q_tilde: ", self.q_tilde)
        #print("n_steps: ", self.n_steps)
        #print("n_paths: ", self.n_paths)
        #print("positions: ", self.parent_client.positions.get("APT", 0))
        gamma, k, fair_value, T, sigma = self.get_avellaneda_stoikov_params()
        
        S0 = fair_value
        
        if fair_value is None:
            return None, None

        constant_term = (2 / gamma) * math.log(1 + gamma / k) 

        print("constant term: ", constant_term)

        reservation_price, sigma = self.calc_reservation_price(symbol, t, sigma, gamma, fair_value)
        if reservation_price is None:
            return None, None
        
        time_remaining = (T - t) / (self.T)
        
        spread = gamma * (sigma ** 2) * (time_remaining) + constant_term

        print("spread: ", spread)

        delta_bid_price = round(reservation_price - spread /2 )
        #print("delta ver. bid price: ", delta_bid_price )
        delta_ask_price = round(reservation_price + spread /2)
        
        #print("deltaBid: ", deltaBid)
        #print("deltaAsk: ", deltaAsk)
        #print("delta ver. ask price: ", delta_ask_price )
        
        # using spread instead 
        #spread = self.calc_bid_ask_spread()
        #print("spread: ", spread)
        #bid_price = int((reservation_price - spread / 2) * 100)
        #print("spead ver. bid price: ", bid_price )
        #ask_price = int((reservation_price + spread / 2) * 100)
        #print("spead ver. ask price: ", ask_price )

        return delta_bid_price, delta_ask_price
    
    def calc_reservation_price(self, symbol, t, sigma, gamma, fair_value):
        """We use the avellinda stoikov model to calculate the reservation price for 
        interday trading p_{\ text{mm}} = s - q \cdot \gamma \sigma^2 (T - t)"""
        S0 = fair_value
        if S0 is None:
            return None, None
        dt = (self.T - t) / self.T # t is the current time stamp 
        
        
        if sigma is None:
            sigma = S0 * 0.02 * 0.01

        positions = self.parent_client.positions.get(symbol, 0)

        reservation_price = S0 - positions * gamma * sigma**2 * dt
        print("reservation price: ", reservation_price) 
        print("sigma: ", sigma)
        print("gamma: ", gamma)
        print("dt: ", dt)
        print("S0: ", S0)
        print("positions: ", positions)

        return reservation_price, sigma
    
    def handle_trade(self, symbol):
        latest_timestamp = None
        latest_timestamp = int(time.time()*100)/100 - self.parent_client.start_time
        if latest_timestamp is None:
            return 
        #print("type of latest_timestamp: ", type(latest_timestamp))
        print("fair_value while trading: ", self.get_fair_value())
        print("symbol: ", symbol)
        bid_price, ask_price = self.calc_bid_ask_price(symbol, latest_timestamp)
        
        best_bid, best_ask = self.calc_best_bid_ask(symbol)
        
        if best_bid is None or best_ask is None:
            #dont perform trade if order book is empty, since numbers are off
            return
        
        if bid_price is None or ask_price is None:
            return
        
        if best_bid > ask_price:
            ask_price = best_bid
            bid_price = None # shouldnt be buying if undervalued
        elif best_ask < bid_price:
            bid_price = best_ask
            ask_price = None # shouldnt be selling if overvalued
            
        if bid_price is not None:
            self.parent_client.trade_queue.put({
            "symbol": symbol,
            "side": xchange_client.Side.BUY,
            "qty": self.get_q_tilde(),
            "price": bid_price
            })
            print(f"Added BUY order to queue: {symbol} {self.get_q_tilde()} @ {bid_price}")
        if ask_price is not None:
            self.parent_client.trade_queue.put({
            "symbol": symbol,
            "side": xchange_client.Side.SELL,
            "qty": self.get_q_tilde(),
            "price": ask_price
            })
            print(f"Added SELL order to queue: {symbol} {self.get_q_tilde()} @ {ask_price}")
        #print("========================================")
        #print("handle_trade for ", symbol)
        #print("Adjusted Bid Price:", bid_price)
        #print("Adjusted Ask Price:", ask_price)
        # Put buy order in queue
        #self.parent_client.trade_queue.put({
        #    "symbol": symbol,
        #    "side": xchange_client.Side.BUY,
        #    "qty": self.get_q_tilde(),
        #    "price": bid_price
        #})
        # print(f"Added BUY order to queue: {symbol} {self.get_q_tilde()} @ {bid_price}")
        
        # # Put sell order in queue
        # self.parent_client.trade_queue.put({
        #     "symbol": symbol,
        #     "side": xchange_client.Side.SELL,
        #     "qty": self.get_q_tilde(),
        #     "price": ask_price
        # })
        # print(f"Added SELL order to queue: {symbol} {self.get_q_tilde()} @ {ask_price}")
        #print("========================================")
        #print("my positions:", self.parent_client.positions)
        
    def increment_trade(self):
        pass
    
    def get_q_tilde(self):
        pass
        
    def unstructured_update(self, news_data):
        pass

    def _update_regression_parameters(self, imbalances, historical_midpoints, historical_spreads, rolling_count):
        """
        Update the regression parameters alpha and beta using fast OLS
        """
        regression_window = self.get_regression_window()
        if len(historical_midpoints) < rolling_count:
            return 0.1, 0.2
        
        # Prepare data for regression using last regression_window points
        midpoints = np.array(list(historical_midpoints[-rolling_count:]))
        spreads = np.array(list(historical_spreads[-rolling_count:]))
        
        # Calculate changes
        delta_midpoints = np.diff(midpoints)
        delta_spreads = np.diff(spreads)
        
        # # Get historical imbalances from the timeseries
        # if len(self.imbalance_timeseries) >= self.regression_window:
        #     # Use the most recent imbalances
        #     imbalances = self.imbalance_timeseries.select("imbalance").tail(self.regression_window-1).to_series().to_numpy()
        # else:
        #     # If we don't have enough imbalance data, use zeros
        #     imbalances = np.zeros(self.regression_window-1)
            
        # print("midpoints: ", midpoints)
        # print("spreads: ", spreads)
        # print("imbalances: ", imbalances)
        # print("delta_midpoints: ", delta_midpoints)
        # print("delta_spreads: ", delta_spreads)
        
        print("delta_midpoints: ", delta_midpoints)
        print("delta_spreads: ", delta_spreads)
        print("imbalances: ", imbalances)
        
        
        # Prepare X matrix (spread changes and imbalances)
        X = np.column_stack((delta_spreads, imbalances))
        
        # Prepare y vector (midpoint changes)
        y = delta_midpoints
        
        # Perform regression if we have enough data
        if len(X) > 0 and len(y) > 0 and len(X) == len(y):
            try:
                # Use numpy's lstsq for fast and stable OLS
                beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
                alpha = beta_hat[0]  # Coefficient for spread changes
                beta = beta_hat[1]   # Coefficient for imbalances
                #print("alpha: ", self.alpha)
                #print("beta: ", self.beta)
            except:
                # If regression fails, use default values
                self.alpha = 0.1
                self.beta = 0.2
        return alpha, beta
                
    def calc_regression_fair_value(self, symbol, rolling_count=None):
        #calculate the fair value using the regression parameters
        # get the last 100 prices
        # get the last 100 spreads
        regression_window = self.get_regression_window()
        # get the last 100 imbalances
        
        historical_midpoints = None
        historical_spreads = None
        imbalance = self._calculate_order_book_imbalance()
        with self.parent_client._lock:
            historical_midpoints = self.parent_client.stock_LOB_timeseries[symbol]["mid_price"]
            historical_spreads = self.parent_client.stock_LOB_timeseries[symbol]["spread"]
            
        imbalances = self.get_imbalances(rolling_count)
        
        if rolling_count is None:
            rolling_count = len(historical_midpoints)
        
        
        # If we don't have enough data, return a default value
        if rolling_count == 0:
            return self.get_fair_value()
        if rolling_count < 5 and len(historical_midpoints) > 3:
            #mark return value as last value in historical_midpoints
            
            return np.average(np.array(historical_midpoints[-rolling_count+2:])) #remove first two values
        elif rolling_count < 5:
            return self.get_fair_value() #let mkj be null for now, too volatile
        
        # Get the current midpoint
        current_midpoint = historical_midpoints[-1] if historical_midpoints.is_empty() is False else 1000.0
        #print("current_midpoint: ", current_midpoint)
        
        
        # Convert historical spreads to numpy array if not empty
        historical_spreads_np = historical_spreads.to_numpy() if not historical_spreads.is_empty() else None
        
        # Calculate spread impact using numpy array
        current_spread = historical_spreads_np[-1] if historical_spreads_np is not None else 1.0
        avg_spread = np.mean(historical_spreads_np) if historical_spreads_np is not None else 1.0
        spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1.0


        
        # Calculate volatility-adjusted fair value
        # Fair value = midpoint + alpha*spread_change + beta*imbalance
        spread_change = spread_ratio - 1.0  # Normalized spread change
        
        print("historical_midpoints: ", len(historical_midpoints))
        print("regression_window: ", regression_window)
        if rolling_count-2 >= regression_window:
            imbalances = self.get_imbalances(rolling_count-2)
            self._update_regression_parameters(imbalances, historical_midpoints, historical_spreads, rolling_count-2)
            # Get previous snapshot values
            prev_spread = historical_spreads[-2]
            spread_change = current_spread - prev_spread
            
            # Get previous imbalance value
            prev_imbalance = imbalances[-2]
            imbalance_change = imbalance - prev_imbalance
            
            # Use current alpha/beta but only look at most recent changes
            fair_value = current_midpoint + self.alpha * spread_change + self.beta * imbalance_change
            #print("fair_value", fair_value)
            #print("current_midpoint", current_midpoint)
            #print("self.alpha", self.alpha)
            #print("self.beta", self.beta) 
            #print("spread_change", spread_change)
            #print("imbalance", imbalance)
        else:
            # Simple heuristic if we don't have enough data for regression
            fair_value = current_midpoint + 0.1 * spread_change + 0.2 * imbalance
        
        
        # Apply confidence bands based on volatility
        # Fair value is within [fair_value - k*volatility, fair_value + k*volatility]
        # lower_band = fair_value - self.k_factor * self.volatility
        # upper_band = fair_value + self.k_factor * self.volatility
        
        # Ensure fair value is within reasonable bounds
        print("fair_value: {symbol} {fair_value}", self.symbol, fair_value)
        fair_value = max(0, min(10000, fair_value))
        
        omega_garch, alpha_garch, beta_garch, sigma = self.get_vol_params()
        
        omega_garch, alpha_garch, beta_garch, sigma = self.calc_volatility(omega_garch, alpha_garch, beta_garch)
        print("sigma: ", self.sigma)
        print("omega: ", self.omega_garch)
        print("alpha: ", self.alpha_garch)
        print("beta: ", self.beta_garch)
        
        #print("fair_value MKJ: ", fair_value)
        #timestamp = int(time.time()*100)/100 - self.parent_client.start_time
        #self._update_fair_value_timeseries(timestamp, fair_value)
        return fair_value
        
    
    
    def calc_volatility(self, omega, alpha, beta, index=None):
        omega, alpha, beta, sigma = self.garch_neg_loglik(omega, alpha, beta, index)
        return omega, alpha, beta, sigma
        
    # GARCH for volatility 
    def garch_neg_loglik(self, omega, alpha, beta, index=None):
        # This method is now only used for the initial calculation
        # The actual optimization happens in fit_garch
        returns = None
        sigma2 = None
        with self.parent_client._lock:
            # Get full series length
            T = len(self.parent_client.stock_LOB_timeseries[self.symbol]["spread"])
            
            # If index specified, only look at last index items
            start_idx = max(0, T - index) if index else 0
            window_size = min(T, index) if index else T
            
            sigma2 = np.zeros(window_size)
            prices = self.parent_client.stock_LOB_timeseries[self.symbol]["mid_price"].to_numpy()[start_idx:]
            returns = np.log(prices[:-1] / prices[1:])
            
        sigma2[0] = np.var(returns)
        
        # Get optimized parameters
        # print("sigma2[0]: ", sigma2[0])
        # print("returns[-1]: ", returns[-1])
        # print("omega: ", sigma2[0] - alpha - beta)
        # print("alpha: ", alpha)
        # print("beta: ", beta)
        
        print("symbol: ", self.symbol)
        
        print("returns: ", returns)
        
        # for numerical stability for optimizer 
        returns = returns 
        
        print("returns: ", returns)
        
        am = arch_model(returns, mean='zero', vol='GARCH', p=1, q=1, dist='normal', rescale=True)
        res = am.fit(disp='off')
        scale = res.scale  # New part of the code
        
        # Extract optimized parameters.
        # Parameter names typically are: 'omega', 'alpha[1]', and 'beta[1]'.
        omega_new = res.params['omega']
        alpha_new = res.params['alpha[1]']
        beta_new = res.params['beta[1]']
        
        print("omega_new: ", omega_new)
        print("alpha_new: ", alpha_new)
        print("beta_new: ", beta_new)
        
        
        sigma_new = np.sqrt(res.forecast(horizon=1, reindex=False).variance.values[-1, 0]/ np.power(scale, 2))
        
        if self.get_fair_value() is None:
            sigma_new *= np.average(prices[-4:])
        
        print("sigma_new: ", sigma_new)
        
        return omega_new, alpha_new, beta_new, sigma_new
        
    #     omega, alpha, beta = self.fit_garch(sigma2[0], alpha, beta, returns)
        
    #     # Calculate final sigma
    #     sigma = np.sqrt(omega + alpha * (returns[-1]**2) + beta * sigma2[-1])
        
    #     if sigma is None or alpha is None or beta is None:
    #         return None, None, None, None
        
    #     return omega, alpha, beta, sigma

    # def fit_garch(self, omega, alpha, beta, returns):
    #     # Initial guess for parameters
    #     initial_guess = [omega, alpha, beta]
    #     bounds = [(1e-8, 100), (0, 1), (0, 1)]
        
    #     # Define the objective function for optimization
    #     def objective(returns, params):
    #         omega, alpha, beta = params
    #         #print("params: ", params)
    #         # Calculate the negative log-likelihood with these parameters
    #         T = len(returns)
    #         sigma2 = np.zeros(T)
    #         sigma2[0] = np.var(returns)
            
    #         # Initialize negative log-likelihood
    #         neg_loglik = 0
            
    #         # Recursively compute the conditional variance and accumulate the log-likelihood
    #         for t in range(1, T):
                
    #             sigma2[t] = omega + alpha * (returns[t-1]**2) + beta * sigma2[t-1]
                
    #             # To avoid numerical issues, ensure sigma2[t] is positive
    #             if sigma2[t] <= 0:
    #                 return 1e5
                
    #             # Contribution to log-likelihood from observation t
    #             neg_loglik += 0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + (returns[t]**2 / sigma2[t]))
    #         #print("neg_loglik: ", neg_loglik)
    #         return neg_loglik
        
    #     # Perform optimization
    #     result = minimize(lambda params: objective(returns, params), initial_guess, bounds=bounds, method='L-BFGS-B')
        
    #     # Extract optimized parameters
    #     omega = result.x[0]
    #     alpha = result.x[1]
    #     beta = result.x[2]
        
    #     return omega, alpha, beta
    