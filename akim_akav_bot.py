from computebot import Compute
import polars as pl
import numpy as np
import math
import time
from utcxchangelib import xchange_client
import asyncio

class AKIMAKAVBot(Compute):
    """Specialized bot for AKIM and AKAV symbols that implements market making, 
    momentum trend trading, and arbitrage using theoretical pricing"""
    
    def __init__(self, parent_client=None, APT_bot=None, MKJ_bot=None, DLR_bot=None):
        super().__init__(parent_client)
        # This bot handles both AKIM and AKAV
        self.symbols = ["AKIM", "AKAV"]
        # Track correlation between the two symbols
        self.correlation = None
        # Track spread between the two symbols
        self.pair_spread = None
        # Historical fair values for each symbol
        self.historical_values = {
            "AKIM": [],
            "AKAV": []
        }
        # Reference to underlying stock bots
        self.APT_bot = APT_bot
        self.MKJ_bot = MKJ_bot
        self.DLR_bot = DLR_bot
        
        self.rolling_count = 0
        self.regression_window = 5
        # Strategy parameters
        self.price_threshold = 0.02  # Threshold for arbitrage (2% deviation)
        self.momentum_window = 5     # Window for momentum calculation
        self.bullish_threshold = 0.01  # Threshold for bullish momentum
        self.bearish_threshold = -0.01  # Threshold for bearish momentum
        self.arb_trade_size = 10     # Size for arbitrage trades
        self.momentum_trade_size = 5  # Size for momentum trades
        self.market_making_size = 3  # Size for market making quotes
        
        self.imbalance_timeseries = pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "imbalance": pl.Float64,
            "bid_volume": pl.Int64,
            "ask_volume": pl.Int64
        })
        self.symbol = "AKAV"
        
        # ETF swap fee
        self.swap_fee = 5
        self.trade_count = 0
        self.trading_frequency = 1000
        self.regression_fair_value = None
        
        # Track historical prices for momentum calculation
        self.price_history = {
            "AKIM": [],
            "AKAV": []
        }
        
        # Track theoretical prices
        self.theoretical_prices = {
            "AKIM": None,
            "AKAV": None
        }
        
        # Track momentum signals
        self.momentum_signals = {
            "AKIM": 0,
            "AKAV": 0
        }
        
        # Track arbitrage signals
        self.arbitrage_signals = {
            "AKIM": None,
            "AKAV": None
        }
        
        # Track market making quotes
        self.market_making_quotes = {
            "AKIM": {"bid": None, "ask": None},
            "AKAV": {"bid": None, "ask": None}
        }
        
        # Track inventory
        self.inventory = {
            "AKIM": 0,
            "AKAV": 0
        }
        
        # Track last update time
        self.last_update_time = time.time()
        
        # Track daily reset times (each day is 90 seconds)
        self.day_length = 90  # seconds
        self.last_day_reset = time.time()
        self.current_day = 0
        
        # Track initial prices for daily reset
        self.initial_prices = {
            "AKIM": None,
            "AKAV": None
        }
        
        # Track time remaining in the day
        self.time_remaining_in_day = self.day_length
        
        # Maximum allowed AKIM position (to ensure we can return to neutral)
        self.max_akim_position = 5
        
        # Flag to track if we're in the final minutes of the day
        self.is_final_minutes = False
        self.final_minutes_threshold = 15  # seconds
        
        # volatility parameters
        self.omega_garch, self.alpha_garch, self.beta_garch = [0.1, 0.05, 0.94]
        self.sigma = None
        
        self.alpha = 0.1
        self.beta = 0.2
        
    def get_fair_value(self):
        """Calculate the theoretical fair value for AKAV based on underlying stocks"""
        # Sum of underlying stock fair values
        apt_fair = self.APT_bot.fair_value_regression if self.APT_bot else 0
        mkj_fair = self.MKJ_bot.fair_value_regression if self.MKJ_bot else 0
        dlr_fair = self.DLR_bot.fair_value_regression if self.DLR_bot else 0
        if apt_fair is None or mkj_fair is None or dlr_fair is None:
            return None
        return apt_fair + mkj_fair + dlr_fair
    
    def calculate_theoretical_price(self, symbol):
        """Calculate the theoretical price for a symbol"""
        if symbol == "AKAV":
            # For AKAV, use the sum of underlying stock fair values
            base_theo = self.regression_fair_value
            if base_theo is not None:
                # Add half the swap fee to the theoretical price
                # This creates a "fair value band" of ±swap_fee/2 around the theoretical price
                # Prices within this band don't present arbitrage opportunities
                return base_theo + (self.swap_fee / 2)
            return None
        elif symbol == "AKIM":
            # For AKIM, use the inverse of AKAV's theoretical price
            akav_theo = self.calculate_theoretical_price("AKAV")
            if akav_theo is not None and self.initial_prices["AKAV"] is not None:
                # Calculate the percentage change in AKAV
                # Subtract the swap fee component before calculating the change
                akav_base = akav_theo - (self.swap_fee / 2)
                akav_change_percent = (akav_base - self.initial_prices["AKAV"]) / self.initial_prices["AKAV"]
                
                # AKIM should move in the opposite direction by the same percentage
                akim_change_percent = -akav_change_percent
                
                # Apply the percentage change to AKIM's initial price
                akim_theo = self.initial_prices["AKIM"] * (1 + akim_change_percent)
                return akim_theo
            return None
        return None
    
    def check_arbitrage_signal(self, symbol, current_price, theo_price):
        """Determine if an arbitrage opportunity exists based on thresholds"""
        if theo_price is None or current_price is None:
            return None
            
        # For AKAV, include the swap fee in the arbitrage calculation
        if symbol == "AKAV":
            # When price is higher than theoretical, we can:
            # 1. Short AKAV
            # 2. Buy underlying stocks
            # 3. Create new AKAV shares and cover short
            # Need price to be higher than theoretical + swap fee for profitable arbitrage
            if current_price > theo_price + self.swap_fee:
                return 'overpriced'
            # When price is lower than theoretical, we can:
            # 1. Buy AKAV
            # 2. Redeem for underlying stocks
            # 3. Sell underlying stocks
            # Need price to be lower than theoretical - swap fee for profitable arbitrage
            elif current_price < theo_price - self.swap_fee:
                return 'underpriced'
        else:  # For AKIM, use regular threshold since no creation/redemption
            pass 
            deviation = (current_price - theo_price) / theo_price
            if deviation > self.price_threshold:
                return 'overpriced'
            elif deviation < -self.price_threshold:
                return 'underpriced'
        
        return None
    
    def compute_momentum(self, symbol):
        """Compute momentum signals for a symbol based on price history"""
        if symbol not in self.price_history or len(self.price_history[symbol]) < self.momentum_window:
            return 0
            
        # Get price history
        prices = self.price_history[symbol]
        
        # Calculate multiple momentum indicators
        
        # 1. Simple price change (original method)
        current_price = prices[-1]
        past_price = prices[-self.momentum_window]
        simple_momentum = (current_price - past_price) / past_price
        
        # 2. Exponential weighted momentum (more recent prices have higher weight)
        if len(prices) >= 5:
            # Calculate returns
            returns = []
            for i in range(1, len(prices)):
                returns.append((prices[i] - prices[i-1]) / prices[i-1])
            
            # Apply exponential weights (more recent returns have higher weight)
            weights = np.exp(np.linspace(-1, 0, len(returns)))
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Calculate weighted momentum
            weighted_momentum = np.sum(np.array(returns) * weights)
        else:
            weighted_momentum = simple_momentum
        
        # 3. Rate of change (ROC) - percentage change over time
        roc_momentum = simple_momentum
        
        # 4. Moving average crossover
        if len(prices) >= 10:
            # Short-term MA (3 periods)
            short_ma = np.mean(prices[-3:])
            # Long-term MA (7 periods)
            long_ma = np.mean(prices[-7:])
            # MA crossover signal
            ma_crossover = (short_ma - long_ma) / long_ma
        else:
            ma_crossover = 0
        
        # Combine all momentum indicators with weights
        # Simple momentum: 20%, Weighted momentum: 40%, ROC: 20%, MA crossover: 20%
        combined_momentum = (
            0.2 * simple_momentum +
            0.4 * weighted_momentum +
            0.2 * roc_momentum +
            0.2 * ma_crossover
        )
        
        # Apply smoothing to reduce noise
        if hasattr(self, 'last_momentum') and symbol in self.last_momentum:
            # Exponential smoothing with alpha=0.3 (higher alpha = more weight to new value)
            alpha = 0.3
            combined_momentum = alpha * combined_momentum + (1 - alpha) * self.last_momentum[symbol]
        
        # Store the current momentum for next calculation
        if not hasattr(self, 'last_momentum'):
            self.last_momentum = {}
        self.last_momentum[symbol] = combined_momentum
        
        return combined_momentum
    
    def adjust_market_making_quotes(self, symbol, current_price, theo_price, momentum_signal):
        """Adjust market making quotes dynamically based on theoretical price and momentum"""
        if current_price is None:
            return None, None
            
        # Base spread calculation
        base_spread = 0.01  # 1% base spread
        
        # For AKAV, incorporate the swap fee into the spread
        if symbol == "AKAV":
            # Add the swap fee to the base spread
            # This ensures our quotes account for the cost of creation/redemption
            base_spread += self.swap_fee / current_price
        
        # Adjust spread based on deviation from theoretical price
        if theo_price is not None:
            deviation = abs((current_price - theo_price) / theo_price)
            # Widen spread as deviation increases
            base_spread += deviation * 0.5
        
        # Adjust spread based on momentum
        # Widen spread when momentum is strong (market is moving)
        momentum_factor = abs(momentum_signal) * 2
        adjusted_spread = base_spread * (1 + momentum_factor)
        
        # Calculate bid and ask prices
        mid_price = current_price
        bid_price = int(mid_price * (1 - adjusted_spread/2))
        ask_price = int(mid_price * (1 + adjusted_spread/2))
        
        # Adjust quotes based on momentum direction
        if momentum_signal > 0:  # Bullish momentum
            # More aggressive on the ask side
            ask_price = int(ask_price * 0.99)  # Lower ask price
        elif momentum_signal < 0:  # Bearish momentum
            # More aggressive on the bid side
            bid_price = int(bid_price * 1.01)  # Higher bid price
        
        return bid_price, ask_price
    
    def update_dynamic_hedges(self):
        """Update hedging positions to maintain inventory neutrality"""
        # Calculate net position across both symbols
        net_position = self.inventory["AKIM"] + self.inventory["AKAV"]
        
        # If net position is significant, hedge
        if abs(net_position) > 5:  # Threshold for hedging
            # Determine which symbol to hedge
            if net_position > 0:
                # # Net long position, need to short
                # if self.inventory["AKIM"] > 0:
                #     # Short AKIM
                #     self.place_hedge_order("AKIM", xchange_client.Side.SELL, abs(net_position))
                # else:
                #     # Short AKAV
                #     self.place_hedge_order("AKAV", xchange_client.Side.SELL, abs(net_position))
                pass
            else:
                pass
                # Net short position, need to go long
                if self.inventory["AKIM"] < 0:
                    # Buy AKIM
                    self.place_hedge_order("AKIM", xchange_client.Side.BUY, abs(net_position))
                else:
                    # Buy AKAV
                    self.place_hedge_order("AKAV", xchange_client.Side.BUY, abs(net_position))
    
    def place_hedge_order(self, symbol, side, qty):
        """Place a hedging order"""
        # Get current market price
        book = self.parent_client.order_books[symbol]
        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        
        if side == xchange_client.Side.BUY and sorted_asks:
            price = sorted_asks[0][0]  # Best ask price
        elif side == xchange_client.Side.SELL and sorted_bids:
            price = sorted_bids[0][0]  # Best bid price
        else:
            return  # Can't place order
            
        # Place the order
        self.parent_client.trade_queue.put({
            "symbol": symbol,
            "side": side,
            "qty": qty,
            "price": price
        })
        print(f"Added HEDGE order to queue: {symbol} {side} {qty} @ {price}")
    
    def _get_best_prices(self, symbol):
        """Helper function to get best bid and ask prices from the order book."""
        book = self.parent_client.order_books.get(symbol)
        if not book:
            return None, None
        
        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        
        best_bid = sorted_bids[0] if sorted_bids else None
        best_ask = sorted_asks[0] if sorted_asks else None
        
        return best_bid, best_ask

    def execute_arbitrage(self, symbol, signal):
        """Execute arbitrage trades based on signal, using swaps for AKAV."""
        if symbol == "AKAV":  # Only AKAV can be created/redeemed via swap
            # Calculate the net profit after swap fee
            theo_price = self.calculate_theoretical_price("AKAV") - (self.swap_fee / 2)  # Remove the spread component
            current_bid, current_ask = self._get_best_prices("AKAV")
            if current_bid is None or current_ask is None:
                print("Could not get current AKAV prices for arbitrage.")
                return
            current_price = (current_bid[0] + current_ask[0]) / 2 # Use mid-price for comparison
            print("current_price: ", current_price)
            print("theo_price: ", theo_price)

            if signal == 'underpriced':
                # Calculate potential profit using the price we'll buy at (ask)
                profit_per_share = (theo_price - current_ask[0] - self.swap_fee)
                if profit_per_share <= 0:
                    print(f"Skipping underpriced AKAV arbitrage, calculated profit {profit_per_share:.2f} <= 0")
                    return  # No profitable opportunity

                print(f"Executing underpriced AKAV arbitrage. Expected profit: {profit_per_share:.2f}")
                # 1. Buy AKAV at the ask price
                buy_price = current_ask[0]
                # 2. Place the swap order to redeem AKAV for underlying assets
                # Use asyncio.create_task as place_swap_order is async
                #asyncio.create_task(self.parent_client.place_swap_order("fromAKAV", self.arb_trade_size))
    

                symbol_to_sell = {}
                
                # 3. Place sell orders for the underlying assets received from the swap
                for underlying_symbol in ["APT", "DLR", "MKJ"]:
                    sell_price, _ = self._get_best_prices(underlying_symbol)
                    
                    symbol_to_sell[underlying_symbol] = sell_price[0]
                    self.arb_trade_size = min(self.arb_trade_size, sell_price[1])
                    
                    if sell_price:
                        print(f"Added ARBITRAGE SELL order for underlying: {underlying_symbol} {self.arb_trade_size} @ {sell_price}")
                    else:
                        print(f"Could not get sell price for {underlying_symbol} to complete arbitrage.")
                        return
                    
                    underlying_cost = 0
                    for symbol in symbol_to_sell:
                        underlying_cost += symbol_to_sell[symbol] * self.arb_trade_size
                    
                    if underlying_cost > buy_price:
                        print(f"Could not complete arbitrage, underlying cost {underlying_cost} is greater than buy price {buy_price}")
                        return
                    
                    self.parent_client.trade_queue.put({
                    "symbol": "AKAV",
                    "side": xchange_client.Side.BUY,
                    "qty": self.arb_trade_size,
                    "price": buy_price,
                    "type": "swap"
                    })
                    print(f"Placed SWAP order: fromAKAV {self.arb_trade_size}")
                    
                    for symbol in symbol_to_sell:
                        self.parent_client.trade_queue.put({
                            "symbol": symbol,
                            "side": xchange_client.Side.SELL,
                            "qty": self.arb_trade_size,
                            "price": symbol_to_sell[symbol]
                        })
                    
                    

            elif signal == 'overpriced':
                # Calculate potential profit using the price we'll sell at (bid)
                profit_per_share = (current_bid[0] - theo_price - self.swap_fee)
                if profit_per_share <= 0:
                    print(f"Skipping overpriced AKAV arbitrage, calculated profit {profit_per_share:.2f} <= 0")
                    return  # No profitable opportunity

                print(f"Executing overpriced AKAV arbitrage. Expected profit: {profit_per_share:.2f}")
                # 1. Buy underlying assets needed for the swap
                underlying_cost = 0
                can_buy_all_underlying = True
                underlying_orders = []
                for underlying_symbol in ["APT", "DLR", "MKJ"]:
                    _, buy_price = self._get_best_prices(underlying_symbol)
                    
                    self.arb_trade_size = min(self.arb_trade_size, buy_price[1])
                    if buy_price:
                        underlying_orders.append({
                            "symbol": underlying_symbol,
                            "side": xchange_client.Side.BUY,
                            "qty": self.arb_trade_size, # Assuming 1:1 swap ratio
                            "price": buy_price[0]
                        })
                    else:
                        print(f"Could not get buy price for {underlying_symbol} for arbitrage.")
                        can_buy_all_underlying = False
                        break # Cannot proceed if any underlying asset is unavailable

                if can_buy_all_underlying:
                    # Place buy orders for all underlying assets
                    
                    for order in underlying_orders:
                        order["qty"] = self.arb_trade_size
                        self.parent_client.trade_queue.put(order)
                        print(f"Added ARBITRAGE BUY order for underlying: {order['symbol']} {order['qty']} @ {order['price']}")

                    # 2. Place the swap order to create AKAV from underlying assets
                    print(f"Placed SWAP order: toAKAV {self.arb_trade_size}")
                    self.parent_client.trade_queue.put({
                        "symbol": "AKAV",
                        "side": xchange_client.Side.BUY,
                        "qty": self.arb_trade_size,
                        "price": buy_price,
                        "type": "swap_to"
                    })

                    # 3. Place sell order for the created AKAV shares at the bid price
                    sell_price = current_bid[0]
                    self.parent_client.trade_queue.put({
                        "symbol": "AKAV",
                        "side": xchange_client.Side.SELL,
                        "qty": self.arb_trade_size,
                        "price": sell_price,
                    })
                    print(f"Added ARBITRAGE SELL order to queue: AKAV {self.arb_trade_size} @ {sell_price}")
                else:
                     print("Could not execute overpriced AKAV arbitrage due to missing underlying prices.")


        elif symbol == "AKIM":  # AKIM cannot be created/redeemed, only traded
            pass
            # Check if we're in the final minutes of the day
            if self.is_final_minutes:
                print("Skipping AKIM arbitrage in final minutes of the day")
                return
                
            # Check if we're approaching the maximum allowed position
            if abs(self.inventory["AKIM"]) >= self.max_akim_position:
                print(f"AKIM position ({self.inventory['AKIM']}) already at maximum allowed ({self.max_akim_position})")
                return
                
            # Calculate remaining capacity
            remaining_capacity = self.max_akim_position - abs(self.inventory["AKIM"])
            trade_size = min(self.arb_trade_size, remaining_capacity)
            
            if trade_size <= 0:
                return
                
            if signal == 'underpriced':
                # AKIM is trading below its theoretical value: buy AKIM
                self.parent_client.trade_queue.put({
                    "symbol": "AKIM",
                    "side": xchange_client.Side.BUY,
                    "qty": trade_size,
                    "price": self.market_making_quotes["AKIM"]["ask"]
                })
                print(f"Added ARBITRAGE BUY order to queue: AKIM {trade_size} @ {self.market_making_quotes['AKIM']['ask']}")
                
            elif signal == 'overpriced':
                # AKIM is trading above its theoretical value: short AKIM
                self.parent_client.trade_queue.put({
                    "symbol": "AKIM",
                    "side": xchange_client.Side.SELL,
                    "qty": trade_size,
                    "price": self.market_making_quotes["AKIM"]["bid"]
                })
                print(f"Added ARBITRAGE SELL order to queue: AKIM {trade_size} @ {self.market_making_quotes['AKIM']['bid']}")
    
    def execute_momentum_trade(self, symbol, momentum_signal):
        """Execute momentum-based trades"""
        # For AKIM, be more conservative with momentum trading
        if symbol == "AKIM":
            # Check if we're in the final minutes of the day
            if self.is_final_minutes:
                print("Skipping AKIM momentum trade in final minutes of the day")
                return
                
            # Check if we're approaching the maximum allowed position
            if abs(self.inventory["AKIM"]) >= self.max_akim_position:
                print(f"AKIM position ({self.inventory['AKIM']}) already at maximum allowed ({self.max_akim_position})")
                return
                
            # Calculate remaining capacity
            remaining_capacity = self.max_akim_position - abs(self.inventory["AKIM"])
            trade_size = min(self.momentum_trade_size, remaining_capacity)
            
            if trade_size <= 0:
                return
                
            # Use a higher threshold for AKIM momentum trades
            if momentum_signal > self.bullish_threshold * 1.5:  # 50% higher threshold
                # Signal bullish on AKIM: buy AKIM
                self.parent_client.trade_queue.put({
                    "symbol": "AKIM",
                    "side": xchange_client.Side.BUY,
                    "qty": trade_size,
                    "price": self.market_making_quotes["AKIM"]["ask"]
                })
                print(f"Added MOMENTUM BUY order to queue: AKIM {trade_size} @ {self.market_making_quotes['AKIM']['ask']}")
                
            elif momentum_signal < self.bearish_threshold * 1.5:  # 50% higher threshold
                # Signal bearish on AKIM: short AKIM
                self.parent_client.trade_queue.put({
                    "symbol": "AKIM",
                    "side": xchange_client.Side.SELL,
                    "qty": trade_size,
                    "price": self.market_making_quotes["AKIM"]["bid"]
                })
                print(f"Added MOMENTUM SELL order to queue: AKIM {trade_size} @ {self.market_making_quotes['AKIM']['bid']}")
        
        # For AKAV, proceed with normal momentum trading
        elif symbol == "AKAV":
            if momentum_signal > self.bullish_threshold:
                # Signal bullish on AKAV: long AKAV
                self.parent_client.trade_queue.put({
                    "symbol": "AKAV",
                    "side": xchange_client.Side.BUY,
                    "qty": self.momentum_trade_size,
                    "price": self.market_making_quotes["AKAV"]["ask"]
                })
                print(f"Added MOMENTUM BUY order to queue: AKAV {self.momentum_trade_size} @ {self.market_making_quotes['AKAV']['ask']}")
                
            elif momentum_signal < self.bearish_threshold:
                # Signal bearish on AKAV: short AKAV
                self.parent_client.trade_queue.put({
                    "symbol": "AKAV",
                    "side": xchange_client.Side.SELL,
                    "qty": self.momentum_trade_size,
                    "price": self.market_making_quotes["AKAV"]["bid"]
                })
                print(f"Added MOMENTUM SELL order to queue: AKAV {self.momentum_trade_size} @ {self.market_making_quotes['AKAV']['bid']}")
    
    def update_market_making_quotes(self, symbol):
        """Update market making quotes for a symbol"""
        # Get current market price
    
        if symbol == "AKIM":
            return
        

        book = self.parent_client.order_books[symbol]
        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        
        if not sorted_bids or not sorted_asks:
            return
            
        current_price = (sorted_bids[0][0] + sorted_asks[0][0]) / 2
        
        # Get theoretical price
        theo_price = self.calculate_theoretical_price(symbol)
        
        # Get momentum signal
        momentum_signal = self.compute_momentum(symbol)
        self.momentum_signals[symbol] = momentum_signal
        
        # Check for arbitrage signal
        arbitrage_signal = self.check_arbitrage_signal(symbol, current_price, theo_price)
        self.arbitrage_signals[symbol] = arbitrage_signal
        
        # Adjust quotes
        bid_price, ask_price = self.adjust_market_making_quotes(symbol, current_price, theo_price, momentum_signal)
        
        if bid_price and ask_price:
            self.market_making_quotes[symbol] = {"bid": bid_price, "ask": ask_price}
            
            # # Place market making orders
            # self.parent_client.trade_queue.put({
            #     "symbol": symbol,
            #     "side": xchange_client.Side.BUY,
            #     "qty": self.market_making_size,
            #     "price": bid_price
            # })
            # print(f"Added MARKET MAKING BUY order to queue: {symbol} {self.market_making_size} @ {bid_price}")
            
            # self.parent_client.trade_queue.put({
            #     "symbol": symbol,
            #     "side": xchange_client.Side.SELL,
            #     "qty": self.market_making_size,
            #     "price": ask_price
            # })
            # print(f"Added MARKET MAKING SELL order to queue: {symbol} {self.market_making_size} @ {ask_price}")
    
    def calc_bid_ask_spread(self, symbol=None, df=None):
        """Calculate bid-ask spread for AKIM or AKAV"""
        if symbol is None:
            return None
            
        book = self.parent_client.order_books[symbol]
        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        
        if not sorted_bids or not sorted_asks:
            return None
            
        best_bid = sorted_bids[0][0]
        best_ask = sorted_asks[0][0]
        
        spread = (best_ask - best_bid) / 100
        return spread
        
    def calc_fair_value(self, symbol=None, df=None):
        """Calculate fair value for AKIM or AKAV"""
        if symbol == "AKAV":
            return self.get_fair_value()
        elif symbol == "AKIM":
            akav_fair = self.get_fair_value()
            if akav_fair is not None and self.initial_prices["AKAV"] is not None:
                # Calculate the percentage change in AKAV
                akav_change_percent = (akav_fair - self.initial_prices["AKAV"]) / self.initial_prices["AKAV"]
                
                # AKIM should move in the opposite direction by the same percentage
                akim_change_percent = -akav_change_percent
                
                # Apply the percentage change to AKIM's initial price
                return self.initial_prices["AKIM"] * (1 + akim_change_percent)
            return None
        return None
    
    def update_pair_correlation(self):
        """Calculate correlation between AKIM and AKAV"""
        if "AKIM" not in self.price_history or "AKAV" not in self.price_history:
            return
            
        if len(self.price_history["AKIM"]) < 10 or len(self.price_history["AKAV"]) < 10:
            return
            
        # Calculate returns
        akim_returns = np.diff(np.log(self.price_history["AKIM"][-10:]))
        akav_returns = np.diff(np.log(self.price_history["AKAV"][-10:]))
        
        # Calculate correlation
        self.correlation = np.corrcoef(akim_returns, akav_returns)[0, 1]
        
        # Calculate spread
        self.pair_spread = np.mean(self.price_history["AKIM"][-10:]) - np.mean(self.price_history["AKAV"][-10:])
        
        print(f"AKIM-AKAV correlation: {self.correlation}, spread: {self.pair_spread}")

    def check_daily_reset(self):
        """Check if it's time for a daily reset and perform the reset if needed"""
        current_time = time.time()
        elapsed_since_reset = current_time - self.last_day_reset
        
        # Update time remaining in the day
        self.time_remaining_in_day = max(0, self.day_length - elapsed_since_reset)
        
        # Check if we're in the final minutes of the day
        self.is_final_minutes = self.time_remaining_in_day <= self.final_minutes_threshold
        
        # If we're in the final minutes, ensure AKIM position is neutral
        if self.is_final_minutes and self.inventory["AKIM"] != 0:
            print(f"FINAL MINUTES: Closing AKIM position of {self.inventory['AKIM']}")
            self.close_akim_position()
        
        # Check if we need to reset (each day is 90 seconds)
        if elapsed_since_reset >= self.day_length:
            # It's time for a daily reset
            self.current_day += 1
            self.last_day_reset = current_time
            
            # Get current prices for reset
            for symbol in self.symbols:
                book = self.parent_client.order_books[symbol]
                sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
                sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
                
                if sorted_bids and sorted_asks:
                    current_price = (sorted_bids[0][0] + sorted_asks[0][0]) / 2
                    self.initial_prices[symbol] = current_price
                else:
                    # If we can't get current prices, use theoretical prices
                    self.initial_prices[symbol] = self.calculate_theoretical_price(symbol)
            
            print(f"Daily reset (Day {self.current_day}): AKAV initial price: {self.initial_prices['AKAV']}, AKIM initial price: {self.initial_prices['AKIM']}")
            
            # Reset price history for the new day
            for symbol in self.symbols:
                if self.initial_prices[symbol] is not None:
                    self.price_history[symbol] = [self.initial_prices[symbol]]
            
            return True
        
        return False

    def close_akim_position(self):
        """Close the AKIM position to return to neutral"""
        akim_position = self.inventory["AKIM"]
        if akim_position == 0:
            return
            
        # Get current market price
        book = self.parent_client.order_books["AKIM"]
        sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
        sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
        
        if not sorted_bids or not sorted_asks:
            return
            
        # Determine side based on position
        if akim_position > 0:
            # Long position, need to sell
            side = xchange_client.Side.SELL
            price = sorted_bids[0][0]  # Best bid price
        else:
            # Short position, need to buy
            side = xchange_client.Side.BUY
            price = sorted_asks[0][0]  # Best ask price
            
        # Place the order
        self.parent_client.trade_queue.put({
            "symbol": "AKIM",
            "side": side,
            "qty": abs(akim_position),
            "price": price
        })
        print(f"CLOSING AKIM POSITION: {side} {abs(akim_position)} @ {price}")
        
    def increment_trade(self):
        self.trade_count += 1
        if self.trade_count % self.trading_frequency == 0:
            self.handle_trade()
            
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

    def handle_trade(self):
        """Process updates for AKIM and AKAV"""
        # Update current time
        current_time = time.time()
        time_since_last_update = current_time - self.last_update_time
        self.last_update_time = current_time
        
        # Check for daily reset
        self.check_daily_reset()
        
        # Process each symbol
        for symbol in self.symbols:
            # Get current market price
            book = self.parent_client.order_books[symbol]
            sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
            sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
            
            if not sorted_bids or not sorted_asks:
                continue
                
            current_price = (sorted_bids[0][0] + sorted_asks[0][0]) / 2
            
            # Update price history
            self.price_history[symbol].append(current_price)
            if len(self.price_history[symbol]) > 100:  # Keep last 100 prices
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Calculate theoretical price
            theo_price = self.calculate_theoretical_price(symbol)
            self.theoretical_prices[symbol] = theo_price
            
            # Calculate momentum signal
            momentum_signal = self.compute_momentum(symbol)
            self.momentum_signals[symbol] = momentum_signal
            
            # Check for arbitrage signal
            arbitrage_signal = self.check_arbitrage_signal(symbol, current_price, theo_price)
            self.arbitrage_signals[symbol] = arbitrage_signal
            
            # Update market making quotes
            self.update_market_making_quotes(symbol)
            
            # Execute arbitrage if signal is present
            if arbitrage_signal is not None:
                self.execute_arbitrage(symbol, arbitrage_signal)
            
            # Execute momentum trades if signal is strong
            # if abs(momentum_signal) > max(self.bullish_threshold, abs(self.bearish_threshold)):
            #     self.execute_momentum_trade(symbol, momentum_signal)
        
        # Update pair correlation
        self.update_pair_correlation()
        
        # Update dynamic hedges
        #self.update_dynamic_hedges()
        
        # Update inventory from parent client positions
        for symbol in self.symbols:
            self.inventory[symbol] = self.parent_client.positions.get(symbol, 0)
            
        # Log position status
        print(f"Current positions: AKIM={self.inventory['AKIM']}, AKAV={self.inventory['AKAV']}, Time remaining: {self.time_remaining_in_day:.1f}s")

    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        """Handle trade messages for AKIM and AKAV"""
        # Update price history with the trade price
        if symbol in self.symbols:
            self.price_history[symbol].append(price)
            if len(self.price_history[symbol]) > 100:  # Keep last 100 prices
                self.price_history[symbol] = self.price_history[symbol][-100:]
            
            # Update inventory
            self.inventory[symbol] = self.parent_client.positions.get(symbol, 0)
            
            # Process the update
            await self.process_update(None)
            
    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        pass
    
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
        timestamp = int(time.time()*100)/100 - self.parent_client.start_time
        #print("imbalance: ", imbalance)
        self._update_imbalance_timeseries(timestamp, imbalance, bid_volume, ask_volume)
        return imbalance
    
    def handle_snapshot(self, symbol: str):
        self.rolling_count += 1
        self.regression_fair_value = self.calc_regression_fair_value("AKAV", self.rolling_count)
        self._calculate_order_book_imbalance()
    
    def get_regression_fair_value(self):
        return self.regression_fair_value
    
    def get_regression_window(self):
        return self.regression_window
    
    def get_imbalances(self, regression_window):
        return self.imbalance_timeseries.select("imbalance").tail(regression_window-1).to_series().to_numpy()
    
    def get_vol_params(self):
        return self.omega_garch, self.alpha_garch, self.beta_garch, self.sigma
    
    def update_vol_params(self, omega_garch, alpha_garch, beta_garch, sigma):
        self.omega_garch = omega_garch
        self.alpha_garch = alpha_garch
        self.beta_garch = beta_garch
        self.sigma = sigma
    
    
