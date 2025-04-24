#%% 

from typing import Optional
import threading
import utcxchangelib
from utcxchangelib import xchange_client
import asyncio
import argparse
import polars as pl
import heapq
import matplotlib.pyplot as plt
from apt_bot import APTBot
from dlr_bot import DLRBot
from mkj_bot import MKJBot
from akim_akav_bot import AKIMAKAVBot
import concurrent.futures
import time
import queue
import signal
import sys
import os

# Global shutdown event
SHUTDOWN_EVENT = threading.Event()

class ComputeThread(threading.Thread):
    """A thread class that runs an asyncio event loop for compute bots"""
    
    def __init__(self, name, bot, queue):
        super().__init__(name=name, daemon=True)
        self.bot = bot
        self.message_queue = queue
        self.stop_event = threading.Event()
        self.loop = None
        self._is_shutting_down = False
        self._force_exit = False
        self._exit_requested = False
        
    def run(self):
        """Run the asyncio event loop in this thread"""
        # Create a new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        
        try:
            # Run the compute bot coroutine
            self.loop.run_until_complete(self._process_updates())
        except Exception as e:
            print(f"{self.name}: Error in thread: {e}")
        finally:
            # Clean up
            try:
                self._is_shutting_down = True
                self._force_exit = True
                self._exit_requested = True
                
                # Cancel all running tasks
                pending = asyncio.all_tasks(self.loop)
                for task in pending:
                    task.cancel()
                
                # Run until all tasks are cancelled
                if pending:
                    self.loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
                
                # Close the loop
                self.loop.close()
            except Exception as e:
                print(f"{self.name}: Error during cleanup: {e}")
            print(f"{self.name}: Thread and event loop stopped")
            return
        
    def stop(self):
        """Signal the thread to stop"""
        self.stop_event.set()
        self._is_shutting_down = True
        self._force_exit = True
        self._exit_requested = True
        
        # If we're waiting on the queue, this will unblock it
        try:
            self.message_queue.put(None)
        except:
            pass
        
    async def _process_updates(self):
        """Process updates from the message queue"""
        print(f"{self.name}: Started processing")
        
        while not (self.stop_event.is_set() or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set()):
            try:
                # Wait for an update with asyncio to allow cancellation
                index = await self._get_update_async()
                
                # None is a signal to check the stop_event
                if index is None:
                    # If stop_event is set, exit the loop
                    if self.stop_event.is_set() or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set():
                        print(f"{self.name}: Stopping compute thread")
                        break
                    continue
             
                # Delegate to the bot's process_update method
                # Each bot class should implement this method
                await self.bot.process_update(index)
                
                # Mark the task as done in the queue (if using standard Queue)
                if hasattr(self.message_queue, 'task_done'):
                    self.message_queue.task_done()
                
            except asyncio.CancelledError:
                print(f"{self.name}: Processing cancelled")
                break
            except Exception as e:
                print(f"{self.name}: Error processing update: {e}")
                import traceback
                traceback.print_exc()
                # If stop_event is set, exit the loop
                if self.stop_event.is_set() or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set():
                    break
                # Otherwise, continue processing
                await asyncio.sleep(0.05)  # Brief pause before retrying
                
        print(f"{self.name}: Stopped processing updates")
        # Ensure the coroutine exits completely
        return
        
    async def _get_update_async(self):
        """Get an update from the queue using asyncio"""
        # Use run_in_executor to move the blocking queue.get to a thread pool
        # This allows us to cancel it if needed
        loop = asyncio.get_event_loop()
        
        def get_from_queue():
            try:
                # Use a timeout to allow checking the stop event
                return self.message_queue.get(timeout=0.05)  # Reduced timeout to 0.05 seconds
            except queue.Empty:
                return None
                
        # Check if we should exit immediately
        if self.stop_event.is_set() or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set():
            return None
            
        try:
            # Check if the event loop is shutting down
            if loop.is_closed() or loop.is_running() == False or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set():
                print(f"{self.name}: Event loop is shutting down, exiting queue get loop")
                return None  # Return immediately instead of continuing the loop
                
            update = await loop.run_in_executor(None, get_from_queue)
            if update is not None or self.stop_event.is_set() or self._is_shutting_down or self._force_exit or self._exit_requested or SHUTDOWN_EVENT.is_set():
                return update
        except asyncio.CancelledError:
            print(f"{self.name}: Queue get cancelled")
            return None
        except Exception as e:
            print(f"{self.name}: Error getting from queue: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(0.05)  # Brief pause before retrying
                
        return None
        


class MyXchangeClient(xchange_client.XChangeClient):
    plot = False
    # Thread lock for synchronizing access to shared data
    _lock = threading.Lock()
    
    # Initialize pnl_timeseries with columns for each symbol
    pnl_timeseries = pl.DataFrame(schema={
        "timestamp": pl.Float64,
        "pnl": pl.Int64,
        "is_news_event": pl.Int64,
        "APT_pnl": pl.Int64,
        "DLR_pnl": pl.Int64,
        "MKJ_pnl": pl.Int64,
        "AKAV_pnl": pl.Int64,
        "AKIM_pnl": pl.Int64,
    })
    
    fair_value_timeseries = {
        "APT": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "fair_value": pl.Int64
        }),
        "DLR": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "fair_value": pl.Int64
        }),
        "MKJ": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "fair_value": pl.Int64
        })
    }
    
    trade_queue = queue.Queue()
    
    stock_LOB_timeseries = { 
        "APT": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "best_bid_px": pl.Int64,
            "best_bid_qt": pl.Int64,
            "best_ask_px": pl.Int64,
            "best_ask_qt": pl.Int64,
            "spread": pl.Int64,
            "mid_price": pl.Float64,
            "2_bid_px": pl.Int64,
            "2_bid_qt": pl.Int64,
            "3_bid_px": pl.Int64,
            "3_bid_qt": pl.Int64,
            "4_bid_px": pl.Int64,
            "4_bid_qt": pl.Int64,
            "2_ask_px": pl.Int64,
            "2_ask_qt": pl.Int64,
            "3_ask_px": pl.Int64,
            "3_ask_qt": pl.Int64,
            "4_ask_px": pl.Int64,
            "4_ask_qt": pl.Int64,
        }),
        "DLR": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "best_bid_px": pl.Int64,
            "best_bid_qt": pl.Int64,
            "best_ask_px": pl.Int64,
            "best_ask_qt": pl.Int64,
            "spread": pl.Int64,
            "mid_price": pl.Float64,
            "2_bid_px": pl.Int64,
            "2_bid_qt": pl.Int64,
            "3_bid_px": pl.Int64,
            "3_bid_qt": pl.Int64,
            "4_bid_px": pl.Int64,
            "4_bid_qt": pl.Int64,
            "2_ask_px": pl.Int64,
            "2_ask_qt": pl.Int64,
            "3_ask_px": pl.Int64,
            "3_ask_qt": pl.Int64,
            "4_ask_px": pl.Int64,
            "4_ask_qt": pl.Int64,
        }),
        "MKJ": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "best_bid_px": pl.Int64,
            "best_bid_qt": pl.Int64,
            "best_ask_px": pl.Int64,
            "best_ask_qt": pl.Int64,
            "spread": pl.Int64,
            "mid_price": pl.Float64,
            "2_bid_px": pl.Int64,
            "2_bid_qt": pl.Int64,
            "3_bid_px": pl.Int64,
            "3_bid_qt": pl.Int64,
            "4_bid_px": pl.Int64,
            "4_bid_qt": pl.Int64,
            "2_ask_px": pl.Int64,
            "2_ask_qt": pl.Int64,
            "3_ask_px": pl.Int64,
            "3_ask_qt": pl.Int64,
            "4_ask_px": pl.Int64,
            "4_ask_qt": pl.Int64,
        }),
        "AKAV": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "best_bid_px": pl.Int64,
            "best_bid_qt": pl.Int64,
            "best_ask_px": pl.Int64,
            "best_ask_qt": pl.Int64,
            "spread": pl.Int64,
            "mid_price": pl.Float64,
            "2_bid_px": pl.Int64,
            "2_bid_qt": pl.Int64,
            "3_bid_px": pl.Int64,
            "3_bid_qt": pl.Int64,
            "4_bid_px": pl.Int64,
            "4_bid_qt": pl.Int64,
            "2_ask_px": pl.Int64,
            "2_ask_qt": pl.Int64,
            "3_ask_px": pl.Int64,
            "3_ask_qt": pl.Int64,
            "4_ask_px": pl.Int64,
            "4_ask_qt": pl.Int64,
        }),
        "AKIM": pl.DataFrame(schema={
            "timestamp": pl.Float64,
            "best_bid_px": pl.Int64,
            "best_bid_qt": pl.Int64,
            "best_ask_px": pl.Int64,
            "best_ask_qt": pl.Int64,
            "spread": pl.Int64,
            "mid_price": pl.Float64,
            "2_bid_px": pl.Int64,
            "2_bid_qt": pl.Int64,
            "3_bid_px": pl.Int64,
            "3_bid_qt": pl.Int64,
            "4_bid_px": pl.Int64,
            "4_bid_qt": pl.Int64,
            "2_ask_px": pl.Int64,
            "2_ask_qt": pl.Int64,
            "3_ask_px": pl.Int64,
            "3_ask_qt": pl.Int64,
            "4_ask_px": pl.Int64,
            "4_ask_qt": pl.Int64,
        }),
    }
    
    # Thread management
    _compute_threads = {}

    def __init__(self, host: str, username: str, password: str):
        super().__init__(host, username, password)
        self.start_time = int(time.time()*100)/100  # just initialize the start time
        self.correct_time = False #use first timestamp as correct time to calibrate the clock
        self.ticks_per_second = 450 #450 in practice round
        
        # Initialize specialized compute bots with self as parent
        self.compute_bots = {
            "APT": APTBot(self),
            "DLR": DLRBot(self),
            "MKJ": MKJBot(self),
        }
        #add the three underlying bots to the AKIM/AKAV bot
        self.compute_bots["AKIM_AKAV"] = AKIMAKAVBot(self, self.compute_bots["APT"], self.compute_bots["MKJ"], self.compute_bots["DLR"])
        
        # Create message queues for each symbol
        self.compute_queues = {
            "APT": queue.Queue(),
            "DLR": queue.Queue(),
            "MKJ": queue.Queue(),
            "AKIM": queue.Queue(),
            "AKAV": queue.Queue()
        }
        
    def start_compute_threads(self):
        """Start separate compute threads, each with its own asyncio event loop"""
        # Create and start compute threads
        self._compute_threads["apt_thread"] = ComputeThread(
            name="APT-ComputeThread",
            bot=self.compute_bots["APT"],
            queue=self.compute_queues["APT"]
        )
        
        self._compute_threads["dlr_thread"] = ComputeThread(
            name="DLR-ComputeThread",
            bot=self.compute_bots["DLR"],
            queue=self.compute_queues["DLR"]
        )
        
        self._compute_threads["mkj_thread"] = ComputeThread(
            name="MKJ-ComputeThread",
            bot=self.compute_bots["MKJ"],
            queue=self.compute_queues["MKJ"]
        )
        
        # AKIM/AKAV bot will handle both symbols
        self._compute_threads["akim_akav_thread"] = ComputeThread(
            name="AKIM-AKAV-ComputeThread",
            bot=self.compute_bots["AKIM_AKAV"],
            queue=self.compute_queues["AKIM"]  # Use AKIM queue as primary
        )
        
        # Start all threads
        for thread_name, thread in self._compute_threads.items():
            thread.start()
            print(f"Started compute thread: {thread.name}")
            
        print(f"All {len(self._compute_threads)} compute threads started")
    
    def stop_compute_threads(self):
        """Signal all compute threads to stop and wait for them to finish"""
        print("Stopping all compute threads...")
        
        # Set the global shutdown event
        SHUTDOWN_EVENT.set()
        
        # Signal all threads to stop
        for thread_name, thread in self._compute_threads.items():
            thread.stop()
            thread._is_shutting_down = True
            
        # Wait for all threads to finish with a timeout
        timeout = 2  # 2 seconds total timeout
        start_time = int(time.time()*100)/100 
        
        for thread_name, thread in self._compute_threads.items():
            if thread.is_alive():
                print(f"Waiting for {thread.name} to finish...")
                
                # Calculate remaining timeout
                elapsed = int(time.time()*100)/100 - start_time
                remaining_timeout = max(0.1, timeout - elapsed)
                
                thread.join(timeout=remaining_timeout)
                
                if thread.is_alive():
                    print(f"WARNING: {thread.name} did not stop cleanly, forcing termination")
                    # Force thread to stop if it's still alive
                    if hasattr(thread, 'loop') and thread.loop is not None:
                        try:
                            # Try to close the loop if it's still open
                            if not thread.loop.is_closed():
                                thread.loop.call_soon_threadsafe(thread.loop.stop)
                        except Exception as e:
                            print(f"Error stopping {thread.name} loop: {e}")
                    
                    # Set the shutdown flag directly
                    thread._is_shutting_down = True
                    
                    # Try joining again with a shorter timeout
                    thread.join(timeout=0.2)
                    
                    # If still alive, this is a last resort
                    if thread.is_alive():
                        print(f"CRITICAL: {thread.name} still alive after forced termination")
                        # We can't do much more here, the thread will eventually exit
                        # when the program terminates
                else:
                    print(f"Successfully stopped {thread.name}")
                    
        # Clear thread dictionary
        self._compute_threads.clear()
        print("All compute threads stopped")


    async def bot_handle_cancel_response(self, order_id: str, success: bool, error: Optional[str]) -> None:
        order = self.open_orders[order_id]
        print(f"{'Market' if order[2] else 'Limit'} Order ID {order_id} cancelled, {order[1]} unfilled")

    async def bot_handle_order_fill(self, order_id: str, qty: int, price: int):
        print("order fill", self.positions)
        for bot in self.compute_bots.values():
            if bot == self.compute_bots["AKIM_AKAV"]:
                continue
            if order_id in bot.open_orders:
                #qty = order
                pass
    async def bot_handle_order_rejected(self, order_id: str, reason: str) -> None:
        print("order rejected because of ", reason)


    async def bot_handle_trade_msg(self, symbol: str, price: int, qty: int):
        """
        Handle trade messages by sending them to the appropriate compute thread.
        """
        if symbol in self.stock_LOB_timeseries:
            # Let data bot process the update first
            if symbol == "AKIM" or symbol == "AKAV":
                await self.compute_bots["AKIM_AKAV"].bot_handle_trade_msg(symbol, price, qty)
            else:
                await self.compute_bots[symbol].bot_handle_trade_msg(symbol, price, qty)

    async def bot_handle_book_update(self, symbol: str):
        if symbol in self.stock_LOB_timeseries:
            # Let data bot process the update first
            if symbol == "AKIM" or symbol == "AKAV":
                #pass for now to prevent etf trading
                self.compute_bots["AKIM_AKAV"].increment_trade()
            else:
                self.compute_bots[symbol].increment_trade()

    async def bot_handle_swap_response(self, swap: str, qty: int, success: bool):
        pass

    async def bot_handle_news(self, news_release: dict):
        # Parsing the message based on what type was received
        timestamp = news_release["timestamp"] # This is in exchange ticks not ISO or Epoch
        news_type = news_release['kind']
        news_data = news_release["new_data"]
        
        if self.correct_time == False:
            timestamp_seconds = timestamp / 5 #450 in practice round per day day is 90 seconds, 5 ticks per second
            self.correct_time = True
            self.start_time = self.start_time - timestamp_seconds #shift epoch time to correct start time
            self.compute_bots["DLR"].update_rounds()
            print("timestamp_seconds: ", timestamp)
            print("update_counter: ", self.compute_bots["DLR"].update_counter)
            if news_type == "structured":
                if news_data["structured_subtype"] != "earnings":
                    print("updating counter")
                    print("timestamp_seconds: ", timestamp)
                    print("update_counter: ", self.compute_bots["DLR"].update_counter)
                    self.compute_bots["DLR"].update_counter -= 1
            print("timestamp_seconds: ", timestamp_seconds)
        
        #print(news_data)

        if news_type == "structured":
            subtype = news_data["structured_subtype"]
            symb = news_data["asset"]
            if subtype == "earnings":
                
                earnings = news_data["value"]

                self.compute_bots["APT"].handle_earnings_update(earnings)
                self.compute_bots["APT"].handle_news_update()
                
            else:
                new_signatures = news_data["new_signatures"]
                cumulative = news_data["cumulative"]
                self.compute_bots["DLR"].handle_news_update()
                self.compute_bots["DLR"].signature_update(new_signatures, cumulative)
        else:
            if news_data == "Ten seconds to EOD.":
                #set flags to all bots to close positions
                for bot in self.compute_bots.values():
                    bot.begin_closing_positions()
            if news_data == "EOD - AKIM has rebalanced":
                #we want to cancel all orders at this point find/create function
                pass
                #for bot in self.compute_bots.values():
                #    bot.cancel_all_orders()
            else:
                for bot in self.compute_bots.values():
                    bot.unstructured_update(news_data)
    

    async def plot_best_bid_ask(self):
        for symbol, df in self.stock_LOB_timeseries.items():
            plt.figure(figsize=(12, 6))
            
            # Thread-safe read of timeseries data
            with self._lock:
                timestamp = df["timestamp"].to_list()
                best_bid_px = df["best_bid_px"].to_list()
                best_ask_px = df["best_ask_px"].to_list()
                bid_ask_spread = [ask - bid for ask, bid in zip(best_ask_px, best_bid_px)]
            
            plt.subplot(2, 1, 1)
            plt.plot(timestamp, best_bid_px, label="Best Bid Price", linestyle="-",markersize=1)
            plt.plot(timestamp, best_ask_px, label="Best Ask Price", linestyle="-",markersize=1)

            plt.legend(["Best Bid Price", "Best Ask Price"])
            plt.grid(True)
            plt.xticks(rotation=45)

            plt.subplot(2, 1, 2)
            plt.plot(timestamp, bid_ask_spread, label="Bid Ask Spread", linestyle="-", markersize=1)
            plt.legend("Bid Ask Spread")
            plt.grid(True)
            plt.xticks(rotation=45)

            # Show plot
            print(f"Saving figure for {symbol}")
            plt.tight_layout()
            plt.savefig(f"data/best_bid_ask_{symbol}.png")
            plt.close()

    async def trade(self):
        await asyncio.sleep(15)
        print("attempting to trade")
        # buy 20 shares of APT
        await self.place_order("APT",1, xchange_client.Side.BUY, int(9))
        await asyncio.sleep(5)
        with self._lock: 
            latest_timestamp = self.stock_LOB_timeseries["APT"].select("timestamp").max().item()
            print("type of latest_timestamp: ", type(latest_timestamp))
            bid_price, ask_price = self.compute_bots["APT"].calc_bid_ask_price(latest_timestamp)
        print("========================================")
        print("Adjusted Bid Price:", bid_price)
        await self.place_order("APT",self.compute_bots["APT"].q_tilde, xchange_client.Side.BUY, bid_price)
        print("Adjusted Ask Price:", ask_price)
        await self.place_order("APT",self.compute_bots["APT"].q_tilde, xchange_client.Side.SELL, ask_price)
        print("my positions:", self.positions)
        
    def calculate_unrealized_pnl(self, symbol_position, sorted_bids, sorted_asks):
        unrealized_pnl = 0
        if symbol_position > 0:  # Long position
            # Use best bid price (highest price we could sell at)
            if sorted_bids:
                # Calculate weighted average price if position is larger than best bid quantity
                remaining_position = symbol_position
                weighted_price = 0
                total_quantity = 0
                
                for bid_price, bid_qty in sorted_bids:
                    if remaining_position <= 0:
                        break
                        
                    # Use the smaller of the remaining position or the bid quantity
                    qty_to_use = min(remaining_position, bid_qty)
                    weighted_price += bid_price * qty_to_use
                    total_quantity += qty_to_use
                    remaining_position -= qty_to_use
                
                if total_quantity > 0:
                    avg_price = weighted_price / total_quantity
                    unrealized_pnl = symbol_position * avg_price
                else:
                    # Fallback to best bid if we couldn't calculate a weighted average
                    unrealized_pnl = symbol_position * sorted_bids[0][0]
                    
        elif symbol_position < 0:  # Short position
            # Use best ask price (lowest price we could buy at)
            if sorted_asks:
                # Calculate weighted average price if position is larger than best ask quantity
                remaining_position = abs(symbol_position)
                weighted_price = 0
                total_quantity = 0
                
                for ask_price, ask_qty in sorted_asks:
                    if remaining_position <= 0:
                        break
                        
                    # Use the smaller of the remaining position or the ask quantity
                    qty_to_use = min(remaining_position, ask_qty)
                    weighted_price += ask_price * qty_to_use
                    total_quantity += qty_to_use
                    remaining_position -= qty_to_use
                
                if total_quantity > 0:
                    avg_price = weighted_price / total_quantity
                    unrealized_pnl = symbol_position * avg_price
                else:
                    # Fallback to best ask if we couldn't calculate a weighted average
                    unrealized_pnl = symbol_position * sorted_asks[0][0]
        return unrealized_pnl

    async def view_books(self):
        # Use polars DataFrame for better performance
        #print("viewing books")
        while True:
            await asyncio.sleep(1)
            current_time = int(time.time()*100)/100 - self.start_time
            
            # Initialize base PNL with cash position
            pnl = self.positions['cash']
            
            # Create a list to store all symbol data for batch processing
            symbol_data = {}
            
            for symbol, book in self.order_books.items():
                # Extract prices where quantity > 0 for printing
                sorted_bids = sorted(((p, q) for p, q in book.bids.items() if q > 0), reverse=True)
                sorted_asks = sorted((p, q) for p, q in book.asks.items() if q > 0)
                
                # Calculate unrealized PnL for this symbol
                symbol_position = self.positions.get(symbol, 0)
                unrealized_pnl = self.calculate_unrealized_pnl(symbol_position, sorted_bids, sorted_asks)
                pnl += unrealized_pnl
                
                # Store symbol data for batch processing
                symbol_data[symbol] = unrealized_pnl
                
                # Create a new row with the first 3 levels of bids and asks
                if symbol in self.stock_LOB_timeseries:
                    # Get current timestamp
                    current_time = int(time.time()*100)/100 - self.start_time 
                    
                    # Create row data with first 3 levels
                    row_data = {
                        "timestamp": current_time,
                        "best_bid_px": sorted_bids[0][0] if sorted_bids else 0,
                        "best_bid_qt": sorted_bids[0][1] if sorted_bids else 0,
                        "best_ask_px": sorted_asks[0][0] if sorted_asks else 0,
                        "best_ask_qt": sorted_asks[0][1] if sorted_asks else 0,
                    }
                    
                    if symbol == "AKIM" or symbol == "AKAV":
                        if len(sorted_bids) > 0 and len(sorted_asks) > 0:
                            row_data["spread"] = sorted_asks[0][0] - sorted_bids[0][0]
                            row_data["mid_price"] = (sorted_asks[0][0] + sorted_bids[0][0]) / 2 
                        elif len(self.stock_LOB_timeseries[symbol]["spread"]) == 0:
                            row_data["spread"] = 6
                            row_data["mid_price"] = 0
                        else:
                            row_data["spread"] = self.stock_LOB_timeseries[symbol]["spread"].tail(1).item()
                            row_data["mid_price"] = self.stock_LOB_timeseries[symbol]["mid_price"].tail(1).item()
                        
                        if row_data["best_bid_px"] == 0 or row_data["best_ask_px"] == 0:
                            if len(self.stock_LOB_timeseries[symbol]["spread"]) == 0:
                                row_data["best_bid_px"] = 0
                                row_data["best_ask_px"] = 0
                            else:
                                row_data["best_bid_px"] = self.stock_LOB_timeseries[symbol]["best_bid_px"].tail(1).item() + 3
                                row_data["best_ask_px"] = self.stock_LOB_timeseries[symbol]["best_ask_px"].tail(1).item() - 3
                        
                        if row_data["best_bid_qt"] == 0 or row_data["best_ask_qt"] == 0:
                            row_data["best_bid_qt"] = 1
                            row_data["best_ask_qt"] = 1
                    else:
                        if len(sorted_bids) > 0 and len(sorted_asks) > 0:
                            row_data["spread"] = sorted_asks[0][0] - sorted_bids[0][0]
                            row_data["mid_price"] = (sorted_asks[0][0] + sorted_bids[0][0]) / 2
                        else:
                            row_data["spread"] = 6
                            row_data["mid_price"] = self.compute_bots[symbol].get_fair_value()
                        
                        if row_data["best_bid_px"] == 0 or row_data["best_ask_px"] == 0:
                            row_data["best_bid_px"] = 0
                            row_data["best_ask_px"] = 0
                        
                        if row_data["best_bid_qt"] == 0 or row_data["best_ask_qt"] == 0:
                            row_data["best_bid_qt"] = 1
                            row_data["best_ask_qt"] = 1
                    
                    #prevent 0 values from being used
                    if row_data["spread"] == 0 or row_data["mid_price"] == 0:
                        row_data["spread"] = self.stock_LOB_timeseries[symbol]["spread"].tail(1).item()
                        row_data["mid_price"] = self.stock_LOB_timeseries[symbol]["mid_price"].tail(1).item()
                    # Add second, third, and fourth levels if available
                    if len(sorted_bids) > 1:
                        row_data["2_bid_px"] = sorted_bids[1][0]
                        row_data["2_bid_qt"] = sorted_bids[1][1]
                    else:
                        row_data["2_bid_px"] = 0
                        row_data["2_bid_qt"] = 0
                    if len(sorted_bids) > 2:
                        row_data["3_bid_px"] = sorted_bids[2][0]
                        row_data["3_bid_qt"] = sorted_bids[2][1]
                    else:
                        row_data["3_bid_px"] = 0
                        row_data["3_bid_qt"] = 0
                    if len(sorted_bids) > 3:
                        row_data["4_bid_px"] = sorted_bids[3][0]
                        row_data["4_bid_qt"] = sorted_bids[3][1]
                    else:
                        row_data["4_bid_px"] = 0
                        row_data["4_bid_qt"] = 0
                        
                    if len(sorted_asks) > 1:
                        row_data["2_ask_px"] = sorted_asks[1][0]
                        row_data["2_ask_qt"] = sorted_asks[1][1]
                    else:
                        row_data["2_ask_px"] = 0
                        row_data["2_ask_qt"] = 0   
                    if len(sorted_asks) > 2:
                        row_data["3_ask_px"] = sorted_asks[2][0]
                        row_data["3_ask_qt"] = sorted_asks[2][1]
                    else:
                        row_data["3_ask_px"] = 0
                        row_data["3_ask_qt"] = 0
                    if len(sorted_asks) > 3:
                        row_data["4_ask_px"] = sorted_asks[3][0]
                        row_data["4_ask_qt"] = sorted_asks[3][1]
                    else:
                        row_data["4_ask_px"] = 0
                        row_data["4_ask_qt"] = 0
                    
                    # Create new row DataFrame
                    new_row = pl.DataFrame([row_data])
                    
                    #print("new_row: ", new_row)
                    # Thread-safe update to the timeseries
                    with self._lock:
                        self.stock_LOB_timeseries[symbol] = pl.concat([
                            self.stock_LOB_timeseries[symbol],
                            new_row
                        ])
                    if symbol == "MKJ" or symbol == "APT" or symbol == "DLR":
                        #print("incrementing trade for MKJ")
                        self.compute_bots[symbol].handle_snapshot()
                    else:
                        self.compute_bots["AKIM_AKAV"].handle_snapshot(symbol)
            
            # Update PnL timeseries with total unrealized PnL
            current_time = int(time.time()*100)/100 - self.start_time
            
            # Create a dictionary with all PNL data
            pnl_data = {
                "timestamp": current_time,
                "pnl": int(pnl),
                "is_news_event": 0
            }
            
            # Add individual symbol PNLs
            for symbol, symbol_pnl in symbol_data.items():
                pnl_data[f"{symbol}_pnl"] = int(symbol_pnl)
                
            print("pnl_data: ", pnl_data)
            
            
            with self._lock:
                # Concatenate the new row
                self.pnl_timeseries = pl.concat([
                    self.pnl_timeseries,
                    pl.DataFrame([pnl_data])
                ])

    async def plot_pnl(self):
        """Plot individual PNL for each asset being traded"""
        # Get all symbols being traded (excluding 'cash')
        symbols = [symbol for symbol in self.positions.keys() if symbol != 'cash']
        
        # Thread-safe read of timeseries data
        with self._lock:
            # Get timestamps for x-axis
            timestamps = self.pnl_timeseries["timestamp"].to_list()
            
            # Get news event timestamps
            unstructured_news_events = self.pnl_timeseries.filter(pl.col("is_news_event") == 1)
            structured_news_events = self.pnl_timeseries.filter(pl.col("is_news_event") == 2)
            
            # Plot PNL for each symbol in separate figures
            for symbol in symbols:
                # Create a new figure for each symbol
                plt.figure(figsize=(12, 6))
                
                # Get PNL data for this symbol
                symbol_pnl = self.pnl_timeseries[f"{symbol}_pnl"].to_list()
                
                # Plot PNL for this symbol
                plt.plot(timestamps, symbol_pnl, label=f"{symbol} PNL", linestyle="-", markersize=1)
                
                # Add vertical lines for news events
                for ts in unstructured_news_events["timestamp"].to_list():
                    plt.axvline(x=ts, color='r', linestyle='--', alpha=0.5)
                
                for ts in structured_news_events["timestamp"].to_list(): 
                    plt.axvline(x=ts, color='g', linestyle='--', alpha=0.5)
                
                plt.title(f"{symbol} PNL")
                plt.legend()
                plt.grid(True)
                plt.tick_params(axis='x', rotation=45)
                
                # Adjust layout and save
                plt.tight_layout()
                print(f"Saving {symbol} PnL figure")
                plt.savefig(f"data/{symbol}_pnl.png")
                plt.close()
            
            # Create a separate figure for total PNL
            plt.figure(figsize=(12, 6))
            total_pnl = self.pnl_timeseries["pnl"].to_list()
            plt.plot(timestamps, total_pnl, label="Total PNL", linestyle="-", markersize=1)
            
            # Add vertical lines for news events
            for ts in unstructured_news_events["timestamp"].to_list():
                plt.axvline(x=ts, color='r', linestyle='--', alpha=0.5)
            
            for ts in structured_news_events["timestamp"].to_list(): 
                plt.axvline(x=ts, color='g', linestyle='--', alpha=0.5)
        
            plt.title("Total PNL (All Assets)")
            plt.legend()
            plt.grid(True)
            plt.tick_params(axis='x', rotation=45)
            
            # Adjust layout and save
            plt.tight_layout()
            print("Saving Total PnL figure")
            plt.savefig("data/total_pnl.png")
            plt.close()

    async def handle_trade(self):
        """Process trades from a queue of pending trades.
        
        This method processes trades from the trade queue without using a timeout.
        """
        # Initialize trade queue if it doesn't exist
        if not hasattr(self, 'trade_queue'):
            self.trade_queue = queue.Queue()
            
        while True:
            try:
                # Get next trade from queue with timeout
                trade = await asyncio.get_event_loop().run_in_executor(
                    None, 
                    self.trade_queue.get
                )
                
                if trade is None:
                    # None signals queue shutdown
                    break
                    
                print(f"Processing trade: {trade}")
                symbol = trade.get("symbol")
                side = trade.get("side") 
                qty = trade.get("qty")
                price = trade.get("price")
                type = trade.get("type")
                
                if type == "swap":
                    print(f"Swapping {symbol} {qty} @ {price}")
                    order_id = await self.place_swap_order("from"+symbol, qty)
                    print(f"Order placed with ID: {order_id}")
                    self.trade_queue.task_done()
                    continue
                
                if type == "swap_to":
                    print(f"Swapping {symbol} {qty} @ {price}")
                    order_id = await self.place_swap_order("to"+symbol, qty)
                    print(f"Order placed with ID: {order_id}")
                    self.trade_queue.task_done()
                    continue
                
                
                if not all([symbol, side, qty, price]):
                    print(f"Invalid trade data: {trade}")
                    self.trade_queue.task_done()
                    continue
                
                # Place the order
                print(f"Placing order: {symbol} {side} {qty} @ {price}")
                order_id = await self.place_order(symbol, qty, side, price)
                print(f"Order placed with ID: {order_id}")
                
                #self.compute_bots[symbol].open_orders.append(order_id)
                # Mark task as done
                self.trade_queue.task_done()
                
            except Exception as e:
                print(f"Error processing trade: {e}")
                import traceback
                traceback.print_exc()
        
    async def start(self, user_interface):
        # Start compute threads
        #self.start_compute_threads()
        #asyncio.create_task(self.trade())
        asyncio.create_task(self.view_books())
        asyncio.create_task(self.handle_trade())
        
        # This is where Phoenixhood will be launched if desired.
        if user_interface:
            self.launch_user_interface()
            asyncio.create_task(self.handle_queued_messages())

        await self.connect()
        
        print("stopping compute threads")
    
    async def plot_fair_value(self):
        plt.figure(figsize=(12, 6))
        
        # Thread-safe read of timeseries data
        for symbol, df in self.fair_value_timeseries.items():
            with self._lock:
                timestamp = df["timestamp"].to_list()
                fair_value = df["fair_value"].to_list()
                midpoint_df = self.stock_LOB_timeseries[symbol]
                midpoint = midpoint_df["mid_price"].to_list()
                timestamp_midpoint = midpoint_df["timestamp"].to_list()
                print("symbol: ", symbol)
                print("midpoint: ", midpoint)
                print("fair_value: ", fair_value)
                print("timestamp: ", timestamp)
                
                plt.plot(timestamp, fair_value, label="Fair Value", linestyle="-", markersize=1)
                plt.plot(timestamp_midpoint, midpoint, label="Midpoint", linestyle="-", markersize=1)
                
                plt.legend()
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.title("Fair Value and Midpoint")
                plt.tight_layout()
                plt.savefig(f"data/fair_value_{symbol}.png")
                plt.close()
                
        



async def main(user_interface: bool):
    # SERVER = '127.0.0.1:8000'   # run locally
    SERVER = '3.138.154.148:3333' # run on sandbox
    TEAMNAME = "yale_buffalo_rutgers_harvard"
    PASSWORD = "mre)2uJdR5"
    my_client = MyXchangeClient(SERVER,TEAMNAME,PASSWORD)
    
    try:
        await my_client.start(user_interface)
    except Exception as e:
        print("Exception encountered in start:", e)
        import traceback
        traceback.print_exc()  # Print the full stack trace
    finally:
        # Plot data at the end of trading
        await my_client.plot_best_bid_ask()
        await my_client.plot_pnl()
        await my_client.plot_fair_value()
        
        # Ensure all threads are stopped
        my_client.stop_compute_threads()
        
        # Set the global shutdown event as a last resort
        SHUTDOWN_EVENT.set()
        
        # Force exit after a timeout if threads are still running
        print("Waiting for threads to terminate...")
        for thread_name, thread in my_client._compute_threads.items():
            if thread.is_alive():
                print(f"Thread {thread.name} is still alive, forcing exit...")
                # We can't do much more here, the thread will eventually exit
                # when the program terminates
    
    return


if __name__ == "__main__":

    # This parsing is unnecessary if you know whether you are using Phoenixhood.
    # It is included here so you can see how one might start the API.

    parser = argparse.ArgumentParser(
        description="Script that connects client to exchange, runs algorithmic trading logic, and optionally deploys Phoenixhood"
    )

    parser.add_argument("--phoenixhood", required=False, default=False, type=bool, help="Starts phoenixhood API if true")
    args = parser.parse_args()

    user_interface = args.phoenixhood

    loop = asyncio.get_event_loop()
    try:
        result = loop.run_until_complete(main(user_interface))
    except KeyboardInterrupt:
        print("Keyboard interrupt received, shutting down...")
        # Set the global shutdown event
        SHUTDOWN_EVENT.set()
    finally:
        # Ensure the loop is closed
        try:
            loop.stop()
            loop.close()
        except:
            pass
        
        # Force exit after a timeout
        print("Forcing exit in 3 seconds...")
        time.sleep(3)
        print("Exiting...")
        os._exit(0)  # Force exit the program

# %%
