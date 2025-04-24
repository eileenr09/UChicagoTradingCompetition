import numpy as np
from datetime import datetime, timedelta
from time_series import (
    TimeSeriesAnalyzer,
    OrderBookSnapshot,
    OrderBookLevel,
    Trade
)

def create_detailed_order_book():
    """Create a detailed order book with varying liquidity."""
    base_price = 100.0
    base_time = datetime.now()
    
    # Create a snapshot with varying liquidity
    bids = []
    asks = []
    
    # Create bids with decreasing liquidity
    for i in range(10):
        price = base_price - i * 0.1
        volume = 1000 - i * 100  # Decreasing volume
        orders = 10 - i  # Decreasing number of orders
        bids.append(OrderBookLevel(price=price, volume=volume, orders=orders))
    
    # Create asks with varying liquidity
    for i in range(10):
        price = base_price + i * 0.1
        volume = 800 - i * 50  # Different volume profile
        orders = 8 - i  # Different order count profile
        asks.append(OrderBookLevel(price=price, volume=volume, orders=orders))
    
    return OrderBookSnapshot(
        timestamp=base_time,
        bids=bids,
        asks=asks,
        depth=10
    )

def create_trades_with_fill_ratios():
    """Create trades with varying fill characteristics."""
    base_time = datetime.now()
    trades = []
    
    # Create a mix of trades with different characteristics
    for i in range(100):
        # Vary the aggressiveness and hit side
        aggressiveness = np.random.choice(['BUY', 'SELL'])
        hit_side = np.random.choice(['BID', 'ASK'])
        
        # Create trades with varying volumes
        volume = int(np.random.uniform(10, 1000))
        
        # Create trades with prices around the mid price
        price = 100.0 + np.random.normal(0, 0.1)
        
        trade = Trade(
            timestamp=base_time + timedelta(seconds=i),
            price=price,
            volume=volume,
            aggressiveness=aggressiveness,
            hit_side=hit_side
        )
        trades.append(trade)
    
    return trades

def analyze_execution_probabilities(analyzer: TimeSeriesAnalyzer):
    """Analyze execution probabilities at different price levels."""
    current_price = 100.0
    price_levels = np.linspace(99.0, 101.0, 21)  # 21 price levels around current price
    
    print("\nExecution Probability Analysis:")
    print("-" * 40)
    print("Price Level | Buy Prob | Sell Prob")
    print("-" * 40)
    
    for price in price_levels:
        buy_prob = analyzer.calculate_execution_probability(price, 'BUY')
        sell_prob = analyzer.calculate_execution_probability(price, 'SELL')
        print(f"{price:10.2f} | {buy_prob:8.2%} | {sell_prob:8.2%}")

def main():
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(order_book_depth=10)
    
    # Create and add order book snapshot
    snapshot = create_detailed_order_book()
    analyzer.add_order_book_snapshot(snapshot)
    
    # Create and add trades
    trades = create_trades_with_fill_ratios()
    for trade in trades:
        analyzer.add_trade(trade)
    
    print("Fill Ratio and Execution Probability Analysis")
    print("=" * 50)
    
    # Calculate fill ratios with different windows
    windows = [50, 100]
    print("\nFill Ratios with Different Windows:")
    print("-" * 30)
    for window in windows:
        fill_ratio = analyzer.calculate_fill_ratio(window=window)
        print(f"Window {window}: {fill_ratio:.2%}")
    
    # Analyze execution probabilities
    analyze_execution_probabilities(analyzer)
    
    # Show detailed order book state
    print("\nOrder Book State:")
    print("-" * 30)
    latest_snapshot = analyzer.order_book_history[-1]
    
    print("\nBids:")
    for bid in latest_snapshot.bids:
        print(f"Price: ${bid.price:.2f}, Volume: {bid.volume}, Orders: {bid.orders}")
    
    print("\nAsks:")
    for ask in latest_snapshot.asks:
        print(f"Price: ${ask.price:.2f}, Volume: {ask.volume}, Orders: {ask.orders}")

if __name__ == "__main__":
    main() 