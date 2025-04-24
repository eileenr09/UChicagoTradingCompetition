import numpy as np
from datetime import datetime, timedelta
from time_series import (
    TimeSeriesAnalyzer,
    OrderBookSnapshot,
    OrderBookLevel,
    Trade
)

def create_sample_data():
    """Create sample order book and trade data for demonstration."""
    # Create sample order book snapshots
    snapshots = []
    base_price = 100.0
    base_time = datetime.now()
    
    for i in range(100):
        # Create bids
        bids = []
        for j in range(10):
            price = base_price - j * 0.1
            volume = 100 - j * 10
            orders = 5 - j
            bids.append(OrderBookLevel(price=price, volume=volume, orders=orders))
        
        # Create asks
        asks = []
        for j in range(10):
            price = base_price + j * 0.1
            volume = 100 - j * 10
            orders = 5 - j
            asks.append(OrderBookLevel(price=price, volume=volume, orders=orders))
        
        snapshot = OrderBookSnapshot(
            timestamp=base_time + timedelta(seconds=i),
            bids=bids,
            asks=asks,
            depth=10
        )
        snapshots.append(snapshot)
    
    # Create sample trades
    trades = []
    for i in range(50):
        trade = Trade(
            timestamp=base_time + timedelta(seconds=i*2),
            price=base_price + np.random.normal(0, 0.1),
            volume=int(np.random.uniform(10, 100)),
            aggressiveness=np.random.choice(['BUY', 'SELL']),
            hit_side=np.random.choice(['BID', 'ASK'])
        )
        trades.append(trade)
    
    return snapshots, trades

def main():
    # Initialize analyzer
    analyzer = TimeSeriesAnalyzer(order_book_depth=10)
    
    # Create and add sample data
    snapshots, trades = create_sample_data()
    
    # Add order book snapshots
    for snapshot in snapshots:
        analyzer.add_order_book_snapshot(snapshot)
    
    # Add trades
    for trade in trades:
        analyzer.add_trade(trade)
    
    # Add some sample news events
    base_time = datetime.now()
    analyzer.add_news_event(base_time + timedelta(minutes=5), "EARNINGS", 0.8)
    analyzer.add_news_event(base_time + timedelta(minutes=10), "REGULATORY", 0.6)
    
    # Calculate various metrics
    print("Time Series Analysis Results:")
    print("-" * 30)
    
    # Order flow imbalance
    imbalance = analyzer.calculate_order_flow_imbalance(window=50)
    print(f"Order Flow Imbalance: {imbalance:.2f}")
    
    # Spread dynamics
    spread_series = analyzer.calculate_spread_dynamics(interval='1min')
    print(f"Average Spread: ${spread_series.mean():.2f}")
    
    # Spoofing detection
    suspicious_events = analyzer.detect_spoofing(window=50)
    print(f"Number of Suspicious Events: {len(suspicious_events)}")
    
    # Market impact
    order_size = 1000
    price_change = 0.5
    impact = analyzer.calculate_market_impact(order_size, price_change)
    print(f"Market Impact: {impact:.4f}")
    
    # News impact analysis
    news_impacts = analyzer.analyze_news_impact("STOCK")
    print("\nNews Impact Analysis:")
    for news_type, impact in news_impacts.items():
        print(f"{news_type}: {impact:.2%}")
    
    # Fill ratio
    fill_ratio = analyzer.calculate_fill_ratio(window=50)
    print(f"Fill Ratio: {fill_ratio:.2%}")
    
    # Execution probability
    current_price = 100.0
    buy_prob = analyzer.calculate_execution_probability(current_price, 'BUY')
    sell_prob = analyzer.calculate_execution_probability(current_price, 'SELL')
    print(f"Execution Probability (Buy): {buy_prob:.2%}")
    print(f"Execution Probability (Sell): {sell_prob:.2%}")

def calculate_inventory_targets(self, symbol: str) -> InventoryTarget:
    """Calculate inventory targets based on asset metrics."""
    metrics = self.asset_metrics[symbol]
    
    # Calculate position size based on risk budget and volatility
    risk_per_share = metrics.volatility * metrics.price
    max_position = self.risk_budget / risk_per_share
    
    # Adjust for liquidity constraints
    max_position = min(max_position, metrics.liquidity * 0.1)  # Max 10% of daily liquidity
    
    # Calculate target inventory based on spread and volume
    spread_cost = metrics.spread / metrics.price
    volume_factor = metrics.volume / (metrics.volume + 1000000)  # Normalize volume
    
    target_inventory = max_position * (1 - spread_cost) * volume_factor

def calculate_optimal_portfolio(self) -> Dict[str, float]:
    """Calculate optimal portfolio weights based on risk budget."""
    weights = {}
    total_risk = 0
    
    # Calculate initial weights based on risk contribution
    for symbol, metrics in self.asset_metrics.items():
        risk_contribution = metrics.volatility * metrics.beta
        weights[symbol] = risk_contribution
        total_risk += risk_contribution
    
    # Normalize weights to sum to 1
    for symbol in weights:
        weights[symbol] /= total_risk
        
    # Adjust for liquidity constraints
    for symbol, weight in weights.items():
        metrics = self.asset_metrics[symbol]
        max_weight = metrics.liquidity / sum(m.liquidity for m in self.asset_metrics.values())
        weights[symbol] = min(weight, max_weight)

if __name__ == "__main__":
    main() 

rebalance_signals = manager.get_rebalance_signals() 