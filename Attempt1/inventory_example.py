import numpy as np
from inventory_management import InventoryManager, AssetMetrics, InventoryTarget

def create_sample_metrics():
    """Create sample asset metrics for demonstration."""
    return {
        'APT': AssetMetrics(
            volatility=0.2,    # 20% volatility
            liquidity=1000000,  # Daily liquidity
            spread=0.1,        # 10 cents spread
            volume=500000,     # Daily volume
            price=50.0,        # Current price
            beta=1.2          # Beta relative to market
        ),
        'DLR': AssetMetrics(
            volatility=0.15,
            liquidity=800000,
            spread=0.15,
            volume=400000,
            price=75.0,
            beta=0.8
        ),
        'MKJ': AssetMetrics(
            volatility=0.3,
            liquidity=600000,
            spread=0.2,
            volume=300000,
            price=25.0,
            beta=1.5
        ),
        'SPY': AssetMetrics(  # ETF
            volatility=0.15,
            liquidity=5000000,
            spread=0.05,
            volume=2000000,
            price=400.0,
            beta=1.0
        )
    }

def main():
    # Initialize inventory manager with $1M risk budget
    manager = InventoryManager(risk_budget=1000000)
    
    # Add asset metrics
    metrics = create_sample_metrics()
    for symbol, metric in metrics.items():
        manager.update_asset_metrics(symbol, metric)
    
    # Update inventory targets
    manager.update_inventory_targets()
    
    print("Inventory Management Analysis")
    print("=" * 50)
    
    # Display inventory targets for each asset
    print("\nInventory Targets:")
    print("-" * 50)
    for symbol, target in manager.inventory_targets.items():
        print(f"\n{symbol}:")
        print(f"  Current Inventory: {target.current_inventory:.0f} shares")
        print(f"  Target Inventory: {target.target_inventory:.0f} shares")
        print(f"  Min Inventory: {target.min_inventory:.0f} shares")
        print(f"  Max Inventory: {target.max_inventory:.0f} shares")
    
    # Calculate optimal portfolio weights
    optimal_weights = manager.calculate_optimal_portfolio()
    print("\nOptimal Portfolio Weights:")
    print("-" * 50)
    for symbol, weight in optimal_weights.items():
        print(f"{symbol}: {weight:.1%}")
    
    # Simulate some trades and check ETF conversion signals
    print("\nETF Conversion Analysis:")
    print("-" * 50)
    
    # Simulate a large trade in APT
    manager.update_positions('APT', 'SPY', 1000)
    
    # Check if ETF conversion is needed
    should_convert, amount = manager.calculate_etf_conversion('APT', 'SPY')
    if should_convert:
        print(f"ETF Conversion Signal:")
        print(f"  Convert {abs(amount):.0f} shares of {'SPY to APT' if amount < 0 else 'APT to SPY'}")
    
    # Get rebalancing signals
    rebalance_signals = manager.get_rebalance_signals()
    if rebalance_signals:
        print("\nRebalancing Signals:")
        print("-" * 50)
        for symbol, action, amount in rebalance_signals:
            print(f"{symbol}: {action} {amount:.0f} shares")

if __name__ == "__main__":
    main() 