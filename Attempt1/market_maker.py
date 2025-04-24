import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class MarketParameters:
    gamma: float  # Risk aversion parameter
    k: float      # Order arrival intensity
    sigma: float  # Volatility
    T: float      # Terminal time
    t: float      # Current time

@dataclass
class HedgeParameters:
    stock_price: float
    etf_price: float
    beta: float
    position_size: float

@dataclass
class AssetMetrics:
    price: float
    volatility: float
    beta: float
    liquidity: float
    volume: float
    spread: float

@dataclass
class InventoryTarget:
    min_inventory: float
    max_inventory: float
    target_inventory: float
    current_inventory: float

class MarketMaker:
    def __init__(self, params: MarketParameters):
        self.params = params
        self.inventory = 0  # Current inventory position
        # Initialize required attributes
        self.inventory_targets: Dict[str, InventoryTarget] = {}
        self.etf_positions: Dict[str, float] = {}
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        
    def calculate_optimal_spread(self) -> float:
        """Calculate the optimal bid-ask spread using the Avellaneda & Stoikov model."""
        T_minus_t = self.params.T - self.params.t
        spread = (self.params.gamma * self.params.sigma**2 * T_minus_t + 
                 2 * self.params.gamma * np.log(1 + self.params.gamma * self.params.k))
        return spread
    
    def calculate_reservation_price(self, current_price: float) -> float:
        """Calculate the reservation price based on inventory position."""
        T_minus_t = self.params.T - self.params.t
        inventory_adjustment = self.inventory * (2 * self.params.gamma * self.params.sigma**2 * T_minus_t) / 2
        return current_price - inventory_adjustment
    
    def calculate_bid_ask_prices(self, current_price: float) -> Tuple[float, float]:
        """Calculate bid and ask prices based on reservation price and spread."""
        spread = self.calculate_optimal_spread()
        reservation_price = self.calculate_reservation_price(current_price)
        
        bid_price = reservation_price - spread / 2
        ask_price = reservation_price + spread / 2
        
        return bid_price, ask_price
    
    def calculate_execution_probability(self, bid_price: float, ask_price: float, alpha: float = 1.0) -> float:
        """Calculate the probability of order execution based on spread."""
        spread = ask_price - bid_price
        prob = min(1, 1 / (1 + alpha * abs(spread)))
        return prob
    
    def update_inventory(self, quantity: int):
        """Update the market maker's inventory position."""
        self.inventory += quantity
    
    def calculate_market_impact(self, order_size: float, liquidity: float, alpha: float = 1.0) -> float:
        """Calculate market impact based on order size and liquidity."""
        impact = alpha * (liquidity / order_size)
        return impact
    
    def calculate_power_law_impact(self, order_size: float, liquidity: float, 
                                 beta: float = 1.0, delta: float = 0.5) -> float:
        """Calculate market impact using the power law model."""
        impact = beta * (liquidity / order_size) ** delta
        return impact

    def calculate_etf_conversion(self, symbol: str, etf_symbol: str) -> Tuple[bool, float]:
        """
        Determine if ETF conversion is needed and calculate the amount.
        
        Trigger Conditions:
        - The system monitors the deviation between current inventory and target inventory
        - A conversion is triggered when the deviation exceeds 20% of the maximum inventory
        - The conversion amount is calculated based on the deviation and the asset's beta
        
        Conversion Direction:
        - If conversion_amount > 0: Convert ETF to shares (when we need more shares)
        - If conversion_amount < 0: Convert shares to ETF (when we need to reduce share exposure)
        """
        # Check if we have the necessary data for both the stock and ETF
        if symbol not in self.inventory_targets or etf_symbol not in self.etf_positions:
            return False, 0.0
        
        # Get current inventory position and target
        target = self.inventory_targets[symbol]
        current = target.current_inventory
        
        # Calculate deviation from target inventory
        deviation = current - target.target_inventory
        
        # Check if deviation exceeds 20% threshold of max inventory
        if abs(deviation) > target.max_inventory * 0.2:  # 20% threshold
            # Calculate conversion amount based on deviation and asset's beta
            conversion_amount = deviation * self.asset_metrics[symbol].beta
            
            # Get current ETF position
            etf_position = self.etf_positions.get(etf_symbol, 0)
            
            # Determine conversion direction and check position availability
            if conversion_amount > 0 and etf_position >= conversion_amount:
                # Convert ETF to shares (need more shares)
                return True, -conversion_amount
            elif conversion_amount < 0 and etf_position <= conversion_amount:
                # Convert shares to ETF (need to reduce share exposure)
                return True, -conversion_amount
                
        return False, 0.0

    def update_positions(self, symbol: str, etf_symbol: str, trade_volume: float):
        """
        Update positions after a trade.
        
        This method updates both the stock and ETF positions after a conversion:
        - Stock position is updated by the trade volume
        - ETF position is updated by the trade volume multiplied by the asset's beta
        """
        # Update stock position
        if symbol in self.inventory_targets:
            target = self.inventory_targets[symbol]
            target.current_inventory += trade_volume
        
        # Update ETF position based on beta
        if etf_symbol in self.etf_positions:
            self.etf_positions[etf_symbol] -= trade_volume * self.asset_metrics[symbol].beta

    def calculate_hedge_ratio(self, params: HedgeParameters) -> float:
        """Calculate the hedge ratio for ETF-based hedging."""
        return params.beta * (params.etf_price / params.stock_price) 
