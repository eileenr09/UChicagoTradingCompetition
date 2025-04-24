import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class AssetMetrics:
    volatility: float
    liquidity: float
    spread: float
    volume: float
    price: float
    beta: float

@dataclass
class InventoryTarget:
    min_inventory: float
    max_inventory: float
    target_inventory: float
    current_inventory: float

class InventoryManager:
    def __init__(self, risk_budget: float = 1000000):
        self.risk_budget = risk_budget
        self.asset_metrics: Dict[str, AssetMetrics] = {}
        self.inventory_targets: Dict[str, InventoryTarget] = {}
        self.etf_positions: Dict[str, float] = {}
        
    def update_asset_metrics(self, symbol: str, metrics: AssetMetrics):
        """Update metrics for a specific asset."""
        self.asset_metrics[symbol] = metrics
        
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
        
        # Set inventory bounds
        min_inventory = -max_position * 0.5  # Allow short positions up to 50% of max
        max_inventory = max_position
        
        return InventoryTarget(
            min_inventory=min_inventory,
            max_inventory=max_inventory,
            target_inventory=target_inventory,
            current_inventory=self.inventory_targets.get(symbol, InventoryTarget(0, 0, 0, 0)).current_inventory
        )
    
    def update_inventory_targets(self):
        """Update inventory targets for all assets."""
        for symbol in self.asset_metrics:
            self.inventory_targets[symbol] = self.calculate_inventory_targets(symbol)
    
    def calculate_etf_conversion(self, symbol: str, etf_symbol: str) -> Tuple[bool, float]:
        """Determine if ETF conversion is needed and calculate the amount."""
        if symbol not in self.inventory_targets or etf_symbol not in self.etf_positions:
            return False, 0.0
            
        target = self.inventory_targets[symbol]
        current = target.current_inventory
        
        # Calculate deviation from target
        deviation = current - target.target_inventory
        
        # Determine if conversion is needed
        if abs(deviation) > target.max_inventory * 0.2:  # 20% threshold
            # Calculate conversion amount
            conversion_amount = deviation * self.asset_metrics[symbol].beta
            
            # Check if we have enough ETF position
            etf_position = self.etf_positions.get(etf_symbol, 0)
            
            if conversion_amount > 0 and etf_position >= conversion_amount:
                return True, -conversion_amount  # Convert ETF to shares
            elif conversion_amount < 0 and etf_position <= conversion_amount:
                return True, -conversion_amount  # Convert shares to ETF
                
        return False, 0.0
    
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
        
        # Re-normalize weights
        total_weight = sum(weights.values())
        for symbol in weights:
            weights[symbol] /= total_weight
            
        return weights
    
    def update_positions(self, symbol: str, etf_symbol: str, trade_volume: float):
        """Update positions after a trade."""
        if symbol in self.inventory_targets:
            target = self.inventory_targets[symbol]
            target.current_inventory += trade_volume
            
        if etf_symbol in self.etf_positions:
            self.etf_positions[etf_symbol] -= trade_volume * self.asset_metrics[symbol].beta
    
    def get_rebalance_signals(self) -> List[Tuple[str, str, float]]:
        """Get signals for portfolio rebalancing."""
        rebalance_signals = []
        optimal_weights = self.calculate_optimal_portfolio()
        
        for symbol, target in self.inventory_targets.items():
            current_weight = target.current_inventory * self.asset_metrics[symbol].price / self.risk_budget
            optimal_weight = optimal_weights[symbol]
            
            if abs(current_weight - optimal_weight) > 0.1:  # 10% threshold
                rebalance_amount = (optimal_weight - current_weight) * self.risk_budget / self.asset_metrics[symbol].price
                rebalance_signals.append((symbol, "REBALANCE", rebalance_amount))
        
        return rebalance_signals 