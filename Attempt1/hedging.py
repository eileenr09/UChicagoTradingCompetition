import numpy as np
from typing import Tuple, List
from dataclasses import dataclass

@dataclass
class HedgeParameters:
    stock_price: float
    etf_price: float
    beta: float
    position_size: float

class HedgingStrategy:
    @staticmethod
    def calculate_hedge_ratio(params: HedgeParameters) -> float:
        """Calculate the hedge ratio for ETF-based hedging."""
        return params.beta * (params.etf_price / params.stock_price)
    
    @staticmethod
    def calculate_etf_position(params: HedgeParameters) -> float:
        """Calculate the required ETF position size for hedging."""
        hedge_ratio = HedgingStrategy.calculate_hedge_ratio(params)
        return -hedge_ratio * params.position_size  # Negative for hedging
    
    @staticmethod
    def calculate_delta_hedge(portfolio_value: float, 
                            price_change: float,
                            delta: float) -> float:
        """Calculate the required hedge position based on delta."""
        return -delta * portfolio_value * price_change
    
    @staticmethod
    def calculate_beta(stock_returns: np.ndarray, 
                      market_returns: np.ndarray) -> float:
        """Calculate beta coefficient between stock and market returns."""
        covariance = np.cov(stock_returns, market_returns)[0,1]
        market_variance = np.var(market_returns)
        return covariance / market_variance
    
    @staticmethod
    def calculate_optimal_hedge_ratio(returns: List[Tuple[float, float]]) -> float:
        """Calculate optimal hedge ratio using historical returns."""
        stock_returns = np.array([r[0] for r in returns])
        market_returns = np.array([r[1] for r in returns])
        
        # Calculate covariance matrix
        cov_matrix = np.cov(stock_returns, market_returns)
        
        # Calculate optimal hedge ratio
        optimal_ratio = cov_matrix[0,1] / cov_matrix[1,1]
        
        return optimal_ratio 