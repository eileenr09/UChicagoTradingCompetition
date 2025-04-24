import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class APTParameters:
    eps: float
    pe_ratio: float
    prev_eps: float

@dataclass
class DLRParameters:
    signatures: int
    threshold: int
    v_success: float
    v_failure: float
    alpha: float = 1.0

@dataclass
class MKJParameters:
    current_price: float
    volatility: float
    sentiment_score: float
    gamma: float = 1.0

class FairValueCalculator:
    @staticmethod
    def calculate_apt_fair_value(params: APTParameters) -> float:
        """Calculate fair value for APT based on P/E ratio and earnings."""
        base_fair_value = params.eps / params.pe_ratio
        
        # Calculate adjustment based on earnings change
        delta_eps = params.eps - params.prev_eps
        adjustment_factor = 1 + (delta_eps / params.prev_eps)
        
        return base_fair_value * adjustment_factor
    
    @staticmethod
    def calculate_dlr_fair_value(params: DLRParameters) -> float:
        """Calculate fair value for DLR based on signatures and probability."""
        # Calculate probability of success
        if params.signatures >= params.threshold:
            p_success = 1.0
        else:
            # Using logistic model for probability
            p_success = 1 / (1 + np.exp(-params.alpha * 
                (params.signatures - 2/params.threshold)))
        
        # Calculate expected value
        fair_value = (p_success * params.v_success + 
                     (1 - p_success) * params.v_failure)
        
        return fair_value
    
    @staticmethod
    def calculate_mkj_fair_value(params: MKJParameters) -> float:
        """Calculate fair value for MKJ based on volatility and sentiment."""
        # Calculate volatility factor
        volatility_factor = params.volatility
        
        # Calculate sentiment-adjusted fair value
        sentiment_adjustment = 1 + (params.gamma * params.sentiment_score)
        
        fair_value = params.current_price * sentiment_adjustment + volatility_factor
        
        return fair_value
    
    @staticmethod
    def calculate_garch_volatility(returns: np.ndarray, 
                                 omega: float = 0.0001,
                                 alpha: float = 0.1,
                                 beta: float = 0.8) -> float:
        """Calculate volatility using GARCH(1,1) model."""
        # Initialize variance
        var = np.var(returns)
        var_history = [var]
        
        # Calculate GARCH variance
        for t in range(1, len(returns)):
            var = (omega + 
                  alpha * returns[t-1]**2 + 
                  beta * var_history[t-1])
            var_history.append(var)
        
        return np.sqrt(var_history[-1]) 