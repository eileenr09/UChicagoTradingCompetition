import numpy as np
from market_maker import MarketMaker, MarketParameters
from fair_value import FairValueCalculator, APTParameters, DLRParameters, MKJParameters
from hedging import HedgingStrategy, HedgeParameters

def main():
    # Example usage of the market making system
    
    # 1. Initialize market maker with parameters
    market_params = MarketParameters(
        gamma=0.1,    # Risk aversion parameter
        k=1.0,        # Order arrival intensity
        sigma=0.2,    # Volatility
        T=1.0,        # Terminal time
        t=0.0         # Current time
    )
    market_maker = MarketMaker(market_params)
    
    # 2. Calculate fair values for different assets
    # APT Fair Value
    apt_params = APTParameters(
        eps=2.5,      # Current EPS
        pe_ratio=15,  # P/E ratio
        prev_eps=2.0  # Previous EPS
    )
    apt_fair_value = FairValueCalculator.calculate_apt_fair_value(apt_params)
    
    # DLR Fair Value
    dlr_params = DLRParameters(
        signatures=800,    # Current signatures
        threshold=1000,    # Required signatures
        v_success=100,     # Value if successful
        v_failure=50,      # Value if failed
        alpha=0.01         # Logistic model parameter
    )
    dlr_fair_value = FairValueCalculator.calculate_dlr_fair_value(dlr_params)
    
    # MKJ Fair Value
    mkj_params = MKJParameters(
        current_price=50.0,
        volatility=0.3,
        sentiment_score=0.2,
        gamma=1.0
    )
    mkj_fair_value = FairValueCalculator.calculate_mkj_fair_value(mkj_params)
    
    # 3. Calculate optimal bid-ask spreads
    current_price = 100.0
    bid_price, ask_price = market_maker.calculate_bid_ask_prices(current_price)
    spread = ask_price - bid_price
    
    # 4. Calculate execution probability
    execution_prob = market_maker.calculate_execution_probability(bid_price, ask_price)
    
    # 5. Calculate market impact
    order_size = 1000
    liquidity = 5000
    market_impact = market_maker.calculate_market_impact(order_size, liquidity)
    
    # 6. Calculate hedge ratios
    hedge_params = HedgeParameters(
        stock_price=100.0,
        etf_price=200.0,
        beta=1.2,
        position_size=1000
    )
    hedge_ratio = HedgingStrategy.calculate_hedge_ratio(hedge_params)
    etf_position = HedgingStrategy.calculate_etf_position(hedge_params)
    
    # Print results
    print("Market Making System Results:")
    print(f"APT Fair Value: ${apt_fair_value:.2f}")
    print(f"DLR Fair Value: ${dlr_fair_value:.2f}")
    print(f"MKJ Fair Value: ${mkj_fair_value:.2f}")
    print(f"Optimal Bid-Ask Spread: ${spread:.2f}")
    print(f"Execution Probability: {execution_prob:.2%}")
    print(f"Market Impact: ${market_impact:.2f}")
    print(f"Hedge Ratio: {hedge_ratio:.2f}")
    print(f"Required ETF Position: {etf_position:.2f}")

if __name__ == "__main__":
    main() 