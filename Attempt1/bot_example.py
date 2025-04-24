import numpy as np
from datetime import datetime
from specialized_bots import (
    APTBot, DLRBot, MKJBot, ETFBot,
    EarningsData, SignatureData, VolatilityMetrics
)

def create_sample_data():
    """Create sample data for demonstration."""
    # APT earnings data
    apt_earnings = EarningsData(
        eps=2.5,
        pe_ratio=15,
        prev_eps=2.0,
        confidence=0.85,
        sentiment_score=0.7
    )
    
    # DLR signature data
    dlr_signatures = SignatureData(
        current_signatures=80000,
        threshold=100000,
        daily_change=1000,
        sentiment_score=0.6
    )
    
    # MKJ volatility metrics
    mkj_vol = VolatilityMetrics(
        garch_vol=0.3,
        historical_vol=0.25,
        implied_vol=0.35,
        sentiment_vol=0.2
    )
    
    return apt_earnings, dlr_signatures, mkj_vol

def main():
    # Initialize bots
    apt_bot = APTBot(risk_budget=1000000)
    dlr_bot = DLRBot(risk_budget=1000000)
    mkj_bot = MKJBot(risk_budget=1000000)
    etf_bot = ETFBot(risk_budget=1000000)
    
    # Create sample data
    apt_earnings, dlr_signatures, mkj_vol = create_sample_data()
    
    print("Specialized Bot Analysis")
    print("=" * 50)
    
    # APT Analysis
    print("\nAPT Analysis:")
    print("-" * 30)
    apt_bot.update_earnings_data(apt_earnings)
    apt_signals = apt_bot.generate_signals(market_price=50.0)
    
    print(f"Fair Value: ${apt_signals['fair_value']:.2f}")
    print(f"Margin of Error: ${apt_signals['margin_of_error']:.2f}")
    print(f"Confidence: {apt_signals['confidence']:.1%}")
    print(f"Valuation Ratio: {apt_signals['valuation_ratio']:.2f}")
    print(f"Adjusted Spread: ${apt_signals['spread']:.2f}")
    
    # DLR Analysis
    print("\nDLR Analysis:")
    print("-" * 30)
    dlr_bot.update_signature_data(dlr_signatures)
    dlr_signals = dlr_bot.generate_signals()
    
    print(f"Success Probability: {dlr_signals['success_probability']:.1%}")
    print(f"Fair Value: ${dlr_signals['fair_value']:.2f}")
    print(f"Position Size: {dlr_signals['position_size']:.0f} shares")
    print(f"Sentiment Score: {dlr_signals['sentiment_score']:.2f}")
    
    # MKJ Analysis
    print("\nMKJ Analysis:")
    print("-" * 30)
    mkj_bot.update_volatility_metrics(mkj_vol)
    mkj_signals = mkj_bot.generate_signals(current_price=25.0)
    
    print(f"Fair Value: ${mkj_signals['fair_value']:.2f}")
    print(f"GARCH Volatility: {mkj_signals['garch_volatility']:.1%}")
    print(f"Historical Volatility: {mkj_signals['historical_volatility']:.1%}")
    print(f"Implied Volatility: {mkj_signals['implied_volatility']:.1%}")
    print(f"Sentiment Score: {mkj_signals['sentiment_score']:.2f}")
    print(f"Price Adjustment: ${mkj_signals['price_adjustment']:.2f}")
    
    # ETF Analysis
    print("\nETF Analysis:")
    print("-" * 30)
    
    # Update correlations
    etf_bot.update_correlation('APT', 0.8)
    etf_bot.update_correlation('DLR', 0.6)
    etf_bot.update_correlation('MKJ', 0.9)
    
    # Generate ETF signals
    stock_prices = {
        'APT': 50.0,
        'DLR': 75.0,
        'MKJ': 25.0
    }
    etf_signals = etf_bot.generate_signals(etf_price=400.0, stock_prices=stock_prices)
    
    print(f"ETF Spread: ${etf_signals['spread']:.2f}")
    print(f"Volatility Drag: {etf_signals['volatility_drag']:.1%}")
    print("\nHedge Ratios:")
    for stock, ratio in etf_signals['hedge_ratios'].items():
        print(f"{stock}: {ratio:.2f}")

if __name__ == "__main__":
    main() 