# UChicago Trading Bot Project

A comprehensive trading bot system designed for market making and automated trading strategies. This project implements specialized bots for different trading scenarios with advanced risk management and market analysis capabilities.

## Project Structure

```
UChicago/
├── specialized_bots.py    # Core bot implementations
├── time_series.py        # Time series analysis and market metrics
├── risk_management.py    # Risk management and position sizing
├── market_analysis.py    # Market analysis and signal generation
├── data_collection.py    # Data collection and preprocessing
├── utils.py             # Utility functions and helpers
└── config.py            # Configuration settings
```

## Core Components

### Specialized Bots

1. **APT Bot**
   - Earnings-based trading strategy
   - Handles earnings surprises and volatility
   - Position sizing based on earnings momentum
   - Risk management for earnings events

2. **DLR Bot**
   - Petition monitoring and analysis
   - Signature tracking with log-normal distribution
   - Position adjustments based on signature momentum
   - Edge case handling for signature spikes

3. **MKJ Bot**
   - News sentiment analysis
   - Price reaction tracking
   - Aggressive hedging strategies
   - Volatility-based position sizing

4. **ETF Bot**
   - ETF spread management
   - Correlation tracking with underlying assets
   - Dynamic hedging ratios
   - Volatility drag consideration

### Market Analysis

- Time series analysis for price patterns
- Volatility tracking and prediction
- Correlation analysis between assets
- Market impact assessment
- Liquidity analysis

### Risk Management

- Position sizing optimization
- Risk budget allocation
- Stop-loss and take-profit management
- Portfolio exposure monitoring
- Drawdown protection

## Features

### Advanced Analytics
- Log-normal distribution modeling
- GARCH volatility estimation
- Sentiment analysis integration
- Correlation-based hedging
- Volatility drag calculation

### Risk Controls
- Dynamic position limits
- Volatility-based adjustments
- Liquidity monitoring
- Market impact assessment
- Edge case handling

### Market Making
- Spread management
- Inventory control
- Liquidity provision
- Order book analysis
- Price impact minimization

## Installation```

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Configure your environment:
```bash
cp config.example.py config.py
# Edit config.py with your settings
```

## Usage

### Basic Bot Initialization

```python
from specialized_bots import APTBot, DLRBot, MKJBot, ETFBot

# Initialize bots with risk budget
apt_bot = APTBot(risk_budget=100000)
dlr_bot = DLRBot(risk_budget=100000)
mkj_bot = MKJBot(risk_budget=100000)
etf_bot = ETFBot(risk_budget=100000)
```

### Market Analysis

```python
from time_series import TimeSeriesAnalyzer

# Initialize analyzer
analyzer = TimeSeriesAnalyzer()

# Update market data
analyzer.update_volatility_metrics("AAPL", 0.2)
analyzer.update_earnings_data((2.5, 25.0, 2.3))
analyzer.update_news_events("AAPL", 150.0, 155.0, 0.05)
```

### Risk Management

```python
from risk_management import RiskManager

# Initialize risk manager
risk_manager = RiskManager(max_position_size=1000)

# Calculate position size
position_size = risk_manager.calculate_position_size(
    price=100.0,
    volatility=0.2,
    risk_budget=10000
)
```

## Configuration

Key configuration parameters in `config.py`:

```python
# Risk Management
MAX_POSITION_SIZE = 1000
RISK_BUDGET = 100000
STOP_LOSS_PERCENTAGE = 0.02

# Market Analysis
VOLATILITY_WINDOW = 20
CORRELATION_WINDOW = 60
SENTIMENT_THRESHOLD = 0.7

# Trading Parameters
MIN_SPREAD = 0.01
MAX_SPREAD = 0.05
LIQUIDITY_THRESHOLD = 1000
```

## Performance Monitoring

The system includes comprehensive performance monitoring:

- Position tracking
- P&L calculation
- Risk metrics
- Market impact analysis
- Liquidity monitoring

## Error Handling

Robust error handling for:
- Market data issues
- Network connectivity
- Order execution failures
- Risk limit breaches
- System errors

