import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from inventory_management import InventoryManager, AssetMetrics
from time_series import TimeSeriesAnalyzer

@dataclass
class EdgeCaseMetrics:
    price_volatility: float
    liquidity_score: float
    market_impact: float
    news_sensitivity: float
    extreme_move_threshold: float
    low_liquidity_threshold: float
    impact_threshold: float
    news_threshold: float

@dataclass
class EarningsData:
    eps: float
    pe_ratio: float
    prev_eps: float
    confidence: float
    sentiment_score: float
    expected_eps: float  # Added for tracking expectations
    revenue: float
    guidance: Optional[float]  # Optional future guidance
    earnings_date: datetime
    market_cap: float

@dataclass
class SignatureData:
    current_signatures: int
    threshold: int
    daily_change: int
    sentiment_score: float
    timestamp: datetime
    expected_daily_change: float  # Added for tracking expectations
    news_impact: float  # Added for tracking news impact on signatures

@dataclass
class VolatilityMetrics:
    garch_vol: float
    historical_vol: float
    implied_vol: float
    realized_vol: float
    volatility_ratio: float  # Ratio of current to historical volatility

@dataclass
class NewsSentiment:
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float
    source: str
    impact_score: float  # 0 to 1
    keywords: List[str]
    volume: int  # Number of similar news items

@dataclass
class NewsEvent:
    timestamp: datetime
    price_before: float
    price_after: float
    price_change: float
    spread_before: float
    spread_after: float
    volatility_before: float
    volatility_after: float

class APTBot:
    def __init__(self, risk_budget: float = 1000000):
        self.inventory_manager = InventoryManager(risk_budget)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.earnings_history: List[EarningsData] = []
        self.news_history: List[NewsEvent] = []
        self.confidence_interval = 0.95
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.05,  # 5% price movement
            low_liquidity_threshold=0.3,   # 30% of average volume
            impact_threshold=0.02,         # 2% price impact
            news_threshold=0.1            # 10% price change
        )
        self.earnings_surprise_threshold = 0.15  # 15% deviation from expectations
        self.position_limits = {
            'max_position': 0.2,  # 20% of risk budget
            'min_position': 0.05,  # 5% of risk budget
            'target_position': 0.1  # 10% of risk budget
        }
        self.spread_thresholds = {
            'high_volatility': 0.15,  # 15% spread triggers inventory collection
            'normal': 0.05,           # 5% normal spread
            'low_volatility': 0.02    # 2% low volatility spread
        }
        
    def handle_news_event(self, news: NewsEvent) -> float:
        """Handle news events based on price reaction and volatility."""
        # Calculate price reaction
        price_reaction = news.price_change / news.price_before
        
        # Calculate volatility change
        vol_change = news.volatility_after / news.volatility_before
        
        # Calculate spread change
        spread_change = news.spread_after / news.spread_before
        
        # If spread is high, focus on volatility rather than news
        if news.spread_after > self.spread_thresholds['high_volatility']:
            return self._handle_high_volatility(vol_change)
        else:
            return self._handle_price_reaction(price_reaction, spread_change)
    
    def _handle_high_volatility(self, vol_change: float) -> float:
        """Handle high volatility periods by collecting inventory."""
        # Increase position size when volatility is high
        position_adjustment = 1 + vol_change * 0.5  # 50% impact of volatility change
        return min(position_adjustment, self.position_limits['max_position'])
    
    def _handle_price_reaction(self, price_reaction: float, spread_change: float) -> float:
        """Handle price reactions to news events."""
        # If price reaction is significant
        if abs(price_reaction) > self.edge_case_metrics.news_threshold:
            # Reduce position size for large price movements
            return 1 - abs(price_reaction) / self.edge_case_metrics.news_threshold
        
        # If spread has increased significantly
        if spread_change > 1.5:  # 50% increase in spread
            # Increase position size to collect inventory
            return 1 + (spread_change - 1) * 0.5
        
        return 1.0
    
    def adjust_spread(self, volatility: float) -> float:
        """Adjust spread based on volatility and news events."""
        base_spread = 0.1  # 10 cents base spread
        
        # Get latest news event if available
        latest_news = self.news_history[-1] if self.news_history else None
        
        # Calculate volatility adjustment
        vol_adjustment = volatility * 2
        
        # Calculate news-based adjustment if available
        news_adjustment = 1.0
        if latest_news:
            # If spread is high, focus on volatility
            if latest_news.spread_after > self.spread_thresholds['high_volatility']:
                news_adjustment = 1 + latest_news.volatility_after * 1.5
            else:
                # Adjust based on price reaction
                price_reaction = latest_news.price_change / latest_news.price_before
                news_adjustment = 1 + abs(price_reaction) * 2
        
        return base_spread * (1 + vol_adjustment) * news_adjustment

class DLRBot:
    def __init__(self, risk_budget: float = 1000000):
        self.inventory_manager = InventoryManager(risk_budget)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.signature_history: List[SignatureData] = []
        self.news_history: List[NewsEvent] = []
        self.v_success = 100.0
        self.v_failure = 50.0
        self.alpha = 0.01
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.1,    # 10% price movement
            low_liquidity_threshold=0.2,    # 20% of average volume
            impact_threshold=0.03,          # 3% price impact
            news_threshold=0.15            # 15% price change
        )
        self.spread_thresholds = {
            'high_volatility': 0.2,  # 20% spread triggers inventory collection
            'normal': 0.08,          # 8% normal spread
            'low_volatility': 0.03   # 3% low volatility spread
        }
    
    def handle_news_event(self, news: NewsEvent) -> float:
        """Handle news events based on price reaction and volatility."""
        # Calculate price reaction
        price_reaction = news.price_change / news.price_before
        
        # Calculate volatility change
        vol_change = news.volatility_after / news.volatility_before
        
        # Calculate spread change
        spread_change = news.spread_after / news.spread_before
        
        # If spread is high, focus on volatility rather than news
        if news.spread_after > self.spread_thresholds['high_volatility']:
            return self._handle_high_volatility(vol_change)
        else:
            return self._handle_price_reaction(price_reaction, spread_change)
    
    def _handle_high_volatility(self, vol_change: float) -> float:
        """Handle high volatility periods by collecting inventory."""
        # Increase position size when volatility is high
        position_adjustment = 1 + vol_change * 0.7  # 70% impact of volatility change
        return min(position_adjustment, self.position_limits['max_position'])
    
    def _handle_price_reaction(self, price_reaction: float, spread_change: float) -> float:
        """Handle price reactions to news events."""
        # If price reaction is significant
        if abs(price_reaction) > self.edge_case_metrics.news_threshold:
            # Reduce position size for large price movements
            return 1 - abs(price_reaction) / self.edge_case_metrics.news_threshold
        
        # If spread has increased significantly
        if spread_change > 1.5:  # 50% increase in spread
            # Increase position size to collect inventory
            return 1 + (spread_change - 1) * 0.7
        
        return 1.0
    
    def adjust_spread(self, volatility: float) -> float:
        """Adjust spread based on volatility and news events."""
        base_spread = 0.1  # 10 cents base spread
        
        # Get latest news event if available
        latest_news = self.news_history[-1] if self.news_history else None
        
        # Calculate volatility adjustment
        vol_adjustment = volatility * 2.5
        
        # Calculate news-based adjustment if available
        news_adjustment = 1.0
        if latest_news:
            # If spread is high, focus on volatility
            if latest_news.spread_after > self.spread_thresholds['high_volatility']:
                news_adjustment = 1 + latest_news.volatility_after * 2
            else:
                # Adjust based on price reaction
                price_reaction = latest_news.price_change / latest_news.price_before
                news_adjustment = 1 + abs(price_reaction) * 2.5
        
        return base_spread * (1 + vol_adjustment) * news_adjustment

class MKJBot:
    def __init__(self, risk_budget: float = 1000000):
        self.inventory_manager = InventoryManager(risk_budget)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.volatility_history: List[VolatilityMetrics] = []
        self.news_history: List[NewsEvent] = []
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.15,    # 15% price movement
            low_liquidity_threshold=0.25,    # 25% of average volume
            impact_threshold=0.04,           # 4% price impact
            news_threshold=0.2              # 20% price change
        )
        self.spread_thresholds = {
            'high_volatility': 0.25,  # 25% spread triggers inventory collection
            'normal': 0.1,            # 10% normal spread
            'low_volatility': 0.05    # 5% low volatility spread
        }
    
    def handle_news_event(self, news: NewsEvent) -> float:
        """Handle news events based on price reaction and volatility."""
        # Calculate price reaction
        price_reaction = news.price_change / news.price_before
        
        # Calculate volatility change
        vol_change = news.volatility_after / news.volatility_before
        
        # Calculate spread change
        spread_change = news.spread_after / news.spread_before
        
        # If spread is high, focus on volatility rather than news
        if news.spread_after > self.spread_thresholds['high_volatility']:
            return self._handle_high_volatility(vol_change)
        else:
            return self._handle_price_reaction(price_reaction, spread_change)
    
    def _handle_high_volatility(self, vol_change: float) -> float:
        """Handle high volatility periods by collecting inventory."""
        # Increase position size when volatility is high
        position_adjustment = 1 + vol_change * 0.8  # 80% impact of volatility change
        return min(position_adjustment, self.position_limits['max_position'])
    
    def _handle_price_reaction(self, price_reaction: float, spread_change: float) -> float:
        """Handle price reactions to news events."""
        # If price reaction is significant
        if abs(price_reaction) > self.edge_case_metrics.news_threshold:
            # Reduce position size for large price movements
            return 1 - abs(price_reaction) / self.edge_case_metrics.news_threshold
        
        # If spread has increased significantly
        if spread_change > 1.5:  # 50% increase in spread
            # Increase position size to collect inventory
            return 1 + (spread_change - 1) * 0.8
        
        return 1.0
    
    def adjust_spread(self, volatility: float) -> float:
        """Adjust spread based on volatility and news events."""
        base_spread = 0.1  # 10 cents base spread
        
        # Get latest news event if available
        latest_news = self.news_history[-1] if self.news_history else None
        
        # Calculate volatility adjustment
        vol_adjustment = volatility * 3
        
        # Calculate news-based adjustment if available
        news_adjustment = 1.0
        if latest_news:
            # If spread is high, focus on volatility
            if latest_news.spread_after > self.spread_thresholds['high_volatility']:
                news_adjustment = 1 + latest_news.volatility_after * 2.5
            else:
                # Adjust based on price reaction
                price_reaction = latest_news.price_change / latest_news.price_before
                news_adjustment = 1 + abs(price_reaction) * 3
        
        return base_spread * (1 + vol_adjustment) * news_adjustment

class ETFBot:
    def __init__(self, risk_budget: float = 1000000):
        self.inventory_manager = InventoryManager(risk_budget)
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.correlation_history: Dict[str, List[float]] = {}
        self.volatility_history: Dict[str, List[VolatilityMetrics]] = {}
        self.news_history: Dict[str, List[NewsEvent]] = {}
        self.volatility_drag = 0.0
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.08,    # 8% price movement
            low_liquidity_threshold=0.35,    # 35% of average volume
            impact_threshold=0.025,          # 2.5% price impact
            news_threshold=0.12             # 12% price change
        )
        self.spread_thresholds = {
            'high_volatility': 0.18,  # 18% spread triggers inventory collection
            'normal': 0.06,          # 6% normal spread
            'low_volatility': 0.02   # 2% low volatility spread
        }
    
    def handle_news_event(self, stock: str, news: NewsEvent) -> float:
        """Handle news events based on price reaction and volatility."""
        if stock not in self.news_history:
            self.news_history[stock] = []
        self.news_history[stock].append(news)
        
        # Calculate price reaction
        price_reaction = news.price_change / news.price_before
        
        # Calculate volatility change
        vol_change = news.volatility_after / news.volatility_before
        
        # Calculate spread change
        spread_change = news.spread_after / news.spread_before
        
        # If spread is high, focus on volatility rather than news
        if news.spread_after > self.spread_thresholds['high_volatility']:
            return self._handle_high_volatility(stock, vol_change)
        else:
            return self._handle_price_reaction(stock, price_reaction, spread_change)
    
    def _handle_high_volatility(self, stock: str, vol_change: float) -> float:
        """Handle high volatility periods by collecting inventory."""
        # Increase position size when volatility is high
        position_adjustment = 1 + vol_change * 0.6  # 60% impact of volatility change
        return min(position_adjustment, self.hedging_limits['max_hedge'])
    
    def _handle_price_reaction(self, stock: str, price_reaction: float, spread_change: float) -> float:
        """Handle price reactions to news events."""
        # If price reaction is significant
        if abs(price_reaction) > self.edge_case_metrics.news_threshold:
            # Reduce position size for large price movements
            return 1 - abs(price_reaction) / self.edge_case_metrics.news_threshold
        
        # If spread has increased significantly
        if spread_change > 1.5:  # 50% increase in spread
            # Increase position size to collect inventory
            return 1 + (spread_change - 1) * 0.6
        
        return 1.0
    
    def adjust_spread(self, volatility: float, liquidity: float) -> float:
        """Adjust spread based on volatility and news events."""
        base_spread = 0.05  # 5 cents base spread
        
        # Calculate volatility adjustment
        vol_adjustment = volatility * 1.5
        
        # Calculate liquidity adjustment
        liquidity_factor = 1 / (1 + liquidity/1000000)
        
        # Calculate volatility drag adjustment
        vol_drag_factor = 1 + abs(self.volatility_drag) * 2
        
        # Calculate news-based adjustment if available
        news_adjustment = 1.0
        for stock, news_list in self.news_history.items():
            if news_list:
                latest_news = news_list[-1]
                # If spread is high, focus on volatility
                if latest_news.spread_after > self.spread_thresholds['high_volatility']:
                    news_adjustment *= (1 + latest_news.volatility_after * 1.5)
                else:
                    # Adjust based on price reaction
                    price_reaction = latest_news.price_change / latest_news.price_before
                    news_adjustment *= (1 + abs(price_reaction) * 1.5)
        
        return base_spread * (1 + vol_adjustment) * liquidity_factor * vol_drag_factor * news_adjustment 