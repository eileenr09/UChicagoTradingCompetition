import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class OrderBookLevel:
    price: float
    volume: int
    orders: int

@dataclass
class OrderBookSnapshot:
    timestamp: datetime
    bids: List[OrderBookLevel]
    asks: List[OrderBookLevel]
    depth: int

@dataclass
class Trade:
    timestamp: datetime
    price: float
    volume: int
    aggressiveness: str  # 'BUY' or 'SELL'
    hit_side: str       # 'BID' or 'ASK'

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
class NewsSentiment:
    timestamp: datetime
    sentiment_score: float  # -1 to 1
    confidence: float
    source: str
    impact_score: float  # 0 to 1
    keywords: List[str]
    volume: int  # Number of similar news items

@dataclass
class VolatilityMetrics:
    garch_vol: float
    historical_vol: float
    implied_vol: float
    realized_vol: float
    volatility_ratio: float  # Ratio of current to historical volatility

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

class TimeSeriesAnalyzer:
    def __init__(self, order_book_depth: int = 10):
        self.order_book_depth = order_book_depth
        self.order_book_history: List[OrderBookSnapshot] = []
        self.trade_history: List[Trade] = []
        self.news_events: List[Tuple[datetime, str, float]] = []  # timestamp, news_type, impact_score
        self.news_history: List[NewsEvent] = []
        self.volatility_history: Dict[str, List[VolatilityMetrics]] = {}
        self.earnings_history: List[Tuple[float, float, float]] = []  # (eps, pe_ratio, prev_eps)
        self.negative_news_threshold = 0.7  # 70% negative news triggers hedging
        self.volatility_spike_threshold = 2.0  # 2x historical volatility
        self.hedging_limits = {
            'max_hedge': 0.5,  # 50% of position
            'min_hedge': 0.1,  # 10% of position
            'target_hedge': 0.3  # 30% of position
        }
        self.spread_thresholds = {
            'high_volatility': 0.15,  # 15% spread triggers inventory collection
            'normal': 0.05,           # 5% normal spread
            'low_volatility': 0.02    # 2% low volatility spread
        }
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.05,
            low_liquidity_threshold=0.3,
            impact_threshold=0.02,
            news_threshold=0.1
        )
        self.position_limits = {
            'max_position': 0.2,
            'min_position': 0.05,
            'target_position': 0.1
        }
        
    def add_order_book_snapshot(self, snapshot: OrderBookSnapshot):
        """Add a new order book snapshot to the history."""
        self.order_book_history.append(snapshot)
        
    def add_trade(self, trade: Trade):
        """Add a new trade to the history."""
        self.trade_history.append(trade)
        
    def add_news_event(self, timestamp: datetime, news_type: str, impact_score: float):
        """Add a news event with its expected impact."""
        self.news_events.append((timestamp, news_type, impact_score))
        
    def calculate_order_flow_imbalance(self, window: int = 100) -> float:
        """Calculate the ratio of aggressive buy vs sell orders."""
        if len(self.trade_history) < window:
            window = len(self.trade_history)
            
        recent_trades = self.trade_history[-window:]
        buy_volume = sum(t.volume for t in recent_trades if t.aggressiveness == 'BUY')
        sell_volume = sum(t.volume for t in recent_trades if t.aggressiveness == 'SELL')
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return 0.0
            
        return (buy_volume - sell_volume) / total_volume
    
    def detect_spoofing(self, window: int = 100) -> List[Tuple[datetime, float]]:
        """Detect potential spoofing based on order cancellations and modifications."""
        if len(self.order_book_history) < window:
            window = len(self.order_book_history)
            
        suspicious_events = []
        recent_snapshots = self.order_book_history[-window:]
        
        for i in range(1, len(recent_snapshots)):
            prev_snapshot = recent_snapshots[i-1]
            curr_snapshot = recent_snapshots[i]
            
            # Calculate cancellation rate
            cancelled_orders = sum(1 for bid in prev_snapshot.bids 
                                 if not any(b.price == bid.price for b in curr_snapshot.bids))
            total_orders = sum(bid.orders for bid in prev_snapshot.bids)
            
            if total_orders > 0:
                cancellation_rate = cancelled_orders / total_orders
                if cancellation_rate > 0.5:  # Threshold for suspicious activity
                    suspicious_events.append((curr_snapshot.timestamp, cancellation_rate))
        
        return suspicious_events
    
    def calculate_spread_dynamics(self, interval: str = '1min') -> pd.Series:
        """Calculate average bid-ask spread over different time intervals."""
        spreads = []
        timestamps = []
        
        for snapshot in self.order_book_history:
            if snapshot.bids and snapshot.asks:
                best_bid = snapshot.bids[0].price
                best_ask = snapshot.asks[0].price
                spread = best_ask - best_bid
                spreads.append(spread)
                timestamps.append(snapshot.timestamp)
        
        df = pd.DataFrame({'timestamp': timestamps, 'spread': spreads})
        df.set_index('timestamp', inplace=True)
        return df.resample(interval).mean()['spread']
    
    def calculate_slippage(self, expected_price: float, actual_price: float) -> float:
        """Calculate slippage as the difference between expected and actual execution price."""
        return actual_price - expected_price
    
    def calculate_market_impact(self, order_size: float, price_change: float) -> float:
        """Calculate market impact based on order size and price change."""
        impact = abs(price_change) / order_size
        if impact > self.edge_case_metrics.impact_threshold:
            return 1 + impact / self.edge_case_metrics.impact_threshold
        return 1.0
    
    def analyze_news_impact(self, asset_class: str) -> Dict[str, float]:
        """Analyze how news events affect pricing in a specific asset class."""
        impacts = []
        for timestamp, news_type, impact_score in self.news_events:
            # Find price change after news event
            price_before = self._get_price_before_news(timestamp)
            price_after = self._get_price_after_news(timestamp)
            if price_before and price_after:
                price_change = (price_after - price_before) / price_before
                impacts.append((news_type, price_change, impact_score))
        
        # Group by news type and calculate average impact
        impact_by_type = {}
        for news_type, price_change, impact_score in impacts:
            if news_type not in impact_by_type:
                impact_by_type[news_type] = []
            impact_by_type[news_type].append(price_change * impact_score)
        
        return {k: np.mean(v) for k, v in impact_by_type.items()}
    
    def calculate_fill_ratio(self, window: int = 100) -> float:
        """Calculate the percentage of orders that get executed."""
        if len(self.trade_history) < window:
            window = len(self.trade_history)
            
        recent_trades = self.trade_history[-window:]
        total_volume = sum(t.volume for t in recent_trades)
        
        # This is a simplified version - in practice, you'd need to track
        # both placed orders and executed orders
        return total_volume / (window * 100)  # Assuming 100 orders per window
    
    def calculate_execution_probability(self, price: float, side: str) -> float:
        """Calculate execution probability based on order placement."""
        if not self.order_book_history:
            return 0.5
            
        latest_snapshot = self.order_book_history[-1]
        
        if side == 'BUY':
            # Calculate probability based on distance from best ask
            best_ask = latest_snapshot.asks[0].price
            distance = price - best_ask
            prob = 1 / (1 + np.exp(distance))
        else:
            # Calculate probability based on distance from best bid
            best_bid = latest_snapshot.bids[0].price
            distance = best_bid - price
            prob = 1 / (1 + np.exp(distance))
            
        return prob
    
    def _get_price_before_news(self, timestamp: datetime) -> Optional[float]:
        """Helper method to get price before a news event."""
        for snapshot in reversed(self.order_book_history):
            if snapshot.timestamp < timestamp:
                if snapshot.bids and snapshot.asks:
                    return (snapshot.bids[0].price + snapshot.asks[0].price) / 2
        return None
    
    def _get_price_after_news(self, timestamp: datetime) -> Optional[float]:
        """Helper method to get price after a news event."""
        for snapshot in self.order_book_history:
            if snapshot.timestamp > timestamp:
                if snapshot.bids and snapshot.asks:
                    return (snapshot.bids[0].price + snapshot.asks[0].price) / 2
        return None

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

    def handle_extreme_price_movement(self, current_price: float, prev_price: float) -> float:
        price_change = (current_price - prev_price) / prev_price
        if abs(price_change) > self.edge_case_metrics.extreme_move_threshold:
            adjustment_factor = 1 + abs(price_change) / self.edge_case_metrics.extreme_move_threshold
            return adjustment_factor
        return 1.0

    def handle_low_liquidity(self, current_volume: float, avg_volume: float) -> float:
        volume_ratio = current_volume / avg_volume
        if volume_ratio < self.edge_case_metrics.low_liquidity_threshold:
            return volume_ratio  # APT
            # return volume_ratio * 0.5  # DLR
            # return volume_ratio * 0.3  # MKJ
            # return volume_ratio * 0.7  # ETF
        return 1.0

    def handle_news_release(self, sentiment_change: float) -> float:
        if abs(sentiment_change) > self.edge_case_metrics.news_threshold:
            return 1 - abs(sentiment_change) / self.edge_case_metrics.news_threshold
            # * 1.5 for DLR
            # * 2.0 for MKJ
            # * 1.2 for ETF
        return 1.0

    def _calculate_sentiment_metrics(self) -> Dict[str, float]:
        if not self.news_history:
            return {
                'average_sentiment': 0.0,
                'negative_ratio': 0.0,
                'impact_score': 0.0,
                'news_volume': 0
            }
        
        recent_news = self.news_history[-20:]  # Last 20 news items
        price_changes = [n.price_change / n.price_before for n in recent_news]
        
        return {
            'average_sentiment': np.mean(price_changes),
            'negative_ratio': sum(1 for p in price_changes if p < 0) / len(price_changes),
            'impact_score': np.mean([abs(p) for p in price_changes]),
            'news_volume': len(recent_news)
        }

    def _handle_negative_news_flood(self, metrics: Dict[str, float]):
        """Handle negative news flood with aggressive hedging."""
        # Calculate hedging ratio based on negative news intensity
        negative_intensity = metrics['negative_ratio'] / self.negative_news_threshold
        impact_factor = metrics['impact_score'] * metrics['news_volume'] / 1000
        
        # Calculate target hedge ratio
        target_hedge = self.hedging_limits['target_hedge'] * (1 + negative_intensity * impact_factor)
        target_hedge = min(target_hedge, self.hedging_limits['max_hedge'])
        target_hedge = max(target_hedge, self.hedging_limits['min_hedge'])

    def _calculate_sentiment_based_spread(self, metrics: Dict[str, float]) -> float:
        """Calculate spread based on sentiment metrics."""
        base_spread = 0.1  # 10 cents base spread
        
        # Adjust spread based on sentiment
        sentiment_factor = 1 + abs(metrics['average_sentiment'])
        
        # Adjust spread based on negative news ratio
        negative_factor = 1 + metrics['negative_ratio'] * 2
        
        # Adjust spread based on impact
        impact_factor = 1 + metrics['impact_score']
        
        # Adjust spread based on news volume
        volume_factor = 1 + min(metrics['news_volume'] / 1000, 1.0)

    def _handle_volatility_spike(self, stock: str, metrics: VolatilityMetrics):
        """Handle volatility spikes with position and spread adjustments."""
        # Calculate volatility spike intensity
        spike_intensity = metrics.volatility_ratio / self.volatility_spike_threshold
        
        # Adjust hedge ratio based on volatility spike
        if stock in self.hedging_limits:
            target_hedge = self.hedging_limits['target_hedge'] * (1 + spike_intensity)
            target_hedge = min(target_hedge, self.hedging_limits['max_hedge'])
            target_hedge = max(target_hedge, self.hedging_limits['min_hedge'])

    def _calculate_volatility_based_spread(self, metrics: VolatilityMetrics) -> float:
        """Calculate spread based on volatility metrics."""
        base_spread = 0.05  # 5 cents base spread
        
        # Adjust spread based on volatility ratio
        vol_ratio_factor = 1 + (metrics.volatility_ratio - 1) * 2
        
        # Adjust spread based on realized volatility
        realized_factor = 1 + metrics.realized_vol * 1.5
        
        # Adjust spread based on implied volatility
        implied_factor = 1 + metrics.implied_vol * 1.2
        
        return base_spread * vol_ratio_factor * realized_factor * implied_factor

    def calculate_volatility_drag(self, returns: np.ndarray) -> float:
        """Calculate volatility drag for inverse ETF."""
        # Calculate realized volatility
        realized_vol = np.std(returns)
        
        # Calculate volatility drag using more accurate formula
        vol_drag = -0.5 * realized_vol**2 * (1 + realized_vol**2/4)
        
        return vol_drag

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

    def calculate_fair_value(self, current_price: float) -> float:
        if not self.earnings_history or not self.volatility_history:
            return current_price
        
        latest_earnings = self.earnings_history[-1]
        latest_vol = self.volatility_history[-1]
        
        # Calculate base fair value
        eps, pe_ratio, _ = latest_earnings
        base_fair_value = eps * pe_ratio
        
        # Calculate volatility adjustment
        combined_vol = (latest_vol.garch_vol + latest_vol.historical_vol + 
                       latest_vol.implied_vol) / 3
        vol_adjustment = combined_vol * current_price
        
        return base_fair_value + vol_adjustment

    def update_volatility_metrics(self, stock: str, metrics: VolatilityMetrics):
        """Update volatility metrics for a stock."""
        if stock not in self.volatility_history:
            self.volatility_history[stock] = []
        self.volatility_history[stock].append(metrics)
        
    def update_earnings_data(self, data: Tuple[float, float, float]):
        """Update earnings data."""
        self.earnings_history.append(data)
        
    def update_news_event(self, news: NewsEvent):
        """Update news event history."""
        self.news_history.append(news)
        
    def calculate_combined_volatility(self, stock: str) -> float:
        """Calculate combined volatility from different metrics."""
        if stock not in self.volatility_history or not self.volatility_history[stock]:
            return 0.0
            
        latest_vol = self.volatility_history[stock][-1]
        return (latest_vol.garch_vol + latest_vol.historical_vol + 
                latest_vol.implied_vol) / 3

class APTBot:
    def __init__(self, risk_budget: float):
        self.risk_budget = risk_budget
        self.earnings_history: List[Tuple[float, float, float]] = []  # (eps, pe_ratio, prev_eps)
        self.earnings_surprise_threshold = 0.15  # 15% deviation from expectations
        self.position_limits = {
            'max_position': 0.2,  # 20% of risk budget
            'min_position': 0.05,  # 5% of risk budget
            'target_position': 0.1  # 10% of risk budget
        }
        self.edge_case_metrics = EdgeCaseMetrics(
            price_volatility=0.0,
            liquidity_score=1.0,
            market_impact=0.0,
            news_sensitivity=0.0,
            extreme_move_threshold=0.05,
            low_liquidity_threshold=0.3,
            impact_threshold=0.02,
            news_threshold=0.1
        )
        self.news_history: List[NewsEvent] = []
        self.volatility_history: List[VolatilityMetrics] = []
        
    def update_earnings_data(self, data: Tuple[float, float, float]):
        """Update earnings data."""
        self.earnings_history.append(data)
        
    def generate_signals(self, market_price: float) -> Dict[str, float]:
        """Generate trading signals based on earnings data."""
        signals = {}
        for eps, pe_ratio, _ in self.earnings_history:
            fair_value = eps / pe_ratio
            if fair_value > market_price:
                signals['BUY'] = fair_value
            elif fair_value < market_price:
                signals['SELL'] = fair_value
        
        # Apply edge case adjustments to position size
        position_adjustment = (
            self.handle_extreme_price_movement(market_price, fair_value) *
            self.handle_low_liquidity(1000, 2000) *
            self.handle_market_impact(1000, market_price - fair_value) *
            self.handle_news_release(0.05)
        )
        
        signals['position_adjustment'] = position_adjustment
        return signals
    
    def handle_extreme_price_movement(self, current_price: float, prev_price: float) -> float:
        """Handle extreme price movements with dynamic adjustment."""
        price_change = (current_price - prev_price) / prev_price
        if abs(price_change) > self.edge_case_metrics.extreme_move_threshold:
            adjustment_factor = 1 + abs(price_change) / self.edge_case_metrics.extreme_move_threshold
            return adjustment_factor
        return 1.0
    
    def handle_low_liquidity(self, current_volume: float, avg_volume: float) -> float:
        """Handle low liquidity periods with position size adjustment."""
        volume_ratio = current_volume / avg_volume
        if volume_ratio < self.edge_case_metrics.low_liquidity_threshold:
            return volume_ratio
        return 1.0
    
    def handle_market_impact(self, order_size: float, price_change: float) -> float:
        """Handle market impact with dynamic spread adjustment."""
        impact = abs(price_change) / order_size
        if impact > self.edge_case_metrics.impact_threshold:
            return 1 + impact / self.edge_case_metrics.impact_threshold
        return 1.0
    
    def handle_news_release(self, sentiment_change: float) -> float:
        """Handle news releases with sentiment-based adjustment."""
        if abs(sentiment_change) > self.edge_case_metrics.news_threshold:
            return 1 - abs(sentiment_change) / self.edge_case_metrics.news_threshold
        return 1.0
    
    def _calculate_sentiment_metrics(self) -> Dict[str, float]:
        """Calculate metrics from recent news sentiment."""
        if not self.news_history:
            return {
                'average_sentiment': 0.0,
                'negative_ratio': 0.0,
                'impact_score': 0.0,
                'news_volume': 0
            }
        
        recent_news = self.news_history[-20:]  # Last 20 news items
        price_changes = [n.price_change / n.price_before for n in recent_news]
        
        return {
            'average_sentiment': np.mean(price_changes),
            'negative_ratio': sum(1 for p in price_changes if p < 0) / len(price_changes),
            'impact_score': np.mean([abs(p) for p in price_changes]),
            'news_volume': len(recent_news)
        }
    
    def calculate_fair_value(self, current_price: float) -> float:
        """Calculate fair value based on earnings and volatility."""
        if not self.earnings_history or not self.volatility_history:
            return current_price
            
        latest_earnings = self.earnings_history[-1]
        latest_vol = self.volatility_history[-1]
        
        # Calculate base fair value
        eps, pe_ratio, _ = latest_earnings
        base_fair_value = eps * pe_ratio
        
        # Calculate volatility adjustment
        combined_vol = (latest_vol.garch_vol + latest_vol.historical_vol + 
                       latest_vol.implied_vol) / 3
        vol_adjustment = combined_vol * current_price
        
        # Get sentiment metrics
        sentiment_metrics = self._calculate_sentiment_metrics()
        
        # Calculate sentiment-based price adjustment
        sentiment_adjustment = sentiment_metrics['average_sentiment'] * current_price * 0.2
        
        # Calculate negative news impact
        negative_impact = -sentiment_metrics['negative_ratio'] * current_price * 0.3
        
        # Calculate news volume impact
        volume_impact = sentiment_metrics['news_volume'] / 1000 * current_price * 0.1
        
        # Calculate final fair value
        fair_value = base_fair_value + sentiment_adjustment + negative_impact + vol_adjustment + volume_impact
        
        return fair_value

class DLRBot:
    def __init__(self, risk_budget: float):
        self.risk_budget = risk_budget
        self.signature_history: List[Tuple[int, int, int]] = []  # (current_signatures, threshold, alpha)
        self.signature_threshold = 100000
        self.signature_spike_threshold = 0.5  # 50% deviation from expected change
        self.position_limits = {
            'max_position': 0.3,  # 30% of risk budget
            'min_position': 0.05,  # 5% of risk budget
            'target_position': 0.15  # 15% of risk budget
        }
        
    def update_signature_data(self, data: Tuple[int, int, int]):
        """Update signature data."""
        self.signature_history.append(data)
        
    def generate_signals(self) -> List[str]:
        """Generate trading signals based on signature data."""
        signals = []
        for current_signatures, threshold, _ in self.signature_history:
            if current_signatures >= threshold:
                signals.append('BUY')
            else:
                signals.append('SELL')
        return signals
    
    def calculate_success_probability(self) -> float:
        """Calculate probability of success using log-normal distribution."""
        if not self.signature_history:
            return 0.0
        
        latest = self.signature_history[-1]
        
        if latest.current_signatures >= self.signature_threshold:
            return 1.0
        
        # Calculate daily signature changes
        daily_changes = [d.daily_change for d in self.signature_history[-20:]]
        
        # Fit log-normal distribution to daily changes
        log_changes = np.log1p(daily_changes)
        mu = np.mean(log_changes)
        sigma = np.std(log_changes)
        
        # Calculate remaining signatures needed
        remaining = self.signature_threshold - latest.current_signatures
        
        # Calculate expected days to reach threshold
        expected_daily = np.exp(mu + sigma**2/2)
        expected_days = remaining / expected_daily
        
        # Calculate probability using log-normal CDF
        z_score = (np.log(expected_days) - mu) / sigma
        probability = 1 - 0.5 * (1 + np.erf(z_score / np.sqrt(2)))
        
        return probability

    def _calculate_signature_momentum(self) -> float:
        """Calculate signature momentum using recent changes."""
        if len(self.signature_history) < 2:
            return 0.0
        
        recent_changes = [d.daily_change for d in self.signature_history[-5:]]
        expected_change = np.mean([d.expected_daily_change for d in self.signature_history[-5:]])
        
        if expected_change == 0:
            return 0.0
        
        return (np.mean(recent_changes) - expected_change) / expected_change

    def _handle_signature_spike(self, momentum: float, data: SignatureData):
        """Handle unexpected signature changes with position adjustments."""
        # Calculate position adjustment based on momentum
        if momentum > 0:  # Positive spike
            target_position = self.position_limits['target_position'] * (1 + momentum)
            target_position = min(target_position, self.position_limits['max_position'])
        else:  # Negative spike
            target_position = self.position_limits['target_position'] * (1 + momentum)
            target_position = max(target_position, self.position_limits['min_position'])

    def _calculate_signature_based_spread(self, data: SignatureData) -> float:
        """Calculate spread based on signature metrics."""
        base_spread = 0.1  # 10 cents base spread
        
        # Adjust spread based on momentum
        momentum = self._calculate_signature_momentum()
        momentum_factor = 1 + abs(momentum)
        
        # Adjust spread based on progress to threshold
        progress = data.current_signatures / self.signature_threshold
        progress_factor = 1 + abs(progress - 0.5)  # Higher spread when closer to threshold
        
        # Adjust spread based on news impact
        news_factor = 1 + abs(data.news_impact)
        
        return base_spread * momentum_factor * progress_factor * news_factor

class MKJBot:
    def __init__(self, risk_budget: float):
        self.risk_budget = risk_budget
        self.volatility_metrics: List[float] = []
        
    def update_volatility_metrics(self, data: float):
        """Update volatility metrics."""
        self.volatility_metrics.append(data)
        
    def generate_signals(self, current_price: float) -> List[str]:
        """Generate trading signals based on volatility metrics."""
        signals = []
        for metric in self.volatility_metrics:
            if metric > current_price:
                signals.append('BUY')
            else:
                signals.append('SELL')
        return signals
    
    def calculate_garch_volatility(self, returns: np.ndarray) -> float:
        """Calculate volatility using GARCH(1,1) model."""
        omega = 0.0001
        alpha = 0.1
        beta = 0.8
        
        var = np.var(returns)
        var_history = [var]
        
        for t in range(1, len(returns)):
            var = omega + alpha * returns[t-1]**2 + beta * var_history[t-1]
            var_history.append(var)

class ETFBot:
    def __init__(self, risk_budget: float):
        self.risk_budget = risk_budget
        self.correlation_history: Dict[str, List[float]] = {}
        
    def update_correlation_data(self, stock: str, data: List[float]):
        """Update correlation data."""
        self.correlation_history[stock] = data
        
    def generate_signals(self, etf_price: float, stock_prices: List[float]) -> List[str]:
        """Generate trading signals based on correlation data."""
        signals = []
        for stock_price in stock_prices:
            if stock_price not in self.correlation_history:
                signals.append('SELL')
            else:
                correlation = np.mean(self.correlation_history[stock_price][-20:])
                hedge_ratio = correlation * (etf_price / stock_price)
                if hedge_ratio > 0.5:
                    signals.append('BUY')
                else:
                    signals.append('SELL')
        return signals
    
    def calculate_hedge_ratio(self, stock: str, etf_price: float, stock_price: float) -> float:
        """Calculate hedge ratio based on correlation and volatility."""
        # Get latest correlation and volatility metrics
        correlation = np.mean(self.correlation_history[stock][-20:])
        vol_metrics = self.volatility_history[stock][-1]
        
        # Calculate base hedge ratio
        base_ratio = correlation * (etf_price / stock_price)
        
        # Adjust for volatility
        vol_adjustment = 1 + (vol_metrics.volatility_ratio - 1) * 0.5
        
        # Calculate final hedge ratio
        hedge_ratio = base_ratio * vol_adjustment

apt_bot = APTBot(risk_budget=1000000)
dlr_bot = DLRBot(risk_budget=1000000)
mkj_bot = MKJBot(risk_budget=1000000)
etf_bot = ETFBot(risk_budget=1000000)

# APT
apt_bot.update_earnings_data(earnings_data)
apt_signals = apt_bot.generate_signals(market_price=50.0)

# DLR
dlr_bot.update_signature_data(signature_data)
dlr_signals = dlr_bot.generate_signals()

# MKJ
mkj_bot.update_volatility_metrics(vol_metrics)
mkj_signals = mkj_bot.generate_signals(current_price=25.0)

# ETF
etf_signals = etf_bot.generate_signals(etf_price=400.0, stock_prices=stock_prices)

# Example usage with proper variable definitions
def example_usage():
    # Create sample data
    earnings_data = EarningsData(
        eps=2.5,
        pe_ratio=20.0,
        prev_eps=2.0,
        confidence=0.95,
        sentiment_score=0.5,
        expected_eps=2.3,
        revenue=1000000.0,
        guidance=2.8,
        earnings_date=datetime.now(),
        market_cap=50000000.0
    )
    
    signature_data = SignatureData(
        current_signatures=50000,
        threshold=100000,
        daily_change=1000,
        sentiment_score=0.3,
        timestamp=datetime.now(),
        expected_daily_change=800,
        news_impact=0.2
    )
    
    vol_metrics = VolatilityMetrics(
        garch_vol=0.2,
        historical_vol=0.18,
        implied_vol=0.22,
        realized_vol=0.19,
        volatility_ratio=1.1
    )
    
    stock_prices = {
        'AAPL': 150.0,
        'MSFT': 280.0,
        'GOOGL': 2700.0
    }
    
    # Initialize bots
    apt_bot = APTBot(risk_budget=1000000)
    dlr_bot = DLRBot(risk_budget=1000000)
    mkj_bot = MKJBot(risk_budget=1000000)
    etf_bot = ETFBot(risk_budget=1000000)
    
    # Generate signals
    apt_bot.update_earnings_data(earnings_data)
    apt_signals = apt_bot.generate_signals(market_price=50.0)
    
    dlr_bot.update_signature_data(signature_data)
    dlr_signals = dlr_bot.generate_signals()
    
    mkj_bot.update_volatility_metrics(vol_metrics)
    mkj_signals = mkj_bot.generate_signals(current_price=25.0)
    
    etf_signals = etf_bot.generate_signals(etf_price=400.0, stock_prices=stock_prices)
    
    return apt_signals, dlr_signals, mkj_signals, etf_signals 