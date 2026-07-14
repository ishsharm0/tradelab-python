"""Live-trading foundations."""

from tradelab.brokers import (
    AlpacaBroker,
    BinanceBroker,
    CoinbaseBroker,
    InteractiveBrokersBroker,
    create_alpaca_broker,
    create_binance_broker,
    create_coinbase_broker,
    create_interactive_brokers_broker,
)

from .broker import BrokerAdapter, SessionBroker
from .candle import CandleAggregator, create_candle_aggregator
from .clock import BrokerClock, create_clock
from .dashboard import DashboardServer, create_dashboard_server
from .engine import LiveEngine, create_live_engine
from .events import LIVE_EVENTS, EventBus, create_event_bus
from .feed import (
    BrokerFeed,
    FeedProvider,
    PollingFeed,
    Subscription,
    create_broker_feed,
    create_polling_feed,
)
from .logger import LiveLogger, create_logger
from .notify import NotifierHandle, attach_notifier
from .orchestrator import LiveOrchestrator, create_live_orchestrator
from .paper import PaperEngine, create_paper_engine
from .risk import RiskManager, create_risk_manager
from .session import SessionManager, TradingSession, create_session_manager
from .state import StateManager, create_state_manager
from .storage import JsonFileStorage, StorageProvider, create_json_file_storage

__all__ = [
    "LIVE_EVENTS",
    "AlpacaBroker",
    "BinanceBroker",
    "BrokerAdapter",
    "BrokerClock",
    "BrokerFeed",
    "CandleAggregator",
    "CoinbaseBroker",
    "DashboardServer",
    "EventBus",
    "FeedProvider",
    "InteractiveBrokersBroker",
    "JsonFileStorage",
    "LiveEngine",
    "LiveLogger",
    "LiveOrchestrator",
    "NotifierHandle",
    "PaperEngine",
    "PollingFeed",
    "RiskManager",
    "SessionBroker",
    "SessionManager",
    "StateManager",
    "StorageProvider",
    "Subscription",
    "TradingSession",
    "attach_notifier",
    "create_alpaca_broker",
    "create_binance_broker",
    "create_broker_feed",
    "create_candle_aggregator",
    "create_clock",
    "create_coinbase_broker",
    "create_dashboard_server",
    "create_event_bus",
    "create_interactive_brokers_broker",
    "create_json_file_storage",
    "create_live_engine",
    "create_live_orchestrator",
    "create_logger",
    "create_paper_engine",
    "create_polling_feed",
    "create_risk_manager",
    "create_session_manager",
    "create_state_manager",
]
