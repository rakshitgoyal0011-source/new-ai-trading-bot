"""Application settings, loaded from environment variables / .env file."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed application configuration. Reads .env once at process start."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore",
        populate_by_name=True,
    )

    # --- run mode ---
    mode: str = Field("demo", alias="DALAL_MODE")  # demo | live

    # --- Kite Connect ---
    kite_api_key: str = Field("", alias="KITE_API_KEY")
    kite_api_secret: str = Field("", alias="KITE_API_SECRET")
    kite_access_token: str = Field("", alias="KITE_ACCESS_TOKEN")
    kite_redirect_url: str = Field(
        "http://127.0.0.1:8000/kite/callback", alias="KITE_REDIRECT_URL"
    )
    kite_totp_secret: str = Field("", alias="KITE_TOTP_SECRET")

    # --- web terminal ---
    host: str = Field("127.0.0.1", alias="DALAL_HOST")
    port: int = Field(8000, alias="DALAL_PORT")

    # --- data providers ---
    history_provider: str = Field("auto", alias="HISTORY_PROVIDER")
    fundamentals_provider: str = Field("yfinance", alias="FUNDAMENTALS_PROVIDER")
    twelvedata_api_key: str = Field("", alias="TWELVEDATA_API_KEY")
    fmp_api_key: str = Field("", alias="FMP_API_KEY")
    news_provider: str = Field("rss", alias="NEWS_PROVIDER")
    newsapi_key: str = Field("", alias="NEWSAPI_KEY")
    finnhub_api_key: str = Field("", alias="FINNHUB_API_KEY")
    sentiment_model: str = Field("lexicon", alias="SENTIMENT_MODEL")

    # --- composite weights ---
    weight_technical: float = Field(0.45, alias="WEIGHT_TECHNICAL")
    weight_fundamental: float = Field(0.35, alias="WEIGHT_FUNDAMENTAL")
    weight_news: float = Field(0.20, alias="WEIGHT_NEWS")

    # --- risk ---
    risk_per_trade_pct: float = Field(1.5, alias="RISK_PER_TRADE_PCT")
    default_capital: float = Field(100_000.0, alias="DEFAULT_CAPITAL")

    # --- storage / logging ---
    db_path: str = Field("data_cache/dalal.sqlite", alias="DB_PATH")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    @property
    def is_live(self) -> bool:
        """True when the terminal should use real Kite data."""
        return self.mode.strip().lower() == "live"

    @property
    def has_kite_creds(self) -> bool:
        return bool(self.kite_api_key and self.kite_api_secret)


@lru_cache
def get_settings() -> Settings:
    """Cached singleton accessor for application settings."""
    return Settings()
