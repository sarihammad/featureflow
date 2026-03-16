from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TOPIC_USER_EVENTS: str = "user-events"
    KAFKA_CONSUMER_GROUP: str = "feature-pipeline"

    REDIS_URL: str = "redis://localhost:6379"

    OFFLINE_STORE_PATH: str = "data/offline"

    FEATURE_TTL_SECONDS: int = 86400  # 24h default TTL

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    LOG_LEVEL: str = "INFO"

    N_USERS: int = 10000
    N_ITEMS: int = 5000
    EVENTS_PER_SECOND: float = 100.0
    BATCH_SIZE: int = 100


settings = Settings()
