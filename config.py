"""
Configuration and constants for FinVoice Autonomous Recovery Agent.
Implements regulatory rules, economic parameters, and integration settings.
"""

import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "FinVoice Autonomous Collections & Prioritization Agent"
    DEBUG: bool = True
    
    # Database Settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/collections.db")
    
    # Regulatory & Compliance Constants (RBI Guidelines)
    CALLING_WINDOW_START_HOUR: int = 8   # 08:00 AM
    CALLING_WINDOW_END_HOUR: int = 19    # 07:00 PM
    MANDATORY_CALL_RECORDING_DISCLOSURE: str = (
        "Namaste, this call is from FinVoice Collections on behalf of your lender. "
        "This call is recorded for quality assurance and regulatory compliance."
    )
    
    # Economic & Prioritization Parameters
    CALL_COST_INR: float = 5.0           # ₹5 per voice call attempt
    WHATSAPP_COST_INR: float = 0.5       # ₹0.50 per WhatsApp message
    EXPLORATION_RATIO: float = 0.10      # 10% epsilon exploration budget to avoid training bias
    DEFAULT_DAILY_CALL_CAPACITY: int = 60 # Standard single caller capacity (or 360 for 6 callers)
    
    # Cashfree Gateway Settings
    CASHFREE_APP_ID: str = os.getenv("CASHFREE_APP_ID", "TEST_CASHFREE_APP_ID")
    CASHFREE_SECRET_KEY: str = os.getenv("CASHFREE_SECRET_KEY", "TEST_CASHFREE_SECRET_KEY")
    CASHFREE_ENV: str = os.getenv("CASHFREE_ENV", "TEST") # TEST or PROD
    CASHFREE_API_VERSION: str = "2023-08-01"
    
    # Model Artifacts Path
    MODEL_PATH: str = "./data/repayment_model.joblib"
    SYNTHETIC_DATA_PATH: str = "./data/synthetic_borrowers_4200.csv"
    
    class Config:
        env_file = ".env"
        extra = "allow"

settings = Settings()
