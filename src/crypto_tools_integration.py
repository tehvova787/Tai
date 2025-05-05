"""
Crypto Tools Integration Module

This module provides integrations for crypto trading bots and data providers:
- Trality
- 3Commas
- Cryptohopper
- Glassnode
- Santiment
- CryptoPredict
- Augmento
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
from abc import ABC, abstractmethod
import json
import requests
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BaseCryptoTool(ABC):
    """Base class for all crypto tools and data providers."""
    
    def __init__(self, config: Dict = None):
        """Initialize the crypto tool.
        
        Args:
            config: Configuration for the tool
        """
        self.config = config or {}
        self.name = self.__class__.__name__
        logger.info(f"Initializing {self.name}")
    
    @abstractmethod
    def get_status(self) -> Dict:
        """Get the connection status of the tool.
        
        Returns:
            Status information
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get a list of tool capabilities.
        
        Returns:
            List of capability strings
        """
        pass

class TralityAPI(BaseCryptoTool):
    """Integration with Trality crypto trading bot platform."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("TRALITY_API_KEY") or self.config.get("api_key")
        self.base_url = "https://api.trality.com/v1"
        self.capabilities = [
            "bot_creation", 
            "bot_backtesting", 
            "strategy_deployment", 
            "performance_monitoring"
        ]
        
        if not self.api_key:
            logger.warning("Trality API key not set - functionality will be limited")
            self.authenticated = False
        else:
            self.authenticated = True
    
    def get_status(self) -> Dict:
        """Get the connection status of the Trality API.
        
        Returns:
            Status information
        """
        if not self.authenticated:
            return {"status": "not_authenticated", "message": "API key not provided"}
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/user/profile", headers=headers)
            
            if response.status_code == 200:
                return {"status": "connected", "message": "Successfully connected to Trality API"}
            else:
                return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to Trality API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Get a list of Trality capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities
    
    def get_bots(self) -> Dict:
        """Get a list of user's bots.
        
        Returns:
            Bot information
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "bots": []}
        
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(f"{self.base_url}/bots", headers=headers)
            
            if response.status_code == 200:
                return {"bots": response.json(), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "bots": []}
        except Exception as e:
            logger.error(f"Error getting Trality bots: {e}")
            return {"error": str(e), "bots": []}
    
    def create_bot(self, name: str, strategy_code: str, exchange: str = "binance") -> Dict:
        """Create a new trading bot.
        
        Args:
            name: Bot name
            strategy_code: Python code for the trading strategy
            exchange: Exchange to trade on
            
        Returns:
            Creation result
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "bot_id": None}
        
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "name": name,
                "exchange": exchange,
                "strategy": strategy_code
            }
            
            response = requests.post(
                f"{self.base_url}/bots", 
                headers=headers,
                json=payload
            )
            
            if response.status_code in (200, 201):
                return {"bot_id": response.json().get("id"), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "bot_id": None}
        except Exception as e:
            logger.error(f"Error creating Trality bot: {e}")
            return {"error": str(e), "bot_id": None}

class ThreeCommasAPI(BaseCryptoTool):
    """Integration with 3Commas trading bot platform."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("THREE_COMMAS_API_KEY") or self.config.get("api_key")
        self.api_secret = os.getenv("THREE_COMMAS_API_SECRET") or self.config.get("api_secret")
        self.base_url = "https://api.3commas.io/public/api"
        self.capabilities = [
            "bot_management", 
            "smart_trading", 
            "portfolio_management", 
            "exchange_integration"
        ]
        
        if not self.api_key or not self.api_secret:
            logger.warning("3Commas API credentials not set - functionality will be limited")
            self.authenticated = False
        else:
            self.authenticated = True
    
    def get_status(self) -> Dict:
        """Get the connection status of the 3Commas API.
        
        Returns:
            Status information
        """
        if not self.authenticated:
            return {"status": "not_authenticated", "message": "API credentials not provided"}
        
        try:
            response = requests.get(
                f"{self.base_url}/ver1/accounts",
                headers={
                    "APIKEY": self.api_key,
                    "SECRET": self.api_secret
                }
            )
            
            if response.status_code == 200:
                return {"status": "connected", "message": "Successfully connected to 3Commas API"}
            else:
                return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to 3Commas API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Get a list of 3Commas capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities
    
    def get_bots(self) -> Dict:
        """Get a list of user's bots.
        
        Returns:
            Bot information
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "bots": []}
        
        try:
            response = requests.get(
                f"{self.base_url}/ver1/bots",
                headers={
                    "APIKEY": self.api_key,
                    "SECRET": self.api_secret
                }
            )
            
            if response.status_code == 200:
                return {"bots": response.json(), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "bots": []}
        except Exception as e:
            logger.error(f"Error getting 3Commas bots: {e}")
            return {"error": str(e), "bots": []}
    
    def create_bot(self, name: str, pair: str, strategy: str, **params) -> Dict:
        """Create a new trading bot.
        
        Args:
            name: Bot name
            pair: Trading pair (e.g., "BTC_USD")
            strategy: Strategy type
            params: Additional parameters
            
        Returns:
            Creation result
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "bot_id": None}
        
        try:
            payload = {
                "name": name,
                "pairs": pair,
                "strategy_list": [strategy],
                **params
            }
            
            response = requests.post(
                f"{self.base_url}/ver1/bots/create_bot",
                headers={
                    "APIKEY": self.api_key,
                    "SECRET": self.api_secret
                },
                json=payload
            )
            
            if response.status_code in (200, 201):
                return {"bot_id": response.json().get("id"), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "bot_id": None}
        except Exception as e:
            logger.error(f"Error creating 3Commas bot: {e}")
            return {"error": str(e), "bot_id": None}

class CryptohopperAPI(BaseCryptoTool):
    """Integration with Cryptohopper trading bot platform."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("CRYPTOHOPPER_API_KEY") or self.config.get("api_key")
        self.base_url = "https://api.cryptohopper.com/v1"
        self.capabilities = [
            "automated_trading", 
            "signal_following", 
            "exchange_management", 
            "strategy_backtesting"
        ]
        
        if not self.api_key:
            logger.warning("Cryptohopper API key not set - functionality will be limited")
            self.authenticated = False
        else:
            self.authenticated = True
    
    def get_status(self) -> Dict:
        """Get the connection status of the Cryptohopper API.
        
        Returns:
            Status information
        """
        if not self.authenticated:
            return {"status": "not_authenticated", "message": "API key not provided"}
        
        try:
            headers = {"API-Key": self.api_key}
            response = requests.get(f"{self.base_url}/hopper", headers=headers)
            
            if response.status_code == 200:
                return {"status": "connected", "message": "Successfully connected to Cryptohopper API"}
            else:
                return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to Cryptohopper API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Get a list of Cryptohopper capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities
    
    def get_hoppers(self) -> Dict:
        """Get a list of user's hoppers.
        
        Returns:
            Hopper information
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "hoppers": []}
        
        try:
            headers = {"API-Key": self.api_key}
            response = requests.get(f"{self.base_url}/hopper", headers=headers)
            
            if response.status_code == 200:
                return {"hoppers": response.json().get("data", []), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "hoppers": []}
        except Exception as e:
            logger.error(f"Error getting Cryptohopper hoppers: {e}")
            return {"error": str(e), "hoppers": []}
    
    def place_order(self, hopper_id: str, coin: str, amount: float, order_type: str = "buy") -> Dict:
        """Place a trading order.
        
        Args:
            hopper_id: Hopper ID
            coin: Coin to trade
            amount: Amount to trade
            order_type: Type of order (buy/sell)
            
        Returns:
            Order result
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "order_id": None}
        
        try:
            headers = {"API-Key": self.api_key}
            
            payload = {
                "type": order_type,
                "amount": amount,
                "market": coin
            }
            
            response = requests.post(
                f"{self.base_url}/hopper/{hopper_id}/order",
                headers=headers,
                json=payload
            )
            
            if response.status_code in (200, 201):
                return {"order_id": response.json().get("orderId"), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "order_id": None}
        except Exception as e:
            logger.error(f"Error placing Cryptohopper order: {e}")
            return {"error": str(e), "order_id": None}

class GlassnodeAPI(BaseCryptoTool):
    """Integration with Glassnode on-chain analytics platform."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("GLASSNODE_API_KEY") or self.config.get("api_key")
        self.base_url = "https://api.glassnode.com/v1"
        self.capabilities = [
            "on_chain_metrics", 
            "market_data", 
            "entity_adjusted_metrics", 
            "network_indicators"
        ]
        
        if not self.api_key:
            logger.warning("Glassnode API key not set - functionality will be limited")
            self.authenticated = False
        else:
            self.authenticated = True
    
    def get_status(self) -> Dict:
        """Get the connection status of the Glassnode API.
        
        Returns:
            Status information
        """
        if not self.authenticated:
            return {"status": "not_authenticated", "message": "API key not provided"}
        
        try:
            params = {"api_key": self.api_key, "a": "BTC"}
            response = requests.get(f"{self.base_url}/metrics/market/price_usd_close", params=params)
            
            if response.status_code == 200:
                return {"status": "connected", "message": "Successfully connected to Glassnode API"}
            else:
                return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to Glassnode API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Get a list of Glassnode capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities
    
    def get_metric(self, metric: str, asset: str = "BTC", since: str = None, until: str = None) -> Dict:
        """Get on-chain metric data.
        
        Args:
            metric: Metric name
            asset: Asset symbol
            since: Start timestamp
            until: End timestamp
            
        Returns:
            Metric data
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "data": []}
        
        try:
            params = {"api_key": self.api_key, "a": asset}
            
            if since:
                params["s"] = since
            if until:
                params["u"] = until
                
            response = requests.get(f"{self.base_url}/metrics/{metric}", params=params)
            
            if response.status_code == 200:
                return {"data": response.json(), "status": "success"}
            else:
                return {"error": f"API returned status code {response.status_code}", "data": []}
        except Exception as e:
            logger.error(f"Error getting Glassnode metric: {e}")
            return {"error": str(e), "data": []}

class SantimentAPI(BaseCryptoTool):
    """Integration with Santiment crypto analytics platform."""
    
    def __init__(self, config: Dict = None):
        super().__init__(config)
        self.api_key = os.getenv("SANTIMENT_API_KEY") or self.config.get("api_key")
        self.base_url = "https://api.santiment.net/graphql"
        self.capabilities = [
            "social_sentiment_analysis", 
            "on_chain_metrics", 
            "development_activity", 
            "market_indicators"
        ]
        
        if not self.api_key:
            logger.warning("Santiment API key not set - functionality will be limited")
            self.authenticated = False
        else:
            self.authenticated = True
    
    def get_status(self) -> Dict:
        """Get the connection status of the Santiment API.
        
        Returns:
            Status information
        """
        if not self.authenticated:
            return {"status": "not_authenticated", "message": "API key not provided"}
        
        try:
            headers = {"Authorization": f"Apikey {self.api_key}"}
            query = """
            {
              currentUser {
                id
                email
              }
            }
            """
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json={"query": query}
            )
            
            if response.status_code == 200 and "data" in response.json():
                return {"status": "connected", "message": "Successfully connected to Santiment API"}
            else:
                return {"status": "error", "message": f"API returned status code {response.status_code}"}
        except Exception as e:
            logger.error(f"Error connecting to Santiment API: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_capabilities(self) -> List[str]:
        """Get a list of Santiment capabilities.
        
        Returns:
            List of capability strings
        """
        return self.capabilities
    
    def get_social_volume(self, asset: str, from_date: str, to_date: str) -> Dict:
        """Get social volume data for an asset.
        
        Args:
            asset: Asset slug
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            Social volume data
        """
        if not self.authenticated:
            return {"error": "Not authenticated", "data": []}
        
        try:
            headers = {"Authorization": f"Apikey {self.api_key}"}
            query = """
            {
              socialVolume(
                slug: "%s"
                from: "%s"
                to: "%s"
                interval: "1d"
                socialDominanceSource: ALL
              ) {
                datetime
                value
              }
            }
            """ % (asset, from_date, to_date)
            
            response = requests.post(
                self.base_url,
                headers=headers,
                json={"query": query}
            )
            
            if response.status_code == 200 and "data" in response.json():
                return {"data": response.json()["data"]["socialVolume"], "status": "success"}
            else:
                return {"error": f"API returned error", "data": []}
        except Exception as e:
            logger.error(f"Error getting Santiment social volume: {e}")
            return {"error": str(e), "data": []}

class CryptoToolFactory:
    """Factory for creating crypto tool instances."""
    
    @staticmethod
    def create_tool(tool_type: str, config: Dict = None) -> BaseCryptoTool:
        """Create and return a crypto tool based on the tool type.
        
        Args:
            tool_type: Type of crypto tool to create
            config: Configuration for the tool
            
        Returns:
            Crypto tool instance
        """
        tool_classes = {
            "trality": TralityAPI,
            "3commas": ThreeCommasAPI,
            "cryptohopper": CryptohopperAPI,
            "glassnode": GlassnodeAPI,
            "santiment": SantimentAPI,
        }
        
        tool_class = tool_classes.get(tool_type.lower())
        
        if tool_class:
            return tool_class(config)
        else:
            logger.error(f"Unknown crypto tool type: {tool_type}")
            raise ValueError(f"Unknown crypto tool type: {tool_type}") 