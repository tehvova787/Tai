"""
TON Blockchain Integration for Lucky Train AI Assistant

This module provides integration with the TON blockchain for the Lucky Train AI assistant,
enabling real-time data access, wallet authentication, and transaction capabilities.
"""

import json
import logging
import os
import time
from typing import Dict, List, Optional, Union, Tuple
import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class TONBlockchainIntegration:
    """Integration with the TON blockchain for the Lucky Train AI assistant."""
    
    def __init__(self, config_path: str = "./config/config.json"):
        """Initialize the TON blockchain integration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.config = self._load_config(config_path)
        
        # Initialize TON API endpoints
        self.ton_api_key = os.getenv("TON_API_KEY")
        self.ton_api_url = self.config.get("ton_api_url", "https://toncenter.com/api/v2/")
        self.ton_explorer_url = self.config.get("ton_explorer_url", "https://tonkeeper.com/")
        
        # Cache for blockchain data
        self.data_cache = {}
        self.cache_expiry = {}
        self.cache_duration = self.config.get("cache_duration", 300)  # 5 minutes by default
        
        logger.info("TON Blockchain Integration initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load the configuration from a JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            The configuration as a dictionary.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load configuration from {config_path}: {e}")
            # Return default configuration
            return {}
    
    def _make_api_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make a request to the TON API.
        
        Args:
            endpoint: The API endpoint to call.
            params: The parameters to pass to the API.
            
        Returns:
            The API response as a dictionary.
        """
        if params is None:
            params = {}
        
        # Check if the result is cached and not expired
        cache_key = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        current_time = time.time()
        
        if cache_key in self.data_cache and current_time < self.cache_expiry.get(cache_key, 0):
            logger.info(f"Using cached data for {endpoint}")
            return self.data_cache[cache_key]
        
        # Make the API request
        headers = {
            "X-API-Key": self.ton_api_key
        }
        
        try:
            url = f"{self.ton_api_url.rstrip('/')}/{endpoint.lstrip('/')}"
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            
            # Cache the result
            self.data_cache[cache_key] = result
            self.cache_expiry[cache_key] = current_time + self.cache_duration
            
            return result
        except Exception as e:
            logger.error(f"Error making API request to {endpoint}: {e}")
            return {"error": str(e)}
    
    def get_blockchain_info(self) -> Dict:
        """Get general information about the TON blockchain.
        
        Returns:
            Information about the TON blockchain.
        """
        return self._make_api_request("getBlockchainInfo")
    
    def get_token_info(self, token_address: str = None) -> Dict:
        """Get information about the LTT token or another token.
        
        Args:
            token_address: The token's contract address.
            
        Returns:
            Information about the token.
        """
        # Use the LTT token address from config if not specified
        if token_address is None:
            token_address = self.config.get("ltt_token_address", "")
        
        return self._make_api_request("getTokenInfo", {"address": token_address})
    
    def get_account_info(self, address: str) -> Dict:
        """Get information about a TON account.
        
        Args:
            address: The account address.
            
        Returns:
            Information about the account.
        """
        return self._make_api_request("getAddressInformation", {"address": address})
    
    def get_account_transactions(self, address: str, limit: int = 10) -> List[Dict]:
        """Get recent transactions for a TON account.
        
        Args:
            address: The account address.
            limit: The maximum number of transactions to return.
            
        Returns:
            Recent transactions for the account.
        """
        return self._make_api_request("getTransactions", {"address": address, "limit": limit})
    
    def get_nft_items(self, owner_address: str = None, collection_address: str = None, limit: int = 10) -> List[Dict]:
        """Get NFT items owned by an address or in a collection.
        
        Args:
            owner_address: The owner's address.
            collection_address: The collection's address.
            limit: The maximum number of items to return.
            
        Returns:
            NFT items owned by the address or in the collection.
        """
        params = {"limit": limit}
        
        if owner_address:
            params["owner"] = owner_address
            
        if collection_address:
            params["collection"] = collection_address
        
        return self._make_api_request("getNFTItems", params)
    
    def verify_wallet_signature(self, address: str, message: str, signature: str) -> bool:
        """Verify a wallet signature for authentication.
        
        Args:
            address: The wallet address.
            message: The message that was signed.
            signature: The signature to verify.
            
        Returns:
            True if the signature is valid, False otherwise.
        """
        try:
            result = self._make_api_request("verifySignature", {
                "address": address,
                "message": message,
                "signature": signature
            })
            
            return result.get("valid", False)
        except Exception as e:
            logger.error(f"Error verifying wallet signature: {e}")
            return False
    
    def prepare_transaction(self, 
                           from_address: str, 
                           to_address: str, 
                           amount: float, 
                           comment: str = "", 
                           token_address: str = None) -> Dict:
        """Prepare a transaction for execution.
        
        Args:
            from_address: The sender's address.
            to_address: The recipient's address.
            amount: The amount to send in TON or tokens.
            comment: A comment for the transaction.
            token_address: The token's contract address for token transfers.
            
        Returns:
            The prepared transaction data.
        """
        # Convert amount to nanotons for TON transfers
        amount_nanotons = int(amount * 1e9)
        
        if token_address:
            # This is a token transfer
            # In a real implementation, this would prepare a token transfer contract call
            return {
                "type": "token_transfer",
                "from": from_address,
                "to": to_address,
                "token": token_address,
                "amount": amount,
                "comment": comment,
                "prepared_tx": f"token_transfer:{from_address}:{to_address}:{token_address}:{amount}:{comment}"
            }
        else:
            # This is a native TON transfer
            return {
                "type": "ton_transfer",
                "from": from_address,
                "to": to_address,
                "amount": amount,
                "amount_nanotons": amount_nanotons,
                "comment": comment,
                "prepared_tx": f"ton_transfer:{from_address}:{to_address}:{amount_nanotons}:{comment}"
            }
    
    def get_wallet_auth_message(self, user_id: str, timestamp: int = None) -> Tuple[str, int]:
        """Generate a message for wallet authentication.
        
        Args:
            user_id: The user's ID.
            timestamp: The current timestamp.
            
        Returns:
            A tuple containing the authentication message and the timestamp.
        """
        if timestamp is None:
            timestamp = int(time.time())
        
        message = f"Lucky Train Authentication\nUser: {user_id}\nTimestamp: {timestamp}"
        return message, timestamp
    
    def generate_deep_link(self, action: str, params: Dict) -> str:
        """Generate a deep link for TON wallet interaction.
        
        Args:
            action: The action type (transfer, nft_purchase, etc.).
            params: The parameters for the action.
            
        Returns:
            A deep link URL for TON wallet interaction.
        """
        base_url = "ton://transfer/"
        
        if action == "transfer":
            address = params.get("address", "")
            amount = params.get("amount", 0)
            comment = params.get("comment", "")
            
            # Calculate amount in nanotons
            amount_nanotons = int(amount * 1e9)
            
            # URL encode the comment
            import urllib.parse
            encoded_comment = urllib.parse.quote_plus(comment)
            
            return f"{base_url}{address}?amount={amount_nanotons}&text={encoded_comment}"
        
        elif action == "nft_purchase":
            # In a real implementation, this would generate a link to purchase an NFT
            nft_address = params.get("nft_address", "")
            price = params.get("price", 0)
            
            # Calculate price in nanotons
            price_nanotons = int(price * 1e9)
            
            return f"ton://transfer/{nft_address}?amount={price_nanotons}&text=NFT+Purchase"
        
        else:
            logger.warning(f"Unknown action type for deep link: {action}")
            return ""
    
    def get_market_price(self, token: str = "TON") -> Dict:
        """Get the current market price for TON or LTT.
        
        Args:
            token: The token symbol (TON or LTT).
            
        Returns:
            The token's market price information.
        """
        token = token.upper()
        
        if token == "TON":
            # Use a public cryptocurrency API for TON price
            try:
                response = requests.get("https://api.coingecko.com/api/v3/simple/price?ids=the-open-network&vs_currencies=usd,eur,rub")
                response.raise_for_status()
                
                data = response.json()
                ton_data = data.get("the-open-network", {})
                
                return {
                    "symbol": "TON",
                    "name": "The Open Network",
                    "price_usd": ton_data.get("usd", 0),
                    "price_eur": ton_data.get("eur", 0),
                    "price_rub": ton_data.get("rub", 0),
                    "source": "CoinGecko",
                    "timestamp": int(time.time())
                }
                
            except Exception as e:
                logger.error(f"Error getting TON price: {e}")
                return {
                    "symbol": "TON",
                    "error": str(e)
                }
        
        elif token == "LTT":
            # For LTT, use a placeholder as it may not be listed yet
            # In a real implementation, you would integrate with the actual token price
            return {
                "symbol": "LTT",
                "name": "Lucky Train Token",
                "price_usd": self.config.get("ltt_price_usd", 0.01),  # Placeholder price
                "price_ton": self.config.get("ltt_price_ton", 0.001),  # Placeholder price in TON
                "source": "Lucky Train",
                "timestamp": int(time.time())
            }
        
        else:
            logger.warning(f"Unknown token: {token}")
            return {
                "error": f"Unknown token: {token}"
            }


# Example usage
if __name__ == "__main__":
    # Initialize the blockchain integration
    blockchain = TONBlockchainIntegration()
    
    # Get blockchain information
    info = blockchain.get_blockchain_info()
    print("Blockchain Info:", json.dumps(info, indent=2))
    
    # Get TON price
    ton_price = blockchain.get_market_price("TON")
    print("TON Price:", json.dumps(ton_price, indent=2))
    
    # Generate a wallet authentication message
    auth_message, timestamp = blockchain.get_wallet_auth_message("test_user")
    print("Auth Message:", auth_message)
    
    # Generate a deep link for a TON transfer
    deep_link = blockchain.generate_deep_link("transfer", {
        "address": "EQD__________________________________________0",
        "amount": 0.1,
        "comment": "Test payment"
    })
    print("Deep Link:", deep_link) 