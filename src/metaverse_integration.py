"""
Metaverse Integration for Lucky Train AI Assistant

This module provides integration capabilities for the Lucky Train AI assistant
to operate within metaverse environments.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Union, BinaryIO
import base64
import time

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'bot'))
from assistant import LuckyTrainAssistant
from blockchain_integration import TONBlockchainIntegration
from multimodal_interface import MultimodalInterface

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LuckyTrainMetaverseAssistant:
    """Metaverse integration for the Lucky Train AI assistant."""
    
    def __init__(self, config_path: str = "./config/config.json"):
        """Initialize the metaverse integration.
        
        Args:
            config_path: Path to the configuration file.
        """
        self.assistant = LuckyTrainAssistant(config_path)
        self.config = self._load_config(config_path)
        
        # Initialize blockchain integration
        self.blockchain = TONBlockchainIntegration(config_path)
        
        # Initialize multimodal interface
        self.multimodal = MultimodalInterface(config_path)
        
        # Queue for pending messages
        self.message_queue = []
        
        # Cache for user language preferences
        self.user_languages = {}
        
        # Cache for authenticated wallet addresses
        self.authenticated_wallets = {}
        
        logger.info("Lucky Train Metaverse Assistant initialized successfully")
    
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
    
    def process_message(self, message: str, user_id: str, location: str = "central_station") -> str:
        """Process a message from a user in the metaverse.
        
        Args:
            message: The user's message.
            user_id: The user's ID in the metaverse.
            location: The location in the metaverse where the user is.
            
        Returns:
            The assistant's response.
        """
        logger.info(f"Received message from user {user_id} at location {location}: {message}")
        
        # Get user's preferred language
        language = self.user_languages.get(user_id, None)
        
        # Add context about the user's location
        context_message = f"{message} (пользователь находится в локации: {location})"
        
        # Process the message with the AI assistant
        response = self.assistant.handle_message(context_message, user_id, "metaverse")
        
        logger.info(f"Generated response for user {user_id}: {response}")
        
        return response
    
    def process_voice_message(self, audio_data: BinaryIO, user_id: str, location: str = "central_station") -> Dict:
        """Process a voice message from a user in the metaverse.
        
        Args:
            audio_data: The audio data as a file-like object.
            user_id: The user's ID in the metaverse.
            location: The location in the metaverse where the user is.
            
        Returns:
            A dictionary containing the text and audio response.
        """
        # Get user's preferred language
        language = self.user_languages.get(user_id, None)
        
        # Convert speech to text
        stt_result = self.multimodal.speech_to_text(audio_data, language)
        
        if not stt_result["success"]:
            error_message = "Не удалось распознать речь. Пожалуйста, попробуйте еще раз."
            if language and language != "ru":
                error_message = self.assistant._translate_text(error_message, language)
            
            return {
                "success": False,
                "error": stt_result.get("error", "Unknown error"),
                "text_response": error_message
            }
        
        # Process the text message
        recognized_text = stt_result["text"]
        language = stt_result["language"]
        
        # Store the user's language preference
        self.user_languages[user_id] = language
        
        # Add context about the user's location
        context_message = f"{recognized_text} (пользователь находится в локации: {location})"
        
        # Process the message with the AI assistant
        text_response = self.assistant.handle_message(context_message, user_id, "metaverse")
        
        # Convert the response to speech
        emotion = "neutral"
        if "!" in text_response:
            emotion = "happy"
        elif "?" in text_response:
            emotion = "professional"
        
        tts_result = self.multimodal.text_to_speech(text_response, language, emotion=emotion)
        
        if not tts_result["success"]:
            return {
                "success": True,
                "recognized_text": recognized_text,
                "text_response": text_response,
                "audio_available": False
            }
        
        return {
            "success": True,
            "recognized_text": recognized_text,
            "text_response": text_response,
            "audio_base64": tts_result.get("audio_base64"),
            "audio_format": tts_result.get("format", "mp3"),
            "audio_available": True
        }
    
    def process_image(self, image_data: BinaryIO, user_id: str, location: str = "central_station") -> Dict:
        """Process an image from a user in the metaverse.
        
        Args:
            image_data: The image data as a file-like object.
            user_id: The user's ID in the metaverse.
            location: The location in the metaverse where the user is.
            
        Returns:
            A dictionary containing the analysis and response.
        """
        # Get user's preferred language
        language = self.user_languages.get(user_id, None)
        
        # Analyze the image
        analysis_result = self.multimodal.analyze_image(image_data)
        
        if not analysis_result["success"]:
            error_message = "Не удалось проанализировать изображение. Пожалуйста, попробуйте еще раз."
            if language and language != "ru":
                error_message = self.assistant._translate_text(error_message, language)
            
            return {
                "success": False,
                "error": analysis_result.get("error", "Unknown error"),
                "text_response": error_message
            }
        
        # Check if it's a QR code for location or NFT
        if analysis_result.get("is_qr_code", False):
            # In a real implementation, this would decode the QR code
            # and determine what action to take
            return {
                "success": True,
                "is_qr_code": True,
                "text_response": "Обнаружен QR-код. Это может быть вход в локацию или информация об NFT."
            }
        
        # Generate a response based on the image content
        prompt = f"Пользователь отправил изображение"
        
        if analysis_result.get("contains_train", False):
            prompt += ", на котором изображен поезд"
        
        if analysis_result.get("contains_nft", False):
            prompt += ", которое может быть связано с NFT"
        
        if analysis_result.get("contains_location", False):
            prompt += ", на котором изображена какая-то локация"
        
        prompt += f". Анализ изображения: {analysis_result.get('analysis', '')}. Пользователь находится в локации: {location}."
        
        # Process the prompt with the AI assistant
        text_response = self.assistant.handle_message(prompt, user_id, "metaverse")
        
        return {
            "success": True,
            "analysis": analysis_result.get("analysis", ""),
            "text_response": text_response
        }
    
    def get_location_info(self, location: str) -> str:
        """Get information about a specific location in the metaverse.
        
        Args:
            location: The location name.
            
        Returns:
            Information about the location.
        """
        location_queries = {
            "central_station": "Расскажи о Центральном вокзале в метавселенной Lucky Train",
            "trading_district": "Расскажи о Торговом квартале в метавселенной Lucky Train",
            "entertainment_district": "Расскажи о Развлекательном районе в метавселенной Lucky Train",
            "business_center": "Расскажи о Бизнес-центре в метавселенной Lucky Train",
            "creative_district": "Расскажи о Творческом районе в метавселенной Lucky Train"
        }
        
        if location not in location_queries:
            return "Информация о данной локации отсутствует."
        
        # Get information about the location
        query = location_queries[location]
        response = self.assistant.handle_message(query, "system", "metaverse")
        
        return response
    
    def get_train_info(self, train_id: str) -> str:
        """Get information about a specific train in the metaverse.
        
        Args:
            train_id: The train ID or name.
            
        Returns:
            Information about the train.
        """
        # Process the query with the AI assistant
        query = f"Расскажи о поезде {train_id} в метавселенной Lucky Train"
        response = self.assistant.handle_message(query, "system", "metaverse")
        
        return response
    
    def get_npc_response(self, npc_id: str, message: str) -> str:
        """Get a response from an NPC in the metaverse.
        
        Args:
            npc_id: The NPC ID or name.
            message: The message sent to the NPC.
            
        Returns:
            The NPC's response.
        """
        # Add context about the NPC
        context_message = f"Пользователь обращается к NPC {npc_id}: {message}"
        
        # Process the message with the AI assistant
        response = self.assistant.handle_message(context_message, f"npc_{npc_id}", "metaverse")
        
        return response
    
    def get_object_info(self, object_id: str) -> str:
        """Get information about a specific object in the metaverse.
        
        Args:
            object_id: The object ID or name.
            
        Returns:
            Information about the object.
        """
        # Process the query with the AI assistant
        query = f"Расскажи об объекте {object_id} в метавселенной Lucky Train"
        response = self.assistant.handle_message(query, "system", "metaverse")
        
        return response
    
    def handle_command(self, command: str, user_id: str, parameters: Dict = None) -> str:
        """Handle a command in the metaverse.
        
        Args:
            command: The command to execute.
            user_id: The user's ID.
            parameters: Additional parameters for the command.
            
        Returns:
            The result of the command.
        """
        if parameters is None:
            parameters = {}
        
        logger.info(f"Received command '{command}' from user {user_id} with parameters: {parameters}")
        
        # Handle different commands
        if command == "help":
            return self._get_help_message()
        
        elif command == "info":
            topic = parameters.get("topic", "project")
            return self._get_topic_info(topic)
        
        elif command == "navigate":
            location = parameters.get("location", "central_station")
            return f"Перемещение в локацию: {location}. {self.get_location_info(location)}"
        
        elif command == "buy_ticket":
            train = parameters.get("train", "express")
            destination = parameters.get("destination", "trading_district")
            return f"Билет на поезд '{train}' до '{destination}' успешно приобретен. Счастливого пути!"
        
        elif command == "check_balance":
            result = self.get_token_balance(user_id)
            
            if result["success"]:
                return f"Ваш текущий баланс:\n- {result['ton_balance']:.2f} TON (${result['ton_usd_value']:.2f})\n- {result['ltt_balance']} LTT (${result['ltt_usd_value']:.2f})"
            else:
                return f"Ошибка при проверке баланса: {result.get('error', 'Неизвестная ошибка')}"
        
        elif command == "view_nfts":
            result = self.get_nft_items(user_id)
            
            if result["success"]:
                if result["count"] > 0:
                    return f"У вас есть {result['count']} NFT. Используйте команду preview_nft [id] для просмотра каждого из них."
                else:
                    return "У вас пока нет NFT. Вы можете приобрести их на маркетплейсе Lucky Train."
            else:
                return f"Ошибка при получении NFT: {result.get('error', 'Неизвестная ошибка')}"
        
        elif command == "preview_location":
            location = parameters.get("location", "central_station")
            return f"Генерация 3D превью локации {location}. Пожалуйста, подождите..."
        
        elif command == "connect_wallet":
            result = self.get_wallet_auth_message(user_id)
            
            if result["success"]:
                return f"Для подключения кошелька, пожалуйста, подпишите следующее сообщение в своем TON кошельке:\n\n{result['message']}"
            else:
                return f"Ошибка при генерации сообщения для аутентификации: {result.get('error', 'Неизвестная ошибка')}"
        
        elif command == "buy_nft":
            nft_id = parameters.get("nft_id", "")
            
            if not nft_id:
                return "Пожалуйста, укажите ID NFT, который вы хотите купить."
            
            result = self.prepare_nft_purchase(user_id, nft_id)
            
            if result["success"]:
                return f"Для покупки NFT #{nft_id} за {result['price']} TON, перейдите по ссылке:\n{result['deep_link']}"
            else:
                return f"Ошибка при подготовке покупки NFT: {result.get('error', 'Неизвестная ошибка')}"
        
        else:
            return f"Неизвестная команда: {command}. Используйте команду 'help' для получения списка доступных команд."
    
    def _get_help_message(self) -> str:
        """Get the help message with available commands.
        
        Returns:
            The help message.
        """
        help_message = (
            "🤖 **Доступные команды в метавселенной Lucky Train:**\n\n"
            "- **help**: Показать список доступных команд\n"
            "- **info [topic]**: Получить информацию о проекте (параметр topic: project, token, metaverse, roadmap, team)\n"
            "- **navigate [location]**: Переместиться в указанную локацию\n"
            "- **buy_ticket [train] [destination]**: Купить билет на поезд\n"
            "- **check_balance**: Проверить баланс токенов LTT\n"
            "- **view_nfts**: Посмотреть ваши NFT\n"
            "- **preview_location [location]**: Показать 3D превью локации\n"
            "- **connect_wallet**: Подключить кошелек TON\n"
            "- **buy_nft [nft_id]**: Купить NFT\n\n"
            "Также вы можете задать любой вопрос о проекте Lucky Train, и я постараюсь на него ответить!"
        )
        
        return help_message
    
    def _get_topic_info(self, topic: str) -> str:
        """Get information about a specific topic.
        
        Args:
            topic: The topic to get information about.
            
        Returns:
            Information about the topic.
        """
        topic_queries = {
            "project": "Расскажи о проекте Lucky Train",
            "token": "Расскажи о токене LTT",
            "metaverse": "Расскажи о метавселенной Lucky Train",
            "roadmap": "Расскажи о дорожной карте проекта",
            "team": "Расскажи о команде проекта Lucky Train"
        }
        
        if topic not in topic_queries:
            return f"Информация о теме '{topic}' отсутствует."
        
        # Get information about the topic
        query = topic_queries[topic]
        response = self.assistant.handle_message(query, "system", "metaverse")
        
        return response
    
    def get_3d_location_preview(self, location: str) -> Dict:
        """Get a 3D preview of a location in the metaverse.
        
        Args:
            location: The location name.
            
        Returns:
            A dictionary containing the preview data.
        """
        # Map the location name to a location type
        location_types = {
            "central_station": "station",
            "trading_district": "district",
            "entertainment_district": "entertainment",
            "business_center": "district",
            "creative_district": "district",
            "natural_reserve": "natural",
            "tech_hub": "landmark"
        }
        
        location_type = location_types.get(location, "station")
        
        # Get the location name in a more readable format
        location_readable = location.replace("_", " ").title()
        
        # Generate the location preview
        return self.multimodal.generate_location_preview(location_readable, location_type)
    
    def preview_nft(self, nft_id: str, background_image: BinaryIO = None) -> Dict:
        """Preview an NFT in AR.
        
        Args:
            nft_id: The NFT ID.
            background_image: Optional background image (e.g., from camera).
            
        Returns:
            A dictionary containing the preview data.
        """
        # In a real implementation, you would fetch the NFT data from the blockchain
        # For now, we'll use some placeholder data
        
        # Get NFT info from the blockchain integration
        nft_data = {
            "id": nft_id,
            "name": f"Lucky Train NFT #{nft_id}",
            "description": "A unique digital asset from the Lucky Train metaverse.",
            "price": "100",
            "owner": "EQD__________________________________________0"
        }
        
        # Create the AR preview
        return self.multimodal.create_ar_preview(nft_data, background_image)
    
    def authenticate_wallet(self, user_id: str, address: str, signature: str, message: str = None) -> Dict:
        """Authenticate a user with their TON wallet.
        
        Args:
            user_id: The user's ID in the metaverse.
            address: The wallet address.
            signature: The signature provided by the wallet.
            message: The message that was signed (if not using the standard message).
            
        Returns:
            A dictionary containing the authentication result.
        """
        try:
            # If no message was provided, use the standard authentication message
            if message is None:
                message, _ = self.blockchain.get_wallet_auth_message(user_id)
            
            # Verify the signature
            is_valid = self.blockchain.verify_wallet_signature(address, message, signature)
            
            if is_valid:
                # Store the authenticated address for this user
                self.authenticated_wallets[user_id] = {
                    "address": address,
                    "authenticated_at": time.time()
                }
                
                return {
                    "success": True,
                    "authenticated": True,
                    "address": address,
                    "user_id": user_id
                }
            else:
                return {
                    "success": False,
                    "authenticated": False,
                    "error": "Invalid signature"
                }
                
        except Exception as e:
            logger.error(f"Error authenticating wallet: {e}")
            return {
                "success": False,
                "authenticated": False,
                "error": str(e)
            }
    
    def get_wallet_auth_message(self, user_id: str) -> Dict:
        """Get a message for wallet authentication.
        
        Args:
            user_id: The user's ID in the metaverse.
            
        Returns:
            A dictionary containing the authentication message.
        """
        try:
            message, timestamp = self.blockchain.get_wallet_auth_message(user_id)
            
            return {
                "success": True,
                "message": message,
                "timestamp": timestamp,
                "user_id": user_id
            }
            
        except Exception as e:
            logger.error(f"Error generating wallet authentication message: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_nft_items(self, user_id: str) -> Dict:
        """Get NFT items owned by a user.
        
        Args:
            user_id: The user's ID in the metaverse.
            
        Returns:
            A dictionary containing the user's NFT items.
        """
        try:
            # Check if the user has an authenticated wallet
            if user_id not in self.authenticated_wallets:
                return {
                    "success": False,
                    "error": "User's wallet not authenticated"
                }
            
            # Get the user's wallet address
            wallet_address = self.authenticated_wallets[user_id]["address"]
            
            # Get NFT items from the blockchain
            nft_items = self.blockchain.get_nft_items(wallet_address)
            
            return {
                "success": True,
                "nft_items": nft_items,
                "count": len(nft_items) if isinstance(nft_items, list) else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting NFT items: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_token_balance(self, user_id: str) -> Dict:
        """Get a user's token balance.
        
        Args:
            user_id: The user's ID in the metaverse.
            
        Returns:
            A dictionary containing the user's token balance.
        """
        try:
            # Check if the user has an authenticated wallet
            if user_id not in self.authenticated_wallets:
                return {
                    "success": False,
                    "error": "User's wallet not authenticated"
                }
            
            # Get the user's wallet address
            wallet_address = self.authenticated_wallets[user_id]["address"]
            
            # Get account info from the blockchain
            account_info = self.blockchain.get_account_info(wallet_address)
            
            # Extract the balance information
            ton_balance = account_info.get("balance", 0)
            
            # Convert from nanotons to TON
            ton_balance_decimal = float(ton_balance) / 1e9 if ton_balance else 0
            
            # In a real implementation, you would also get the LTT token balance
            # For now, we'll use a placeholder
            ltt_balance = 1000
            
            # Get token price information
            ton_price = self.blockchain.get_market_price("TON")
            ltt_price = self.blockchain.get_market_price("LTT")
            
            # Calculate USD value
            ton_usd_value = ton_balance_decimal * ton_price.get("price_usd", 0)
            ltt_usd_value = ltt_balance * ltt_price.get("price_usd", 0)
            
            return {
                "success": True,
                "address": wallet_address,
                "ton_balance": ton_balance_decimal,
                "ltt_balance": ltt_balance,
                "ton_usd_value": ton_usd_value,
                "ltt_usd_value": ltt_usd_value,
                "total_usd_value": ton_usd_value + ltt_usd_value
            }
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def prepare_nft_purchase(self, user_id: str, nft_id: str) -> Dict:
        """Prepare an NFT purchase transaction.
        
        Args:
            user_id: The user's ID in the metaverse.
            nft_id: The NFT ID to purchase.
            
        Returns:
            A dictionary containing the transaction details and deep link.
        """
        try:
            # Check if the user has an authenticated wallet
            if user_id not in self.authenticated_wallets:
                return {
                    "success": False,
                    "error": "User's wallet not authenticated"
                }
            
            # Get the user's wallet address
            wallet_address = self.authenticated_wallets[user_id]["address"]
            
            # In a real implementation, you would get the NFT details from the blockchain
            # For now, we'll use a placeholder
            nft_price = 100  # in TON
            nft_address = "EQD__________________________________________1"
            
            # Generate a deep link for the purchase
            deep_link = self.blockchain.generate_deep_link("nft_purchase", {
                "nft_address": nft_address,
                "price": nft_price
            })
            
            return {
                "success": True,
                "nft_id": nft_id,
                "price": nft_price,
                "deep_link": deep_link,
                "wallet_address": wallet_address
            }
            
        except Exception as e:
            logger.error(f"Error preparing NFT purchase: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def run_interactive_console(self):
        """Run an interactive console for testing the metaverse integration."""
        print("Lucky Train Metaverse Assistant - Interactive Console")
        print("Type 'exit' to quit")
        print()
        
        user_id = "console_user"
        location = "central_station"
        
        while True:
            user_input = input(f"[{location}] > ")
            
            if user_input.lower() == "exit":
                break
            
            # Check if it's a command
            if user_input.startswith("/"):
                command_parts = user_input[1:].split()
                command = command_parts[0]
                
                if command == "location":
                    if len(command_parts) > 1:
                        location = command_parts[1]
                        print(f"Перемещение в локацию: {location}")
                    else:
                        print(f"Текущая локация: {location}")
                    continue
                
                parameters = {}
                for part in command_parts[1:]:
                    if "=" in part:
                        key, value = part.split("=", 1)
                        parameters[key] = value
                
                response = self.handle_command(command, user_id, parameters)
            else:
                response = self.process_message(user_input, user_id, location)
            
            print(response)
            print()


def main():
    """Main function to run the metaverse integration."""
    assistant = LuckyTrainMetaverseAssistant()
    assistant.run_interactive_console()


if __name__ == "__main__":
    main() 