"""
Streaming Response Handling for Lucky Train AI Assistant

This module provides functionality for streaming AI responses in real-time
to improve user experience with faster feedback.
"""

import logging
import time
from typing import Callable, Dict, Generator, List, Optional, Union
import queue
import threading
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StreamingHandler:
    """Handler for streaming AI responses."""
    
    def __init__(self):
        """Initialize the streaming handler."""
        self.active_streams = {}
        self.message_queues = {}
        self.stream_threads = {}
    
    def generate_streaming_response(
        self,
        openai_client: openai.OpenAI,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        session_id: str = None
    ) -> str:
        """Generate a complete response using streaming.
        
        Args:
            openai_client: The OpenAI client instance.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
            session_id: The session ID to use for tracking the stream.
            
        Returns:
            The complete response as a string.
        """
        if session_id is None:
            session_id = f"stream_{int(time.time())}"
        
        # Create a queue for storing message chunks
        self.message_queues[session_id] = queue.Queue()
        
        # Create and start the streaming thread
        stream_thread = threading.Thread(
            target=self._stream_response,
            args=(openai_client, messages, model, temperature, max_tokens, session_id),
            daemon=True
        )
        self.stream_threads[session_id] = stream_thread
        stream_thread.start()
        
        # Collect all message chunks from the queue
        full_response = ""
        
        while True:
            try:
                # Get message chunk from the queue with a timeout
                chunk = self.message_queues[session_id].get(timeout=30)
                
                # Check if this is the end of the stream
                if chunk is None:
                    break
                
                full_response += chunk
                
            except queue.Empty:
                logger.warning(f"Timeout waiting for response chunks in session {session_id}")
                break
            except Exception as e:
                logger.error(f"Error getting response chunks: {e}")
                break
        
        # Clean up
        self._cleanup_stream(session_id)
        
        return full_response
    
    def get_response_generator(
        self,
        openai_client: openai.OpenAI,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        session_id: str = None
    ) -> Generator[str, None, None]:
        """Get a generator for streaming responses.
        
        Args:
            openai_client: The OpenAI client instance.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
            session_id: The session ID to use for tracking the stream.
            
        Returns:
            A generator that yields response chunks.
        """
        if session_id is None:
            session_id = f"stream_{int(time.time())}"
        
        # Create a queue for storing message chunks
        self.message_queues[session_id] = queue.Queue()
        
        # Create and start the streaming thread
        stream_thread = threading.Thread(
            target=self._stream_response,
            args=(openai_client, messages, model, temperature, max_tokens, session_id),
            daemon=True
        )
        self.stream_threads[session_id] = stream_thread
        stream_thread.start()
        
        # Yield message chunks from the queue
        try:
            while True:
                # Get message chunk from the queue with a timeout
                chunk = self.message_queues[session_id].get(timeout=30)
                
                # Check if this is the end of the stream
                if chunk is None:
                    break
                
                yield chunk
                
        except queue.Empty:
            logger.warning(f"Timeout waiting for response chunks in session {session_id}")
        except Exception as e:
            logger.error(f"Error getting response chunks: {e}")
        finally:
            # Clean up
            self._cleanup_stream(session_id)
    
    def _stream_response(
        self,
        openai_client: openai.OpenAI,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        session_id: str
    ):
        """Stream the response from the OpenAI API and add chunks to the queue.
        
        Args:
            openai_client: The OpenAI client instance.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
            session_id: The session ID to use for tracking the stream.
        """
        try:
            # Create a streaming response
            stream = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True
            )
            
            # Mark this stream as active
            self.active_streams[session_id] = True
            
            # Process the stream
            for chunk in stream:
                if not self.active_streams.get(session_id, False):
                    # Stream was cancelled
                    break
                
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        # Add the content to the queue
                        self.message_queues[session_id].put(content)
            
            # Signal the end of the stream
            self.message_queues[session_id].put(None)
            
        except Exception as e:
            logger.error(f"Error streaming response: {e}")
            # Signal the end of the stream
            self.message_queues[session_id].put(None)
        finally:
            # Mark this stream as inactive
            self.active_streams[session_id] = False
    
    def cancel_stream(self, session_id: str):
        """Cancel an active stream.
        
        Args:
            session_id: The session ID of the stream to cancel.
        """
        if session_id in self.active_streams:
            self.active_streams[session_id] = False
            logger.info(f"Cancelled stream {session_id}")
    
    def _cleanup_stream(self, session_id: str):
        """Clean up resources associated with a stream.
        
        Args:
            session_id: The session ID of the stream to clean up.
        """
        if session_id in self.active_streams:
            del self.active_streams[session_id]
        
        if session_id in self.message_queues:
            del self.message_queues[session_id]
        
        if session_id in self.stream_threads:
            del self.stream_threads[session_id]
        
        logger.info(f"Cleaned up stream {session_id}")
        
class WebSocketHandler:
    """Handler for WebSocket connections for streaming responses."""
    
    def __init__(self):
        """Initialize the WebSocket handler."""
        self.streaming_handler = StreamingHandler()
        self.active_connections = {}
    
    def register_connection(
        self,
        connection,
        session_id: str,
        message_callback: Callable[[str, str], None]
    ):
        """Register a WebSocket connection for receiving streaming responses.
        
        Args:
            connection: The WebSocket connection object.
            session_id: The session ID to associate with the connection.
            message_callback: A callback function that takes the session ID and message
                as arguments and sends the message to the WebSocket client.
        """
        self.active_connections[session_id] = {
            "connection": connection,
            "callback": message_callback
        }
        logger.info(f"Registered WebSocket connection for session {session_id}")
    
    def unregister_connection(self, session_id: str):
        """Unregister a WebSocket connection.
        
        Args:
            session_id: The session ID associated with the connection.
        """
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"Unregistered WebSocket connection for session {session_id}")
            
            # Cancel any active streams for this session
            self.streaming_handler.cancel_stream(session_id)
    
    def stream_response_to_websocket(
        self,
        openai_client: openai.OpenAI,
        messages: List[Dict[str, str]],
        session_id: str,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Stream a response to a WebSocket connection.
        
        Args:
            openai_client: The OpenAI client instance.
            messages: The messages to use for generating the response.
            session_id: The session ID associated with the connection.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
        """
        if session_id not in self.active_connections:
            logger.warning(f"No active WebSocket connection for session {session_id}")
            return
        
        # Get the message callback
        message_callback = self.active_connections[session_id]["callback"]
        
        # Create a generator for streaming responses
        generator = self.streaming_handler.get_response_generator(
            openai_client=openai_client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id
        )
        
        # Stream the response to the WebSocket connection
        try:
            for chunk in generator:
                message_callback(session_id, chunk)
                
            # Signal the end of the stream
            message_callback(session_id, None)
            
        except Exception as e:
            logger.error(f"Error streaming response to WebSocket: {e}")
            # Signal an error
            message_callback(session_id, {"error": str(e)})
            
class StreamingOutput:
    """Class for managing streaming output to different interfaces."""
    
    def __init__(self):
        """Initialize the streaming output manager."""
        self.streaming_handler = StreamingHandler()
        self.websocket_handler = WebSocketHandler()
    
    def stream_to_telegram(
        self,
        openai_client: openai.OpenAI,
        bot: object,
        chat_id: int,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500,
        update_interval: float = 1.0
    ):
        """Stream a response to a Telegram chat.
        
        Args:
            openai_client: The OpenAI client instance.
            bot: The Telegram bot instance.
            chat_id: The Telegram chat ID.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
            update_interval: The interval in seconds between message updates.
        """
        session_id = f"telegram_{chat_id}_{int(time.time())}"
        
        # Send an initial message
        sent_message = bot.send_message(
            chat_id=chat_id,
            text="⌛ Генерирую ответ..."
        )
        
        # Create a generator for streaming responses
        generator = self.streaming_handler.get_response_generator(
            openai_client=openai_client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id
        )
        
        # Stream the response with periodic updates
        full_response = ""
        last_update_time = time.time()
        
        try:
            for chunk in generator:
                full_response += chunk
                
                # Update the message periodically
                current_time = time.time()
                if current_time - last_update_time >= update_interval:
                    bot.edit_message_text(
                        chat_id=chat_id,
                        message_id=sent_message.message_id,
                        text=full_response + "▌"
                    )
                    last_update_time = current_time
            
            # Final update
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=sent_message.message_id,
                text=full_response
            )
            
        except Exception as e:
            logger.error(f"Error streaming response to Telegram: {e}")
            # Update with an error message
            bot.edit_message_text(
                chat_id=chat_id,
                message_id=sent_message.message_id,
                text="❌ Произошла ошибка при генерации ответа. Пожалуйста, попробуйте еще раз."
            )
    
    def stream_to_console(
        self,
        openai_client: openai.OpenAI,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Stream a response to the console.
        
        Args:
            openai_client: The OpenAI client instance.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
        """
        session_id = f"console_{int(time.time())}"
        
        # Create a generator for streaming responses
        generator = self.streaming_handler.get_response_generator(
            openai_client=openai_client,
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            session_id=session_id
        )
        
        # Stream the response to the console
        print("⌛ Генерирую ответ...", end="\r")
        
        try:
            # Clear the loading message
            print(" " * 30, end="\r")
            
            for chunk in generator:
                print(chunk, end="", flush=True)
            
            print()  # Add a newline at the end
            
        except Exception as e:
            logger.error(f"Error streaming response to console: {e}")
            print("\n❌ Произошла ошибка при генерации ответа")
    
    def register_websocket(self, connection, session_id: str, message_callback: Callable[[str, str], None]):
        """Register a WebSocket connection for streaming responses.
        
        Args:
            connection: The WebSocket connection object.
            session_id: The session ID to associate with the connection.
            message_callback: A callback function that takes the session ID and message
                as arguments and sends the message to the WebSocket client.
        """
        self.websocket_handler.register_connection(connection, session_id, message_callback)
    
    def unregister_websocket(self, session_id: str):
        """Unregister a WebSocket connection.
        
        Args:
            session_id: The session ID associated with the connection.
        """
        self.websocket_handler.unregister_connection(session_id)
    
    def stream_to_websocket(
        self,
        openai_client: openai.OpenAI,
        session_id: str,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """Stream a response to a WebSocket connection.
        
        Args:
            openai_client: The OpenAI client instance.
            session_id: The session ID associated with the connection.
            messages: The messages to use for generating the response.
            model: The model to use.
            temperature: The temperature to use.
            max_tokens: The maximum number of tokens to generate.
        """
        self.websocket_handler.stream_response_to_websocket(
            openai_client=openai_client,
            messages=messages,
            session_id=session_id,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        ) 