"""
Analytics and Monitoring for Lucky Train AI Assistant

This module provides functionality for tracking user interactions, collecting feedback,
and generating analytics about the usage of the Lucky Train AI assistant.
"""

import logging
import time
import json
import os
import csv
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
import threading
import queue
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AnalyticsEvent:
    """A class representing an analytics event."""
    
    def __init__(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ):
        """Initialize an analytics event.
        
        Args:
            event_type: The type of event (e.g., 'message', 'feedback', 'error').
            user_id: The ID of the user associated with the event.
            platform: The platform where the event occurred (e.g., 'telegram', 'website').
            properties: Additional properties for the event.
            timestamp: The timestamp of the event (defaults to current time).
        """
        self.event_type = event_type
        self.user_id = user_id or "anonymous"
        self.platform = platform or "unknown"
        self.properties = properties or {}
        self.timestamp = timestamp or time.time()
        self.event_id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the event to a dictionary.
        
        Returns:
            The event as a dictionary.
        """
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "user_id": self.user_id,
            "platform": self.platform,
            "properties": self.properties,
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsEvent':
        """Create an event from a dictionary.
        
        Args:
            data: The dictionary data.
            
        Returns:
            A new AnalyticsEvent instance.
        """
        event = cls(
            event_type=data.get("event_type", "unknown"),
            user_id=data.get("user_id"),
            platform=data.get("platform"),
            properties=data.get("properties", {}),
            timestamp=data.get("timestamp")
        )
        event.event_id = data.get("event_id", event.event_id)
        return event

class AnalyticsManager:
    """Manager for tracking and processing analytics events."""
    
    def __init__(self, config: Dict = None):
        """Initialize the analytics manager.
        
        Args:
            config: Analytics configuration dictionary.
        """
        self.config = config or {}
        
        # Get analytics settings
        self.enabled = self.config.get("enabled", True)
        self.log_user_interactions = self.config.get("log_user_interactions", True)
        self.collect_feedback = self.config.get("collect_feedback", True)
        
        # Set up storage paths
        self.storage_dir = self.config.get("storage_dir", "./analytics_data")
        self.events_file = os.path.join(self.storage_dir, "events.jsonl")
        self.feedback_file = os.path.join(self.storage_dir, "feedback.jsonl")
        self.daily_stats_dir = os.path.join(self.storage_dir, "daily_stats")
        
        # Create storage directories
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.daily_stats_dir, exist_ok=True)
        
        # Set up event processing queue
        self.event_queue = queue.Queue()
        self.processing_thread = None
        self.should_stop = False
        
        # Initialize metrics
        self.metrics = {
            "total_messages": 0,
            "total_users": set(),
            "platform_counts": {},
            "daily_counts": {},
            "error_counts": {},
            "popular_topics": {},
            "response_times": []
        }
        
        # Load existing metrics if available
        self._load_metrics()
        
        if self.enabled:
            # Start event processing thread
            self._start_processing_thread()
            logger.info("Analytics manager initialized and started")
        else:
            logger.info("Analytics manager initialized (disabled)")
    
    def _start_processing_thread(self):
        """Start the event processing thread."""
        if self.processing_thread is not None and self.processing_thread.is_alive():
            return
        
        self.should_stop = False
        self.processing_thread = threading.Thread(target=self._process_events, daemon=True)
        self.processing_thread.start()
    
    def _process_events(self):
        """Process events from the queue."""
        while not self.should_stop:
            try:
                # Get an event from the queue with a timeout
                try:
                    event = self.event_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Process the event
                self._process_event(event)
                
                # Mark the event as processed
                self.event_queue.task_done()
                
            except Exception as e:
                logger.error(f"Error processing analytics event: {e}")
    
    def _process_event(self, event: AnalyticsEvent):
        """Process an analytics event.
        
        Args:
            event: The event to process.
        """
        # Save the event
        self._save_event(event)
        
        # Update metrics based on the event
        self._update_metrics(event)
    
    def _save_event(self, event: AnalyticsEvent):
        """Save an event to storage.
        
        Args:
            event: The event to save.
        """
        # Save to the appropriate file based on event type
        if event.event_type == "feedback":
            file_path = self.feedback_file
        else:
            file_path = self.events_file
        
        try:
            # Append to the file
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
        except Exception as e:
            logger.error(f"Error saving analytics event: {e}")
    
    def _update_metrics(self, event: AnalyticsEvent):
        """Update metrics based on an event.
        
        Args:
            event: The event to process.
        """
        # Get the date string for daily metrics
        event_date = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d')
        
        # Initialize daily counts if needed
        if event_date not in self.metrics["daily_counts"]:
            self.metrics["daily_counts"][event_date] = {"messages": 0, "users": set(), "platforms": {}}
        
        # Update metrics based on event type
        if event.event_type == "message":
            # Update total messages
            self.metrics["total_messages"] += 1
            
            # Update total users
            self.metrics["total_users"].add(event.user_id)
            
            # Update platform counts
            if event.platform not in self.metrics["platform_counts"]:
                self.metrics["platform_counts"][event.platform] = 0
            self.metrics["platform_counts"][event.platform] += 1
            
            # Update daily counts
            self.metrics["daily_counts"][event_date]["messages"] += 1
            self.metrics["daily_counts"][event_date]["users"].add(event.user_id)
            
            if event.platform not in self.metrics["daily_counts"][event_date]["platforms"]:
                self.metrics["daily_counts"][event_date]["platforms"][event.platform] = 0
            self.metrics["daily_counts"][event_date]["platforms"][event.platform] += 1
            
            # Update popular topics
            topic = event.properties.get("topic", "general")
            if topic not in self.metrics["popular_topics"]:
                self.metrics["popular_topics"][topic] = 0
            self.metrics["popular_topics"][topic] += 1
            
            # Update response times
            response_time = event.properties.get("response_time")
            if response_time is not None:
                self.metrics["response_times"].append(response_time)
        
        elif event.event_type == "error":
            # Update error counts
            error_type = event.properties.get("error_type", "unknown")
            if error_type not in self.metrics["error_counts"]:
                self.metrics["error_counts"][error_type] = 0
            self.metrics["error_counts"][error_type] += 1
        
        # Save updated metrics periodically
        # This is a simple approach; in a production system, you might want to 
        # save metrics less frequently to reduce I/O
        self._save_metrics()
    
    def _save_metrics(self):
        """Save metrics to storage."""
        try:
            # Convert sets to lists for JSON serialization
            serializable_metrics = self.metrics.copy()
            serializable_metrics["total_users"] = list(self.metrics["total_users"])
            
            for date, counts in serializable_metrics["daily_counts"].items():
                serializable_metrics["daily_counts"][date]["users"] = list(counts["users"])
            
            # Save to file
            metrics_file = os.path.join(self.storage_dir, "metrics.json")
            with open(metrics_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_metrics, f, indent=2)
            
            # Save daily stats if a day has completed
            today = datetime.now().strftime('%Y-%m-%d')
            for date in list(self.metrics["daily_counts"].keys()):
                if date != today:
                    self._save_daily_stats(date)
                    # Remove old daily data from in-memory metrics
                    if date in self.metrics["daily_counts"]:
                        del self.metrics["daily_counts"][date]
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")
    
    def _save_daily_stats(self, date: str):
        """Save daily statistics to a separate file.
        
        Args:
            date: The date string in YYYY-MM-DD format.
        """
        try:
            if date not in self.metrics["daily_counts"]:
                return
            
            daily_data = self.metrics["daily_counts"][date]
            serializable_data = daily_data.copy()
            serializable_data["users"] = list(daily_data["users"])
            
            # Save to file
            daily_file = os.path.join(self.daily_stats_dir, f"{date}.json")
            with open(daily_file, 'w', encoding='utf-8') as f:
                json.dump(serializable_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error saving daily stats for {date}: {e}")
    
    def _load_metrics(self):
        """Load metrics from storage."""
        metrics_file = os.path.join(self.storage_dir, "metrics.json")
        
        if not os.path.exists(metrics_file):
            return
        
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                loaded_metrics = json.load(f)
            
            # Convert lists back to sets
            loaded_metrics["total_users"] = set(loaded_metrics["total_users"])
            
            for date, counts in loaded_metrics["daily_counts"].items():
                loaded_metrics["daily_counts"][date]["users"] = set(counts["users"])
            
            self.metrics.update(loaded_metrics)
            
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    def track_event(
        self,
        event_type: str,
        user_id: Optional[str] = None,
        platform: Optional[str] = None,
        properties: Optional[Dict[str, Any]] = None
    ):
        """Track an analytics event.
        
        Args:
            event_type: The type of event.
            user_id: The ID of the user.
            platform: The platform where the event occurred.
            properties: Additional properties for the event.
        """
        if not self.enabled:
            return
        
        # Create a new event
        event = AnalyticsEvent(event_type, user_id, platform, properties)
        
        # Add to processing queue
        self.event_queue.put(event)
    
    def track_message(
        self,
        user_id: str,
        platform: str,
        message: str,
        response: str,
        topic: str = "general",
        response_time: Optional[float] = None
    ):
        """Track a user message and its response.
        
        Args:
            user_id: The ID of the user.
            platform: The platform where the message was sent.
            message: The user's message.
            response: The assistant's response.
            topic: The topic of the message.
            response_time: The time taken to generate the response, in seconds.
        """
        if not self.enabled or not self.log_user_interactions:
            return
        
        properties = {
            "message_length": len(message),
            "response_length": len(response),
            "topic": topic
        }
        
        if response_time is not None:
            properties["response_time"] = response_time
        
        self.track_event("message", user_id, platform, properties)
    
    def track_feedback(
        self,
        user_id: str,
        platform: str,
        message_id: str,
        rating: int,
        comments: Optional[str] = None
    ):
        """Track user feedback.
        
        Args:
            user_id: The ID of the user.
            platform: The platform where the feedback was given.
            message_id: The ID of the message being rated.
            rating: The rating (e.g., 1-5).
            comments: Optional comments from the user.
        """
        if not self.enabled or not self.collect_feedback:
            return
        
        properties = {
            "message_id": message_id,
            "rating": rating
        }
        
        if comments:
            properties["comments"] = comments
        
        self.track_event("feedback", user_id, platform, properties)
    
    def track_error(
        self,
        error_type: str,
        error_message: str,
        user_id: Optional[str] = None,
        platform: Optional[str] = None
    ):
        """Track an error.
        
        Args:
            error_type: The type of error.
            error_message: The error message.
            user_id: The ID of the user who encountered the error.
            platform: The platform where the error occurred.
        """
        if not self.enabled:
            return
        
        properties = {
            "error_type": error_type,
            "error_message": error_message
        }
        
        self.track_event("error", user_id, platform, properties)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the current metrics.
        
        Returns:
            A dictionary of metrics.
        """
        # Return a copy of the metrics with sets converted to lists
        serializable_metrics = self.metrics.copy()
        serializable_metrics["total_users"] = list(self.metrics["total_users"])
        serializable_metrics["total_user_count"] = len(self.metrics["total_users"])
        
        for date, counts in serializable_metrics["daily_counts"].items():
            serializable_metrics["daily_counts"][date]["users"] = list(counts["users"])
            serializable_metrics["daily_counts"][date]["user_count"] = len(counts["users"])
        
        # Calculate average response time
        if self.metrics["response_times"]:
            serializable_metrics["avg_response_time"] = sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
        else:
            serializable_metrics["avg_response_time"] = 0
        
        return serializable_metrics
    
    def get_daily_stats(self, date: str) -> Optional[Dict[str, Any]]:
        """Get statistics for a specific day.
        
        Args:
            date: The date string in YYYY-MM-DD format.
            
        Returns:
            A dictionary of daily statistics, or None if not found.
        """
        # Check if the date is in the current metrics
        if date in self.metrics["daily_counts"]:
            daily_data = self.metrics["daily_counts"][date]
            serializable_data = daily_data.copy()
            serializable_data["users"] = list(daily_data["users"])
            serializable_data["user_count"] = len(daily_data["users"])
            return serializable_data
        
        # Check if there's a file for the date
        daily_file = os.path.join(self.daily_stats_dir, f"{date}.json")
        if os.path.exists(daily_file):
            try:
                with open(daily_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading daily stats for {date}: {e}")
        
        return None
    
    def export_events_to_csv(self, output_file: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        """Export events to a CSV file.
        
        Args:
            output_file: The output file path.
            start_date: The start date in YYYY-MM-DD format (inclusive).
            end_date: The end date in YYYY-MM-DD format (inclusive).
        """
        try:
            # Convert date strings to timestamps
            start_timestamp = None
            end_timestamp = None
            
            if start_date:
                start_timestamp = datetime.strptime(start_date, '%Y-%m-%d').timestamp()
            
            if end_date:
                # Set end timestamp to the end of the day
                end_timestamp = datetime.strptime(end_date, '%Y-%m-%d').replace(
                    hour=23, minute=59, second=59
                ).timestamp()
            
            # Collect events
            events = []
            
            with open(self.events_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        timestamp = event_data.get("timestamp")
                        
                        # Filter by date range
                        if start_timestamp and timestamp < start_timestamp:
                            continue
                        
                        if end_timestamp and timestamp > end_timestamp:
                            continue
                        
                        events.append(event_data)
                        
                    except Exception as e:
                        logger.error(f"Error parsing event line: {e}")
            
            # Write to CSV
            if events:
                # Get all unique fields
                fields = set()
                for event in events:
                    fields.update(event.keys())
                    fields.update(f"property_{key}" for key in event.get("properties", {}).keys())
                
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow(sorted(fields))
                    
                    # Write rows
                    for event in events:
                        row = []
                        properties = event.get("properties", {})
                        
                        for field in sorted(fields):
                            if field.startswith("property_"):
                                # Get property value
                                prop_name = field[len("property_"):]
                                value = properties.get(prop_name, "")
                            else:
                                # Get event field value
                                value = event.get(field, "")
                            
                            row.append(value)
                        
                        writer.writerow(row)
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error exporting events to CSV: {e}")
            return False
    
    def stop(self):
        """Stop the analytics manager."""
        if not self.enabled:
            return
        
        logger.info("Stopping analytics manager")
        self.should_stop = True
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        # Save metrics one last time
        self._save_metrics()
        
        logger.info("Analytics manager stopped")

class FeedbackCollector:
    """Class for collecting and managing user feedback."""
    
    def __init__(self, analytics_manager: AnalyticsManager):
        """Initialize the feedback collector.
        
        Args:
            analytics_manager: The analytics manager to use for tracking feedback.
        """
        self.analytics_manager = analytics_manager
    
    def create_feedback_buttons(self, message_id: str) -> Dict[str, Any]:
        """Create feedback buttons for a message.
        
        Args:
            message_id: The ID of the message to get feedback for.
            
        Returns:
            A dictionary with button configuration.
        """
        return {
            "type": "feedback",
            "message_id": message_id,
            "buttons": [
                {"label": "👍", "value": 5, "action": "rate"},
                {"label": "👎", "value": 1, "action": "rate"},
                {"label": "💬", "value": "comment", "action": "comment"}
            ]
        }
    
    def process_feedback(
        self,
        user_id: str,
        platform: str,
        message_id: str,
        rating: Union[int, str],
        comments: Optional[str] = None
    ):
        """Process user feedback.
        
        Args:
            user_id: The ID of the user.
            platform: The platform where the feedback was given.
            message_id: The ID of the message being rated.
            rating: The rating value or action.
            comments: Optional comments from the user.
        """
        # Track the feedback
        if isinstance(rating, int):
            self.analytics_manager.track_feedback(user_id, platform, message_id, rating, comments)
        elif comments:
            # If rating is not an integer but we have comments, track as feedback with a neutral rating
            self.analytics_manager.track_feedback(user_id, platform, message_id, 3, comments)
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Get a summary of feedback.
        
        Returns:
            A dictionary with feedback statistics.
        """
        feedback_file = self.analytics_manager.feedback_file
        
        if not os.path.exists(feedback_file):
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "rating_counts": {},
                "recent_comments": []
            }
        
        try:
            # Collect feedback events
            feedback_events = []
            
            with open(feedback_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        event_data = json.loads(line)
                        if event_data.get("event_type") == "feedback":
                            feedback_events.append(event_data)
                    except Exception as e:
                        logger.error(f"Error parsing feedback line: {e}")
            
            # Calculate statistics
            total_feedback = len(feedback_events)
            
            if total_feedback == 0:
                return {
                    "total_feedback": 0,
                    "average_rating": 0,
                    "rating_counts": {},
                    "recent_comments": []
                }
            
            ratings = [
                event["properties"].get("rating") 
                for event in feedback_events 
                if isinstance(event["properties"].get("rating"), (int, float))
            ]
            
            average_rating = sum(ratings) / len(ratings) if ratings else 0
            
            # Count ratings
            rating_counts = {}
            for rating in ratings:
                if rating not in rating_counts:
                    rating_counts[rating] = 0
                rating_counts[rating] += 1
            
            # Get recent comments
            comments = [
                {
                    "user_id": event["user_id"],
                    "platform": event["platform"],
                    "rating": event["properties"].get("rating"),
                    "comment": event["properties"].get("comments"),
                    "timestamp": event["timestamp"],
                    "datetime": event.get("datetime")
                }
                for event in feedback_events
                if event["properties"].get("comments")
            ]
            
            # Sort by timestamp descending
            comments.sort(key=lambda x: x["timestamp"], reverse=True)
            
            # Get the 10 most recent comments
            recent_comments = comments[:10]
            
            return {
                "total_feedback": total_feedback,
                "average_rating": average_rating,
                "rating_counts": rating_counts,
                "recent_comments": recent_comments
            }
            
        except Exception as e:
            logger.error(f"Error getting feedback summary: {e}")
            return {
                "total_feedback": 0,
                "average_rating": 0,
                "rating_counts": {},
                "recent_comments": []
            } 