"""
Calendar and Schedule Optimization Service

This module handles:
- Google Calendar integration
- AI-powered schedule optimization
- Event management with user preferences
- Smart reminder scheduling

Based on research requirements for calendar AI optimization and user control.
"""

import os
import json
import logging
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Tuple
from google import genai
from google.genai import types
import spacy
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from flask import current_app, session

logger = logging.getLogger(__name__)

class CalendarOptimizationService:
    """
    AI-powered calendar optimization service
    Implements research features: schedule optimization, user control, AI recommendations
    """
    
    def __init__(self):
        """Initialize calendar service"""
        self.nlp = None
        self.gemini_model = None
        self.calendar_service = None
        self._services_setup = False
    
    def _ensure_services_setup(self):
        """Ensure services are set up (deferred until Flask context is available)"""
        if not self._services_setup:
            self.setup_services()
            self._services_setup = True
    
    def setup_services(self):
        """Setup AI and calendar services"""
        try:
            # Setup Gemini AI
            api_key = current_app.config.get('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                
                # Use dictionary format for generation config (compatible with current version)
                generation_config = {
                    "response_mime_type": "application/json",
                    "temperature": 0.7,
                    "top_p": 0.8,
                    "top_k": 40
                }
                
                self.gemini_model = genai.GenerativeModel(
                    'gemini-2.5-flash',
                    generation_config=generation_config
                )
            
            # Setup spaCy for event analysis
            try:
                self.nlp = spacy.load('en_core_web_sm')
            except OSError:
                logger.warning("spaCy model not available for event analysis")
        
        except Exception as e:
            logger.error(f"Error setting up calendar services: {e}")
    
    def analyze_event_text(self, event_text: str) -> Dict:
        """
        Analyze event text to extract type and scheduling preferences
        Uses spaCy NLP to understand event context
        """
        self._ensure_services_setup()
        if not self.nlp:
            return self._basic_event_analysis(event_text)
        
        doc = self.nlp(event_text.lower())
        
        # Event type detection based on keywords
        event_types = {
            'work': ['meeting', 'work', 'office', 'conference', 'presentation', 'deadline'],
            'exercise': ['gym', 'workout', 'run', 'exercise', 'fitness', 'yoga', 'sports'],
            'meal': ['lunch', 'dinner', 'breakfast', 'meal', 'eat', 'restaurant'],
            'health': ['doctor', 'appointment', 'checkup', 'therapy', 'medical'],
            'personal': ['family', 'friend', 'personal', 'hobby', 'shopping'],
            'sleep': ['sleep', 'rest', 'nap', 'bedtime']
        }
        
        detected_type = 'personal'  # default
        confidence = 0.0
        
        for event_type, keywords in event_types.items():
            matches = sum(1 for token in doc if token.text in keywords)
            type_confidence = matches / len(keywords)
            if type_confidence > confidence:
                confidence = type_confidence
                detected_type = event_type
        
        # Priority detection
        priority_indicators = {
            5: ['urgent', 'critical', 'important', 'deadline', 'must'],
            4: ['high', 'priority', 'essential'],
            3: ['normal', 'regular'],
            2: ['low', 'optional', 'maybe'],
            1: ['if time', 'flexible', 'whenever']
        }
        
        priority = 3  # default
        for level, indicators in priority_indicators.items():
            if any(indicator in event_text.lower() for indicator in indicators):
                priority = level
                break
        
        return {
            'event_type': detected_type,
            'priority': priority,
            'confidence': confidence,
            'suggested_duration': self._suggest_duration(detected_type)
        }
    
    def _basic_event_analysis(self, event_text: str) -> Dict:
        """Basic event analysis without spaCy"""
        event_text_lower = event_text.lower()
        
        if any(word in event_text_lower for word in ['meeting', 'work', 'office']):
            return {'event_type': 'work', 'priority': 4, 'suggested_duration': 60}
        elif any(word in event_text_lower for word in ['gym', 'workout', 'exercise']):
            return {'event_type': 'exercise', 'priority': 3, 'suggested_duration': 45}
        else:
            return {'event_type': 'personal', 'priority': 3, 'suggested_duration': 30}
    
    def _suggest_duration(self, event_type: str) -> int:
        """Suggest duration in minutes based on event type"""
        durations = {
            'work': 60,
            'exercise': 45,
            'meal': 30,
            'health': 45,
            'personal': 30,
            'sleep': 480  # 8 hours
        }
        return durations.get(event_type, 30)
    
    def optimize_schedule(self, user_events: List, health_data: Dict, preferences: Dict) -> Dict:
        """
        AI-powered schedule optimization
        Implements research: AI schedule optimization with user control
        """
        self._ensure_services_setup()
        try:
            if not self.gemini_model:
                logger.warning("Gemini model not available, using basic optimization")
                return self._basic_schedule_optimization(user_events, health_data, preferences)
            
            logger.info(f"🤖 CALLING GEMINI AI for schedule optimization with {len(user_events)} events")
            
            # Prepare context for AI
            context = self._prepare_schedule_context(user_events, health_data, preferences)
            
            prompt = f"""
            You are an AI scheduling assistant. Optimize the user's schedule based on their health data and preferences.
            
            IMPORTANT RULES:
            1. NEVER move events marked as 'is_fixed_time': True
            2. NEVER move events marked as 'is_ai_modifiable': False
            3. Respect user's quiet hours for reminders
            4. Prioritize high-priority events
            5. Consider health patterns for optimal timing
            
            Context:
            {context}
            
            RETURN ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS, NO CODE BLOCKS
            Provide optimization suggestions in JSON format:
            {{
                "schedule_changes": [
                    {{
                        "event_id": "id",
                        "suggested_time": "YYYY-MM-DDTHH:MM:SS",
                        "reason": "explanation"
                    }}
                ],
                "new_reminders": [
                    {{
                        "type": "water/exercise/sleep",
                        "time": "YYYY-MM-DDTHH:MM:SS",
                        "message": "reminder text"
                    }}
                ],
                "insights": ["list of insights about the schedule"]
            }}
            
            RETURN ONLY THE JSON OBJECT - NO ADDITIONAL TEXT
            Only suggest realistic, beneficial changes.
            """
            
            logger.info("📡 Sending request to Gemini AI...")
            response = self.gemini_model.generate_content(prompt)
            logger.info(f"📨 Gemini AI response received: {len(response.text) if response.text else 0} characters")
            
            if response.text:
                try:
                    # Parse AI response
                    optimization_result = json.loads(response.text)
                    logger.info(f"✅ Gemini AI optimization successful: {len(optimization_result.get('schedule_changes', []))} changes, {len(optimization_result.get('new_reminders', []))} reminders")
                    return optimization_result
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Failed to parse AI optimization response: {e}")
                    logger.error(f"Raw response: {response.text[:200]}...")
                    return self._basic_schedule_optimization(user_events, health_data, preferences)
            else:
                logger.warning("🔕 Gemini AI returned empty response")
                return self._basic_schedule_optimization(user_events, health_data, preferences)
            
        except Exception as e:
            logger.error(f"Error in AI schedule optimization: {e}")
            return self._basic_schedule_optimization(user_events, health_data, preferences)
    
    def _prepare_schedule_context(self, user_events: List, health_data: Dict, preferences: Dict) -> str:
        """Prepare context for AI schedule optimization"""
        context_parts = []
        
        # Current events
        events_info = []
        for event in user_events:
            # Handle both dictionary and object formats
            if isinstance(event, dict):
                event_info = {
                    'id': event.get('id'),
                    'title': event.get('title'),
                    'start': event.get('start_time'),
                    'end': event.get('end_time'),
                    'type': event.get('event_type'),
                    'priority': event.get('priority_level', 3),
                    'fixed': event.get('is_fixed_time', False),
                    'ai_modifiable': event.get('is_ai_modifiable', True)
                }
            else:
                event_info = {
                    'id': event.id,
                    'title': event.title,
                    'start': event.start_time.isoformat(),
                    'end': event.end_time.isoformat(),
                    'type': event.event_type,
                    'priority': event.priority_level,
                    'fixed': event.is_fixed_time,
                    'ai_modifiable': event.is_ai_modifiable
                }
            events_info.append(event_info)
        
        context_parts.append(f"Current events: {json.dumps(events_info, indent=2)}")
        
        # Health data context
        if health_data:
            context_parts.append(f"Health patterns: {json.dumps(health_data, indent=2)}")
        
        # User preferences
        if preferences:
            # Handle both dictionary and object formats
            if isinstance(preferences, dict):
                prefs_info = {
                    'water_goal': preferences.get('daily_water_goal', 2.5),
                    'sleep_goal': preferences.get('daily_sleep_goal', 8),
                    'steps_goal': preferences.get('daily_steps_goal', 10000),
                    'quiet_hours_start': preferences.get('quiet_hours_start', '22:00'),
                    'quiet_hours_end': preferences.get('quiet_hours_end', '07:00'),
                    'reminders_enabled': {
                        'water': preferences.get('reminder_water', True),
                        'exercise': preferences.get('reminder_exercise', True),
                        'sleep': preferences.get('reminder_sleep', True),
                        'mindfulness': preferences.get('reminder_mindfulness', True)
                    }
                }
            else:
                prefs_info = {
                    'water_goal': preferences.daily_water_goal,
                    'sleep_goal': preferences.daily_sleep_goal,
                    'steps_goal': preferences.daily_steps_goal,
                    'quiet_hours_start': preferences.quiet_hours_start.strftime('%H:%M'),
                    'quiet_hours_end': preferences.quiet_hours_end.strftime('%H:%M'),
                    'reminders_enabled': {
                        'water': preferences.reminder_water,
                        'exercise': preferences.reminder_exercise,
                        'sleep': preferences.reminder_sleep,
                        'mindfulness': preferences.reminder_mindfulness
                    }
                }
            context_parts.append(f"User preferences: {json.dumps(prefs_info, indent=2)}")
        
        return '\n'.join(context_parts)
    
    def _basic_schedule_optimization(self, user_events: List, health_data: Dict, preferences: Dict) -> Dict:
        """Basic schedule optimization without AI"""
        optimizations = {
            'schedule_changes': [],
            'new_reminders': [],
            'insights': []
        }
        
        # Generate basic reminders based on preferences
        if preferences and preferences.reminder_water:
            # Add water reminders every 2 hours during active hours
            current_time = datetime.now()
            for hour in range(8, 20, 2):  # 8 AM to 6 PM, every 2 hours
                reminder_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
                if self._is_time_available(reminder_time, user_events):
                    optimizations['new_reminders'].append({
                        'type': 'water',
                        'time': reminder_time.isoformat(),
                        'message': 'Time for a glass of water! Stay hydrated.'
                    })
        
        # Basic insights
        optimizations['insights'] = [
            'Schedule analysis complete',
            'Basic reminders added based on your preferences'
        ]
        
        return optimizations
    
    def _is_time_available(self, target_time: datetime, events: List) -> bool:
        """Check if a time slot is available (no conflicting events)"""
        for event in events:
            if event.start_time <= target_time <= event.end_time:
                return False
        return True
    
    def generate_smart_reminders(self, user_events: List, health_data: Dict, preferences: Dict) -> List[Dict]:
        """
        Generate smart reminders based on schedule and health patterns
        Implements research: adaptive reminders based on user behavior
        """
        self._ensure_services_setup()
        reminders = []
        
        if not preferences:
            return reminders
        
        current_time = datetime.now()
        
        # Water reminders - avoid during meetings/work events
        if preferences.reminder_water:
            water_times = self._calculate_optimal_water_times(user_events, preferences)
            for water_time in water_times:
                reminders.append({
                    'type': 'water',
                    'time': water_time,
                    'message': self._generate_water_reminder_message(health_data),
                    'priority': 2
                })
        
        # Exercise reminders - suggest based on free time and energy patterns
        if preferences.reminder_exercise:
            exercise_time = self._find_optimal_exercise_time(user_events, health_data)
            if exercise_time:
                reminders.append({
                    'type': 'exercise',
                    'time': exercise_time,
                    'message': self._generate_exercise_reminder_message(health_data),
                    'priority': 3
                })
        
        # Sleep reminders - based on sleep goal and patterns
        if preferences.reminder_sleep:
            sleep_reminder_time = self._calculate_sleep_reminder_time(preferences, health_data)
            if sleep_reminder_time:
                reminders.append({
                    'type': 'sleep',
                    'time': sleep_reminder_time,
                    'message': self._generate_sleep_reminder_message(preferences),
                    'priority': 4
                })
        
        return reminders
    
    def _calculate_optimal_water_times(self, events: List, preferences: Dict) -> List[datetime]:
        """Calculate optimal times for water reminders"""
        water_times = []
        current_time = datetime.now()
        
        # Generate reminders every 2 hours during active day
        for hour in range(8, 20, 2):
            reminder_time = current_time.replace(hour=hour, minute=0, second=0, microsecond=0)
            
            # Skip if during quiet hours
            if self._is_quiet_hours(reminder_time.time(), preferences):
                continue
            
            # Skip if user has events at this time
            if not self._is_time_available(reminder_time, events):
                # Try 30 minutes later
                alternative_time = reminder_time + timedelta(minutes=30)
                if self._is_time_available(alternative_time, events):
                    water_times.append(alternative_time)
            else:
                water_times.append(reminder_time)
        
        return water_times[:6]  # Limit to 6 reminders per day
    
    def _is_quiet_hours(self, check_time: time, preferences: Dict) -> bool:
        """Check if time falls within user's quiet hours"""
        if not preferences or not preferences.quiet_hours_start or not preferences.quiet_hours_end:
            return False
        
        start = preferences.quiet_hours_start
        end = preferences.quiet_hours_end
        
        # Handle overnight quiet hours (e.g., 22:00 to 08:00)
        if start > end:
            return check_time >= start or check_time <= end
        else:
            return start <= check_time <= end
    
    def _find_optimal_exercise_time(self, events: List, health_data: Dict) -> Optional[datetime]:
        """Find optimal time for exercise based on schedule and energy patterns"""
        current_time = datetime.now()
        
        # Preferred exercise times (morning or evening)
        preferred_times = [
            current_time.replace(hour=7, minute=0, second=0, microsecond=0),   # 7 AM
            current_time.replace(hour=18, minute=0, second=0, microsecond=0),  # 6 PM
            current_time.replace(hour=19, minute=0, second=0, microsecond=0),  # 7 PM
        ]
        
        for exercise_time in preferred_times:
            if self._is_time_available(exercise_time, events):
                return exercise_time
        
        return None
    
    def _calculate_sleep_reminder_time(self, preferences: Dict, health_data: Dict) -> Optional[datetime]:
        """Calculate when to remind user about sleep"""
        if not preferences or not preferences.daily_sleep_goal:
            return None
        
        # Remind 30 minutes before target bedtime
        target_sleep_time = time(22, 0)  # Default 10 PM
        reminder_time = datetime.now().replace(
            hour=target_sleep_time.hour,
            minute=target_sleep_time.minute,
            second=0,
            microsecond=0
        ) - timedelta(minutes=30)
        
        return reminder_time
    
    def _generate_water_reminder_message(self, health_data: Dict) -> str:
        """Generate personalized water reminder message"""
        messages = [
            "💧 Time to hydrate! Your body needs water to function optimally.",
            "🥤 Drink a glass of water to stay energized and focused.",
            "💦 Hydration check! Keep up your healthy water intake.",
        ]
        
        # Personalize based on health data if available
        if health_data and health_data.get('recent_activity_level', 0) > 7:
            return "💧 High activity detected! Extra hydration needed to recover properly."
        
        return messages[datetime.now().second % len(messages)]
    
    def _generate_exercise_reminder_message(self, health_data: Dict) -> str:
        """Generate personalized exercise reminder message"""
        messages = [
            "🏃‍♂️ Ready for some movement? Even 15 minutes makes a difference!",
            "💪 Your body is designed to move. Time for some activity!",
            "🚶‍♀️ Take a break and get your blood flowing with some exercise.",
        ]
        
        if health_data and health_data.get('recent_steps', 0) < 5000:
            return "👟 Low step count today. How about a quick walk or some stretching?"
        
        return messages[datetime.now().second % len(messages)]
    
    def _generate_sleep_reminder_message(self, preferences: Dict) -> str:
        """Generate personalized sleep reminder message"""
        sleep_goal = preferences.daily_sleep_goal if preferences else 8
        
        return f"😴 Wind down time! Aim for {sleep_goal} hours of quality sleep tonight. Consider dimming lights and avoiding screens."

# Remove module-level instantiation to avoid app context errors
# Use get_calendar_service() from ai_services/__init__.py instead
