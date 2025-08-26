"""
Calendar and Schedule Optimization Service with OR-Tools

This module handles:
- Google Calendar integration  
- AI-powered schedule optimization using OR-Tools
- Event management with user preferences
- Smart reminder scheduling based on health data
- Context-aware hydration and exercise reminders

Based on research requirements for calendar AI optimization with user control.
Implements OR-Tools for constraint-based schedule optimization.
"""

import os
import json
import logging
import socket
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# Google Services
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2
from google.auth.transport import urllib3 as google_urllib3
from google.auth.transport.urllib3 import Request as UrllibRequest

# AI and ML
import google.generativeai as genai
from google.genai import types
import spacy

# Optimization
from ortools.sat.python import cp_model

# Flask
from flask import current_app, session

logger = logging.getLogger(__name__)

# Simple timeout wrapper for API calls on Windows
def execute_with_timeout(func, timeout_seconds=30):
    """Execute a function with timeout using threading - improved version"""
    import threading
    import queue
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func()
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        logger.warning(f"Operation timed out after {timeout_seconds} seconds")
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    if not exception_queue.empty():
        exception = exception_queue.get()
        logger.error(f"Exception in threaded operation: {exception}")
        raise exception
    
    if not result_queue.empty():
        return result_queue.get()
    
    raise Exception("Unknown error occurred during execution")

class ScheduleOptimizer:
    """
    OR-Tools based schedule optimizer for calendar events
    Implements constraint-based optimization for health-aware scheduling
    """
    
    def __init__(self):
        """Initialize OR-Tools optimizer"""
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
    
    def optimize_schedule(self, events: List[Dict], health_data: Dict, 
                         preferences: Dict) -> Dict[str, Any]:
        """
        Optimize schedule using OR-Tools constraint programming
        
        Args:
            events: List of calendar events to optimize
            health_data: User's health data for context-aware scheduling
            preferences: User preferences and constraints
            
        Returns:
            Optimized schedule with health-aware recommendations
        """
        try:
            # Reset model
            self.model = cp_model.CpModel()
            
            # Time slots (15-minute intervals from 6 AM to 11 PM)
            start_hour = 6
            end_hour = 23
            slot_duration = 15  # minutes
            total_slots = (end_hour - start_hour) * 60 // slot_duration
            
            # Categorize events
            fixed_events = [e for e in events if e.get('is_fixed', False)]
            flexible_events = [e for e in events if not e.get('is_fixed', False)]
            
            # Create variables for flexible events
            event_vars = {}
            for i, event in enumerate(flexible_events):
                # Start time variable (slot index)
                duration_slots = event.get('duration_minutes', 60) // slot_duration
                max_start_slot = total_slots - duration_slots
                
                start_var = self.model.NewIntVar(0, max_start_slot, f'event_{i}_start')
                event_vars[i] = {
                    'start': start_var,
                    'duration': duration_slots,
                    'event': event
                }
            
            # Add constraints
            self._add_time_constraints(event_vars, total_slots)
            self._add_health_constraints(event_vars, health_data)
            self._add_preference_constraints(event_vars, preferences)
            self._add_fixed_event_constraints(event_vars, fixed_events, start_hour, slot_duration)
            
            # Create objective function
            self._create_optimization_objective(event_vars, health_data)
            
            # Solve
            status = self.solver.Solve(self.model)
            
            if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
                return self._extract_solution(event_vars, start_hour, slot_duration, health_data)
            else:
                logger.warning("Schedule optimization failed, returning original schedule")
                return self._fallback_schedule(events, health_data)
                
        except Exception as e:
            logger.error(f"Error in schedule optimization: {e}")
            return self._fallback_schedule(events, health_data)
    
    def _add_time_constraints(self, event_vars: Dict, total_slots: int):
        """Add basic time constraints"""
        # No overlapping events
        for i in range(len(event_vars)):
            for j in range(i + 1, len(event_vars)):
                var_i = event_vars[i]
                var_j = event_vars[j]
                
                # Event i ends before event j starts OR event j ends before event i starts
                self.model.AddBoolOr([
                    var_i['start'] + var_i['duration'] <= var_j['start'],
                    var_j['start'] + var_j['duration'] <= var_i['start']
                ])
    
    def _add_health_constraints(self, event_vars: Dict, health_data: Dict):
        """Add health-based constraints"""
        try:
            # Extract health insights
            health_metrics = health_data.get('health_metrics', {})
            calendar_suggestions = health_data.get('calendar_suggestions', {})
            
            # Exercise timing constraints
            optimal_exercise_times = calendar_suggestions.get('optimal_exercise_times', ['morning'])
            exercise_events = [i for i, var in event_vars.items() 
                             if 'exercise' in var['event'].get('type', '').lower()]
            
            for event_idx in exercise_events:
                if 'morning' in optimal_exercise_times:
                    # Morning: 6-10 AM (slots 0-96)
                    self.model.Add(event_vars[event_idx]['start'] <= 96)
                elif 'evening' in optimal_exercise_times:
                    # Evening: 6-8 PM (slots 480-576)
                    self.model.Add(event_vars[event_idx]['start'] >= 480)
                    self.model.Add(event_vars[event_idx]['start'] <= 576)
            
        except Exception as e:
            logger.warning(f"Could not apply health constraints: {e}")
    
    def _add_preference_constraints(self, event_vars: Dict, preferences: Dict):
        """Add user preference constraints"""
        try:
            # Work hours constraint
            work_start = preferences.get('work_start_hour', 9)
            work_end = preferences.get('work_end_hour', 17)
            
            work_events = [i for i, var in event_vars.items() 
                          if var['event'].get('type') == 'work']
            
            for event_idx in work_events:
                # Convert to slot indices
                start_slot = (work_start - 6) * 4  # 6 AM is slot 0, 15-min intervals
                end_slot = (work_end - 6) * 4
                
                self.model.Add(event_vars[event_idx]['start'] >= start_slot)
                self.model.Add(event_vars[event_idx]['start'] + 
                             event_vars[event_idx]['duration'] <= end_slot)
            
        except Exception as e:
            logger.warning(f"Could not apply preference constraints: {e}")
    
    def _add_fixed_event_constraints(self, event_vars: Dict, fixed_events: List[Dict], 
                                   start_hour: int, slot_duration: int):
        """Add constraints for fixed events that cannot be moved"""
        for fixed_event in fixed_events:
            try:
                # Convert fixed event time to slots
                event_start = datetime.fromisoformat(fixed_event['start_time'])
                event_minutes = event_start.hour * 60 + event_start.minute
                base_minutes = start_hour * 60
                
                if event_minutes >= base_minutes:
                    fixed_start_slot = (event_minutes - base_minutes) // slot_duration
                    fixed_duration_slots = fixed_event.get('duration_minutes', 60) // slot_duration
                    
                    # Ensure no flexible events overlap with fixed events
                    for var in event_vars.values():
                        self.model.AddBoolOr([
                            var['start'] + var['duration'] <= fixed_start_slot,
                            var['start'] >= fixed_start_slot + fixed_duration_slots
                        ])
                        
            except Exception as e:
                logger.warning(f"Could not add constraint for fixed event: {e}")
    
    def _create_optimization_objective(self, event_vars: Dict, health_data: Dict):
        """Create objective function to maximize schedule quality"""
        try:
            objective_terms = []
            
            # Prefer spacing between events (avoid back-to-back scheduling)
            for i in range(len(event_vars)):
                for j in range(i + 1, len(event_vars)):
                    var_i = event_vars[i]
                    var_j = event_vars[j]
                    
                    # Bonus for having gaps between events
                    gap_var = self.model.NewIntVar(0, 1000, f'gap_{i}_{j}')
                    
                    # Gap = |end_i - start_j| or |end_j - start_i|
                    self.model.AddMaxEquality(gap_var, [
                        var_j['start'] - (var_i['start'] + var_i['duration']),
                        var_i['start'] - (var_j['start'] + var_j['duration'])
                    ])
                    
                    objective_terms.append(gap_var)
            
            # Maximize the objective (better spacing)
            if objective_terms:
                self.model.Maximize(sum(objective_terms))
                
        except Exception as e:
            logger.warning(f"Could not create optimization objective: {e}")
    
    def _extract_solution(self, event_vars: Dict, start_hour: int, 
                         slot_duration: int, health_data: Dict) -> Dict:
        """Extract optimized schedule from OR-Tools solution"""
        optimized_events = []
        
        for var_data in event_vars.values():
            start_slot = self.solver.Value(var_data['start'])
            start_minutes = start_slot * slot_duration
            start_time = datetime.now().replace(
                hour=start_hour + start_minutes // 60,
                minute=start_minutes % 60,
                second=0,
                microsecond=0
            )
            
            event = var_data['event'].copy()
            event['optimized_start_time'] = start_time.isoformat()
            optimized_events.append(event)
        
        # Add health-aware reminders
        health_reminders = self._generate_health_reminders(health_data, optimized_events)
        
        return {
            'success': True,
            'optimized_events': optimized_events,
            'health_reminders': health_reminders,
            'optimization_score': self.solver.ObjectiveValue() if self.solver.ObjectiveValue() else 0,
            'solver_status': 'optimal' if self.solver.status == cp_model.OPTIMAL else 'feasible'
        }
    
    def _fallback_schedule(self, events: List[Dict], health_data: Dict) -> Dict:
        """Fallback when optimization fails"""
        health_reminders = self._generate_health_reminders(health_data, events)
        
        return {
            'success': False,
            'original_events': events,
            'health_reminders': health_reminders,
            'message': 'Could not optimize schedule, using original times'
        }
    
    def _generate_health_reminders(self, health_data: Dict, events: List[Dict]) -> List[Dict]:
        """Generate context-aware health reminders based on research requirements"""
        reminders = []
        
        try:
            calendar_suggestions = health_data.get('calendar_suggestions', {})
            
            # Hydration reminders (can overlap with other events)
            hydration_times = calendar_suggestions.get('hydration_reminders', 
                                                     ['09:00', '12:00', '15:00', '18:00'])
            for time_str in hydration_times:
                reminders.append({
                    'type': 'hydration',
                    'time': time_str,
                    'message': 'Time to drink water! 💧',
                    'can_overlap': True,  # Important: can overlap with work
                    'priority': 'medium'
                })
            
            # Exercise reminders (specific events, not breaks)
            exercise_data = health_data.get('health_metrics', {}).get('exercise', {})
            if exercise_data:
                optimal_times = calendar_suggestions.get('optimal_exercise_times', ['morning'])
                for time_period in optimal_times:
                    if time_period == 'morning':
                        time_str = '07:00'
                    elif time_period == 'afternoon':
                        time_str = '14:00'
                    else:  # evening
                        time_str = '18:00'
                    
                    reminders.append({
                        'type': 'exercise',
                        'time': time_str,
                        'message': 'Optimal time for exercise based on your health data 🏃‍♂️',
                        'can_overlap': False,
                        'priority': 'high'
                    })
            
            # Sleep optimization reminder
            sleep_schedule = calendar_suggestions.get('sleep_schedule', {})
            if sleep_schedule.get('recommended_bedtime'):
                reminders.append({
                    'type': 'sleep',
                    'time': sleep_schedule['recommended_bedtime'],
                    'message': 'Time to start winding down for optimal sleep 😴',
                    'can_overlap': False,
                    'priority': 'high'
                })
            
        except Exception as e:
            logger.error(f"Error generating health reminders: {e}")
        
        return reminders


class CalendarOptimizationService:
    """
    Enhanced calendar optimization service with Gemini 2.5 Flash and OR-Tools
    Implements research features: schedule optimization, user control, AI recommendations
    """
    
    def __init__(self):
        """Initialize calendar service"""
        self.nlp = None
        self.gemini_model = None
        self.calendar_service = None
        self.optimizer = ScheduleOptimizer()
        self.setup_services()
    
    def setup_services(self):
        """Setup AI and calendar services"""
        try:
            # Setup Gemini 2.5 Flash
            api_key = current_app.config.get('GEMINI_API_KEY')
            if api_key and api_key not in ['demo_api_key', 'your-actual-gemini-api-key-here', 'test_api_key', 'your_gemini_api_key_here']:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini 2.5 Flash initialized for calendar optimization")
            else:
                if api_key in ['demo_api_key', 'your-actual-gemini-api-key-here', 'test_api_key', 'your_gemini_api_key_here']:
                    logger.info("Using demo API key - calendar AI features will use fallback responses")
                else:
                    logger.warning("Gemini API key not found for calendar service")
                self.gemini_model = None
            
            # Setup spaCy for event analysis
            model_name = current_app.config.get('SPACY_MODEL', 'en_core_web_sm')
            try:
                self.nlp = spacy.load(model_name)
                logger.info("spaCy loaded for calendar text analysis")
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not available for event analysis")
        
        except Exception as e:
            logger.error(f"Error setting up calendar services: {e}")
    
    def analyze_event_text(self, event_text: str) -> Dict:
        """
        Analyze event text to extract type and scheduling preferences
        Uses spaCy NLP to understand event context
        """
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
        try:
            if not self.gemini_model:
                return self._basic_schedule_optimization(user_events, health_data, preferences)
            
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
            
            Only suggest realistic, beneficial changes.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                try:
                    # Parse AI response
                    optimization_result = json.loads(response.text)
                    return optimization_result
                except json.JSONDecodeError:
                    logger.error("Failed to parse AI optimization response")
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
            prefs_info = {
                'water_goal': preferences.daily_water_goal,
                'sleep_goal': preferences.daily_sleep_goal,
                'steps_goal': preferences.daily_steps_goal,
                'quiet_hours_start': preferences.quiet_hours_start.strftime('%H:%M'),
                'quiet_hours_end': preferences.quiet_hours_end.strftime('%H:%M'),
                'reminders_enabled': {
                    'water': preferences.reminder_water,
                    'exercise': preferences.reminder_exercise,
                    'sleep': preferences.reminder_sleep
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

# Singleton instance
calendar_service = CalendarOptimizationService()

def get_calendar_service():
    """Get calendar optimization service instance"""
    return calendar_service


class GoogleCalendarService:
    """
    Google Calendar integration service
    Handles OAuth, sync, and calendar operations with Google Calendar API
    """
    
    def __init__(self):
        """Initialize Google Calendar service"""
        self.service = None
        self.credentials = None
        self.setup_credentials()
    
    def setup_credentials(self):
        """Setup Google Calendar API credentials"""
        try:
            self.client_id = current_app.config.get('GOOGLE_CALENDAR_CLIENT_ID')
            self.client_secret = current_app.config.get('GOOGLE_CALENDAR_CLIENT_SECRET')
            
            if not self.client_id or not self.client_secret:
                logger.warning("Google Calendar API credentials not configured")
        except Exception as e:
            logger.error(f"Error setting up Google Calendar credentials: {e}")
    
    def get_authorization_url(self, redirect_uri: str) -> str:
        """Get Google Calendar authorization URL"""
        # Ensure credentials are set up
        if not hasattr(self, 'client_id') or not self.client_id:
            self.setup_credentials()
        
        if not self.client_id or not self.client_secret:
            raise ValueError("Google Calendar API credentials not configured. Please set GOOGLE_CALENDAR_CLIENT_ID and GOOGLE_CALENDAR_CLIENT_SECRET in environment variables.")
        
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                },
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            
            flow.redirect_uri = redirect_uri
            auth_url, _ = flow.authorization_url(prompt='consent')
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Error generating Google Calendar auth URL: {e}")
            raise
    
    def exchange_code_for_token(self, authorization_code: str, redirect_uri: str) -> Dict:
        """Exchange authorization code for tokens"""
        try:
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [redirect_uri]
                    }
                },
                scopes=['https://www.googleapis.com/auth/calendar']
            )
            
            flow.redirect_uri = redirect_uri
            flow.fetch_token(code=authorization_code)
            
            credentials = flow.credentials
            
            return {
                'access_token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
        except Exception as e:
            logger.error(f"Error exchanging code for Google Calendar token: {e}")
            return {}
    
    def sync_events_to_database(self, user_id: int, credentials_dict: Dict, days_ahead: int = 30) -> Dict:
        """Sync Google Calendar events to our database with robust timeout handling"""
        try:
            from app import db
            from app.models import CalendarEvent
            import socket
            from google.auth.transport.requests import Request as GoogleRequest
            import urllib3
            
            logger.info("Starting Google Calendar sync with enhanced timeout handling")
            
            # Build service with proper credentials handling and timeout configuration
            credentials = Credentials.from_authorized_user_info(credentials_dict)
            
            # Refresh credentials if needed
            if credentials.expired and credentials.refresh_token:
                request = GoogleRequest()
                credentials.refresh(request)
                logger.info("Refreshed Google Calendar credentials")
            
            # Create HTTP adapter with timeout settings
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            
            # Method 1: Try with httplib2 with explicit timeout
            try:
                # Create httplib2 Http object with timeout
                http = httplib2.Http(timeout=25)  # 25 second timeout
                http = credentials.authorize(http)
                
                # Build service with the custom HTTP client
                service = build('calendar', 'v3', http=http, cache_discovery=False)
                logger.info("Built Google Calendar service with httplib2 timeout")
                
            except Exception as http_error:
                logger.warning(f"Failed to create service with httplib2: {http_error}")
                # Fallback: Use default method
                service = build('calendar', 'v3', credentials=credentials, cache_discovery=False)
                logger.info("Using fallback Google Calendar service build")
            
            # Calculate date range - reduce to 7 days to minimize data transfer
            start_date = datetime.now()
            end_date = start_date + timedelta(days=min(days_ahead, 7))  # Limit to 7 days
            
            logger.info(f"Fetching events from {start_date.date()} to {end_date.date()}")
            
            # Get events from Google Calendar with aggressive timeout and retry handling
            events_result = None
            max_retries = 3
            retry_delay = 3  # Longer delay between retries
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"Attempting to fetch calendar events (attempt {attempt + 1}/{max_retries})")
                    
                    # Direct API call without threading timeout (let httplib2 handle it)
                    events_result = service.events().list(
                        calendarId='primary',
                        timeMin=start_date.isoformat() + 'Z',
                        timeMax=end_date.isoformat() + 'Z',
                        maxResults=5,  # Very small number to minimize timeout risk
                        singleEvents=True,
                        orderBy='startTime'
                    ).execute()
                    
                    logger.info(f"Successfully fetched {len(events_result.get('items', []))} events")
                    break  # Success, exit retry loop
                    
                except socket.timeout:
                    logger.warning(f"Socket timeout on attempt {attempt + 1}")
                    if attempt == max_retries - 1:  # Last attempt
                        logger.error("Google Calendar API request timed out after all retry attempts")
                        return {
                            'success': False,
                            'error': 'Connection timeout while fetching calendar events. Your internet connection may be slow or Google servers are busy. Please try again later.',
                            'synced_count': 0
                        }
                    time.sleep(retry_delay)
                    
                except httplib2.ServerNotFoundError:
                    logger.error("Server not found error - network connectivity issue")
                    return {
                        'success': False,
                        'error': 'Network connection error. Please check your internet connection.',
                        'synced_count': 0
                    }
                    
                except HttpError as api_error:
                    status_code = api_error.resp.status if hasattr(api_error, 'resp') else 0
                    logger.warning(f"HTTP error {status_code} on attempt {attempt + 1}: {api_error}")
                    
                    if status_code in [429, 500, 502, 503, 504]:  # Rate limit or server errors
                        if attempt == max_retries - 1:  # Last attempt
                            logger.error(f"Google Calendar API HTTP error: {api_error}")
                            return {
                                'success': False,
                                'error': f'Google Calendar service error (HTTP {status_code}). Please try again later.',
                                'synced_count': 0
                            }
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        # Non-retryable error
                        logger.error(f"Non-retryable Google Calendar API HTTP error: {api_error}")
                        return {
                            'success': False,
                            'error': f'Google Calendar API error: {str(api_error)}',
                            'synced_count': 0
                        }
                        
                except Exception as api_error:
                    error_str = str(api_error).lower()
                    logger.error(f"General Google Calendar API error: {api_error}")
                    
                    if any(keyword in error_str for keyword in ['timeout', 'timed out', 'connection']):
                        logger.warning(f"Timeout-related error detected in attempt {attempt + 1}")
                        if attempt == max_retries - 1:  # Last attempt
                            return {
                                'success': False,
                                'error': 'Connection timeout while fetching calendar events. Please check your internet connection and try again.',
                                'synced_count': 0
                            }
                        time.sleep(retry_delay)
                    else:
                        return {
                            'success': False,
                            'error': f'Failed to fetch calendar events: {str(api_error)}',
                            'synced_count': 0
                        }
            
            # Process the events and sync to database
            events = events_result.get('items', [])
            synced_count = 0
            
            logger.info(f"Processing {len(events)} events for database sync")
            
            for google_event in events:
                try:
                    # Parse event data
                    event_start = google_event.get('start', {}).get('dateTime')
                    event_end = google_event.get('end', {}).get('dateTime')
                    
                    if not event_start or not event_end:
                        continue
                    
                    start_time = datetime.fromisoformat(event_start.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(event_end.replace('Z', '+00:00'))
                    
                    # Check if event already exists
                    existing_event = CalendarEvent.query.filter_by(
                        user_id=user_id,
                        google_calendar_id=google_event['id']
                    ).first()
                    
                    if existing_event:
                        # Update existing event
                        existing_event.title = google_event.get('summary', 'Untitled Event')
                        existing_event.description = google_event.get('description', '')
                        existing_event.start_time = start_time
                        existing_event.end_time = end_time
                        existing_event.updated_at = datetime.utcnow()
                    else:
                        # Create new event
                        new_event = CalendarEvent(
                            user_id=user_id,
                            title=google_event.get('summary', 'Untitled Event'),
                            description=google_event.get('description', ''),
                            start_time=start_time,
                            end_time=end_time,
                            google_calendar_id=google_event['id'],
                            is_ai_modifiable=False,  # Google Calendar events are not modifiable by default
                            event_type='work'  # Default type, can be analyzed later
                        )
                        db.session.add(new_event)
                    
                    synced_count += 1
                    
                except Exception as e:
                    logger.error(f"Error processing Google Calendar event: {e}")
                    continue
            
            db.session.commit()
            
            return {
                'success': True,
                'synced_events': synced_count,
                'message': f'Successfully synced {synced_count} events from Google Calendar'
            }
            
        except Exception as e:
            logger.error(f"Error syncing Google Calendar events: {e}")
            db.session.rollback()
            return {
                'success': False,
                'error': str(e)
            }

# Google Calendar service singleton
google_calendar_service = GoogleCalendarService()

def get_google_calendar_service():
    """Get Google Calendar service instance"""
    return google_calendar_service
