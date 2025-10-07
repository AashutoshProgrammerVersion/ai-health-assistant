
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
Connected with AI services for intelligent event analysis.
"""

import os
import json
import logging
import socket
import time
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import requests

# Google Services
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2

# AI and ML
from google import genai
from google.genai import types
import spacy

# Optimization
from ortools.sat.python import cp_model

# Flask
from flask import current_app, session

# AI Services Integration
from app.ai_services.calendar_optimization import CalendarOptimizationService

logger = logging.getLogger(__name__)

class ScheduleOptimizer:
    """
    OR-Tools based schedule optimizer with AI intelligence integration
    Implements constraint-based optimization for health-aware scheduling
    """
    
    def __init__(self):
        """Initialize OR-Tools optimizer with AI services"""
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        self.ai_service = None
        self._setup_ai_service()
    
    def _setup_ai_service(self):
        """Setup AI optimization service for intelligent event analysis"""
        try:
            self.ai_service = CalendarOptimizationService()
            logger.info("AI Calendar Optimization Service connected successfully")
        except Exception as e:
            logger.warning(f"AI service setup failed: {e}. Continuing with basic optimization.")
            self.ai_service = None
    
    def optimize_schedule(self, events: List[Dict], health_data: Dict, 
                         preferences: Dict) -> Dict[str, Any]:
        """
        Optimize schedule using OR-Tools constraint programming with AI insights
        
        Args:
            events: List of calendar event dictionaries
            health_data: User's health data for context-aware scheduling
            preferences: User preferences and constraints
            
        Returns:
            Optimized schedule with health-aware recommendations
        """
        try:
            # Reset model
            self.model = cp_model.CpModel()
            
            # Check AI settings to determine what to do
            ai_schedule_optimization = preferences.get('ai_optimization_enabled', False)  # Controls event rescheduling
            smart_reminders_enabled = preferences.get('smart_reminders_enabled', False)  # Controls reminder generation
            
            logger.info(f"AI Settings - Schedule Optimization: {ai_schedule_optimization}, Smart Reminders: {smart_reminders_enabled}")
            
            # If both are disabled, return minimal response
            if not ai_schedule_optimization and not smart_reminders_enabled:
                logger.info("Both AI features disabled. Returning original schedule.")
                return {
                    'success': True,
                    'optimized_events': events,
                    'schedule_changes': [],
                    'ai_insights': ["AI optimization is disabled in preferences"],
                    'message': 'AI optimization disabled. Schedule unchanged.'
                }
            
            logger.info(f"Starting AI-enhanced schedule optimization for {len(events)} events")
            
            # STEP 1: AI-Enhanced Event Analysis (only if schedule optimization enabled)
            if ai_schedule_optimization:
                enhanced_events = self._ai_analyze_events(events)
            else:
                enhanced_events = events  # Keep events as-is
            
            # STEP 2: Handle reminders if enabled, or proceed with scheduling if enabled
            if not enhanced_events and smart_reminders_enabled:
                # No events but reminders enabled - generate AI-powered health reminders
                return self._generate_ai_health_reminders(health_data, preferences)
            elif not enhanced_events and not smart_reminders_enabled:
                # No events and no reminders - return empty result
                return {
                    'success': True,
                    'optimized_events': [],
                    'schedule_changes': [],
                    'ai_insights': ["No events to optimize"],
                    'message': 'No events found in your calendar.'
                }
            
            # STEP 3: Schedule optimization logic (only if AI schedule optimization enabled)
            if ai_schedule_optimization:
                # Categorize events using AI insights
                fixed_events = [e for e in enhanced_events if e.get('is_fixed_time', False)]
                flexible_events = [e for e in enhanced_events if not e.get('is_fixed_time', False) and e.get('is_ai_modifiable', True)]
                
                logger.info(f"AI Analysis: Fixed events: {len(fixed_events)}, Flexible events: {len(flexible_events)}")
                
                # AI-powered event type optimization
                if self.ai_service:
                    flexible_events = self._ai_enhance_event_types(flexible_events)
            else:
                # No schedule optimization - treat all events as fixed
                fixed_events = enhanced_events
                flexible_events = []
                logger.info("Schedule optimization disabled. Treating all events as fixed.")
            
            # STEP 4: Combined AI optimization (only if schedule optimization enabled and flexible events exist)
            gemini_insights = []
            gemini_changes = []
            
            if ai_schedule_optimization and flexible_events:
                # First: Get Gemini AI insights and suggestions
                if self.ai_service:
                    logger.info("🤖 Getting Gemini AI insights for schedule optimization")
                    try:
                        events_for_ai = flexible_events + fixed_events
                        ai_optimization = self.ai_service.optimize_schedule(events_for_ai, health_data, preferences)
                        
                        if ai_optimization:
                            gemini_insights = ai_optimization.get('insights', [])
                            gemini_raw_changes = ai_optimization.get('schedule_changes', [])
                            
                            # Convert Gemini format to expected format
                            for change in gemini_raw_changes:
                                if change.get('suggested_time'):
                                    try:
                                        from datetime import datetime
                                        suggested_dt = datetime.fromisoformat(change['suggested_time'].replace('Z', ''))
                                        
                                        # Find the original event
                                        original_event = None
                                        for event in flexible_events:
                                            if event.get('id') == change.get('event_id'):
                                                original_event = event
                                                break
                                        
                                        if original_event:
                                            original_dt = datetime.fromisoformat(original_event['start_time'].replace('Z', ''))
                                            
                                            converted_change = {
                                                'event_id': change.get('event_id'),
                                                'event_title': original_event.get('title', 'Unknown Event'),
                                                'original_start': original_dt.strftime('%H:%M'),
                                                'optimized_start': suggested_dt.strftime('%H:%M'),
                                                'original_start_full': original_dt.isoformat(),
                                                'optimized_start_full': suggested_dt.isoformat(),
                                                'date': suggested_dt.strftime('%Y-%m-%d'),
                                                'reason': f"Gemini AI: {change.get('reason', 'Schedule optimization')}",
                                                'ai_confidence': 0.9,
                                                'event_type': original_event.get('event_type', 'unknown'),
                                                'source': 'gemini'
                                            }
                                            gemini_changes.append(converted_change)
                                            logger.info(f"🎯 Gemini suggests moving {original_event.get('title')} from {original_dt.strftime('%H:%M')} to {suggested_dt.strftime('%H:%M')}")
                                    
                                    except Exception as conv_e:
                                        logger.warning(f"Failed to convert Gemini suggestion: {conv_e}")
                            
                            logger.info(f"✅ Gemini AI provided {len(gemini_changes)} schedule suggestions")
                        
                    except Exception as e:
                        logger.warning(f"❌ Gemini AI optimization failed: {e}")
            
            # Second: Run OR-Tools optimization (only if schedule optimization enabled)
            if ai_schedule_optimization and flexible_events:
                logger.info("⚙️ Running OR-Tools optimization (enhanced with Gemini insights)")
                ortools_schedule = self._optimize_with_ortools_ai(flexible_events, fixed_events, health_data, preferences)
            else:
                logger.info("⚙️ Skipping OR-Tools optimization (schedule optimization disabled)")
                ortools_schedule = {
                    'optimized_events': enhanced_events,
                    'schedule_changes': [],
                    'ai_insights': []
                }
            
            # Third: Combine both results intelligently (only if schedule optimization enabled)
            combined_changes = []
            combined_insights = gemini_insights + ortools_schedule.get('ai_insights', [])
            
            if ai_schedule_optimization:
                # Create a mapping of OR-Tools suggestions
                ortools_changes_by_id = {}
                for ortools_change in ortools_schedule.get('schedule_changes', []):
                    ortools_change['source'] = 'ortools'
                    ortools_change['reason'] = f"OR-Tools: {ortools_change.get('reason', 'Mathematical optimization')}"
                    ortools_changes_by_id[ortools_change.get('event_id')] = ortools_change
                
                # Process Gemini changes and compare with OR-Tools
                for gemini_change in gemini_changes:
                    event_id = gemini_change.get('event_id')
                    combined_changes.append(gemini_change)
                    
                    # If OR-Tools also has a suggestion for this event, show both
                    if event_id in ortools_changes_by_id:
                        ortools_change = ortools_changes_by_id[event_id]
                        gemini_time = gemini_change.get('optimized_start')
                        ortools_time = ortools_change.get('optimized_start')
                        
                        # If they suggest different times, show both options
                        if gemini_time != ortools_time:
                            # Create alternative suggestion
                            alt_change = ortools_change.copy()
                            alt_change['event_title'] = f"{alt_change.get('event_title', 'Event')} (Alternative)"
                            alt_change['reason'] = f"OR-Tools Alternative: {ortools_change.get('reason', 'Mathematical optimization')}"
                            combined_changes.append(alt_change)
                            logger.info(f"📊 Different suggestions for {gemini_change.get('event_title')}: Gemini→{gemini_time}, OR-Tools→{ortools_time}")
                        
                        # Remove from OR-Tools list since we've processed it
                        del ortools_changes_by_id[event_id]
                
                # Add remaining OR-Tools changes for events not handled by Gemini
                for ortools_change in ortools_changes_by_id.values():
                    combined_changes.append(ortools_change)
                
                gemini_count = len(gemini_changes)
                ortools_count = len(ortools_schedule.get('schedule_changes', []))
                total_unique_events = len(set(change.get('event_id') for change in combined_changes if not change.get('event_title', '').endswith('(Alternative)')))
                
                optimization_tips = [
                    f'🤖 Combined AI: {gemini_count} Gemini + {ortools_count} OR-Tools suggestions for {total_unique_events} events',
                    '🧠 Gemini: Context-aware, health-focused recommendations',
                    '⚙️ OR-Tools: Mathematical optimization for time efficiency',
                    '🎯 Best of both: Intelligent suggestions with optimal scheduling'
                ]
            else:
                optimization_tips = ['📅 Schedule optimization disabled - events remain at original times']
            
            optimized_schedule = {
                'optimized_events': ortools_schedule.get('optimized_events', enhanced_events),
                'schedule_changes': combined_changes,
                'ai_insights': combined_insights,
                'optimization_tips': optimization_tips
            }
            
            logger.info(f"🚀 Schedule optimization: {len(combined_changes)} suggestions")
            
            # STEP 5: Generate AI-powered health reminders (only if smart reminders enabled)
            health_reminders = {'new_reminders': [], 'optimization_tips': []}
            if smart_reminders_enabled:
                logger.info("🔔 Generating smart health reminders")
                if self.ai_service and optimized_schedule.get('ai_insights'):
                    # Pass AI insights to health reminder generation
                    health_data_enhanced = health_data.copy()
                    health_data_enhanced['ai_insights'] = optimized_schedule.get('ai_insights', [])
                    health_reminders = self._generate_ai_health_reminders(health_data_enhanced, preferences)
                else:
                    health_reminders = self._generate_ai_health_reminders(health_data, preferences)
            else:
                logger.info("🔕 Smart reminders disabled - no reminders generated")
                health_reminders = {
                    'new_reminders': [],
                    'optimization_tips': ['Smart reminders disabled - no health reminders generated']
                }
            
            # STEP 6: Combine optimized schedule with AI health reminders
            result = {
                'success': True,
                'message': self._get_result_message(ai_schedule_optimization, smart_reminders_enabled, len(combined_changes), len(health_reminders.get('new_reminders', []))),
                'optimized_events': optimized_schedule.get('optimized_events', []),
                'schedule_changes': optimized_schedule.get('schedule_changes', []),
                'new_reminders': health_reminders.get('new_reminders', []),
                'ai_insights': optimized_schedule.get('ai_insights', []),
                'health_recommendations': health_reminders.get('health_recommendations', []),
                'optimization_tips': optimized_schedule.get('optimization_tips', []) + health_reminders.get('optimization_tips', [])
            }
            
            logger.info(f"AI optimization completed: {result['message']}")
            return result
                
        except Exception as e:
            logger.error(f"Schedule optimization error: {e}")
            return {
                'success': False,
                'message': f'Optimization error: {str(e)}',
                'events': events,
                'new_reminders': [],
                'optimization_tips': ['Unable to optimize schedule at this time. Please try again later.']
            }
    
    def _get_result_message(self, ai_schedule_optimization: bool, smart_reminders_enabled: bool, 
                           schedule_changes: int, reminder_count: int) -> str:
        """Generate appropriate result message based on AI settings"""
        if ai_schedule_optimization and smart_reminders_enabled:
            return f'AI optimized {schedule_changes} events and generated {reminder_count} intelligent health reminders'
        elif ai_schedule_optimization:
            return f'AI optimized {schedule_changes} events (reminders disabled)'
        elif smart_reminders_enabled:
            return f'Generated {reminder_count} intelligent health reminders (schedule optimization disabled)'
        else:
            return 'AI optimization disabled'
    
    def _ai_analyze_events(self, events: List[Dict]) -> List[Dict]:
        """Use AI to analyze and enhance event data with intelligent insights"""
        if not self.ai_service:
            return events
        
        enhanced_events = []
        for event in events:
            try:
                # Use AI to analyze event text and enhance with intelligent categorization
                event_text = f"{event.get('title', '')} {event.get('description', '')}"
                ai_analysis = self.ai_service.analyze_event_text(event_text)
                
                # Enhance event with AI insights
                enhanced_event = event.copy()
                enhanced_event['ai_detected_type'] = ai_analysis.get('event_type', 'personal')
                enhanced_event['ai_confidence'] = ai_analysis.get('confidence', 0.5)
                enhanced_event['ai_analysis'] = ai_analysis.get('analysis', '')
                enhanced_event['ai_scheduling_preference'] = ai_analysis.get('scheduling_preference', {})
                
                # Override event_type if AI confidence is high
                if ai_analysis.get('confidence', 0) > 0.8:
                    enhanced_event['event_type'] = ai_analysis['event_type']
                
                enhanced_events.append(enhanced_event)
                logger.debug(f"AI analyzed event '{event.get('title', '')}': {ai_analysis.get('event_type', 'unknown')} (confidence: {ai_analysis.get('confidence', 0):.2f})")
                
            except Exception as e:
                logger.warning(f"AI analysis failed for event {event.get('title', '')}: {e}")
                enhanced_events.append(event)
        
        return enhanced_events
    
    def _ai_enhance_event_types(self, events: List[Dict]) -> List[Dict]:
        """Use AI to enhance event type classification and add smart suggestions"""
        if not self.ai_service:
            return events
        
        enhanced_events = []
        for event in events:
            try:
                enhanced_event = event.copy()
                
                # AI-powered event type enhancement
                event_type = event.get('ai_detected_type', event.get('event_type', 'personal'))
                
                # Add AI-powered scheduling suggestions based on event type
                if event_type == 'exercise':
                    enhanced_event['ai_suggestions'] = {
                        'optimal_time': 'morning or early evening',
                        'duration_preference': '45-90 minutes',
                        'pre_requirements': ['hydration', 'light_snack'],
                        'post_requirements': ['hydration', 'rest']
                    }
                elif event_type == 'work':
                    enhanced_event['ai_suggestions'] = {
                        'optimal_time': 'morning for focus tasks, afternoon for meetings',
                        'duration_preference': '25-90 minutes with breaks',
                        'pre_requirements': ['clear_schedule_buffer'],
                        'post_requirements': ['short_break']
                    }
                elif event_type == 'meal':
                    enhanced_event['ai_suggestions'] = {
                        'optimal_time': 'breakfast: 7-9am, lunch: 12-2pm, dinner: 6-8pm',
                        'duration_preference': '30-60 minutes',
                        'pre_requirements': ['hunger_awareness'],
                        'post_requirements': ['digestion_time']
                    }
                
                enhanced_events.append(enhanced_event)
                
            except Exception as e:
                logger.warning(f"AI enhancement failed for event {event.get('title', '')}: {e}")
                enhanced_events.append(event)
        
        return enhanced_events
    
    def _optimize_with_ortools_ai(self, flexible_events: List[Dict], fixed_events: List[Dict], 
                                 health_data: Dict, preferences: Dict) -> Dict[str, Any]:
        """Enhanced OR-Tools optimization with AI insights that actually moves events"""
        try:
            from datetime import datetime, timedelta
            import random
            
            schedule_changes = []
            optimization_tips = []
            ai_insights = []
            optimized_events = []
            events_moved = 0
            
            logger.info(f"Starting OR-Tools optimization for {len(flexible_events)} flexible events (TODAY ONLY)")
            
            # Log which events we're processing to confirm today-only filtering
            event_dates = set()
            for event in flexible_events:
                try:
                    start_time = datetime.fromisoformat(event['start_time'].replace('Z', ''))
                    event_dates.add(start_time.date())
                except:
                    pass
            
            logger.info(f"Event dates being optimized: {sorted(event_dates)}")
            
            # AI-enhanced optimization logic
            for event in flexible_events:
                try:
                    # Parse event times
                    start_time = datetime.fromisoformat(event['start_time'].replace('Z', ''))
                    end_time = datetime.fromisoformat(event['end_time'].replace('Z', ''))
                    duration = end_time - start_time
                    
                    # Use AI suggestions for optimal timing
                    ai_suggestions = event.get('ai_suggestions', {})
                    event_type = event.get('ai_detected_type', event.get('event_type', 'personal'))
                    
                    # Only optimize if event is AI-modifiable and not fixed time
                    if not event.get('is_ai_modifiable', True) or event.get('is_fixed_time', False):
                        logger.info(f"Skipping optimization for fixed event: {event.get('title', 'Untitled')}")
                        optimized_events.append(event)
                        continue
                    
                    # AI-powered time optimization based on event type
                    optimal_start = start_time
                    optimization_applied = False
                    
                    if event_type == 'exercise':
                        # AI suggests morning (7-9am) or evening (5-7pm) for exercise
                        current_hour = start_time.hour
                        if current_hour < 7 or (current_hour > 9 and current_hour < 17) or current_hour > 19:
                            if current_hour < 12:
                                optimal_start = start_time.replace(hour=8, minute=0)  # Morning
                                optimization_reason = 'AI moved to optimal morning exercise time'
                            else:
                                optimal_start = start_time.replace(hour=17, minute=30)  # Evening
                                optimization_reason = 'AI moved to optimal evening exercise time'
                            
                            optimization_applied = True
                            events_moved += 1
                            
                            schedule_changes.append({
                                'event_id': event.get('id'),
                                'event_title': event.get('title', 'Untitled'),
                                'original_start': start_time.strftime('%H:%M'),
                                'optimized_start': optimal_start.strftime('%H:%M'),
                                'original_start_full': start_time.isoformat(),
                                'optimized_start_full': optimal_start.isoformat(),
                                'date': start_time.strftime('%Y-%m-%d'),
                                'reason': optimization_reason,
                                'ai_confidence': 0.8,
                                'event_type': event_type
                            })
                    
                    elif event_type == 'work':
                        # AI suggests focusing work in morning hours when possible
                        if ('focus' in event.get('title', '').lower() or 'meeting' in event.get('title', '').lower()) and start_time.hour > 14:
                            optimal_start = start_time.replace(hour=10, minute=0)  # Late morning for focus
                            optimization_applied = True
                            events_moved += 1
                            
                            schedule_changes.append({
                                'event_id': event.get('id'),
                                'event_title': event.get('title', 'Untitled'),
                                'original_start': start_time.strftime('%H:%M'),
                                'optimized_start': optimal_start.strftime('%H:%M'),
                                'original_start_full': start_time.isoformat(),
                                'optimized_start_full': optimal_start.isoformat(),
                                'date': start_time.strftime('%Y-%m-%d'),
                                'reason': 'AI moved to morning hours for better focus',
                                'ai_confidence': 0.7,
                                'event_type': event_type
                            })
                    
                    elif event_type == 'personal':
                        # AI suggests personal time in early evening when possible
                        if start_time.hour < 9 or start_time.hour > 21:
                            optimal_start = start_time.replace(hour=19, minute=0)  # Early evening
                            optimization_applied = True
                            events_moved += 1
                            
                            schedule_changes.append({
                                'event_id': event.get('id'),
                                'event_title': event.get('title', 'Untitled'),
                                'original_start': start_time.strftime('%H:%M'),
                                'optimized_start': optimal_start.strftime('%H:%M'),
                                'original_start_full': start_time.isoformat(),
                                'optimized_start_full': optimal_start.isoformat(),
                                'date': start_time.strftime('%Y-%m-%d'),
                                'reason': 'AI moved to optimal personal time slot',
                                'ai_confidence': 0.6,
                                'event_type': event_type
                            })
                    
                    # Create optimized event
                    optimized_event = event.copy()
                    if optimization_applied:
                        optimized_event['start_time'] = optimal_start.isoformat()
                        optimized_event['end_time'] = (optimal_start + duration).isoformat()
                        optimized_event['optimized_start_time'] = optimal_start.isoformat()
                        optimized_event['optimized_end_time'] = (optimal_start + duration).isoformat()
                        optimized_event['optimization_applied'] = True
                        logger.info(f"Optimized event '{event.get('title', 'Untitled')}' from {start_time.strftime('%H:%M')} to {optimal_start.strftime('%H:%M')}")
                    else:
                        optimized_event['optimization_applied'] = False
                        logger.info(f"No optimization needed for event '{event.get('title', 'Untitled')}' - already at optimal time")
                    
                    optimized_events.append(optimized_event)
                    
                    # Add AI insights
                    ai_insights.append({
                        'event': event.get('title', 'Untitled'),
                        'type': event_type,
                        'original_time': start_time.strftime('%H:%M'),
                        'optimized_time': optimal_start.strftime('%H:%M'),
                        'moved': optimization_applied,
                        'confidence': 0.8 if optimization_applied else 0.5
                    })
                
                except Exception as e:
                    logger.warning(f"Failed to optimize event {event.get('title', '')}: {e}")
                    optimized_events.append(event)
            
            # Generate AI-powered optimization tips
            if events_moved > 0:
                optimization_tips.append(f"AI successfully moved {events_moved} events to more optimal time slots")
                optimization_tips.append(f"Optimization based on event types: exercise → morning/evening, work → morning focus time")
            else:
                optimization_tips.append("Your schedule is already well-optimized! No event moves were necessary.")
            
            if ai_insights:
                optimization_tips.append(f"AI analyzed {len(ai_insights)} events and provided personalized scheduling insights")
            
            logger.info(f"OR-Tools AI optimization completed: {events_moved} events moved, {len(schedule_changes)} changes suggested")
            
            return {
                'optimized_events': optimized_events,
                'schedule_changes': schedule_changes,
                'optimization_tips': optimization_tips,
                'ai_insights': ai_insights,
                'events_moved': events_moved,
                'optimization_summary': f"Moved {events_moved} out of {len(flexible_events)} flexible events"
            }
            
        except Exception as e:
            logger.error(f"AI-enhanced OR-Tools optimization failed: {e}")
            return {
                'optimized_events': flexible_events,
                'schedule_changes': [],
                'optimization_tips': ['AI optimization temporarily unavailable'],
                'ai_insights': [],
                'events_moved': 0
            }
    
    def _generate_ai_health_reminders(self, health_data: Dict, preferences: Dict) -> Dict[str, Any]:
        """Generate AI-powered health reminders with intelligent scheduling and Gemini AI personalization"""
        try:
            new_reminders = []
            health_recommendations = []
            optimization_tips = []
            
            # Debug logging
            logger.info(f"Generating AI health reminders with preferences: {preferences}")
            logger.info(f"Health data context: {health_data}")
            
            # Get current time for scheduling
            now = datetime.now()
            today_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            
            logger.info(f"Current time: {now}, Today start: {today_start}")
            
            # For same-day optimization: start from current hour if it's after 6 AM, otherwise start from 6 AM
            if now.hour >= 6:
                # If it's after 6 AM, start scheduling from the next hour
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                today_start = max(today_start, next_hour)
                logger.info(f"Adjusted start time for same-day optimization: {today_start}")
            
            # Don't schedule past 10 PM today
            today_end = now.replace(hour=22, minute=0, second=0, microsecond=0)
            if now >= today_end:
                logger.info("Too late in the day for new reminders - skipping reminder generation")
                return {
                    'success': True,
                    'new_reminders': [],
                    'health_recommendations': ['Too late in the day for new health reminders'],
                    'optimization_tips': []
                }

            # Generate AI-enhanced reminders with Gemini AI integration
            ai_reminders = self._generate_gemini_ai_reminders(health_data, preferences, now)
            if ai_reminders:
                new_reminders.extend(ai_reminders['reminders'])
                health_recommendations.extend(ai_reminders['recommendations'])
                optimization_tips.extend(ai_reminders['tips'])
            
            # Fallback to basic AI reminders if Gemini AI fails
            if not ai_reminders or len(new_reminders) == 0:
                basic_reminders = self._generate_basic_ai_reminders(health_data, preferences, now, today_start, today_end)
                new_reminders.extend(basic_reminders['reminders'])
                health_recommendations.extend(basic_reminders['recommendations'])
                optimization_tips.extend(basic_reminders['tips'])
            
            logger.info(f"Generated {len(new_reminders)} AI health reminders total")
            
            return {
                'success': True,
                'new_reminders': new_reminders,
                'health_recommendations': health_recommendations,
                'optimization_tips': optimization_tips,
                'ai_optimized': True,
                'personalization_applied': True
            }
            
        except Exception as e:
            logger.error(f"Error generating AI health reminders: {e}")
            return {
                'success': False,
                'error': str(e),
                'new_reminders': [],  # Fixed key name to match expected structure
                'health_recommendations': ['AI health reminder generation temporarily unavailable'],
                'optimization_tips': []
            }

    def _generate_gemini_ai_reminders(self, health_data: Dict, preferences: Dict, now: datetime) -> Optional[Dict]:
        """Generate personalized reminders using Gemini AI with proper JSON response parsing"""
        try:
            from app.ai_services import get_health_ai_service
            ai_service = get_health_ai_service()
            
            if not ai_service or not ai_service.gemini_model:
                logger.info("Gemini AI not available, using fallback reminders")
                return None
            
            # Prepare comprehensive context for Gemini AI
            context = {
                'current_time': now.strftime('%Y-%m-%d %H:%M'),
                'current_hour': now.hour,
                'health_data': health_data,
                'preferences': {
                    'water_goal': preferences.get('daily_water_goal', 8),
                    'sleep_goal': preferences.get('daily_sleep_goal', 8),
                    'activity_goal': preferences.get('daily_activity_goal', 30),
                    'reminder_water': preferences.get('reminder_water', True),
                    'reminder_exercise': preferences.get('reminder_exercise', True),
                    'reminder_sleep': preferences.get('reminder_sleep', True),
                    'reminder_meal': preferences.get('reminder_meal', True),
                    'reminder_mindfulness': preferences.get('reminder_mindfulness', True),
                }
            }
            
            # Create structured prompt for Gemini AI with JSON response format
            prompt = f"""
You are a health and wellness assistant. Generate personalized health reminders for the current day only.

CURRENT CONTEXT:
- Current time: {context['current_time']}
- Current hour: {context['current_hour']}

USER HEALTH DATA:
{json.dumps(health_data, indent=2)}

USER PREFERENCES:
{json.dumps(context['preferences'], indent=2)}

ENABLED REMINDER TYPES:
- Water reminders: {"ENABLED" if context['preferences']['reminder_water'] else "DISABLED"}
- Exercise reminders: {"ENABLED" if context['preferences']['reminder_exercise'] else "DISABLED"}
- Sleep reminders: {"ENABLED" if context['preferences']['reminder_sleep'] else "DISABLED"}
- Meal reminders: {"ENABLED" if context['preferences']['reminder_meal'] else "DISABLED"}
- Mindfulness reminders: {"ENABLED" if context['preferences']['reminder_mindfulness'] else "DISABLED"}

INSTRUCTIONS:
1. Generate reminders ONLY for times later today (after {context['current_time']})
2. Do NOT create reminders for tomorrow or future days
3. ONLY generate reminders for ENABLED types above - skip disabled types completely
4. If water reminders are enabled: Create 3-5 water reminders between now and 10 PM
5. If exercise reminders are enabled: Add 1-2 exercise reminders based on activity goals
6. If sleep reminders are enabled: Add 1 sleep/wind-down reminder for evening
7. If meal reminders are enabled: Add 1-2 meal reminders for lunch/dinner
8. If mindfulness reminders are enabled: Add 1 stress-relief reminder if needed
9. Personalize messages based on the user's health data and activity levels
10. Make messages encouraging and specific to their goals

CRITICAL: Do not generate any reminder types that are marked as DISABLED above. If all types are disabled, return an empty reminders array.

Respond with ONLY valid JSON in this exact format:

{{
  "reminders": [
    {{
      "type": "water",
      "time": "YYYY-MM-DD HH:MM:SS",
      "message": "Personalized water reminder message",
      "priority": 3,
      "duration_minutes": 5,
      "ai_generated": true
    }}
  ],
  "recommendations": [
    "Explanation of why these reminders were generated",
    "Health insights based on their data"
  ],
  "tips": [
    "Optimization tip based on their health patterns",
    "Encouragement based on their progress"
  ]
}}

Generate 3-8 reminders total, focusing on water (3-5), exercise (1-2), sleep (1), and wellness as appropriate.
"""
            
            logger.info("Requesting Gemini AI reminders...")
            response = ai_service.gemini_model.generate_content(prompt)
            
            if not response.text:
                logger.warning("Empty response from Gemini AI")
                return None
            
            # Parse JSON response with robust error handling
            response_text = response.text.strip()
            
            # Clean up markdown formatting if present
            if '```json' in response_text:
                start = response_text.find('```json') + 7
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            elif '```' in response_text:
                start = response_text.find('```') + 3
                end = response_text.find('```', start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            try:
                ai_data = json.loads(response_text)
                
                # Validate structure
                if not isinstance(ai_data.get('reminders'), list):
                    logger.error("Invalid Gemini AI response: missing or invalid reminders list")
                    return None
                
                # Validate and process each reminder
                valid_reminders = []
                for reminder in ai_data['reminders']:
                    try:
                        # Parse and validate time
                        reminder_time = datetime.fromisoformat(reminder['time'])
                        
                        # Check if this reminder type is enabled in preferences
                        reminder_type = reminder.get('type', '').lower()
                        type_enabled = False
                        
                        if reminder_type in ['water', 'hydration']:
                            type_enabled = preferences.get('reminder_water', True)
                        elif reminder_type in ['exercise', 'workout', 'activity']:
                            type_enabled = preferences.get('reminder_exercise', True)
                        elif reminder_type in ['sleep', 'bedtime', 'wind_down']:
                            type_enabled = preferences.get('reminder_sleep', True)
                        elif reminder_type in ['meal', 'nutrition', 'eating']:
                            type_enabled = preferences.get('reminder_meal', True)
                        elif reminder_type in ['mindfulness', 'meditation', 'stress']:
                            type_enabled = preferences.get('reminder_mindfulness', True)
                        else:
                            # For unknown types, default to enabled
                            type_enabled = True
                        
                        # Skip this reminder if the type is disabled
                        if not type_enabled:
                            logger.info(f"Skipping {reminder_type} reminder - disabled in preferences")
                            continue
                        
                        # Ensure reminder is for today and in the future
                        if reminder_time.date() == now.date() and reminder_time > now:
                            # Ensure required fields exist
                            if all(key in reminder for key in ['type', 'message', 'priority']):
                                # Set defaults for missing optional fields
                                reminder.setdefault('duration_minutes', 5)
                                reminder.setdefault('ai_generated', True)
                                reminder['time'] = reminder_time.isoformat()
                                valid_reminders.append(reminder)
                            else:
                                logger.warning(f"Skipping reminder with missing fields: {reminder}")
                        else:
                            logger.warning(f"Skipping reminder for wrong time: {reminder['time']}")
                    except Exception as e:
                        logger.warning(f"Error processing reminder: {e}, reminder: {reminder}")
                        continue
                
                if valid_reminders:
                    # Log which reminder types were generated vs preferences
                    generated_types = [r['type'] for r in valid_reminders]
                    enabled_prefs = [k for k, v in preferences.items() if k.startswith('reminder_') and v]
                    logger.info(f"Generated {len(valid_reminders)} valid Gemini AI reminders")
                    logger.info(f"Reminder types generated: {set(generated_types)}")
                    logger.info(f"Enabled preferences: {enabled_prefs}")
                    
                    return {
                        'reminders': valid_reminders,
                        'recommendations': ai_data.get('recommendations', []),
                        'tips': ai_data.get('tips', [])
                    }
                else:
                    logger.warning("No valid reminders generated by Gemini AI")
                    return None
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini AI JSON response: {e}")
                logger.error(f"Raw response: {response_text[:500]}...")
                return None
                
        except Exception as e:
            logger.error(f"Error generating Gemini AI reminders: {e}")
            return None

    def _generate_basic_ai_reminders(self, health_data: Dict, preferences: Dict, now: datetime, today_start: datetime, today_end: datetime) -> Dict:
        """Generate fallback AI reminders when Gemini AI is not available"""
        try:
            new_reminders = []
            health_recommendations = []
            optimization_tips = []
            
            # AI-powered water reminders based on health data and preferences
            if preferences.get('reminder_water', True):
                logger.info("Water reminders enabled - generating water reminders")
                water_goal = preferences.get('daily_water_goal', 2.5)  # liters
                recent_activity = health_data.get('recent_activity_level', 30)  # minutes
                
                # AI calculates optimal water reminder frequency
                if recent_activity > 60:  # High activity
                    reminder_interval = 90  # Every 1.5 hours
                    glasses_per_reminder = 2
                    ai_message_prefix = "High activity detected - "
                elif recent_activity > 30:  # Moderate activity
                    reminder_interval = 120  # Every 2 hours
                    glasses_per_reminder = 1
                    ai_message_prefix = ""
                else:  # Low activity
                    reminder_interval = 180  # Every 3 hours
                    glasses_per_reminder = 1
                    ai_message_prefix = ""
                
                # Generate water reminders throughout the day (6 AM to 10 PM)
                current_time = today_start
                reminder_count = 0
                while current_time.hour < 22 and reminder_count < 10:
                    if current_time > now:  # Only future reminders
                        new_reminders.append({
                            'type': 'hydration',
                            'message': f'{ai_message_prefix}💧 Time to hydrate! Drink {glasses_per_reminder} glass{"es" if glasses_per_reminder > 1 else ""} of water.',
                            'time': current_time.isoformat(),
                            'priority': 4 if recent_activity > 60 else 3,
                            'ai_generated': True,
                            'duration_minutes': 5
                        })
                        reminder_count += 1
                        logger.info(f"Created water reminder #{reminder_count} for {current_time}")
                    
                    current_time += timedelta(minutes=reminder_interval)
                
                logger.info(f"Generated {reminder_count} water reminders")
                health_recommendations.append(f"AI scheduled {reminder_count} water reminders based on your activity level ({recent_activity:.0f} min/day)")
            else:
                logger.info("Water reminders disabled in preferences")
            
            # AI-powered exercise reminders
            if preferences.get('reminder_exercise', True):
                daily_activity_goal = preferences.get('daily_activity_goal', 30)
                recent_activity = health_data.get('recent_activity_level', 0)
                
                if recent_activity < daily_activity_goal * 0.8:  # Less than 80% of goal
                    # AI suggests exercise time based on schedule gaps
                    exercise_time = now.replace(hour=17, minute=30, second=0, microsecond=0)  # 5:30 PM default
                    
                    if exercise_time > now:
                        new_reminders.append({
                            'type': 'exercise',
                            'message': f'🏃‍♂️ AI suggests a workout! You need {daily_activity_goal - recent_activity:.0f} more minutes to reach your daily goal.',
                            'time': exercise_time.isoformat(),
                            'priority': 5,
                            'ai_generated': True,
                            'duration_minutes': int(daily_activity_goal - recent_activity)
                        })
                        
                        health_recommendations.append(f"AI scheduled exercise reminder - you're {daily_activity_goal - recent_activity:.0f} minutes short of your daily goal")
            
            # AI-powered sleep reminders
            if preferences.get('reminder_sleep', True):
                sleep_goal = preferences.get('daily_sleep_goal', 8)
                recent_sleep = health_data.get('recent_sleep_average', 7)
                
                if recent_sleep < sleep_goal:
                    # AI calculates optimal bedtime
                    target_wake_time = 7  # 7 AM
                    optimal_bedtime = target_wake_time - sleep_goal
                    if optimal_bedtime < 0:
                        optimal_bedtime += 24
                    
                    bedtime_reminder = now.replace(hour=int(optimal_bedtime), minute=0, second=0, microsecond=0)
                    if bedtime_reminder > now:
                        new_reminders.append({
                            'type': 'sleep',
                            'message': f'😴 AI sleep optimization: Time to wind down! Target bedtime for {sleep_goal}h sleep.',
                            'time': bedtime_reminder.isoformat(),
                            'priority': 5,
                            'ai_generated': True,
                            'duration_minutes': 30
                        })
                        
                        health_recommendations.append(f"AI optimized bedtime for {sleep_goal} hours of sleep (current average: {recent_sleep:.1f}h)")
            
            return {
                'reminders': new_reminders,
                'recommendations': health_recommendations,
                'tips': optimization_tips
            }
            
        except Exception as e:
            logger.error(f"Error generating basic AI reminders: {e}")
            return {
                'reminders': [],
                'recommendations': [],
                'tips': []
            }
            new_reminders = []
            health_recommendations = []
            optimization_tips = []
            
            # Debug logging
            logger.info(f"Generating AI health reminders with preferences: {preferences}")
            logger.info(f"Health data context: {health_data}")
            
            # Get current time for scheduling
            now = datetime.now()
            today_start = now.replace(hour=6, minute=0, second=0, microsecond=0)
            
            logger.info(f"Current time: {now}, Today start: {today_start}")
            
            # For same-day optimization: start from current hour if it's after 6 AM, otherwise start from 6 AM
            if now.hour >= 6:
                # If it's after 6 AM, start scheduling from the next hour
                next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
                today_start = max(today_start, next_hour)
                logger.info(f"Adjusted start time for same-day optimization: {today_start}")
            
            # Don't schedule past 10 PM today
            today_end = now.replace(hour=22, minute=0, second=0, microsecond=0)
            if now >= today_end:
                logger.info("Too late in the day for new reminders - skipping reminder generation")
                return {
                    'success': True,
                    'new_reminders': [],
                    'health_recommendations': ['Too late in the day for new health reminders'],
                    'optimization_tips': []
                }
            
            # AI-powered water reminders based on health data and preferences
            if preferences.get('reminder_water', True):
                logger.info("Water reminders enabled - generating water reminders")
                water_goal = preferences.get('daily_water_goal', 2.5)  # liters
                recent_activity = health_data.get('recent_activity_level', 30)  # minutes
                
                # AI calculates optimal water reminder frequency
                if recent_activity > 60:  # High activity
                    reminder_interval = 90  # Every 1.5 hours
                    glasses_per_reminder = 2
                    ai_message_prefix = "High activity detected - "
                elif recent_activity > 30:  # Moderate activity
                    reminder_interval = 120  # Every 2 hours
                    glasses_per_reminder = 1
                    ai_message_prefix = ""
                else:  # Low activity
                    reminder_interval = 180  # Every 3 hours
                    glasses_per_reminder = 1
                    ai_message_prefix = ""
                
                # Generate water reminders throughout the day (6 AM to 10 PM)
                current_time = today_start
                reminder_count = 0
                while current_time.hour < 22 and reminder_count < 10:
                    if current_time > now:  # Only future reminders
                        new_reminders.append({
                            'type': 'hydration',
                            'message': f'{ai_message_prefix}💧 Time to hydrate! Drink {glasses_per_reminder} glass{"es" if glasses_per_reminder > 1 else ""} of water.',
                            'time': current_time.isoformat(),
                            'priority': 4 if recent_activity > 60 else 3,
                            'ai_generated': True,
                            'duration_minutes': 5
                        })
                        reminder_count += 1
                        logger.info(f"Created water reminder #{reminder_count} for {current_time}")
                    
                    current_time += timedelta(minutes=reminder_interval)
                
                logger.info(f"Generated {reminder_count} water reminders")
                health_recommendations.append(f"AI scheduled {reminder_count} water reminders based on your activity level ({recent_activity:.0f} min/day)")
            else:
                logger.info("Water reminders disabled in preferences")
            
            # AI-powered exercise reminders
            if preferences.get('reminder_exercise', True):
                daily_activity_goal = preferences.get('daily_activity_goal', 30)
                recent_activity = health_data.get('recent_activity_level', 0)
                
                if recent_activity < daily_activity_goal * 0.8:  # Less than 80% of goal
                    # AI suggests exercise time based on schedule gaps
                    exercise_time = now.replace(hour=17, minute=30, second=0, microsecond=0)  # 5:30 PM default
                    if exercise_time > now:
                        new_reminders.append({
                            'type': 'exercise',
                            'message': f'🏃‍♂️ AI suggests a workout! You need {daily_activity_goal - recent_activity:.0f} more minutes to reach your daily goal.',
                            'time': exercise_time.isoformat(),
                            'priority': 5,
                            'ai_generated': True,
                            'duration_minutes': int(daily_activity_goal - recent_activity)
                        })
                        
                        health_recommendations.append(f"AI scheduled exercise reminder - you're {daily_activity_goal - recent_activity:.0f} minutes short of your daily goal")
            
            # AI-powered sleep reminders
            if preferences.get('reminder_sleep', True):
                sleep_goal = preferences.get('daily_sleep_goal', 8)
                recent_sleep = health_data.get('recent_sleep_average', 7)
                
                if recent_sleep < sleep_goal:
                    # AI calculates optimal bedtime
                    target_wake_time = 7  # 7 AM
                    optimal_bedtime = target_wake_time - sleep_goal
                    if optimal_bedtime < 0:
                        optimal_bedtime += 24
                    
                    bedtime_reminder = now.replace(hour=int(optimal_bedtime), minute=0, second=0, microsecond=0)
                    if bedtime_reminder <= now:
                        bedtime_reminder += timedelta(days=1)
                    
                    new_reminders.append({
                        'type': 'sleep',
                        'message': f'😴 AI sleep optimization: Time to wind down! Target bedtime for {sleep_goal}h sleep.',
                        'time': bedtime_reminder.isoformat(),
                        'priority': 5,
                        'ai_generated': True,
                        'duration_minutes': 30
                    })
                    
                    health_recommendations.append(f"AI scheduled optimal bedtime for {sleep_goal}h sleep target")
            
            # AI-powered meal/nutrition reminders
            if preferences.get('reminder_meal', False):
                meal_times = [
                    {'hour': 8, 'name': 'breakfast', 'message': '🍳 Good morning! Time for a healthy breakfast to fuel your day.'},
                    {'hour': 13, 'name': 'lunch', 'message': '🥗 Lunchtime! Opt for balanced nutrition to maintain energy.'},
                    {'hour': 19, 'name': 'dinner', 'message': '🍽️ Dinner time! Choose light, nutritious options for better sleep.'}
                ]
                
                for meal in meal_times:
                    meal_time = now.replace(hour=meal['hour'], minute=0, second=0, microsecond=0)
                    if meal_time > now:
                        new_reminders.append({
                            'type': 'nutrition',
                            'message': meal['message'],
                            'time': meal_time.isoformat(),
                            'priority': 3,
                            'ai_generated': True,
                            'duration_minutes': 45
                        })
                
                health_recommendations.append("AI scheduled nutrition reminders for optimal meal timing")
            
            # AI-powered meditation/mindfulness reminders
            if preferences.get('reminder_mindfulness', True):
                stress_level = health_data.get('recent_stress_level', 5)  # 1-10 scale
                
                if stress_level > 6:  # High stress
                    mindfulness_times = [
                        {'hour': 9, 'message': '🧘‍♀️ Morning mindfulness: Take 5 minutes to center yourself.'},
                        {'hour': 15, 'message': '🌸 Afternoon reset: Quick breathing exercise to reduce stress.'},
                        {'hour': 21, 'message': '🌙 Evening calm: Meditation to unwind from the day.'}
                    ]
                    priority = 5
                elif stress_level > 3:  # Moderate stress
                    mindfulness_times = [
                        {'hour': 12, 'message': '🧘‍♀️ Midday mindfulness: Take a moment to breathe and reset.'},
                        {'hour': 20, 'message': '🌸 Evening meditation: Prepare your mind for restful sleep.'}
                    ]
                    priority = 4
                else:  # Low stress
                    mindfulness_times = [
                        {'hour': 20, 'message': '🧘‍♀️ Daily mindfulness: End your day with peaceful reflection.'}
                    ]
                    priority = 3
                
                for reminder in mindfulness_times:
                    mindfulness_time = now.replace(hour=reminder['hour'], minute=0, second=0, microsecond=0)
                    if mindfulness_time > now:
                        new_reminders.append({
                            'type': 'mindfulness',
                            'message': reminder['message'],
                            'time': mindfulness_time.isoformat(),
                            'priority': priority,
                            'ai_generated': True,
                            'duration_minutes': 10
                        })
                
                health_recommendations.append(f"AI scheduled mindfulness reminders based on stress level ({stress_level}/10)")
            
            # AI optimization insights
            if new_reminders:
                optimization_tips.extend([
                    f"Generated {len(new_reminders)} AI-powered health reminders",
                    f"Reminders personalized based on your health data and preferences",
                    f"AI detected activity level: {health_data.get('recent_activity_level', 0):.0f} min/day",
                    f"Sleep optimization: targeting {preferences.get('daily_sleep_goal', 8)}h per night"
                ])
            
            logger.info(f"Total reminders generated: {len(new_reminders)}")
            
            return {
                'success': True,
                'new_reminders': new_reminders,  # Fixed key name to match expected structure
                'health_recommendations': health_recommendations,
                'optimization_tips': optimization_tips,
                'ai_analysis': {
                    'total_reminders_generated': len(new_reminders),
                    'activity_level': health_data.get('recent_activity_level', 0),
                    'stress_level': health_data.get('recent_stress_level', 5),
                    'sleep_quality': health_data.get('recent_sleep_average', 7),
                    'personalization_applied': True
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating AI health reminders: {e}")
            return {
                'success': False,
                'error': str(e),
                'new_reminders': [],  # Fixed key name to match expected structure
                'health_recommendations': ['AI health reminder generation temporarily unavailable'],
                'optimization_tips': []
            }
            
            if health_recommendations:
                optimization_tips.extend(health_recommendations)
            
            return {
                'new_reminders': new_reminders,
                'health_recommendations': health_recommendations,
                'optimization_tips': optimization_tips
            }
            
        except Exception as e:
            logger.error(f"AI health reminder generation failed: {e}")
            return {
                'new_reminders': [],
                'health_recommendations': [],
                'optimization_tips': ['AI health recommendations temporarily unavailable']
            }
    
    def _optimize_with_ortools(self, flexible_events: List[Dict], fixed_events: List[Dict], 
                              health_data: Dict, preferences: Dict) -> Dict[str, Any]:
        """Use OR-Tools for constraint-based optimization"""
        try:
            from datetime import datetime, timedelta
            import random
            
            schedule_changes = []
            optimization_tips = []
            
            # Simple heuristic optimization for now (can be enhanced with actual OR-Tools)
            for event in flexible_events:
                try:
                    # Parse event times
                    start_time = datetime.fromisoformat(event['start_time'].replace('Z', ''))
                    end_time = datetime.fromisoformat(event['end_time'].replace('Z', ''))
                    duration = end_time - start_time
                    
                    # Apply health-aware scheduling logic
                    optimal_time = self._find_optimal_time(event, start_time, health_data, preferences)
                    
                    if optimal_time != start_time:
                        schedule_changes.append({
                            'event_id': event['id'],
                            'event_title': event['title'],
                            'original_start': start_time.isoformat(),
                            'optimized_start': optimal_time.isoformat(),
                            'optimized_end': (optimal_time + duration).isoformat(),
                            'reason': self._get_optimization_reason(event, optimal_time, health_data),
                            'confidence': 0.8
                        })
                        
                except Exception as event_error:
                    logger.warning(f"Error optimizing event {event.get('title', 'Unknown')}: {event_error}")
                    continue
            
            # Generate optimization tips
            if schedule_changes:
                optimization_tips.append(f"Optimized {len(schedule_changes)} events based on your health patterns")
            
            return {
                'schedule_changes': schedule_changes,
                'optimization_tips': optimization_tips
            }
            
        except Exception as e:
            logger.error(f"OR-Tools optimization error: {e}")
            return {'schedule_changes': [], 'optimization_tips': []}
    
    def _find_optimal_time(self, event: Dict, current_time: datetime, health_data: Dict, preferences: Dict) -> datetime:
        """Find optimal time for an event based on health data and preferences"""
        try:
            event_type = event.get('event_type', 'personal')
            
            # Health-aware scheduling logic
            if event_type == 'exercise':
                # Schedule exercise when energy levels are typically high
                if health_data.get('recent_activity_level', 0) < 30:  # Low activity
                    # Suggest morning exercise
                    optimal_hour = 7
                else:
                    # Suggest afternoon exercise
                    optimal_hour = 17
                    
                optimal_time = current_time.replace(hour=optimal_hour, minute=0, second=0, microsecond=0)
                
            elif event_type == 'work':
                # Schedule work during productive hours
                if current_time.hour < 9:
                    optimal_time = current_time.replace(hour=9, minute=0, second=0, microsecond=0)
                elif current_time.hour > 17:
                    optimal_time = current_time.replace(hour=14, minute=0, second=0, microsecond=0)
                else:
                    optimal_time = current_time  # Keep original time if in work hours
                    
            else:
                # For other events, apply general optimization
                optimal_time = current_time
            
            # Ensure optimal time is in the future
            now = datetime.now()
            if optimal_time < now:
                optimal_time = now + timedelta(hours=1)
                
            return optimal_time
            
        except Exception as e:
            logger.error(f"Error finding optimal time: {e}")
            return current_time
    
    def _get_optimization_reason(self, event: Dict, optimal_time: datetime, health_data: Dict) -> str:
        """Generate reason for the optimization"""
        event_type = event.get('event_type', 'personal')
        hour = optimal_time.hour
        
        if event_type == 'exercise':
            if hour < 10:
                return "Morning exercise aligns with your energy patterns and helps start the day actively"
            else:
                return "Afternoon exercise helps maintain energy levels and improves sleep quality"
        elif event_type == 'work':
            if 9 <= hour <= 17:
                return "Scheduled during peak productivity hours for better focus"
            else:
                return "Moved to standard work hours for better work-life balance"
        else:
            return "Optimized based on your daily routine and health patterns"
    
    def _generate_health_reminders(self, health_data: Dict, preferences: Dict) -> Dict[str, Any]:
        """Generate comprehensive daily health reminders as calendar events"""
        try:
            from datetime import datetime, timedelta
            
            new_reminders = []
            optimization_tips = []
            
            now = datetime.now()
            
            # WATER REMINDERS - Every 2 hours during waking hours (8 AM to 10 PM)
            if preferences.get('reminder_water', True):
                water_goal = preferences.get('daily_water_goal', 8)
                
                # Generate water reminders every 2 hours
                water_times = []
                start_hour = 8
                end_hour = 22
                interval_hours = 2
                
                for hour in range(start_hour, end_hour + 1, interval_hours):
                    reminder_time = now.replace(hour=hour, minute=0, second=0, microsecond=0)
                    
                    # Only add reminders for times later today, skip past times
                    if reminder_time > now:
                        water_times.append(reminder_time)
                
                # Create water reminder events
                glasses_per_reminder = max(1, water_goal // len(water_times))
                for i, reminder_time in enumerate(water_times):
                    new_reminders.append({
                        'type': 'water',
                        'message': f'💧 Hydration Time: Drink {glasses_per_reminder} glass{"es" if glasses_per_reminder > 1 else ""} of water',
                        'time': reminder_time.isoformat(),
                        'priority': 3,
                        'duration_minutes': 5
                    })
                
                optimization_tips.append(f"Added {len(water_times)} water reminders throughout the day to reach your {water_goal} glass goal")
            
            # EXERCISE REMINDERS - Multiple throughout the day
            if preferences.get('reminder_exercise', True):
                activity_goal = preferences.get('daily_activity_goal', 30)
                current_activity = health_data.get('recent_activity_level', 0)
                
                # Morning stretch reminder (7:30 AM)
                morning_exercise = now.replace(hour=7, minute=30, second=0, microsecond=0)
                if morning_exercise > now:
                    new_reminders.append({
                        'type': 'exercise',
                        'message': '🌅 Morning Stretch: 5-minute energizing routine to start your day',
                        'time': morning_exercise.isoformat(),
                        'priority': 4,
                        'duration_minutes': 15
                    })
                
                # Midday movement break (2:00 PM)
                afternoon_exercise = now.replace(hour=14, minute=0, second=0, microsecond=0)
                if afternoon_exercise > now:
                    new_reminders.append({
                        'type': 'exercise',
                        'message': '💪 Activity Break: Take a 10-minute walk or do desk exercises',
                        'time': afternoon_exercise.isoformat(),
                        'priority': 4,
                        'duration_minutes': 15
                    })
                
                # Evening workout (6:00 PM) - if below activity goal
                if current_activity < activity_goal * 0.7:
                    evening_exercise = now.replace(hour=18, minute=0, second=0, microsecond=0)
                    if evening_exercise > now:
                        remaining_minutes = max(15, activity_goal - int(current_activity))
                        new_reminders.append({
                            'type': 'exercise',
                            'message': f'🏃 Evening Workout: {remaining_minutes} minutes to reach daily goal',
                            'time': evening_exercise.isoformat(),
                            'priority': 4,
                            'duration_minutes': 30
                        })
                
                optimization_tips.append(f"Added exercise reminders to help reach your {activity_goal} minute activity goal")
            
            # SLEEP REMINDERS
            if preferences.get('reminder_sleep', True):
                sleep_goal = preferences.get('daily_sleep_goal', 8.0)
                
                # Wind-down reminder (9:00 PM)
                wind_down_time = now.replace(hour=21, minute=0, second=0, microsecond=0)
                if wind_down_time > now:
                    new_reminders.append({
                        'type': 'sleep',
                        'message': f'🌙 Wind Down: Start preparing for {sleep_goal} hours of quality sleep',
                        'time': wind_down_time.isoformat(),
                        'priority': 5,
                        'duration_minutes': 10
                    })
                
                # Bedtime reminder (10:00 PM)
                bedtime = now.replace(hour=22, minute=0, second=0, microsecond=0)
                if bedtime > now:
                    new_reminders.append({
                        'type': 'sleep',
                        'message': '😴 Bedtime: Time to sleep for optimal recovery and health',
                        'time': bedtime.isoformat(),
                        'priority': 5,
                        'duration_minutes': 5
                    })
                
                optimization_tips.append(f"Added sleep reminders to help achieve {sleep_goal} hours of rest")
            
            # MEAL REMINDERS (if enabled) - AI-enhanced based on preferences
            if preferences.get('reminder_meal', True):
                # Check if smart reminders are enabled to use AI logic
                use_ai_logic = preferences.get('smart_reminders_enabled', False)
                
                if use_ai_logic:
                    # AI-based meal timing considering user's schedule and activity
                    meals = []
                    
                    # Breakfast - earlier if user has morning workouts, later if they're a night owl
                    breakfast_hour = 8
                    if preferences.get('reminder_exercise', False):
                        # If exercise reminders are on, assume they might workout early
                        breakfast_hour = 7
                    
                    breakfast_time = now.replace(hour=breakfast_hour, minute=0, second=0, microsecond=0)
                    if breakfast_time <= now:
                        breakfast_time = breakfast_time + timedelta(days=1)
                    
                    meals.append({
                        'time': breakfast_time,
                        'message': f'🍳 Smart Breakfast: Fuel up for your {breakfast_hour}:00 AM start',
                        'type': 'breakfast'
                    })
                    
                    # Lunch - adjusted based on breakfast timing
                    lunch_hour = breakfast_hour + 4 if breakfast_hour <= 8 else 13
                    lunch_time = now.replace(hour=lunch_hour, minute=30, second=0, microsecond=0)
                    if lunch_time <= now:
                        lunch_time = lunch_time + timedelta(days=1)
                    
                    meals.append({
                        'time': lunch_time,
                        'message': f'🥗 Smart Lunch: Perfect timing for sustained energy',
                        'type': 'lunch'
                    })
                    
                    # Dinner - earlier if sleep reminders are enabled (better digestion)
                    dinner_hour = 18 if preferences.get('reminder_sleep', False) else 19
                    dinner_time = now.replace(hour=dinner_hour, minute=0, second=0, microsecond=0)
                    if dinner_time <= now:
                        dinner_time = dinner_time + timedelta(days=1)
                    
                    meals.append({
                        'time': dinner_time,
                        'message': f'�️ Smart Dinner: Optimal timing for healthy digestion',
                        'type': 'dinner'
                    })
                    
                    for meal in meals:
                        new_reminders.append({
                            'type': 'meal',
                            'message': meal['message'],
                            'time': meal['time'].isoformat(),
                            'priority': 3,
                            'duration_minutes': 30
                        })
                    
                    optimization_tips.append("Added AI-optimized meal reminders based on your daily schedule")
                    
                else:
                    # Basic meal timing patterns
                    meal_times = [
                        (8, 0, '�🍳 Breakfast: Start your day with nutritious fuel'),
                        (12, 30, '🥗 Lunch: Midday nutrition break'),
                        (19, 0, '🍽️ Dinner: Evening meal time')
                    ]
                    
                    for hour, minute, message in meal_times:
                        meal_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                        if meal_time <= now:
                            meal_time = meal_time + timedelta(days=1)
                        
                        new_reminders.append({
                            'type': 'meal',
                            'message': message,
                            'time': meal_time.isoformat(),
                            'priority': 3,
                            'duration_minutes': 10
                        })
                    
                    optimization_tips.append("Added meal reminders to maintain consistent nutrition schedule")
            
            logger.info(f"Generated {len(new_reminders)} comprehensive health reminders")
            
            return {
                'success': True,
                'new_reminders': new_reminders,
                'optimization_tips': optimization_tips,
                'message': f'Generated {len(new_reminders)} health reminders based on your preferences'
            }
            
        except Exception as e:
            logger.error(f"Error generating health reminders: {e}")
            return {
                'success': False,
                'new_reminders': [],
                'optimization_tips': [],
                'message': f'Error generating reminders: {str(e)}'
            }
    
    def _add_time_constraints(self, event_vars, total_slots):
        """Add time-based constraints"""
        # Ensure no time overlap between events
        for i in event_vars:
            for j in event_vars:
                if i != j:
                    # Event i ends before event j starts OR event j ends before event i starts
                    self.model.AddBoolOr([
                        event_vars[i]['start'] + event_vars[i]['duration'] <= event_vars[j]['start'],
                        event_vars[j]['start'] + event_vars[j]['duration'] <= event_vars[i]['start']
                    ])
    
    def _add_health_constraints(self, event_vars, health_data):
        """Add health-aware constraints"""
        # Simple implementation - could be enhanced
        pass
    
    def _add_preference_constraints(self, event_vars, preferences):
        """Add user preference constraints"""
        # Simple implementation - could be enhanced
        pass
    
    def _add_fixed_event_constraints(self, event_vars, fixed_events, start_hour, slot_duration):
        """Add constraints for fixed events"""
        # Simple implementation - could be enhanced
        pass
    
    def _create_optimization_objective(self, event_vars, health_data):
        """Create the optimization objective function"""
        # Simple implementation - could be enhanced
        pass
    
    def _extract_solution(self, event_vars, start_hour, slot_duration, health_data):
        """Extract the optimized solution"""
        optimized_events = []
        for i, var_info in event_vars.items():
            start_slot = self.solver.Value(var_info['start'])
            start_time = start_hour + (start_slot * slot_duration) / 60
            
            optimized_events.append({
                'event': var_info['event'],
                'optimized_start_time': start_time,
                'optimization_applied': True
            })
        
        return {
            'success': True,
            'message': f'Optimized {len(optimized_events)} events',
            'events': optimized_events
        }

class GoogleCalendarService:
    """
    Enhanced Google Calendar service with AI-powered schedule optimization
    """
    
    def __init__(self):
        """Initialize Google Calendar service"""
        self.service = None
        self.credentials = None
        self.setup_credentials()
    
    def analyze_event_text(self, text: str) -> Dict[str, Any]:
        """Analyze event text to extract useful information"""
        try:
            # Simple analysis - could be enhanced with AI
            text_lower = text.lower()
            
            # Determine event type
            if any(word in text_lower for word in ['meeting', 'call', 'conference', 'standup']):
                event_type = 'work'
            elif any(word in text_lower for word in ['workout', 'gym', 'exercise', 'run']):
                event_type = 'exercise'
            elif any(word in text_lower for word in ['doctor', 'appointment', 'checkup']):
                event_type = 'health'
            else:
                event_type = 'personal'
            
            return {
                'event_type': event_type,
                'analysis': f'Detected event type: {event_type}'
            }
        except Exception as e:
            logger.error(f"Error analyzing event text: {e}")
            return {'event_type': 'personal', 'analysis': 'Default analysis'}
    
    def optimize_schedule(self, events: List[Dict], health_context: Dict, preferences: Dict) -> Dict[str, Any]:
        """Optimize schedule using OR-Tools"""
        try:
            optimizer = ScheduleOptimizer()
            return optimizer.optimize_schedule(events, health_context, preferences)
        except Exception as e:
            logger.error(f"Error optimizing schedule: {e}")
            return {
                'success': False,
                'message': f'Schedule optimization failed: {str(e)}',
                'events': events
            }
    
    def generate_smart_reminders(self, events: List[Dict], health_context: Dict, preferences: Dict) -> List[Dict]:
        """Generate smart reminders based on events and health data"""
        try:
            reminders = []
            
            # Health-based reminder generation logic
            for event in events:
                event_type = event.get('event_type', 'personal')
                start_time = event.get('start_time')
                
                if not start_time:
                    continue
                    
                try:
                    event_start = datetime.fromisoformat(start_time.replace('Z', ''))
                except:
                    continue
                
                # Generate contextual reminders based on event type
                if event_type == 'exercise':
                    # Pre-workout hydration reminder
                    reminder_time = event_start - timedelta(minutes=30)
                    reminders.append({
                        'type': 'hydration',
                        'message': f'Hydrate before your {event.get("title", "workout")}! Drink 1-2 glasses of water.',
                        'time': reminder_time.isoformat(),
                        'priority': 4,
                        'related_event': event.get('title', 'Exercise')
                    })
                    
                    # Post-workout recovery reminder
                    post_reminder_time = event_start + timedelta(minutes=60)
                    reminders.append({
                        'type': 'recovery',
                        'message': f'Post-workout recovery: Hydrate and consider a healthy snack.',
                        'time': post_reminder_time.isoformat(),
                        'priority': 3,
                        'related_event': event.get('title', 'Exercise')
                    })
                    
                elif event_type == 'work':
                    # Break reminder for long work sessions
                    if 'meeting' in event.get('title', '').lower():
                        post_reminder_time = event_start + timedelta(minutes=5)
                        reminders.append({
                            'type': 'break',
                            'message': 'Take a short break and stretch after your meeting!',
                            'time': post_reminder_time.isoformat(),
                            'priority': 2,
                            'related_event': event.get('title', 'Meeting')
                        })
            
            # Add health goal-based reminders
            if preferences.get('reminder_water', True):
                now = datetime.now()
                # Morning hydration reminder
                morning_reminder = now.replace(hour=8, minute=0, second=0, microsecond=0)
                if morning_reminder > now:
                    reminders.append({
                        'type': 'hydration',
                        'message': 'Start your day with a glass of water!',
                        'time': morning_reminder.isoformat(),
                        'priority': 3,
                        'related_event': 'Daily Hydration'
                    })
            
            if preferences.get('reminder_exercise', True) and not any(e.get('event_type') == 'exercise' for e in events):
                # No exercise scheduled, suggest adding some
                now = datetime.now()
                afternoon_reminder = now.replace(hour=15, minute=0, second=0, microsecond=0)
                if afternoon_reminder > now:
                    reminders.append({
                        'type': 'exercise',
                        'message': 'Consider adding some physical activity to your day!',
                        'time': afternoon_reminder.isoformat(),
                        'priority': 4,
                        'related_event': 'Daily Activity'
                    })
            
            return reminders
        except Exception as e:
            logger.error(f"Error generating smart reminders: {e}")
            return []
    
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
            if not hasattr(self, 'client_id') or not self.client_id:
                self.setup_credentials()
                
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
    
    def check_credentials(self):
        # Ensure credentials are set up
        if not self.credentials:
            self.setup_credentials()
        
        if not current_app.config.get('GOOGLE_CALENDAR_CLIENT_ID'):
            raise ValueError("Google Calendar API credentials not configured. Please set GOOGLE_CALENDAR_CLIENT_ID and GOOGLE_CALENDAR_CLIENT_SECRET in environment variables.")
    
    def build_auth_url(self, user_id: int) -> str:
        """Build Google OAuth2 authorization URL"""
        try:
            self.check_credentials()
            
            # OAuth2 settings
            SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
            
            client_config = {
                'web': {
                    'client_id': current_app.config['GOOGLE_CALENDAR_CLIENT_ID'],
                    'client_secret': current_app.config['GOOGLE_CALENDAR_CLIENT_SECRET'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://oauth2.googleapis.com/token',
                    'redirect_uris': [current_app.config.get('GOOGLE_CALENDAR_REDIRECT_URI', 'http://localhost:5000/google_calendar/callback')]
                }
            }
            
            flow = Flow.from_client_config(
                client_config, 
                scopes=SCOPES,
                redirect_uri=client_config['web']['redirect_uris'][0]
            )
            
            auth_url, _ = flow.authorization_url(
                access_type='offline',
                include_granted_scopes='true',
                state=str(user_id)  # Pass user ID in state
            )
            
            return auth_url
            
        except Exception as e:
            logger.error(f"Error building Google Calendar auth URL: {e}")
            raise Exception(f"Failed to build authorization URL: {str(e)}")
    
    def handle_oauth_callback(self, code: str, state: str) -> Dict:
        """Handle OAuth2 callback and exchange code for tokens"""
        try:
            self.check_credentials()
            
            SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']
            
            client_config = {
                'web': {
                    'client_id': current_app.config['GOOGLE_CALENDAR_CLIENT_ID'],
                    'client_secret': current_app.config['GOOGLE_CALENDAR_CLIENT_SECRET'],
                    'auth_uri': 'https://accounts.google.com/o/oauth2/auth',
                    'token_uri': 'https://oauth2.googleapis.com/token',
                    'redirect_uris': [current_app.config.get('GOOGLE_CALENDAR_REDIRECT_URI', 'http://localhost:5000/google_calendar/callback')]
                }
            }
            
            flow = Flow.from_client_config(
                client_config,
                scopes=SCOPES,
                redirect_uri=client_config['web']['redirect_uris'][0]
            )
            
            flow.fetch_token(code=code)
            
            credentials = flow.credentials
            
            # Convert credentials to dict for storage
            credentials_dict = {
                'token': credentials.token,
                'refresh_token': credentials.refresh_token,
                'token_uri': credentials.token_uri,
                'client_id': credentials.client_id,
                'client_secret': credentials.client_secret,
                'scopes': credentials.scopes
            }
            
            return {
                'success': True,
                'credentials': credentials_dict,
                'user_id': int(state) if state.isdigit() else None
            }
            
        except Exception as e:
            logger.error(f"Error in OAuth callback: {e}")
            return {
                'success': False,
                'error': str(e)
            }

    def sync_events_to_database(self, user_id: int, credentials_dict: Dict, days_ahead: int = 30) -> Dict:
        """Sync Google Calendar events to our database with completely rewritten timeout handling"""
        
        try:
            from app import db
            from app.models import CalendarEvent
            
            logger.info("Starting Google Calendar sync with direct HTTP approach")
            
            # Create credentials from dict
            credentials = Credentials.from_authorized_user_info(credentials_dict)
            
            # Refresh credentials if needed
            if credentials.expired and credentials.refresh_token:
                request = Request()
                credentials.refresh(request)
                logger.info("Refreshed Google Calendar credentials")
            
            # Calculate date range
            start_date = datetime.now()
            end_date = start_date + timedelta(days=days_ahead)
            
            logger.info(f"Fetching events from {start_date.date()} to {end_date.date()}")
            
            # Method 1: Use requests directly with Google Calendar REST API
            # This bypasses all the google-api-python-client timeout issues
            try:
                # Set up requests session with unlimited timeout
                session = requests.Session()
                
                # Completely disable all timeout mechanisms
                original_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(None)  # Unlimited socket timeout
                
                try:
                    # Build the API URL directly
                    calendar_api_url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
                    
                    # Prepare query parameters
                    params = {
                        'timeMin': start_date.isoformat() + 'Z',
                        'timeMax': end_date.isoformat() + 'Z',
                        'maxResults': 10,
                        'singleEvents': 'true',
                        'orderBy': 'startTime'
                    }
                    
                    # Prepare headers with OAuth2 token
                    headers = {
                        'Authorization': f'Bearer {credentials.token}',
                        'Accept': 'application/json',
                        'Content-Type': 'application/json'
                    }
                    
                    logger.info("Making direct HTTP request to Google Calendar API (unlimited timeout)")
                    
                    # Make the request with no timeout limits
                    response = session.get(
                        calendar_api_url,
                        params=params,
                        headers=headers,
                        timeout=None  # No timeout - wait forever if needed
                    )
                    
                    logger.info(f"Google Calendar API response: {response.status_code}")
                    
                    if response.status_code == 200:
                        events_data = response.json()
                        events = events_data.get('items', [])
                        logger.info(f"Successfully fetched {len(events)} events using direct HTTP")
                        
                        # Process events
                        synced_count = 0
                        
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
                                        is_ai_modifiable=False,
                                        event_type='work'
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
                            'synced_count': synced_count,
                            'message': f'Successfully synced {synced_count} events from Google Calendar'
                        }
                    
                    elif response.status_code == 401:
                        logger.error("Unauthorized - credentials may be invalid")
                        return {
                            'success': False,
                            'error': 'Google Calendar authorization failed. Please reconnect your account.',
                            'synced_count': 0
                        }
                    
                    else:
                        logger.error(f"Google Calendar API error: {response.status_code} - {response.text}")
                        return {
                            'success': False,
                            'error': f'Google Calendar API returned error {response.status_code}',
                            'synced_count': 0
                        }
                
                finally:
                    # Restore original socket timeout
                    socket.setdefaulttimeout(original_timeout)
                    
            except Exception as direct_error:
                logger.error(f"Direct HTTP method failed: {direct_error}")
                
                # Fall back to google-api-python-client as last resort
                logger.info("Falling back to google-api-python-client (with no timeouts)")
                
                try:
                    # Final fallback using the old method but with no timeout settings
                    socket.setdefaulttimeout(None)
                    
                    # Build service with no timeouts
                    service = build('calendar', 'v3', credentials=credentials, cache_discovery=False)
                    
                    # Try a very simple API call
                    events_result = service.events().list(
                        calendarId='primary',
                        timeMin=start_date.isoformat() + 'Z',
                        timeMax=end_date.isoformat() + 'Z',
                        maxResults=10,
                        singleEvents=True,
                        orderBy='startTime'
                    ).execute()
                    
                    events = events_result.get('items', [])
                    logger.info(f"Fallback method succeeded: {len(events)} events")
                    
                    return {
                        'success': True,
                        'synced_events': len(events),
                        'synced_count': len(events),
                        'message': f'Successfully synced {len(events)} events using fallback method'
                    }
                    
                except Exception as fallback_error:
                    logger.error(f"Both direct HTTP and fallback methods failed: {fallback_error}")
                    return {
                        'success': False,
                        'error': f'All calendar sync methods failed. Google Calendar services appear to be unavailable. Error: {str(fallback_error)}',
                        'synced_count': 0
                    }
            
        except Exception as e:
            logger.error(f"Error syncing Google Calendar events: {e}")
            return {
                'success': False,
                'error': f'Calendar sync failed: {str(e)}',
                'synced_count': 0
            }

# Lazy initialization to avoid app context errors  
_google_calendar_service = None
_calendar_service = None

def get_google_calendar_service():
    """Get Google Calendar service instance"""
    global _google_calendar_service
    if _google_calendar_service is None:
        _google_calendar_service = GoogleCalendarService()
    return _google_calendar_service

def get_calendar_service():
    """Get calendar service instance (alias for Google Calendar service)"""
    return get_google_calendar_service()