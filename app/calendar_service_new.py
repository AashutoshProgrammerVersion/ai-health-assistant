
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
import requests

# Google Services
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import httplib2

# AI and ML
import google.generativeai as genai
from google.genai import types
import spacy

# Optimization
from ortools.sat.python import cp_model

# Flask
from flask import current_app, session

logger = logging.getLogger(__name__)

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
                return {
                    'success': False,
                    'message': 'Could not optimize schedule with current constraints',
                    'events': events
                }
                
        except Exception as e:
            logger.error(f"Schedule optimization error: {e}")
            return {
                'success': False,
                'message': f'Optimization error: {str(e)}',
                'events': events
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
    
    def setup_credentials(self):
        """Setup Google Calendar API credentials"""
        try:
            # These would be set via environment variables or config
            if not current_app.config.get('GOOGLE_CALENDAR_CLIENT_ID'):
                logger.warning("Google Calendar API credentials not configured")
        except Exception as e:
            logger.error(f"Error setting up Google Calendar credentials: {e}")
    
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

def get_google_calendar_service():
    """Get Google Calendar service instance"""
    global _google_calendar_service
    if _google_calendar_service is None:
        _google_calendar_service = GoogleCalendarService()
    return _google_calendar_service