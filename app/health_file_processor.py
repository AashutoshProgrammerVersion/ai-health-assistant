"""
Health File Processor with Gemini 2.0 Flash Integration

This module processes health data files from multiple wearable devices:
- Samsung Health (Galaxy Watch, Galaxy Ring, etc.)
- Apple Health
- Fitbit
- Garmin
- Other fitness trackers

Uses Gemini 2.0 Flash for intelligent data extraction and analysis.
Based on successful test implementation and research requirements.
"""

import os
import json
import tempfile
import time
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging
import threading
from dataclasses import dataclass
from collections import deque

from flask import current_app, flash, session
from google import genai
from google.genai import types
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import spacy

from app import db
from app.models import HealthData, User

logger = logging.getLogger(__name__)

# Global progress store
PROGRESS_STORE = {}

@dataclass
class TokenUsage:
    """Track token usage for rate limiting"""
    timestamp: float
    tokens: int

class ProcessingLock:
    """Global lock to ensure only one user processes at a time"""
    _lock = threading.Lock()
    _current_user = None
    
    @classmethod
    def get_lock(cls):
        """Get the global processing lock for context manager usage"""
        return cls._lock
    
    @classmethod
    def acquire(cls, user_id: int) -> bool:
        """Try to acquire processing lock for user"""
        with cls._lock:
            if cls._current_user is None:
                cls._current_user = user_id
                return True
            return False
    
    @classmethod
    def release(cls, user_id: int) -> None:
        """Release processing lock"""
        with cls._lock:
            if cls._current_user == user_id:
                cls._current_user = None

class RateLimiter:
    """Post-response rate limiting with model-specific waiting times"""
    def __init__(self, target_tokens_per_request: int = 950000):  # Just under 1M limit
        self.target_tokens_per_request = target_tokens_per_request
        self.last_response_time = None
        self.last_used_model = None  # Track which model was used last
        self._lock = threading.Lock()
    
    def wait_for_next_call(self, next_model: str) -> None:
        """Wait appropriate time based on previous and next model combination"""
        with self._lock:
            if self.last_response_time is None or self.last_used_model is None:
                # First response - record time but no wait needed
                print(f"🚀 First API call - no waiting required")
                return
            
            # Determine wait time based on model transition
            if self.last_used_model == 'gemini-2.5-flash-lite':
                # Previous response was from 2.5 Flash-Lite
                if next_model == 'gemini-2.5-flash-lite':
                    wait_duration = 10.0  # 10 seconds between Flash-Lite calls
                    print(f"⏳ Flash-Lite → Flash-Lite: waiting {wait_duration} seconds")
                else:  # next_model == 'gemini-2.0-flash'
                    wait_duration = 60.0  # 60 seconds to switch to 2.0 Flash
                    print(f"⏳ Flash-Lite → 2.0 Flash: waiting {wait_duration} seconds")
            else:  # self.last_used_model == 'gemini-2.0-flash'
                # Previous response was from 2.0 Flash
                if next_model == 'gemini-2.0-flash':
                    wait_duration = 60.0  # 60 seconds between 2.0 Flash calls
                    print(f"⏳ 2.0 Flash → 2.0 Flash: waiting {wait_duration} seconds")
                else:  # next_model == 'gemini-2.5-flash-lite'
                    wait_duration = 10.0  # 10 seconds to switch to Flash-Lite
                    print(f"⏳ 2.0 Flash → Flash-Lite: waiting {wait_duration} seconds")
            
            # Calculate actual wait time based on elapsed time
            elapsed = time.time() - self.last_response_time
            actual_wait = max(0, wait_duration - elapsed)
            
            if actual_wait > 0:
                print(f"⏳ Waiting {actual_wait:.1f} seconds (elapsed: {elapsed:.1f}s)")
                time.sleep(actual_wait)
            else:
                print(f"✅ No wait needed - {elapsed:.1f}s have already passed")
    
    def mark_response_received(self, model_used: str) -> None:
        """Mark that we received a response from a specific model"""
        with self._lock:
            self.last_response_time = time.time()
            self.last_used_model = model_used
            print(f"✅ Response received from {model_used}")
    
    def record_usage(self, tokens: int) -> None:
        """Record that we used tokens (for compatibility)"""
        print(f"📊 Used {tokens:,} tokens in this request")
    
    # Legacy method compatibility - now uses model-aware waiting
    def wait_after_response(self) -> None:
        """Legacy compatibility - use wait_for_next_call instead"""
        print(f"⚠️  Using legacy wait_after_response - specify model for better control")
        with self._lock:
            if self.last_response_time is None:
                return
            elapsed = time.time() - self.last_response_time
            wait_time = max(0, 60.0 - elapsed)  # Default 1 minute wait
            if wait_time > 0:
                print(f"⏳ Legacy wait: {wait_time:.1f} seconds")
                time.sleep(wait_time)
    
    def wait_for_next_request(self) -> None:
        """Legacy compatibility - now waits after response"""
        self.wait_after_response()

class HealthDataValidator:
    """Validates and fixes JSON responses to match expected health data structure"""
    
    @staticmethod
    def get_expected_fields():
        """Return the expected health data field structure"""
        return {
            # Core fields
            'date_logged': str,
            'data_source': str,
            
            # Activity metrics
            'steps': (int, type(None)),
            'distance_km': (float, type(None)),
            'calories_total': (int, type(None)),  # Updated field name
            'active_minutes': (int, type(None)),
            'workout_type': (str, type(None)),
            'workout_duration_minutes': (int, type(None)),
            'floors_climbed': (int, type(None)),
            
            # Heart rate metrics
            'heart_rate_avg': (int, type(None)),
            'heart_rate_resting': (int, type(None)),
            'heart_rate_max': (int, type(None)),
            'heart_rate_variability': (float, type(None)),
            
            # Sleep metrics
            'sleep_duration_hours': (float, type(None)),
            'sleep_quality_score': (int, type(None)),
            'sleep_deep_minutes': (int, type(None)),  # Updated to minutes
            'sleep_rem_minutes': (int, type(None)),  # Updated to minutes
            'sleep_awake_minutes': (int, type(None)),
            
            # Vital signs
            'blood_oxygen_percent': (int, type(None)),  # Updated field name
            'systolic_bp': (int, type(None)),
            'diastolic_bp': (int, type(None)),
            'stress_level': (int, type(None)),
            'body_temperature': (float, type(None)),
            
            # Nutrition
            'calories_consumed': (int, type(None)),
            'protein_grams': (float, type(None)),
            'carbs_grams': (float, type(None)),
            'fat_grams': (float, type(None)),
            'fiber_grams': (float, type(None)),
            'water_intake_liters': (float, type(None)),
            
            # Body composition
            'weight_kg': (float, type(None)),
            'body_fat_percent': (float, type(None)),  # Updated field name
            'muscle_mass_kg': (float, type(None)),
            'bmi': (float, type(None)),
            
            # Wellness metrics
            'mood_score': (int, type(None)),
            'energy_level': (int, type(None)),
            'meditation_minutes': (int, type(None)),
            'screen_time_hours': (float, type(None)),
            'social_interactions': (int, type(None)),
            
            # Exercise session details
            'workout_intensity': (str, type(None)),
            'workout_calories': (int, type(None)),
            
            # Notes
            'notes': (str, type(None))
        }
    
    @staticmethod
    def validate_and_fix_health_data(data):
        """Validate and fix health data structure to match expected format"""
        try:
            print("Validating and fixing JSON structure...")
            
            # If data is not a dict, try to extract health_data
            if not isinstance(data, dict):
                print("Data is not a dictionary")
                return {"health_data": []}
            
            # Try to find health data in various possible structures
            health_entries = []
            
            # Case 1: data has 'health_data' key with list
            if 'health_data' in data and isinstance(data['health_data'], list):
                health_entries = data['health_data']
                print(f"Found health_data list with {len(health_entries)} entries")
            
            # Case 2: data has 'health_data' key with dict (old format)
            elif 'health_data' in data and isinstance(data['health_data'], dict):
                old_data = data['health_data']
                print(f"🔄 Converting old format with categories: {list(old_data.keys())}")
                
                # Convert from old categorized format to new flat format
                date_entries = {}
                
                for category, entries in old_data.items():
                    if isinstance(entries, list):
                        for entry in entries:
                            if isinstance(entry, dict) and 'date' in entry:
                                date = entry['date']
                                if date not in date_entries:
                                    date_entries[date] = {
                                        'date_logged': date,
                                        'data_source': 'file_upload'
                                    }
                                
                                # Map old field names to new field names
                                field_mapping = {
                                    # Activity
                                    'steps': 'steps',
                                    'distance_km': 'distance_km', 
                                    'calories': 'calories_total',  # Map to database field
                                    'calories_burned': 'calories_total',  # Map to database field
                                    'calories_total': 'calories_total',  # Direct mapping
                                    'active_minutes': 'active_minutes',
                                    'exercise_type': 'workout_type',
                                    'workout_type': 'workout_type',
                                    'exercise_duration_minutes': 'workout_duration_minutes',
                                    'workout_duration_minutes': 'workout_duration_minutes',
                                    'floors_climbed': 'floors_climbed',
                                    
                                    # Heart rate
                                    'heart_rate_avg': 'heart_rate_avg',
                                    'avg_hr': 'heart_rate_avg',
                                    'heart_rate_resting': 'heart_rate_resting',
                                    'resting_hr': 'heart_rate_resting',
                                    'heart_rate_max': 'heart_rate_max',
                                    'max_hr': 'heart_rate_max',
                                    'heart_rate_variability': 'heart_rate_variability',
                                    
                                    # Sleep
                                    'sleep_duration_hours': 'sleep_duration_hours',
                                    'duration_hours': 'sleep_duration_hours',
                                    'sleep_quality_score': 'sleep_quality_score',
                                    'quality': 'sleep_quality_score',
                                    'deep_sleep_hours': 'deep_sleep_hours_temp',  # Convert later
                                    'sleep_deep_minutes': 'sleep_deep_minutes',  # Direct mapping
                                    'rem_sleep_hours': 'rem_sleep_hours_temp',  # Convert later
                                    'sleep_rem_minutes': 'sleep_rem_minutes',  # Direct mapping
                                    'sleep_awake_minutes': 'sleep_awake_minutes',
                                    
                                    # Vitals
                                    'oxygen_saturation': 'blood_oxygen_percent',  # Map to database field
                                    'blood_oxygen_percent': 'blood_oxygen_percent',  # Direct mapping
                                    'systolic_bp': 'systolic_bp',
                                    'bp_systolic': 'systolic_bp',
                                    'diastolic_bp': 'diastolic_bp',
                                    'bp_diastolic': 'diastolic_bp',
                                    'stress_level': 'stress_level',
                                    'body_temperature': 'body_temperature',
                                    
                                    # Nutrition
                                    'calories_consumed': 'calories_consumed',
                                    'calories_in': 'calories_consumed',
                                    'protein_grams': 'protein_grams',
                                    'carbs_grams': 'carbs_grams',
                                    'fat_grams': 'fat_grams',
                                    'fiber_grams': 'fiber_grams',
                                    'water_intake_liters': 'water_intake_liters',
                                    'water_ml': 'water_ml_temp',  # Convert later
                                    
                                    # Body composition
                                    'weight_kg': 'weight_kg',
                                    'body_fat_percentage': 'body_fat_percent',  # Map to database field
                                    'body_fat_percent': 'body_fat_percent',  # Direct mapping
                                    'muscle_mass_kg': 'muscle_mass_kg',
                                    'bmi': 'bmi',
                                    
                                    # Wellness
                                    'mood_score': 'mood_score',
                                    'energy_level': 'energy_level',
                                    'meditation_minutes': 'meditation_minutes',
                                    'screen_time_hours': 'screen_time_hours',
                                    'social_interactions': 'social_interactions',
                                    
                                    # Exercise
                                    'workout_intensity': 'workout_intensity',
                                    'workout_calories': 'workout_calories',
                                    
                                    # Notes
                                    'notes': 'notes'
                                }
                                
                                # Copy fields from entry to date_entries
                                for old_field, new_field in field_mapping.items():
                                    if old_field in entry and entry[old_field] is not None:
                                        if new_field == 'water_ml_temp':
                                            # Convert ml to liters
                                            date_entries[date]['water_intake_liters'] = entry[old_field] / 1000.0
                                        elif new_field == 'deep_sleep_hours_temp':
                                            # Convert hours to minutes
                                            date_entries[date]['sleep_deep_minutes'] = int(entry[old_field] * 60)
                                        elif new_field == 'rem_sleep_hours_temp':
                                            # Convert hours to minutes
                                            date_entries[date]['sleep_rem_minutes'] = int(entry[old_field] * 60)
                                        else:
                                            date_entries[date][new_field] = entry[old_field]
                
                health_entries = list(date_entries.values())
                print(f"Converted to {len(health_entries)} entries")
            
            # Case 2.5: data has 'health_data' key with 'activity' array (2.0 Flash format)
            elif 'health_data' in data and isinstance(data['health_data'], dict) and 'activity' in data['health_data']:
                activity_entries = data['health_data']['activity']
                if isinstance(activity_entries, list):
                    print(f"Found health_data.activity array with {len(activity_entries)} entries")
                    # Convert activity entries to standard format
                    health_entries = []
                    for entry in activity_entries:
                        if isinstance(entry, dict):
                            # Convert date field if needed
                            if 'date' in entry and 'date_logged' not in entry:
                                entry['date_logged'] = entry['date']
                            # Ensure data_source is set
                            if 'data_source' not in entry:
                                entry['data_source'] = 'file_upload'
                            health_entries.append(entry)
                    print(f"Converted {len(health_entries)} activity entries")
                else:
                    print("activity field is not a list")
                    health_entries = []
            
            # Case 3: data is directly a list
            elif isinstance(data, list):
                health_entries = data
                print(f"Data is directly a list with {len(health_entries)} entries")
            
            # Case 4: data is a single entry dict
            elif 'date' in data or 'date_logged' in data:
                health_entries = [data]
                print("Data is a single entry")
            
            else:
                print("Could not find health data in response")
                return {"health_data": []}
            
            # Fix each entry structure
            expected_fields = HealthDataValidator.get_expected_fields()
            fixed_entries = []
            
            for entry in health_entries:
                if not isinstance(entry, dict):
                    continue
                
                fixed_entry = {}
                
                # Ensure required fields exist
                fixed_entry['date_logged'] = entry.get('date_logged') or entry.get('date') or '2025-09-21'
                fixed_entry['data_source'] = entry.get('data_source', 'file_upload')
                
                # Fix all other fields
                for field, expected_type in expected_fields.items():
                    if field in ['date_logged', 'data_source']:
                        continue  # Already handled
                    
                    value = entry.get(field)
                    
                    if value is not None:
                        # Try to convert to expected type
                        if isinstance(expected_type, tuple):
                            # Field can be multiple types (e.g., int or None)
                            if int in expected_type and isinstance(value, (int, float)):
                                fixed_entry[field] = int(value)
                            elif float in expected_type and isinstance(value, (int, float)):
                                fixed_entry[field] = float(value)
                            elif str in expected_type:
                                fixed_entry[field] = str(value)
                            else:
                                fixed_entry[field] = value
                        else:
                            # Single expected type
                            if expected_type == int and isinstance(value, (int, float)):
                                fixed_entry[field] = int(value)
                            elif expected_type == float and isinstance(value, (int, float)):
                                fixed_entry[field] = float(value)
                            elif expected_type == str:
                                fixed_entry[field] = str(value)
                            else:
                                fixed_entry[field] = value
                    else:
                        fixed_entry[field] = None
                
                fixed_entries.append(fixed_entry)
            
            result = {"health_data": fixed_entries}
            print(f"Fixed structure: {len(fixed_entries)} valid entries")
            return result
            
        except Exception as e:
            print(f"❌ Error validating JSON structure: {e}")
            return {"health_data": []}

class JSONRepair:
    """Repairs broken JSON responses from Flash 2.0 model"""
    
    @staticmethod
    def repair_broken_json(response_text: str) -> str:
        """
        Repair broken JSON by removing incomplete entries and fixing common issues
        """
        try:
            print("Attempting to repair broken JSON...")
            
            if not response_text or not response_text.strip():
                print("Empty response text")
                return '{"health_data": []}'
            
            # Clean up common issues
            cleaned = response_text.strip()
            
            # Remove markdown code blocks if present
            if cleaned.startswith('```json'):
                cleaned = cleaned[7:]
            if cleaned.startswith('```'):
                cleaned = cleaned[3:]
            if cleaned.endswith('```'):
                cleaned = cleaned[:-3]
            
            # Try to parse as-is first
            try:
                json.loads(cleaned)
                print("JSON is already valid after cleanup")
                return cleaned
            except json.JSONDecodeError:
                pass
            
            # Look for the health_data structure
            if '"health_data"' in cleaned and '"activity"' in cleaned:
                # Find the start of the activity array
                activity_start = cleaned.find('"activity": [')
                if activity_start != -1:
                    # Extract everything before the activity array
                    prefix = cleaned[:activity_start + len('"activity": [')]
                    
                    # Find the part after the activity array start
                    array_content = cleaned[activity_start + len('"activity": ['):]
                    
                    # Parse individual entries by looking for complete objects
                    entries = []
                    current_pos = 0
                    brace_count = 0
                    in_string = False
                    entry_start = 0
                    
                    for i, char in enumerate(array_content):
                        if char == '"' and (i == 0 or array_content[i-1] != '\\'):
                            in_string = not in_string
                        
                        if not in_string:
                            if char == '{':
                                if brace_count == 0:
                                    entry_start = i
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    # We have a complete entry
                                    entry = array_content[entry_start:i+1].strip()
                                    if entry:
                                        entries.append(entry)
                                    # Look for comma after this entry
                                    next_pos = i + 1
                                    while next_pos < len(array_content) and array_content[next_pos] in ' \n\t':
                                        next_pos += 1
                                    if next_pos < len(array_content) and array_content[next_pos] == ',':
                                        next_pos += 1
                                    # Skip to next entry
                                    while next_pos < len(array_content) and array_content[next_pos] in ' \n\t':
                                        next_pos += 1
                                    i = next_pos - 1  # -1 because the loop will increment
                    
                # Reconstruct the JSON with complete entries
                if entries:
                    print(f"Found {len(entries)} complete entries")
                    
                    # Build the repaired JSON
                    entries_json = ',\n      '.join(entries)
                    repaired_content = f'{{\n  "health_data": {{\n    "activity": [\n      {entries_json}\n    ]\n  }}\n}}'
                    
                    # Validate the repaired JSON
                    try:
                        parsed = json.loads(repaired_content)
                        print(f"Repaired JSON with {len(entries)} complete entries")
                        return repaired_content
                    except json.JSONDecodeError as e:
                        print(f"First repair attempt failed: {e}")
                        
                        # Try alternative format
                        simple_content = f'{{"health_data": [{",".join(entries)}]}}'
                        try:
                            parsed = json.loads(simple_content)
                            print(f"Repaired with alternative format: {len(entries)} entries")
                            return simple_content
                        except json.JSONDecodeError as e2:
                            print(f"Alternative format failed: {e2}")
            
            # If we can't repair, try to extract any valid JSON objects
            print("Attempting to extract valid JSON objects...")
            valid_objects = []
            
            # Look for individual JSON objects that might be valid
            lines = cleaned.split('\n')
            current_object = ""
            brace_count = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                current_object += line + "\n"
                
                # Count braces to detect complete objects
                for char in line:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                
                # If we have a complete object, try to parse it
                if brace_count == 0 and current_object.strip():
                    try:
                        obj = json.loads(current_object.strip())
                        valid_objects.append(obj)
                        current_object = ""
                    except json.JSONDecodeError:
                        # Not valid JSON, continue
                        pass
            
            if valid_objects:
                print(f"Extracted {len(valid_objects)} valid objects")
                return json.dumps({"health_data": valid_objects})
            
            # Last resort: return empty structure
            print("Could not repair JSON, returning empty structure")
            return '{"health_data": []}'
            
        except Exception as e:
            print(f"Error repairing JSON: {e}")
            return '{"health_data": []}'

class HealthFileProcessor:
    """
    Process health data files using Gemini 2.0 Flash with intelligent chunking,
    rate limiting, and incremental data building for latest 7 days extraction.
    """
    
    # Global rate limiter instance
    _rate_limiter = None
    
    def __init__(self):
        """Initialize health file processor with 1M token optimization"""
        # Initialize rate limiter for 1M token optimization (shared across all instances)
        if HealthFileProcessor._rate_limiter is None:
            HealthFileProcessor._rate_limiter = RateLimiter(target_tokens_per_request=950000)
        
        self.rate_limiter = HealthFileProcessor._rate_limiter
        self.client = None
        self.nlp = None
        self.scaler = StandardScaler()
        self.setup_ai_services()
    
    def update_progress(self, user_id: int, step: int, total_steps: int, message: str, start_time: float = None, chunks_completed: int = 0, total_chunks: int = 0):
        """Update processing progress in session with detailed tracking"""
        try:
            percentage = int((step / total_steps) * 100)
            
            # For chunk processing, add sub-progress within the step
            if step == 2 and total_chunks > 0:  # Step 2 is chunk processing
                chunk_progress = (chunks_completed / total_chunks) * 25  # 25% of total progress is for chunk processing
                percentage = 50 + int(chunk_progress)  # Start from 50% and add chunk progress
            
            progress_data = {
                'step': step,
                'total_steps': total_steps,
                'message': message,
                'percentage': percentage,
                'start_time': start_time or time.time(),
                'chunks_completed': chunks_completed,
                'total_chunks': total_chunks,
                'type': 'progress'
            }
            
            # Calculate estimated time based on actual progress
            if step > 0 and start_time:
                elapsed = time.time() - start_time
                
                if step < total_steps:
                    # Calculate time per step and estimate remaining
                    if chunks_completed > 0 and total_chunks > 0 and step == 2:
                        # During chunk processing, calculate based on chunk completion rate
                        chunk_rate = chunks_completed / elapsed if elapsed > 0 else 0
                        remaining_chunks = total_chunks - chunks_completed
                        chunk_time_remaining = remaining_chunks / chunk_rate if chunk_rate > 0 else 0
                        
                        # Add time for remaining steps (merge + final)
                        steps_remaining = total_steps - step
                        avg_step_time = elapsed / (step + (chunks_completed / total_chunks))
                        step_time_remaining = steps_remaining * avg_step_time
                        
                        estimated_seconds = chunk_time_remaining + step_time_remaining
                    else:
                        # Standard step-based calculation
                        rate = step / elapsed if elapsed > 0 else 0
                        if rate > 0:
                            remaining_steps = total_steps - step
                            estimated_seconds = remaining_steps / rate
                        else:
                            estimated_seconds = None
                    
                    progress_data['estimated_time'] = estimated_seconds
                else:
                    progress_data['estimated_time'] = 0  # Complete
            
            session[f'health_data_progress_{user_id}'] = progress_data
            # Also store in global progress store for reliable access
            PROGRESS_STORE[f'health_data_progress_{user_id}'] = progress_data
            
            print(f"✅ PROGRESS UPDATE: {percentage}% - {message}")
            
            # For chunk completion, also log the chunk details
            if chunks_completed > 0:
                print(f"  └─ Chunks: {chunks_completed}/{total_chunks} completed")
            
        except Exception as e:
            logger.error(f"Error updating progress: {e}")

    def setup_ai_services(self):
        """Setup Gemini 2.0 Flash and spaCy for health data processing"""
        try:
            # Initialize Gemini 2.0 Flash
            api_key = current_app.config.get('GEMINI_API_KEY')
            print(f"DEBUG: API key from config: {api_key}")  # Debug line
            logger.info(f"API key from config: {api_key[:10]}..." if api_key else "No API key found")
            
            if api_key and api_key != 'your_gemini_api_key_here' and api_key.strip():
                try:
                    self.client = genai.Client(api_key=api_key)
                    logger.info("Gemini 2.0 Flash initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize Gemini client: {e}")
                    self.client = None
            else:
                logger.warning("Gemini API key not configured or invalid")
                self.client = None
            
            # Initialize spaCy for NLP processing
            model_name = current_app.config.get('SPACY_MODEL', 'en_core_web_sm')
            try:
                self.nlp = spacy.load(model_name)
                logger.info(f"spaCy model '{model_name}' loaded successfully")
            except OSError:
                logger.warning(f"spaCy model '{model_name}' not found. Install with: python -m spacy download {model_name}")
                self.nlp = None
                
        except Exception as e:
            logger.error(f"Error setting up AI services: {e}")
    
    def _get_mime_type(self, filename: str) -> str:
        """Get MIME type for a file based on its extension"""
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type:
            return mime_type
        
        # Default MIME types for common health data formats
        extension = filename.lower().split('.')[-1] if '.' in filename else ''
        mime_type_map = {
            'json': 'application/json',
            'csv': 'text/csv',
            'txt': 'text/plain',
            'xml': 'application/xml',
            'pdf': 'application/pdf'
        }
        return mime_type_map.get(extension, 'application/octet-stream')
    
    def validate_health_files(self, files) -> Dict[str, Any]:
        """
        Validate uploaded health data files
        Returns validation results and file information
        """
        validation_result = {
            'valid': True,
            'files': [],
            'errors': [],
            'total_size': 0
        }
        
        allowed_extensions = current_app.config.get('ALLOWED_HEALTH_FILE_EXTENSIONS', 
                                                   {'csv', 'json', 'txt', 'xml', 'pdf'})
        max_file_size = current_app.config.get('MAX_SINGLE_FILE_SIZE', 50 * 1024 * 1024)  # 50MB per file
        max_total_size = current_app.config.get('MAX_TOTAL_SIZE', 100 * 1024 * 1024)  # 100MB total
        
        for file in files:
            if file.filename == '':
                continue
                
            file_info = {
                'filename': file.filename,
                'size': 0,
                'extension': '',
                'valid': True,
                'error': None
            }
            
            # Check file size (read current position to get size)
            file.seek(0, 2)  # Seek to end
            file_size = file.tell()
            file.seek(0)  # Reset to beginning
            file_info['size'] = file_size
            validation_result['total_size'] += file_size
            
            # Validate individual file size for Gemini File API
            if file_size > max_file_size:
                file_info['valid'] = False
                file_info['error'] = f"File size ({file_size / (1024*1024):.1f}MB) exceeds maximum allowed size ({max_file_size / (1024*1024):.0f}MB)"
                validation_result['errors'].append(f"{file.filename}: File too large")
            
            # Check file extension
            if '.' in file.filename:
                extension = file.filename.rsplit('.', 1)[1].lower()
                file_info['extension'] = extension
                
                if extension not in allowed_extensions:
                    file_info['valid'] = False
                    file_info['error'] = f"File type '{extension}' not supported"
                    validation_result['errors'].append(f"{file.filename}: Unsupported file type")
            else:
                file_info['valid'] = False
                file_info['error'] = "No file extension found"
                validation_result['errors'].append(f"{file.filename}: No file extension")
            
            validation_result['files'].append(file_info)
        
        # Check total size
        if validation_result['total_size'] > max_total_size:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Total file size ({validation_result['total_size'] / (1024*1024):.1f}MB) exceeds maximum allowed ({max_total_size / (1024*1024):.0f}MB)")
        
        if validation_result['errors']:
            validation_result['valid'] = False
        
        return validation_result
    
    def process_health_files(self, files, user_id: int) -> Dict[str, Any]:
        """
        Process multiple health data files using Gemini 2.0 Flash
        Main method for extracting and analyzing health data
        """
        if not self.client:
            return {
                'success': False,
                'error': 'Gemini API not configured. Please set GEMINI_API_KEY in environment.'
            }
        
        start_time = time.time()
        total_steps = 4  # File selection, chunking, processing, merging
        
        try:
            # Initialize progress
            self.update_progress(user_id, 0, total_steps, "Validating uploaded files...", start_time)
            
            # Validate files first
            validation = self.validate_health_files(files)
            if not validation['valid']:
                return {
                    'success': False,
                    'error': 'File validation failed',
                    'validation_errors': validation['errors']
                }
            
            # Save files temporarily
            temp_files = []
            upload_folder = Path(current_app.config.get('UPLOAD_FOLDER', 'uploads'))
            upload_folder.mkdir(exist_ok=True)
            
            print("\n" + "="*60)
            print("PROCESSING HEALTH DATA FILES")
            print("="*60)
            
            for file in files:
                if file.filename == '':
                    continue
                    
                # Create unique filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{file.filename}"
                file_path = upload_folder / safe_filename
                
                file.save(str(file_path))
                temp_files.append(str(file_path))
                print(f"✓ Saved file: {safe_filename}")
                logger.info(f"Saved file: {safe_filename}")
            
            print(f"\nTotal files to process: {len(temp_files)}")
            print("="*60)
            
            if not temp_files:
                return {
                    'success': False,
                    'error': 'No valid files were uploaded'
                }
            
            # Update progress: Start AI processing
            self.update_progress(user_id, 1, total_steps, "Starting Gemini 2.0 Flash and 2.5 Flash-Lite data extraction...", start_time)
            
            # Process files with Gemini 2.0 Flash
            print(f"\n🤖 Starting Gemini 2.0 Flash and 2.5 Flash-Lite processing...")
            logger.info(f"Processing {len(temp_files)} files with Gemini 2.0 Flash...")
            health_data = self._process_with_gemini(temp_files, user_id, start_time)
            
            if health_data.get('success'):
                print("\n✅ Gemini processing completed successfully!")
                
                # Update progress: Analyzing patterns
                self.update_progress(user_id, 3, total_steps, "Analyzing health patterns and generating insights...", start_time)
                
                # Analyze patterns with scikit-learn
                analyzed_data = self._analyze_health_patterns(health_data['data'])
                
                # Update progress: Saving data
                self.update_progress(user_id, 4, total_steps, "Saving processed health data...", start_time)
                
                # Save to database (data is already in the correct flat format from merge prompt)
                self._save_health_data(analyzed_data, user_id)
                
                # Clean up temporary files
                self._cleanup_files(temp_files)
                
                print(f"✅ Health data processing complete! Processed {len(temp_files)} files")
                print("="*60 + "\n")
                
                # Clear progress
                session.pop(f'health_data_progress_{user_id}', None)
                
                return {
                    'success': True,
                    'data': analyzed_data,
                    'files_processed': len(temp_files),
                    'processing_time': datetime.now().isoformat(),
                    'message': f"Successfully processed {len(temp_files)} health data files"
                }
            else:
                print("\n❌ Gemini processing failed!")
                print("="*60 + "\n")
                # Clean up on failure
                self._cleanup_files(temp_files)
                return health_data
            
        except Exception as e:
            print(f"\n❌ Error processing health files: {e}")
            print("="*60 + "\n")
            logger.error(f"Error processing health files: {e}")
            # Clean up on error
            if 'temp_files' in locals():
                self._cleanup_files(temp_files)
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def _process_with_gemini(self, file_paths: List[str], user_id: int, start_time: float = None) -> Dict[str, Any]:
        """
        Process files using incremental approach with CountTokens API for latest 7 days.
        Each chunk builds upon previous results using sophisticated token management.
        """
        with ProcessingLock.get_lock():  # Single user processing at a time
            try:
                logger.info(f"Starting separate chunk processing of {len(file_paths)} files for user {user_id}")
                print(f"\n🎯 Starting separate chunk processing with CountTokens API...")
                
                # PHASE 0: AI File Selection
                print(f"🔍 Phase 0: AI analyzing filenames to select relevant health data files...")
                if start_time:
                    self.update_progress(user_id, 1, 4, "AI selecting relevant health data files...", start_time)
                selected_files = self._ai_select_relevant_files(file_paths)
                
                if not selected_files:
                    logger.warning("No relevant files selected by AI")
                    return {'success': False, 'error': 'No relevant health data files found'}
                
                print(f"✅ AI selected {len(selected_files)} relevant files from {len(file_paths)} total files")
                if start_time:
                    self.update_progress(user_id, 1, 4, f"✅ Selected {len(selected_files)} relevant files from {len(file_paths)} total", start_time)
                
                # PHASE 1: Read and prepare file content
                print(f"📖 Phase 1: Reading {len(selected_files)} selected files...")
                if start_time:
                    self.update_progress(user_id, 1, 4, f"Reading {len(selected_files)} selected files and creating chunks...", start_time)
                all_files_content = []
                total_content_size = 0
                
                for i, file_path in enumerate(selected_files, 1):
                    try:
                        filename = Path(file_path).name
                        print(f"  📄 Reading {i}/{len(selected_files)}: {filename}")
                        
                        # Read file content with proper encoding
                        content = self._read_file_with_encoding(file_path)
                        if content is None:
                            continue
                        
                        file_entry = {
                            "filename": filename,
                            "content": content,
                            "size": len(content)
                        }
                        
                        all_files_content.append(file_entry)
                        total_content_size += len(content)
                            
                    except Exception as e:
                        logger.warning(f"Error reading file {filename}: {e}")
                        continue
                
                print(f"✅ Read {len(all_files_content)} files ({total_content_size/(1024*1024):.1f}MB total)")
                
                # PHASE 2: Create intelligent chunks using CountTokens API
                print(f"\n🧩 Phase 2: Creating token-optimized chunks...")
                chunks = self._create_token_based_chunks(all_files_content)
                print(f"✅ Created {len(chunks)} chunks for separate processing")
                if start_time:
                    self.update_progress(user_id, 2, 4, f"✅ Created {len(chunks)} chunks for processing", start_time, 0, len(chunks))
                
                # PHASE 3: Process each chunk with post-response rate limiting
                print(f"\n🎯 Phase 3: Processing {len(chunks)} chunks with post-response rate limiting...")
                if start_time:
                    self.update_progress(user_id, 2, 4, f"Extracting data from {len(chunks)} chunks using Gemini 2.0 Flash and 2.5 Flash-Lite...", start_time)
                print(f"⏱️  Strategy: Send → Receive → Wait 1 minute → Repeat")
                chunk_responses = []
                
                for i, chunk in enumerate(chunks, 1):
                    print(f"  📊 Processing chunk {i}/{len(chunks)} ({len(chunk['files'])} files)...")
                    if start_time:
                        self.update_progress(user_id, 2, 4, f"Processing chunk {i}/{len(chunks)} with Gemini AI...", start_time)
                    
                    try:
                        # Process chunk independently
                        chunk_result = self._process_chunk_separately(
                            chunk, i, chunks_total=len(chunks)
                        )
                        
                        if chunk_result and chunk_result.get('success'):
                            chunk_responses.append({
                                'chunk_number': i,
                                'data': chunk_result['data'],
                                'files_count': len(chunk['files']),
                                'token_usage': chunk_result.get('token_usage', 0)
                            })
                            print(f"  ✅ Chunk {i} processed successfully - {chunk_result.get('token_usage', 0):,} tokens used")
                            # Update progress after each chunk completion
                            if start_time:
                                self.update_progress(user_id, 2, 4, f"✅ Completed chunk {i}/{len(chunks)} - {chunk_result.get('token_usage', 0):,} tokens used", start_time, i, len(chunks))
                        else:
                            print(f"  ⚠️  Chunk {i} failed: {chunk_result.get('error', 'Unknown error')}")
                            if start_time:
                                self.update_progress(user_id, 2, 4, f"⚠️ Chunk {i}/{len(chunks)} failed: {chunk_result.get('error', 'Unknown error')}", start_time, i, len(chunks))
                            
                    except Exception as e:
                        logger.error(f"Error processing chunk {i}: {e}")
                        print(f"  ❌ Chunk {i} failed with error: {e}")
                        continue
                
                # PHASE 4: Final merge with post-response waiting
                print(f"\n🔗 Phase 4: AI-powered final merge of {len(chunk_responses)} chunk responses...")
                if start_time:
                    self.update_progress(user_id, 3, 4, f"Merging {len(chunk_responses)} processed chunks with AI...", start_time)
                
                if not chunk_responses:
                    return {
                        'success': False,
                        'error': 'No health data could be extracted from any chunks'
                    }
                
                # Use AI to intelligently merge all chunk responses
                final_data = self._merge_all_chunks_with_ai(chunk_responses)
                
                if not final_data.get('success'):
                    return {
                        'success': False,
                        'error': f"Final merge failed: {final_data.get('error', 'Unknown error')}"
                    }
                
                print(f"✅ Separate chunk processing completed successfully!")
                if start_time:
                    self.update_progress(user_id, 4, 4, "✅ AI merge completed - Processing finished!", start_time)
                merged_health_data = final_data.get('data', {}).get('health_data', [])
                
                # Debug: Show final merged data structure
                print(f"\n🔍 FINAL MERGED DATA STRUCTURE:")
                print("="*80)
                print(f"Final data keys: {list(final_data.get('data', {}).keys())}")
                if merged_health_data:
                    if isinstance(merged_health_data, list):
                        print(f"health_data: {len(merged_health_data)} entries (new list format)")
                        if merged_health_data:
                            print(f"  Sample entry: {merged_health_data[0] if merged_health_data else 'None'}")
                            print(f"  Date range: {merged_health_data[0].get('date_logged', 'unknown')} to {merged_health_data[-1].get('date_logged', 'unknown')}")
                    else:
                        # Legacy format handling
                        print(f"health_data: {len(merged_health_data)} categories (legacy dict format)")
                        for category, entries in merged_health_data.items():
                            print(f"{category}: {len(entries) if isinstance(entries, list) else 'N/A'} entries")
                            if entries and isinstance(entries, list):
                                print(f"  Sample entry: {entries[0] if entries else 'None'}")
                print("="*80)
                
                return {
                    'success': True,
                    'data': final_data.get('data', {})
                }
                
            except Exception as e:
                logger.error(f"Error in separate chunk processing: {e}")
                print(f"❌ Separate chunk processing failed: {e}")
                return {
                    'success': False,
                    'error': f"Separate chunk processing failed: {str(e)}"
                }
    
    def _create_token_based_chunks(self, all_files_content: List[Dict]) -> List[Dict]:
        """Create chunks based on actual token count using CountTokens API with 250k threshold"""
        try:
            chunks = []
            current_chunk = {"files": [], "content": "", "estimated_tokens": 0}
            max_tokens_per_chunk = 200000  # Keep under 250k to use Gemini 2.5 Pro with thinking
            
            base_prompt = "Extract health data from the following files:\n\n"
            base_prompt_tokens = self._count_tokens(base_prompt)
            
            for file_entry in all_files_content:
                # Format file content for inclusion
                file_text = f"=== FILE: {file_entry['filename']} ===\n{file_entry['content']}\n\n"
                
                # Count tokens for this file
                file_tokens = self._count_tokens(file_text)
                
                # Check if adding this file would exceed the limit
                total_tokens = base_prompt_tokens + current_chunk["estimated_tokens"] + file_tokens
                
                if total_tokens > max_tokens_per_chunk and current_chunk["files"]:
                    # Start a new chunk
                    chunks.append(current_chunk)
                    print(f"    📦 Chunk {len(chunks)}: {len(current_chunk['files'])} files, ~{current_chunk['estimated_tokens']:,} tokens")
                    current_chunk = {"files": [], "content": "", "estimated_tokens": 0}
                
                # Add file to current chunk
                current_chunk["files"].append(file_entry)
                current_chunk["content"] += file_text
                current_chunk["estimated_tokens"] += file_tokens
            
            # Add the last chunk if it has files
            if current_chunk["files"]:
                chunks.append(current_chunk)
                print(f"    📦 Chunk {len(chunks)}: {len(current_chunk['files'])} files, ~{current_chunk['estimated_tokens']:,} tokens")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating token-based chunks: {e}")
            # Fallback: simple size-based chunking
            return self._create_size_based_chunks(all_files_content)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's CountTokens API"""
        try:
            response = self.client.models.count_tokens(
                model='gemini-2.0-flash',  # Use a consistent model for counting
                contents=[types.Part(text=text)]
            )
            return response.total_tokens
            
        except Exception as e:
            logger.warning(f"CountTokens API failed: {e}")
            # Fallback: rough estimation (1 token ≈ 4 characters)
            return len(text) // 4
    
    def _create_size_based_chunks(self, all_files_content: List[Dict]) -> List[Dict]:
        """Fallback method: create chunks based on character count"""
        chunks = []
        current_chunk = {"files": [], "content": "", "estimated_tokens": 0}
        max_chars_per_chunk = 3200000  # Roughly 800K tokens
        
        for file_entry in all_files_content:
            file_text = f"=== FILE: {file_entry['filename']} ===\n{file_entry['content']}\n\n"
            
            if len(current_chunk["content"]) + len(file_text) > max_chars_per_chunk and current_chunk["files"]:
                chunks.append(current_chunk)
                current_chunk = {"files": [], "content": "", "estimated_tokens": 0}
            
            current_chunk["files"].append(file_entry)
            current_chunk["content"] += file_text
            current_chunk["estimated_tokens"] = len(current_chunk["content"]) // 4
        
        if current_chunk["files"]:
            chunks.append(current_chunk)
        
        return chunks
    
    def _process_chunk_for_data_extraction(self, chunk: Dict, chunk_number: int) -> Dict[str, Any]:
        """Process a single chunk to extract raw health data using optimized model selection"""
        try:
            # Determine model and configuration based on chunk token count
            chunk_tokens = chunk.get('estimated_tokens', 0)
            
            if chunk_tokens < 250000:
                # Use Gemini 2.5 Flash-Lite (NO thinking) for smaller chunks - cleaner JSON output
                model_name = 'gemini-2.5-flash-lite'
                max_tokens = 65536  # Maximum output tokens for Flash-Lite
                use_thinking = False  # Flash-Lite doesn't support thinking
                print(f"  🚀 Using Gemini 2.5 Flash-Lite (no thinking) (~{chunk_tokens:,} tokens)")
            else:
                # Use Gemini 2.0 Flash for larger chunks
                model_name = 'gemini-2.0-flash'
                max_tokens = 8192  # Maximum output tokens for 2.0 Flash
                use_thinking = False
                print(f"  ⚡ Using Gemini 2.0 Flash (~{chunk_tokens:,} tokens)")
            
            # This entire extraction method is now unused - processing happens through _process_chunk_separately
            return {
                'success': False,
                'error': 'This extraction method is deprecated - use _process_chunk_separately instead'
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Deprecated extraction method error: {str(e)}'
            }
    
    def _clean_json_response(self, response_text: str) -> str:
        """
        Clean JSON response text by removing markdown code blocks and other formatting.
        
        Thinking models often return JSON wrapped in ```json ... ``` blocks,
        which need to be removed before parsing.
        """
        if not response_text:
            return response_text
        
        # Remove leading/trailing whitespace
        cleaned = response_text.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith('```json'):
            # Find the end of the opening ```json
            start_idx = cleaned.find('\n', 7)  # 7 = len('```json')
            if start_idx != -1:
                cleaned = cleaned[start_idx + 1:]
        elif cleaned.startswith('```'):
            # Generic code block
            start_idx = cleaned.find('\n')
            if start_idx != -1:
                cleaned = cleaned[start_idx + 1:]
        
        # Remove closing code block
        if cleaned.endswith('```'):
            end_idx = cleaned.rfind('\n```')
            if end_idx != -1:
                cleaned = cleaned[:end_idx]
        
        # Remove any remaining backticks at start/end
        cleaned = cleaned.strip('`').strip()
        
        return cleaned

    def _is_json_complete(self, json_text: str) -> tuple[bool, str]:
        """
        Check if JSON text appears to be complete by counting braces and brackets.
        
        Returns:
            tuple: (is_complete, diagnostic_message)
        """
        if not json_text:
            return False, "Empty JSON text"
        
        # Count opening and closing braces/brackets
        open_braces = json_text.count('{')
        close_braces = json_text.count('}')
        open_brackets = json_text.count('[')
        close_brackets = json_text.count(']')
        
        # Check if they match
        braces_match = open_braces == close_braces
        brackets_match = open_brackets == close_brackets
        
        if not braces_match or not brackets_match:
            return False, f"Mismatched braces ({open_braces}/{close_braces}) or brackets ({open_brackets}/{close_brackets})"
        
        # Check if it ends properly (not in the middle of a field)
        stripped = json_text.strip()
        if not (stripped.endswith('}') or stripped.endswith(']')):
            return False, f"JSON doesn't end with }} or ], ends with: {repr(stripped[-20:])}"
        
        return True, "JSON appears complete"

    def _extract_response_text(self, response, model_name: str, use_thinking: bool) -> str:
        """
        Robust response text extraction that handles both standard and thinking models.
        
        The new Google GenAI SDK may have different response structures for thinking models.
        This method tries multiple access patterns to ensure we get the response text.
        """
        try:
            # First try the standard response.text approach
            if response.text is not None:
                return response.text
            
            # If response.text is None, try accessing through candidates structure
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    if hasattr(candidate.content, 'parts') and candidate.content.parts:
                        # Try to get text from first part
                        first_part = candidate.content.parts[0]
                        if hasattr(first_part, 'text') and first_part.text:
                            return first_part.text
                        
                        # For thinking models, look for non-thought parts
                        if use_thinking:
                            for part in candidate.content.parts:
                                if hasattr(part, 'thought') and not part.thought:
                                    # This is a non-thought part (the actual response)
                                    if hasattr(part, 'text') and part.text:
                                        return part.text
                                elif hasattr(part, 'text') and part.text and not hasattr(part, 'thought'):
                                    # This part doesn't have thought attribute, so it's likely the response
                                    return part.text
            
            # Log debug information if we can't extract text
            print(f"⚠️  Could not extract response text for {model_name} (thinking: {use_thinking})")
            print(f"  response.text: {getattr(response, 'text', 'NOT_FOUND')}")
            print(f"  response type: {type(response)}")
            
            if hasattr(response, 'candidates'):
                print(f"  candidates: {len(response.candidates) if response.candidates else 0}")
                if response.candidates and len(response.candidates) > 0:
                    candidate = response.candidates[0]
                    print(f"  candidate.content: {getattr(candidate, 'content', 'NOT_FOUND')}")
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        parts = candidate.content.parts
                        print(f"  parts: {len(parts) if parts else 0}")
                        for i, part in enumerate(parts or []):
                            print(f"    part[{i}]: text={hasattr(part, 'text')}, thought={hasattr(part, 'thought')}")
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            print(f"❌ Error in _extract_response_text: {e}")
            return None

    def _ai_select_relevant_files(self, file_paths: List[str]) -> List[str]:
        """Use AI to analyze filenames and select only relevant health data files"""
        try:
            # Create list of just filenames for analysis
            filenames = [Path(fp).name for fp in file_paths]
            
            selection_prompt = f"""Analyze these {len(filenames)} filenames and select ONLY files that contain health/fitness/wellness data from ANY device or platform.

CRITICAL: Go through ALL the lines of ALL the file names provided. Make sure you review EVERY SINGLE file name in the list - do not skip any.

TAKE YOUR TIME: Process this thoroughly and carefully. Spend as much time as needed to ensure you examine every filename completely. Think through each filename systematically and comprehensively.

INCLUDE files containing health data from ANY source:
- Heart rate: heart_rate, hr, cardiac, pulse, heartbeat, bpm
- Sleep: sleep, slp, bedtime, rest, dream, wake, nap
- Activity: steps, step, walk, run, exercise, workout, activity, calories, cal, distance, pace, speed
- Body metrics: weight, bmi, body, fat, muscle, mass, height, composition
- Vitals: blood, bp, pressure, oxygen, spo2, temperature, temp, respiratory, breath
- Nutrition: nutrition, food, water, hydration, diet, meal, calorie, macro, vitamin
- Health tracking: health, medical, vital, symptom, medication, glucose, insulin

DEVICE/PLATFORM patterns (include ALL):
- Samsung Health: samsung.health, shealth, s_health
- Apple Health: apple.health, healthkit, hk, health_export
- Fitbit: fitbit, fb, fit_bit
- Garmin: garmin, grm, connect
- Google Fit: google.fit, gfit, googlefit
- Polar: polar, pol
- Strava: strava, str
- MyFitnessPal: myfitnesspal, mfp
- Withings: withings, wit
- Oura: oura, ring
- Any other health/fitness app data

EXCLUDE non-health files:
- System: config, settings, preferences, dashboard, ui, interface, theme
- Technical: cache, temp, log, debug, error, backup, sync, metadata, index
- Generic: .blob, .cache, system files without health indicators

PROCESS ALL FILE NAMES: Read through every single line of file names. Check each one individually against the criteria above. Take your time to be thorough and complete.

SYSTEMATIC APPROACH: Examine each filename methodically. Look for health-related keywords, device names, and data types. Be comprehensive in your analysis.

Return ONLY the selected filenames, one per line, no extra text or explanations:

{chr(10).join(filenames)}"""

            # Wait for appropriate time before making the request
            self.rate_limiter.wait_for_next_call('gemini-2.0-flash')
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[types.Part(text=selection_prompt)],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=8192  # Maximum output tokens for 2.0 Flash
                )
            )
            
            # Mark response received with model information
            self.rate_limiter.mark_response_received('gemini-2.0-flash')
            logger.info("✅ File selection response received from AI")
            
            # Robust response text extraction (file selection uses Gemini 2.0 Flash without thinking)
            response_text = self._extract_response_text(response, 'gemini-2.0-flash', False)
            
            # Log raw response for debugging
            print("\n" + "="*80)
            print("🔍 RAW FILENAME SELECTION RESPONSE:")
            print("="*80)
            if response_text:
                print(response_text)
            else:
                print("❌ FILENAME SELECTION RESPONSE TEXT IS NONE OR EMPTY!")
            print("="*80 + "\n")
            
            self.rate_limiter.wait_after_response()
            
            if not response_text:
                logger.warning("No response text from AI file selection, using all files")
                return file_paths
            
            # Parse the response to get selected filenames
            selected_filenames = []
            for line in response_text.strip().split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and not line.startswith('Selected'):
                    # Clean up any extra formatting
                    line = line.replace('- ', '').replace('* ', '').strip()
                    if line in filenames:
                        selected_filenames.append(line)
            
            # Convert back to full file paths
            selected_paths = []
            for filepath in file_paths:
                if Path(filepath).name in selected_filenames:
                    selected_paths.append(filepath)
            
            logger.info(f"AI selected {len(selected_paths)} relevant files from {len(file_paths)} total")
            
            # Show some examples of selected vs excluded files
            excluded_count = len(file_paths) - len(selected_paths)
            if excluded_count > 0:
                excluded_examples = [Path(fp).name for fp in file_paths if fp not in selected_paths][:5]
                print(f"  📋 Selected: {len(selected_paths)} files")
                print(f"  🗑️  Excluded: {excluded_count} files (e.g., {', '.join(excluded_examples[:3])}{'...' if len(excluded_examples) > 3 else ''})")
            
            return selected_paths
            
        except Exception as e:
            logger.warning(f"AI file selection failed: {e}")
            # Even if request fails, wait 1 minute before proceeding
            self.rate_limiter.wait_after_response()
            # Fallback: use basic filename filtering
            print(f"  ⚠️  AI selection failed, using basic filename filtering...")
            return self._basic_file_filter(file_paths)
    
    def _basic_file_filter(self, file_paths: List[str]) -> List[str]:
        """Fallback method: basic filename filtering for health data from any platform"""
        health_keywords = [
            # Heart & cardiovascular
            'heart', 'hr', 'cardiac', 'pulse', 'bpm', 'heartbeat',
            # Sleep
            'sleep', 'slp', 'bedtime', 'rest', 'dream', 'wake', 'nap',
            # Activity & exercise  
            'step', 'walk', 'run', 'exercise', 'workout', 'activity', 'calorie', 'cal',
            'distance', 'pace', 'speed', 'active', 'move',
            # Body metrics
            'weight', 'bmi', 'body', 'fat', 'muscle', 'mass', 'height', 'composition',
            # Vitals
            'blood', 'pressure', 'bp', 'oxygen', 'spo2', 'temperature', 'temp', 'glucose',
            # Nutrition
            'nutrition', 'food', 'water', 'hydration', 'diet', 'meal', 'macro', 'vitamin',
            # Health platforms
            'health', 'vital', 'samsung.health', 'shealth', 'apple.health', 'healthkit',
            'fitbit', 'garmin', 'google.fit', 'polar', 'strava', 'myfitnesspal', 'withings', 'oura'
        ]
        
        excluded_keywords = [
            'config', 'setting', 'preference', 'dashboard', 'cache', 'temp',
            'log', 'debug', 'error', 'backup', 'sync', 'metadata', 'index',
            'ui', 'interface', 'theme', 'layout', '.blob', '.cache', 'system'
        ]
        
        selected_files = []
        for filepath in file_paths:
            filename = Path(filepath).name.lower()
            
            # Check if any health keywords are present
            has_health_keyword = any(keyword in filename for keyword in health_keywords)
            
            # Check if any excluded keywords are present
            has_excluded_keyword = any(keyword in filename for keyword in excluded_keywords)
            
            if has_health_keyword and not has_excluded_keyword:
                selected_files.append(filepath)
        
        logger.info(f"Basic filter selected {len(selected_files)} files from {len(file_paths)} total")
        return selected_files

    def _combine_all_files_to_text(self, file_paths: List[str]) -> str:
        """Combine all files into one text string - no size limits"""
        combined_parts = []
        
        for i, file_path in enumerate(file_paths, 1):
            try:
                filename = Path(file_path).name
                logger.info(f"Reading file {i}/{len(file_paths)}: {filename}")
                
                # Try different encodings
                content = None
                for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        with open(file_path, 'r', encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    logger.warning(f"Could not read {filename} with any encoding, trying binary mode")
                    try:
                        with open(file_path, 'rb') as f:
                            binary_content = f.read()
                        # Try to decode as UTF-8 with error handling
                        content = binary_content.decode('utf-8', errors='replace')
                    except Exception as e:
                        logger.warning(f"Could not read {filename} at all: {e}")
                        continue
                
                if content and content.strip():
                    # Add file header and content
                    file_section = f"""
=== FILE {i}: {filename} ===
{content}
=== END OF FILE {i} ===

"""
                    combined_parts.append(file_section)
                    logger.info(f"Successfully read {filename} ({len(content)} characters)")
                else:
                    logger.warning(f"File {filename} is empty or unreadable")
                    
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {e}")
                continue
        
        combined_content = "\n".join(combined_parts)
        logger.info(f"Combined {len(combined_parts)} files into {len(combined_content)} characters")
        
        return combined_content

    def _analyze_health_patterns(self, health_data: Dict) -> Dict:
        """Analyze health patterns using scikit-learn for enhanced insights"""
        try:
            if not health_data or 'health_metrics' not in health_data:
                return health_data
            
            metrics = health_data['health_metrics']
            
            # Extract time series data for pattern analysis
            daily_features = self._extract_daily_features(metrics)
            
            if daily_features:
                # Cluster analysis for identifying health patterns
                patterns = self._identify_health_patterns(daily_features)
                health_data['ml_analysis'] = patterns
            
            # NLP analysis of text data
            if self.nlp:
                text_insights = self._analyze_text_insights(health_data)
                health_data['nlp_insights'] = text_insights
            
            return health_data
            
        except Exception as e:
            logger.error(f"Error in pattern analysis: {e}")
            return health_data
    
    def _extract_daily_features(self, metrics: Dict) -> Optional[np.ndarray]:
        """Extract numerical features for ML analysis"""
        try:
            features = []
            
            # Extract available metrics
            if 'steps' in metrics and 'daily_data' in metrics['steps']:
                steps_data = [day.get('steps', 0) for day in metrics['steps']['daily_data']]
                features.extend(steps_data)
            
            if 'sleep' in metrics and 'daily_data' in metrics['sleep']:
                sleep_data = [day.get('duration_hours', 0) for day in metrics['sleep']['daily_data']]
                features.extend(sleep_data)
            
            if 'heart_rate' in metrics and 'daily_data' in metrics['heart_rate']:
                hr_data = [day.get('resting_hr', 0) for day in metrics['heart_rate']['daily_data']]
                features.extend(hr_data)
            
            if features:
                # Reshape for scikit-learn
                return np.array(features).reshape(1, -1)
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _identify_health_patterns(self, features: np.ndarray) -> Dict:
        """Use scikit-learn to identify health patterns"""
        try:
            # Normalize features
            normalized_features = self.scaler.fit_transform(features)
            
            # Simple pattern analysis (expand based on needs)
            patterns = {
                'feature_analysis': {
                    'mean_values': normalized_features.mean(axis=1).tolist(),
                    'variance': normalized_features.var(axis=1).tolist(),
                    'consistency_score': float(1.0 / (1.0 + normalized_features.var()))
                },
                'health_trend': 'stable',  # Could be enhanced with more data
                'recommendation_priority': 'moderate'
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in pattern identification: {e}")
            return {'error': 'Pattern analysis failed'}
    
    def _analyze_text_insights(self, health_data: Dict) -> Dict:
        """Use spaCy for text analysis of health insights"""
        try:
            insights = health_data.get('insights', {})
            text_content = ' '.join(insights.get('key_findings', []) + 
                                  insights.get('recommendations', []))
            
            if not text_content or not self.nlp:
                return {}
            
            doc = self.nlp(text_content)
            
            # Extract key entities and concepts
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            keywords = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
            
            return {
                'key_entities': entities,
                'health_keywords': list(set(keywords)),
                'sentiment_indicators': 'positive' if 'good' in text_content.lower() else 'neutral'
            }
            
        except Exception as e:
            logger.error(f"Error in NLP analysis: {e}")
            return {}
    
    def _transform_to_daily_format(self, health_data: Dict) -> Dict:
        """Transform extracted health data into daily format expected by database"""
        try:
            logger.info("Transforming health data to daily format...")
            
            # Get the raw health data categories
            raw_data = health_data.get('health_data', {})
            
            # Collect all unique dates from all categories
            all_dates = set()
            for category, entries in raw_data.items():
                if isinstance(entries, list):
                    for entry in entries:
                        if isinstance(entry, dict) and 'date' in entry:
                            all_dates.add(entry['date'])
            
            # Create daily records
            daily_health_data = []
            for date in sorted(all_dates):
                daily_record = {
                    'date': date,
                    'activity': {},
                    'heart_rate': {},
                    'sleep': {},
                    'exercise_sessions': [],
                    'advanced_metrics': {},
                    'body_composition': {},
                    'lifestyle': {}
                }
                
                # Process heart rate data for this date
                heart_rate_entries = [entry for entry in raw_data.get('heart_rate', []) 
                                    if entry.get('date') == date]
                if heart_rate_entries:
                    hr_entry = heart_rate_entries[0]  # Take first entry for this date
                    daily_record['heart_rate'] = {
                        'avg': hr_entry.get('avg'),
                        'max': hr_entry.get('max'),
                        'min': hr_entry.get('min'),
                        'resting': hr_entry.get('resting'),
                        'variability': hr_entry.get('variability')
                    }
                
                # Process exercise/activity data for this date
                exercise_entries = [entry for entry in raw_data.get('exercise', []) 
                                  if entry.get('date') == date]
                for exercise in exercise_entries:
                    session = {
                        'workout_type': exercise.get('type'),
                        'duration_minutes': exercise.get('duration_min'),
                        'calories': exercise.get('calories'),
                        'intensity': exercise.get('intensity')
                    }
                    daily_record['exercise_sessions'].append(session)
                
                # Process steps data for this date
                steps_entries = [entry for entry in raw_data.get('steps', []) 
                               if entry.get('date') == date]
                if steps_entries:
                    steps_entry = steps_entries[0]
                    daily_record['activity']['steps'] = steps_entry.get('count', steps_entry.get('steps'))
                    daily_record['activity']['distance_km'] = steps_entry.get('distance_km')
                    if not daily_record['activity'].get('calories_total'):  # Don't override if already set
                        daily_record['activity']['calories_total'] = steps_entry.get('calories')
                    daily_record['activity']['active_minutes'] = steps_entry.get('active_minutes')
                    daily_record['activity']['floors_climbed'] = steps_entry.get('floors_climbed')
                
                # Process sleep data for this date
                sleep_entries = [entry for entry in raw_data.get('sleep', []) 
                               if entry.get('date') == date]
                if sleep_entries:
                    sleep_entry = sleep_entries[0]
                    daily_record['sleep'] = {
                        'duration_hours': sleep_entry.get('duration_hours'),
                        'quality_score': sleep_entry.get('quality_score'),
                        'deep_minutes': sleep_entry.get('deep_minutes'),
                        'light_minutes': sleep_entry.get('light_minutes'),
                        'rem_minutes': sleep_entry.get('rem_minutes'),
                        'awake_minutes': sleep_entry.get('awake_minutes')
                    }
                
                # Process nutrition data for this date
                nutrition_entries = [entry for entry in raw_data.get('nutrition', []) 
                                   if entry.get('date') == date]
                if nutrition_entries:
                    nutrition_entry = nutrition_entries[0]
                    daily_record['lifestyle']['water_intake_liters'] = nutrition_entry.get('water_liters', nutrition_entry.get('water_ml', 0) / 1000 if nutrition_entry.get('water_ml') else None)
                    if not daily_record['activity'].get('calories_total'):  # Don't override calories from activity
                        daily_record['activity']['calories_total'] = nutrition_entry.get('calories_consumed', nutrition_entry.get('calories'))
                
                # Process vitals/wellness data for this date
                vitals_entries = [entry for entry in raw_data.get('vitals', []) 
                                if entry.get('date') == date]
                if vitals_entries:
                    vitals_entry = vitals_entries[0]
                    daily_record['advanced_metrics']['blood_oxygen_percent'] = vitals_entry.get('blood_oxygen', vitals_entry.get('oxygen_saturation'))
                    daily_record['advanced_metrics']['stress_level'] = vitals_entry.get('stress_level')
                    daily_record['advanced_metrics']['body_temperature'] = vitals_entry.get('temperature', vitals_entry.get('body_temp'))
                    daily_record['body_composition']['weight_kg'] = vitals_entry.get('weight_kg', vitals_entry.get('weight'))
                    daily_record['body_composition']['body_fat_percent'] = vitals_entry.get('body_fat', vitals_entry.get('body_fat_percent'))
                    daily_record['body_composition']['muscle_mass_kg'] = vitals_entry.get('muscle_mass_kg', vitals_entry.get('muscle_mass'))
                
                wellness_entries = [entry for entry in raw_data.get('wellness', []) 
                                  if entry.get('date') == date]
                if wellness_entries:
                    wellness_entry = wellness_entries[0]
                    daily_record['lifestyle']['mood_score'] = wellness_entry.get('mood_score')
                    daily_record['lifestyle']['energy_level'] = wellness_entry.get('energy_level')
                
                daily_health_data.append(daily_record)
            
            # Preserve original structure but add daily_health_data
            transformed_data = health_data.copy()
            transformed_data['daily_health_data'] = daily_health_data
            
            logger.info(f"Transformed data into {len(daily_health_data)} daily records")
            print(f"📈 TRANSFORMED {len(daily_health_data)} DAILY RECORDS")
            
            return transformed_data
            
        except Exception as e:
            logger.error(f"Error transforming data to daily format: {e}")
            return health_data  # Return original data if transformation fails

    def _save_health_data(self, health_data: Dict, user_id: int):
        """Save processed health data to database with comprehensive metrics"""
        try:
            # Save overall summary record
            summary_record = HealthData(
                user_id=user_id,
                data_source='file_upload',
                processed_data=json.dumps(health_data),
                extraction_date=datetime.utcnow(),
                health_score=health_data.get('health_insights', {}).get('overall_health_score', 0.0),
                device_type=', '.join(health_data.get('extraction_summary', {}).get('devices_detected', []))
            )
            
            # Process daily health data if available
            daily_data = health_data.get('health_data', [])  # Changed from 'daily_health_data' to 'health_data'
            for day_record in daily_data:
                try:
                    date_logged = datetime.strptime(day_record['date_logged'], '%Y-%m-%d').date()  # Changed from 'date' to 'date_logged'
                    
                    # Check if record already exists for this date
                    existing_record = HealthData.query.filter_by(
                        user_id=user_id, 
                        date_logged=date_logged
                    ).first()
                    
                    if existing_record:
                        # Update existing record with new data
                        daily_health_record = existing_record
                    else:
                        # Create new record
                        daily_health_record = HealthData(
                            user_id=user_id,
                            date_logged=date_logged,
                            data_source='file_upload',
                            extraction_date=datetime.utcnow()
                        )
                    
                    # Extract and save activity metrics (direct from day_record)
                    daily_health_record.steps = day_record.get('steps')
                    daily_health_record.distance_km = day_record.get('distance_km')
                    daily_health_record.calories_total = day_record.get('calories_total')  # Now using correct field name
                    daily_health_record.active_minutes = day_record.get('active_minutes')
                    daily_health_record.floors_climbed = day_record.get('floors_climbed')
                    
                    # Extract and save workout details (direct from day_record)
                    daily_health_record.workout_type = day_record.get('workout_type')
                    daily_health_record.workout_duration_minutes = day_record.get('workout_duration_minutes')
                    daily_health_record.workout_intensity = day_record.get('workout_intensity')
                    daily_health_record.workout_calories = day_record.get('workout_calories')
                    
                    # Extract and save heart rate metrics (direct from day_record)
                    daily_health_record.heart_rate_avg = day_record.get('heart_rate_avg')
                    daily_health_record.heart_rate_resting = day_record.get('heart_rate_resting')
                    daily_health_record.heart_rate_max = day_record.get('heart_rate_max')
                    daily_health_record.heart_rate_variability = day_record.get('heart_rate_variability')
                    
                    # Extract and save sleep metrics (direct from day_record) - Now using correct field names
                    daily_health_record.sleep_duration_hours = day_record.get('sleep_duration_hours')
                    daily_health_record.sleep_quality_score = day_record.get('sleep_quality_score')
                    daily_health_record.sleep_deep_minutes = day_record.get('sleep_deep_minutes')  # Now in minutes directly
                    daily_health_record.sleep_rem_minutes = day_record.get('sleep_rem_minutes')  # Now in minutes directly
                    daily_health_record.sleep_awake_minutes = day_record.get('sleep_awake_minutes')
                    
                    # Extract and save advanced metrics (direct from day_record) - Now using correct field names
                    daily_health_record.blood_oxygen_percent = day_record.get('blood_oxygen_percent')  # Now using correct field name
                    daily_health_record.stress_level = day_record.get('stress_level')
                    daily_health_record.body_temperature = day_record.get('body_temperature')
                    
                    # Extract and save body composition metrics (direct from day_record)
                    daily_health_record.weight_kg = day_record.get('weight_kg')
                    daily_health_record.body_fat_percent = day_record.get('body_fat_percent')  # Now using correct field name
                    daily_health_record.muscle_mass_kg = day_record.get('muscle_mass_kg')
                    
                    # Extract and save nutrition metrics (direct from day_record)
                    daily_health_record.water_intake_liters = day_record.get('water_intake_liters')
                    daily_health_record.calories_consumed = day_record.get('calories_consumed')
                    daily_health_record.protein_grams = day_record.get('protein_grams')
                    daily_health_record.carbs_grams = day_record.get('carbs_grams')
                    daily_health_record.fat_grams = day_record.get('fat_grams')
                    daily_health_record.fiber_grams = day_record.get('fiber_grams')
                    
                    # Extract and save vital signs (direct from day_record)
                    daily_health_record.systolic_bp = day_record.get('systolic_bp')
                    daily_health_record.diastolic_bp = day_record.get('diastolic_bp')
                    
                    # Extract and save body composition (direct from day_record)
                    daily_health_record.bmi = day_record.get('bmi')
                    
                    # Extract and save wellness metrics (direct from day_record)
                    daily_health_record.mood_score = day_record.get('mood_score')
                    daily_health_record.energy_level = day_record.get('energy_level')
                    daily_health_record.meditation_minutes = day_record.get('meditation_minutes')
                    daily_health_record.screen_time_hours = day_record.get('screen_time_hours')
                    daily_health_record.social_interactions = day_record.get('social_interactions')
                    
                    # Extract any notes
                    daily_health_record.notes = day_record.get('notes')
                    
                    # Set device information
                    daily_health_record.device_type = ', '.join(health_data.get('extraction_summary', {}).get('devices_detected', []))
                    
                    if not existing_record:
                        db.session.add(daily_health_record)
                    
                except Exception as e:
                    logger.error(f"Error processing daily record for {day_record.get('date', 'unknown')}: {e}")
                    continue
            
            # Add summary record
            db.session.add(summary_record)
            db.session.commit()
            
            logger.info(f"Saved {len(daily_data)} daily health records and 1 summary record for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error saving health data: {e}")
            db.session.rollback()
            raise
    
    def _read_file_with_encoding(self, file_path: str) -> str:
        """Read file content with proper encoding handling"""
        filename = Path(file_path).name
        content = None
        
        for encoding in ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
        
        if content is None:
            logger.warning(f"Could not read {filename} with any encoding")
            return None
            
        return content
    
    def _create_token_based_chunks(self, files_content: List[Dict]) -> List[Dict]:
        """
        Dynamic Model Selection: Create optimized chunks for model selection
        Uses CountTokens API for precision - <250K for 2.5 Flash with thinking, >250K for Flash 2.0
        """
        chunks = []
        current_chunk = {"files": [], "estimated_content_tokens": 0}
        
        # Target under 250K total tokens for Gemini 2.5 Flash with thinking
        max_content_tokens = 200000  # Content tokens - leave room for prompt
        base_prompt_tokens = 50000   # Estimated prompt overhead
        
        print(f"🎯 Dynamic Model Selection: Creating optimized chunks...")
        print(f"   Target: <250K tokens for Gemini 2.5 Flash with thinking, >250K for Flash 2.0")
        print(f"   Base prompt: {base_prompt_tokens:,} tokens")
        
        for file_data in files_content:
            filename = file_data["filename"]
            content = file_data["content"]
            
            # Use CountTokens API for precise measurement
            try:
                # For large files, sample and extrapolate to avoid excessive CountTokens calls
                if len(content) > 500000:  # If content > 500K chars
                    sample_content = content[:200000]  # Sample first 200K chars
                    
                    token_response = self.client.models.count_tokens(
                        model='gemini-2.0-flash',  # Use consistent model for counting
                        contents=[sample_content]
                    )
                    
                    # Extrapolate tokens for full content
                    sample_ratio = len(sample_content) / len(content)
                    estimated_tokens = int(token_response.total_tokens / sample_ratio)
                    
                else:
                    # For smaller files, count exact tokens
                    token_response = self.client.models.count_tokens(
                        model='gemini-2.0-flash',
                        contents=[content]
                    )
                    estimated_tokens = token_response.total_tokens
                
                print(f"  📊 {filename}: {estimated_tokens:,} tokens")
                
            except Exception as e:
                logger.warning(f"CountTokens failed for {filename}: {e}")
                # Fallback: estimation (1 token ≈ 4 characters for health data)
                estimated_tokens = len(content) // 4
                print(f"  📊 {filename}: ~{estimated_tokens:,} tokens (estimated)")
            
            # Check if adding this file would exceed our target
            if (current_chunk["estimated_content_tokens"] + estimated_tokens) > max_content_tokens and current_chunk["files"]:
                # Current chunk is full - save it and start new one
                chunks.append(current_chunk)
                print(f"  📦 Chunk {len(chunks)} complete: {len(current_chunk['files'])} files, {current_chunk['estimated_content_tokens']:,} content tokens")
                current_chunk = {"files": [], "estimated_content_tokens": 0}
            
            # Add file to current chunk
            current_chunk["files"].append(file_data)
            current_chunk["estimated_content_tokens"] += estimated_tokens
        
        # Add the final chunk if it has files
        if current_chunk["files"]:
            chunks.append(current_chunk)
            print(f"  📦 Chunk {len(chunks)} complete: {len(current_chunk['files'])} files, {current_chunk['estimated_content_tokens']:,} content tokens")
        
        print(f"✅ Created {len(chunks)} maximum-sized chunks for 1M token optimization")
        for i, chunk in enumerate(chunks, 1):
            total_estimated = chunk['estimated_content_tokens'] + base_prompt_tokens
            print(f"  🎯 Chunk {i}: {chunk['estimated_content_tokens']:,} content + {base_prompt_tokens:,} prompt = ~{total_estimated:,} total tokens")
        
        return chunks
    
    def _process_chunk_separately(self, chunk: Dict, chunk_num: int, chunks_total: int) -> Dict:
        """
        Post-Response Rate Limiting: Combine files, send request, wait 1 minute AFTER response
        """
        try:
            print(f"\n🎯 Processing chunk {chunk_num}/{chunks_total} with combined file content...")
            
            # STEP 1: Combine all files in chunk into single content
            print(f"  📁 Combining {len(chunk['files'])} files into single content...")
            combined_content = self._combine_chunk_files(chunk['files'])
            
            # STEP 2: Create comprehensive extraction prompt
            extraction_prompt = self._create_comprehensive_extraction_prompt(chunk_num, chunks_total)
            
            # STEP 3: Create complete request content
            full_request_content = f"{extraction_prompt}\n\n=== CHUNK {chunk_num} COMBINED HEALTH DATA ===\n{combined_content}"
            
            # STEP 4: Get precise token count for the complete request and select model
            try:
                print(f"  📊 Measuring tokens for complete request...")
                token_response = self.client.models.count_tokens(
                    model='gemini-2.0-flash',  # Use consistent model for counting
                    contents=[full_request_content]
                )
                total_tokens = token_response.total_tokens
                print(f"  🎯 Request will use {total_tokens:,} tokens")
                
                if total_tokens > 950000:
                    print(f"  ⚠️  WARNING: Request exceeds 950K token target!")
                
            except Exception as e:
                logger.warning(f"CountTokens failed for chunk {chunk_num}: {e}")
                total_tokens = len(full_request_content) // 4
                print(f"  📊 Estimated {total_tokens:,} tokens (CountTokens failed)")
            
            # STEP 5: Select model based on token count - UPDATED to use non-thinking models
            if total_tokens < 250000:
                # Use Gemini 2.5 Flash-Lite (NO thinking) for smaller chunks - cleaner JSON output
                model_name = 'gemini-2.5-flash-lite'
                max_tokens = 65536  # Maximum output tokens for Flash-Lite
                use_thinking = False  # Flash-Lite doesn't support thinking
                print(f"  🚀 Using Gemini 2.5 Flash-Lite (no thinking) (~{total_tokens:,} tokens)")
            else:
                # Use Gemini 2.0 Flash for larger chunks
                model_name = 'gemini-2.0-flash'
                max_tokens = 8192  # Maximum output tokens for 2.0 Flash
                use_thinking = False
                print(f"  ⚡ Using Gemini 2.0 Flash (~{total_tokens:,} tokens)")
            
            # STEP 6: Wait for appropriate time before making the request
            self.rate_limiter.wait_for_next_call(model_name)
            print(f"  🚀 Sending request to {model_name}...")
            start_time = time.time()
            
            # Configure generation - Use response_mime_type for clean JSON
            config = types.GenerateContentConfig(
                temperature=0,  # Deterministic for health data
                max_output_tokens=max_tokens,
                response_mime_type="application/json"  # Forces clean JSON output
            )

            # Process with selected model
            response = self.client.models.generate_content(
                model=model_name,
                contents=[full_request_content],
                config=config
            )
            
            processing_time = time.time() - start_time
            print(f"  ✅ Response received in {processing_time:.1f} seconds")
            
            # Robust response text extraction for both standard and thinking models
            response_text = self._extract_response_text(response, model_name, use_thinking)
            
            # Log raw response for debugging
            print("\n" + "="*80)
            print(f"🔍 RAW CHUNK {chunk_num} RESPONSE:")
            print("="*80)
            if response_text:
                print(response_text)
            else:
                print("❌ RESPONSE TEXT IS NONE OR EMPTY!")
                print("Response object debug info:")
                print(f"  response.text: {response.text}")
                print(f"  response type: {type(response)}")
                if hasattr(response, 'candidates'):
                    print(f"  candidates count: {len(response.candidates) if response.candidates else 0}")
            print("="*80 + "\n")
            
            # STEP 6: Record response received with model information
            self.rate_limiter.mark_response_received(model_name)
            self.rate_limiter.record_usage(total_tokens)
            
            # STEP 7: Parse and validate response
            if not response_text:
                print(f"  ❌ No response text extracted for chunk {chunk_num}")
                if chunk_num < chunks_total:
                    print(f"  ⏳ Waiting 1 minute after receiving response...")
                    self.rate_limiter.wait_after_response()
                return {
                    'success': False,
                    'error': 'No response text could be extracted from API response',
                    'token_usage': total_tokens
                }
            
            try:
                # Clean the response text to remove markdown code blocks
                cleaned_response = self._clean_json_response(response_text)
                
                # Add debug info about cleaning
                if cleaned_response != response_text:
                    print(f"  🧹 Cleaned response (removed markdown formatting)")
                
                # Check if JSON appears complete
                is_complete, completeness_msg = self._is_json_complete(cleaned_response)
                if not is_complete:
                    print(f"  ⚠️  JSON appears incomplete: {completeness_msg}")
                    # Try to repair the JSON before giving up
                    print(f"  🔧 Attempting JSON repair...")
                    repaired_response = JSONRepair.repair_broken_json(cleaned_response)
                    try:
                        chunk_data = json.loads(repaired_response)
                        print(f"  ✅ JSON repair successful for chunk {chunk_num}")
                    except json.JSONDecodeError as repair_error:
                        print(f"  ❌ JSON repair failed: {repair_error}")
                        return {
                            'success': False,
                            'error': f'JSON repair failed: {completeness_msg}',
                            'raw_response': response_text[:1000] if response_text else 'No response text',
                            'cleaned_response': cleaned_response[:1000] if cleaned_response else 'No cleaned response',
                            'repaired_response': repaired_response[:1000] if repaired_response else 'No repaired response',
                            'token_usage': total_tokens
                        }
                else:
                    # Try normal parsing first
                    try:
                        chunk_data = json.loads(cleaned_response)
                        print(f"  ✅ JSON parsing successful for chunk {chunk_num}")
                    except json.JSONDecodeError as parse_error:
                        print(f"  ❌ JSON parsing failed: {parse_error}")
                        print(f"  🔧 Attempting JSON repair...")
                        repaired_response = JSONRepair.repair_broken_json(cleaned_response)
                        try:
                            chunk_data = json.loads(repaired_response)
                            print(f"  ✅ JSON repair successful for chunk {chunk_num}")
                        except json.JSONDecodeError as repair_error:
                            print(f"  ❌ JSON repair failed: {repair_error}")
                            return {
                                'success': False,
                                'error': f'JSON parsing and repair failed: {parse_error}',
                                'raw_response': response_text[:1000] if response_text else 'No response text',
                                'cleaned_response': cleaned_response[:1000] if cleaned_response else 'No cleaned response',
                                'repaired_response': repaired_response[:1000] if repaired_response else 'No repaired response',
                                'token_usage': total_tokens
                            }
                
                # Validate and fix the structure
                chunk_data = HealthDataValidator.validate_and_fix_health_data(chunk_data)
                
                # Show extraction statistics
                if 'health_data' in chunk_data:
                    health_entries = chunk_data['health_data']
                    if isinstance(health_entries, list):
                        print(f"  📊 Extracted {len(health_entries)} health entries (fixed structure)")
                        # Show sample entry fields
                        if health_entries:
                            sample_fields = [k for k, v in health_entries[0].items() if v is not None]
                            print(f"  📂 Sample fields: {', '.join(sample_fields[:10])}")
                    else:
                        # Old format handling
                        categories = list(health_entries.keys()) if isinstance(health_entries, dict) else []
                        total_entries = sum(len(entries) if isinstance(entries, list) else 0 
                                          for entries in health_entries.values()) if isinstance(health_entries, dict) else 0
                        print(f"  📊 Extracted {total_entries} health entries across {len(categories)} categories")
                        print(f"  📂 Categories: {', '.join(categories)}")
                
                # STEP 8: Wait 1 minute AFTER receiving response (if not last chunk)
                if chunk_num < chunks_total:
                    print(f"  ⏳ Waiting 1 minute after receiving response...")
                    self.rate_limiter.wait_after_response()
                
                return {
                    'success': True,
                    'data': chunk_data,
                    'token_usage': total_tokens,
                    'processing_time': processing_time
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for chunk {chunk_num}: {e}")
                print(f"  ❌ JSON parsing failed for chunk {chunk_num}")
                print(f"  🔍 Error details: {str(e)}")
                print(f"  📝 Raw response length: {len(response_text) if response_text else 0}")
                print(f"  🧹 Cleaned response length: {len(cleaned_response) if 'cleaned_response' in locals() else 0}")
                if response_text:
                    print(f"  📄 First 200 chars of raw response: {repr(response_text[:200])}")
                if 'cleaned_response' in locals() and cleaned_response:
                    print(f"  🧽 First 200 chars of cleaned response: {repr(cleaned_response[:200])}")
                return {
                    'success': False, 
                    'error': 'Invalid JSON response',
                    'json_error': str(e),
                    'raw_response': response_text[:1000] if response_text else 'No response text',  # First 1K chars for debugging
                    'cleaned_response': cleaned_response[:1000] if 'cleaned_response' in locals() and cleaned_response else 'No cleaned response',
                    'token_usage': total_tokens
                }
                
        except Exception as e:
            logger.error(f"Error processing chunk {chunk_num}: {e}")
            print(f"  ❌ Error processing chunk {chunk_num}: {e}")
            # Even if request fails, wait 1 minute before proceeding
            self.rate_limiter.wait_after_response()
            return {'success': False, 'error': str(e)}
    
    def _combine_chunk_files(self, files_data: List[Dict]) -> str:
        """Combine all files in a chunk into single content string"""
        combined_parts = []
        
        for i, file_data in enumerate(files_data, 1):
            filename = file_data['filename']
            content = file_data['content']
            
            if content and content.strip():
                # Add file section with clear delimiters
                file_section = f"""
=== FILE {i}: {filename} ===
{content}
=== END OF FILE {i}: {filename} ===

"""
                combined_parts.append(file_section)
                print(f"    📄 Added {filename} ({len(content):,} characters)")
            else:
                print(f"    ⚠️  Skipped empty file: {filename}")
        
        combined_content = "\n".join(combined_parts)
        print(f"  ✅ Combined {len(combined_parts)} files into {len(combined_content):,} characters")
        
        return combined_content
    
    def _create_comprehensive_extraction_prompt(self, chunk_num: int, chunks_total: int) -> str:
        """Create comprehensive extraction prompt for independent chunk processing"""
        
        return f"""Extract health data from chunk {chunk_num} of {chunks_total}.

🎯 CRITICAL PRIORITY: ONE RECORD PER DAY - PROPER AGGREGATION
- If multiple readings exist for the SAME DATE, aggregate them into ONE record per day
- ⚠️  STRICT DATE BOUNDARIES: ONLY aggregate records from the EXACT SAME DATE (YYYY-MM-DD)
- Never mix data from different dates - each day must be completely separate
- Use proper aggregation method for each metric type:

📊 AGGREGATION RULES BY METRIC TYPE:
**EXAMPLE:** If you find steps: 3000 at 9am, 2500 at 2pm, 1800 at 6pm all on 2024-01-15, 
→ Create ONE record for 2024-01-15 with steps: 7300 (3000+2500+1800)

**CUMULATIVE METRICS (ADD all values for same date):**
- steps: Add all step counts for the same date
- distance_km: Add all distance values for the same date
- calories_total: Add all total calorie values for the same date  
- calories_consumed: Add all food calories for the same date
- active_minutes: Add all active time for the same date
- floors_climbed: Add all floors climbed for the same date
- workout_duration_minutes: Add all workout durations for the same date
- workout_calories: Add all workout calories for the same date
- water_intake_liters: Add all water intake for the same date
- protein_grams, carbs_grams, fat_grams, fiber_grams: Add all nutrition for the same date
- meditation_minutes: Add all meditation time for the same date
- screen_time_hours: Add all screen time for the same date
- sleep_deep_minutes, sleep_rem_minutes, sleep_awake_minutes: Add all sleep phase minutes for the same date

**VITAL SIGNS (AVERAGE all values for same date):**
- heart_rate_avg, heart_rate_resting, heart_rate_max: Average all readings for the same date
- systolic_bp, diastolic_bp: Average all blood pressure readings for the same date
- blood_oxygen_percent: Average all SpO2 readings for the same date
- stress_level: Average all stress measurements for the same date
- body_temperature: Average all temperature readings for the same date

**SLEEP METRICS (COMBINE for same date):**
- sleep_duration_hours: Add all sleep sessions for the same date
- sleep_deep_minutes, sleep_rem_minutes: Add all deep/REM time for the same date
- sleep_quality_score: Average all quality scores for the same date

**BODY COMPOSITION (LATEST value for same date):**
- weight_kg: Use the latest/last measurement for the same date
- body_fat_percent: Use the latest measurement for the same date
- muscle_mass_kg: Use the latest measurement for the same date
- bmi: Calculate from latest weight for the same date

**WELLNESS SCORES (AVERAGE for same date):**
- mood_score: Average all mood entries for the same date
- energy_level: Average all energy scores for the same date
- social_interactions: Count/add all social interactions for the same date

**EXERCISE SESSION METRICS (COMBINE for same date):**
- workout_type: Use primary/longest workout type for the same date
- workout_intensity: Use highest intensity for the same date

🔍 THOROUGH PROCESSING REQUIRED:
- Read EVERY line in EVERY file provided
- Process ALL files completely to find the latest 14 days of data
- Take maximum time for quality - thoroughness over speed
- Go through all content multiple times if needed

📊 DATA EXTRACTION RULES:
- Only extract values that actually exist in the files - NO hallucination
- Use exact values from the files - NO approximation
- 🚨 CRITICAL: Use ONLY the exact value found for that SPECIFIC day - DO NOT copy/repeat the same value across different dates
- Each day record must contain ONLY the actual data recorded for that specific date
- If a day has no recorded data for a metric, OMIT that field completely (don't include null/empty fields)
- 🎯 FOCUS: Extract ONLY the latest 14 days of actual data from the files
- Works with ANY health platform: Samsung Health, Apple Health, Fitbit, Garmin, etc.

🎯 OUTPUT: Return ONLY valid JSON (no markdown, no explanations):

{{
  "health_data": [
    {{
      "date_logged": "YYYY-MM-DD",
      "data_source": "file_upload",
      
      // SLEEP & REST METRICS
      "sleep_duration_hours": 7.5,
      "sleep_quality_score": 8,
      "sleep_deep_minutes": 90,
      "sleep_rem_minutes": 105,
      "sleep_awake_minutes": 15,
      
      // PHYSICAL ACTIVITY METRICS
      "steps": 10000,
      "active_minutes": 45,
      "calories_total": 450,
      "distance_km": 6.5,
      "workout_type": "Running",
      "workout_duration_minutes": 60,
      "floors_climbed": 5,
      
      // VITAL SIGNS
      "heart_rate_avg": 72,
      "heart_rate_resting": 60,
      "systolic_bp": 120,
      "diastolic_bp": 80,
      "blood_oxygen_percent": 98,
      "body_temperature": 36.8,
      
      // NUTRITION & HYDRATION
      "water_intake_liters": 2.5,
      "calories_consumed": 2000,
      "protein_grams": 80.5,
      "carbs_grams": 250.0,
      "fat_grams": 65.5,
      "fiber_grams": 25.0,
      
      // BODY COMPOSITION
      "weight_kg": 70.5,
      "body_fat_percent": 18.5,
      "muscle_mass_kg": 45.2,
      "bmi": 22.5,
      
      // MENTAL HEALTH & LIFESTYLE
      "mood_score": 8,
      "stress_level": 3,
      "energy_level": 7,
      "meditation_minutes": 20,
      "screen_time_hours": 6.5,
      "social_interactions": 5,
      
      // EXERCISE SESSION DETAILS
      "workout_intensity": "moderate",
      "workout_calories": 450,
      
      // ADDITIONAL HEART RATE METRICS
      "heart_rate_max": 180,
      "heart_rate_variability": 32.5,
      
      // NOTES
      "notes": "Optional health data notes"
    }}
  ],
  "extraction_summary": {{
    "files_processed": {chunk_num},
    "devices_detected": ["Samsung Health", "Apple Health", etc.],
    "date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
    "total_data_points": number
  }}
}}

Focus on the latest 14 days of actual data. Take your time for thorough, quality processing.
🚨 REMEMBER: Each day must contain ONLY the exact data for that specific date - DO NOT repeat identical values across different days."""
    
    def _merge_all_chunks_with_ai(self, chunk_responses: List[Dict]) -> Dict:
        """Use AI to intelligently merge all chunk responses into final health data"""
        try:
            print(f"🤖 AI merging {len(chunk_responses)} chunk responses with 1M token optimization...")
            
            # Prepare merge request data
            merge_data = {
                "total_chunks": len(chunk_responses),
                "chunk_responses": chunk_responses
            }
            
            # Create merge prompt
            merge_prompt = f"""You are a health data integration specialist. Merge {len(chunk_responses)} separate chunk responses into a single comprehensive health dataset.

CRITICAL: DO NOT ADD, MODIFY, OR INVENT ANY DATA. Only combine existing data from chunks.

CRITICAL: Go through ALL the lines in ALL the responses provided. Make sure you process EVERY SINGLE line of EVERY SINGLE response - do not skip any content.

CRITICAL: Extract ONLY the latest 14 days of data for each unique person/user found in the data.
SCANNING STRATEGY: Even if chunks contain 30+ days of data, extract only the most recent 14 days for each person.
DUPLICATE HANDLING: If multiple chunks have the same date/person data, merge them intelligently (don't duplicate).
VALUE CONVERSION: Apply sensible unit conversion for unrealistic values (excessive active minutes, distance, etc.)

TAKE MAXIMUM TIME: Process this merge as thoroughly as possible. Spend extensive time analyzing every piece of data from all chunks. This is critical for producing the highest quality final output.

COMPREHENSIVE MERGING PROCESS:
1. Read through every single chunk response completely
2. Identify all unique data points across all chunks
3. Focus on the latest 14 days of data for each person (scan wider ranges if available)
4. Carefully merge data from the same dates across chunks, avoiding duplicates
5. Verify no data is lost or duplicated incorrectly
6. Cross-reference all merged data for consistency
7. Take your time - thorough merging is essential for quality

TASK: Intelligently combine health data from {len(chunk_responses)} processing chunks into final structured format.

INPUT: {len(chunk_responses)} chunk responses with extracted health data from various platforms (Samsung Health, Apple Health, Fitbit, Garmin, etc.)

STRICT MERGE REQUIREMENTS:
1. ONLY use data present in the chunk responses
2. DO NOT create new dates, values, or data points
3. COMBINE entries from same date across chunks (if multiple exist)
4. DEDUPLICATE identical entries, keep most complete data
5. PRESERVE exact dates and values from chunks
6. SORT by date (newest first) for each category
7. DO NOT estimate, approximate, or calculate new values
8. PROCESS ALL RESPONSES COMPLETELY: Read through every line of every response
9. SYSTEMATIC MERGING: Take extensive time to ensure perfect data integration
10. 🎯 EXTRACT ONLY THE LATEST 14 days of data for each person (search wider to ensure coverage)
11. 🔧 CONVERT VALUES SENSIBLY: Apply unit conversion for unrealistic values from any health platform
12. 🚨 NULL FIELD HANDLING: For any fields that have no data, OMIT them completely from the JSON - do NOT include fields with null, 0, or empty values

EXACT JSON STRUCTURE REQUIRED (match this precisely):
{{
  "health_data": [
    {{
      "date_logged": "YYYY-MM-DD",
      "data_source": "file_upload",
      
      // SLEEP & REST METRICS
      "sleep_duration_hours": 7.5,
      "sleep_quality_score": 8,
      "sleep_deep_minutes": 90,
      "sleep_rem_minutes": 105,
      "sleep_awake_minutes": 15,
      
      // PHYSICAL ACTIVITY METRICS
      "steps": 10000,
      "active_minutes": 45,
      "calories_total": 450,
      "distance_km": 6.5,
      "workout_type": "Running",
      "workout_duration_minutes": 60,
      "floors_climbed": 5,
      
      // VITAL SIGNS
      "heart_rate_avg": 72,
      "heart_rate_resting": 60,
      "heart_rate_max": 180,
      "heart_rate_variability": 32.5,
      "systolic_bp": 120,
      "diastolic_bp": 80,
      "blood_oxygen_percent": 98,
      "body_temperature": 36.8,
      
      // NUTRITION & HYDRATION
      "water_intake_liters": 2.5,
      "calories_consumed": 2000,
      "protein_grams": 80.5,
      "carbs_grams": 250.0,
      "fat_grams": 65.5,
      "fiber_grams": 25.0,
      
      // BODY COMPOSITION
      "weight_kg": 70.5,
      "body_fat_percent": 18.5,
      "muscle_mass_kg": 45.2,
      "bmi": 22.5,
      
      // MENTAL HEALTH & LIFESTYLE
      "mood_score": 8,
      "stress_level": 3,
      "energy_level": 7,
      "meditation_minutes": 20,
      "screen_time_hours": 6.5,
      "social_interactions": 5,
      
      // EXERCISE SESSION DETAILS
      "workout_intensity": "moderate",
      "workout_calories": 450
    }},
  ],
  "processing_summary": {{
    "total_chunks_merged": {len(chunk_responses)},
    "final_date_range": {{"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"}},
    "total_entries": 14,
    "latest_14_days_only": true
  }}
}}

IMPORTANT MERGE RULES:
1. For missing fields: OMIT them completely (don't include null/empty fields)
2. Ensure dates are in YYYY-MM-DD format  
3. All numeric fields must be actual numbers
4. Set data_source to "file_upload" for all entries
5. If multiple readings per day, aggregate appropriately (avg for heart rate, sum for steps, etc.)
6. ONLY include the latest 14 days of data for each unique person (scan wider ranges to ensure coverage)
7. 🚨 DEDUPLICATE: Remove duplicate entries from same date/time/person across chunks
8. 🔧 CONVERT VALUES SENSIBLY: Fix unrealistic values from any platform (Samsung, Apple, Garmin, etc.):
   - Active minutes: Must be 0-1440 (max minutes per day). If >1440, likely seconds → convert to minutes
   - Distance: Must be realistic daily values (0-50km typically). If >100km, likely in different units → convert appropriately  
   - Steps: Reasonable daily range (0-50000). If excessive, check for data errors
   - Heart rate: 30-220 BPM range. Values outside this are likely errors
   - Sleep duration: 0-24 hours max. Convert if in wrong units
   - Sleep minutes: Deep/REM/awake should be in minutes, not hours
   - Calories: Reasonable daily ranges (0-10000). Convert if necessary
   - Body fat: Should be percentage (5-60%), not decimal
   - All metrics: Apply common sense unit conversion and error detection

CHUNK RESPONSES TO MERGE:
{json.dumps(merge_data, indent=1)}

RETURN ONLY VALID JSON - NO MARKDOWN, NO EXPLANATIONS, NO CODE BLOCKS
Return the final merged health dataset for the latest 14 days (with wider scanning for coverage) using the exact structure specified:"""

            # Get precise token count for merge request
            try:
                print(f"  📊 Measuring tokens for merge request...")
                token_response = self.client.models.count_tokens(
                    model='gemini-2.0-flash',  # Use consistent model for counting
                    contents=[merge_prompt]
                )
                total_tokens = token_response.total_tokens
                print(f"  🎯 Merge request will use {total_tokens:,} tokens")
            except Exception as e:
                logger.warning(f"CountTokens failed for merge: {e}")
                total_tokens = len(merge_prompt) // 4
                print(f"  📊 Estimated {total_tokens:,} tokens for merge (CountTokens failed)")
            
            # Wait for appropriate time before making the merge request
            self.rate_limiter.wait_for_next_call('gemini-2.0-flash')
            print(f"  🚀 Sending merge request to Gemini 2.0 Flash...")
            start_time = time.time()
            
            # Process merge with Gemini 2.0 Flash (maximum tokens)
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=[merge_prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                    max_output_tokens=8192  # Maximum output tokens for 2.0 Flash
                )
            )
            
            processing_time = time.time() - start_time
            print(f"  ✅ Merge response received in {processing_time:.1f} seconds")
            
            # Robust response text extraction (merge uses Gemini 2.0 Flash without thinking)
            response_text = self._extract_response_text(response, 'gemini-2.0-flash', False)
            
            # Log raw response for debugging
            print("\n" + "="*80)
            print("🔍 RAW MERGE RESPONSE:")
            print("="*80)
            if response_text:
                print(response_text)
            else:
                print("❌ MERGE RESPONSE TEXT IS NONE OR EMPTY!")
            print("="*80 + "\n")
            
            # Record response received with model information
            self.rate_limiter.mark_response_received('gemini-2.0-flash')
            # Token usage tracking
            self.rate_limiter.record_usage(total_tokens)
            
            # Final response completed - no additional wait needed (handled by next request)
            
            # Parse final response
            if not response_text:
                return {
                    'success': False,
                    'error': 'No response text could be extracted from merge API response'
                }
            
            try:
                final_data = json.loads(response_text)
                print(f"✅ Successfully parsed merge response")
                
                # Validate and fix the final structure
                final_data = HealthDataValidator.validate_and_fix_health_data(final_data)
                print(f"✅ Successfully merged and validated {len(chunk_responses)} chunks")
                
                # Show final statistics
                if 'health_data' in final_data:
                    health_entries = final_data['health_data']
                    if isinstance(health_entries, list):
                        print(f"📊 Final dataset: {len(health_entries)} health entries (latest 14 days)")
                        if health_entries:
                            dates = [entry.get('date_logged', 'unknown') for entry in health_entries]
                            print(f"📅 Date range: {min(dates)} to {max(dates)}")
                    else:
                        # Fallback for old format
                        total_entries = sum(len(entries) for entries in health_entries.values()) if isinstance(health_entries, dict) else 0
                        print(f"📊 Final dataset: {total_entries} total health entries across {len(health_entries)} categories")
                
                return {
                    'success': True,
                    'data': final_data
                }
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse final merge response: {e}")
                return {'success': False, 'error': 'Invalid JSON in final merge response'}
                
        except Exception as e:
            logger.error(f"Error in AI chunk merging: {e}")
            # Even if request fails, wait 1 minute before proceeding
            self.rate_limiter.wait_after_response()
            return {'success': False, 'error': str(e)}
    
    def _cleanup_files(self, file_paths: List[str]):
        """Clean up temporary files"""
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Could not delete {file_path}: {e}")

# Global processor instance - created lazily within Flask context
_health_processor = None

def get_health_processor():
    """Get health processor instance, creating it if needed within Flask context"""
    global _health_processor
    if _health_processor is None:
        _health_processor = HealthFileProcessor()
    return _health_processor
