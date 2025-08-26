"""
Health File Processor with Gemini 2.5 Flash Integration

This module processes health data files from multiple wearable devices:
- Samsung Health (Galaxy Watch, Galaxy Ring, etc.)
- Apple Health
- Fitbit
- Garmin
- Other fitness trackers

Uses Gemini 2.5 Flash for intelligent data extraction and analysis.
Based on successful test implementation and research requirements.
"""

import os
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from flask import current_app, flash
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

class HealthFileProcessor:
    """
    Process health data files using Gemini 2.5 Flash and scikit-learn
    Implements research requirements for multi-device health data integration
    """
    
    def __init__(self):
        """Initialize health file processor with AI models"""
        self.client = None
        self.nlp = None
        self.scaler = StandardScaler()
        self.setup_ai_services()
    
    def setup_ai_services(self):
        """Setup Gemini 2.5 Flash and spaCy for health data processing"""
        try:
            # Initialize Gemini 2.5 Flash
            api_key = current_app.config.get('GEMINI_API_KEY')
            print(f"DEBUG: API key from config: {api_key}")  # Debug line
            logger.info(f"API key from config: {api_key[:10]}..." if api_key else "No API key found")
            
            if api_key and api_key != 'your_gemini_api_key_here' and api_key.strip():
                try:
                    self.client = genai.Client(api_key=api_key)
                    logger.info("Gemini 2.5 Flash initialized successfully")
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
        max_size = current_app.config.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024)
        
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
        
        if validation_result['errors']:
            validation_result['valid'] = False
        
        return validation_result
    
    def process_health_files(self, files, user_id: int) -> Dict[str, Any]:
        """
        Process multiple health data files using Gemini 2.5 Flash
        Main method for extracting and analyzing health data
        """
        if not self.client:
            return {
                'success': False,
                'error': 'Gemini API not configured. Please set GEMINI_API_KEY in environment.'
            }
        
        try:
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
            
            for file in files:
                if file.filename == '':
                    continue
                    
                # Create unique filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                safe_filename = f"{timestamp}_{file.filename}"
                file_path = upload_folder / safe_filename
                
                file.save(str(file_path))
                temp_files.append(str(file_path))
                logger.info(f"Saved file: {safe_filename}")
            
            if not temp_files:
                return {
                    'success': False,
                    'error': 'No valid files were uploaded'
                }
            
            # Process files with Gemini 2.5 Flash
            logger.info(f"Processing {len(temp_files)} files with Gemini 2.5 Flash...")
            health_data = self._process_with_gemini(temp_files)
            
            if health_data.get('success'):
                # Analyze patterns with scikit-learn
                analyzed_data = self._analyze_health_patterns(health_data['data'])
                
                # Save to database
                self._save_health_data(analyzed_data, user_id)
                
                # Clean up temporary files
                self._cleanup_files(temp_files)
                
                return {
                    'success': True,
                    'data': analyzed_data,
                    'files_processed': len(temp_files),
                    'processing_time': datetime.now().isoformat(),
                    'message': f"Successfully processed {len(temp_files)} health data files"
                }
            else:
                # Clean up on failure
                self._cleanup_files(temp_files)
                return health_data
            
        except Exception as e:
            logger.error(f"Error processing health files: {e}")
            # Clean up on error
            if 'temp_files' in locals():
                self._cleanup_files(temp_files)
            return {
                'success': False,
                'error': f"Processing failed: {str(e)}"
            }
    
    def _process_with_gemini(self, file_paths: List[str]) -> Dict[str, Any]:
        """Process all files by combining them into one text and sending to Gemini 2.5 Flash"""
        try:
            logger.info(f"Processing {len(file_paths)} files by combining into single text")
            
            # Combine all files into one text
            combined_content = self._combine_all_files_to_text(file_paths)
            
            if not combined_content.strip():
                return {
                    'success': False,
                    'error': 'No readable content found in any files'
                }
            
            # Create comprehensive health data extraction prompt
            prompt = f"""
{self._create_health_extraction_prompt()}

IMPORTANT: Please analyze ALL the content below thoroughly. Go through every line and extract all available health data. The content includes data from {len(file_paths)} health data files.

Here is the combined health data content to analyze:

{combined_content}

Please analyze ALL of the above content completely and extract all health metrics according to the JSON schema provided.
"""
            
            logger.info(f"Sending combined content ({len(combined_content)} characters) to Gemini")
            
            # Process with Gemini 2.5 Flash
            response = self.client.models.generate_content(
                model=current_app.config.get('GEMINI_MODEL', 'gemini-2.5-flash'),
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    thinking_config=types.ThinkingConfig(thinking_budget=0)  # Disable thinking for speed
                )
            )
            
            # Parse response
            health_data = json.loads(response.text)
            logger.info("Successfully parsed Gemini response")
            
            return {
                'success': True,
                'data': health_data
            }
            
        except Exception as e:
            logger.error(f"Error in Gemini processing: {e}")
            return {
                'success': False,
                'error': f"Gemini processing failed: {str(e)}"
            }
    
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
    
    def _create_health_extraction_prompt(self) -> str:
        """Create comprehensive prompt for Gemini 2.5 Flash health data extraction"""
        return """
You are a health data extraction specialist. Analyze the provided health data files from wearable devices (Samsung Health, Apple Health, Fitbit, Garmin, etc.) and extract structured information.

IMPORTANT: Please analyze ALL the content provided. Go through every single line of data and extract all available health metrics. Do not skip any sections or files.

Extract and organize the following health metrics with proper data types and ranges:

**ESSENTIAL ACTIVITY METRICS:**
1. Steps - Daily step count (integer, 0-50,000)
2. Distance - Distance traveled in kilometers (float, 0-100.0)
3. Calories - Total calories burned (integer, 0-8,000)
4. Active Minutes - Minutes of movement/exercise (integer, 0-1440)
5. Floors Climbed - Flights of stairs (integer, 0-200)

**HEART RATE METRICS:**
1. Average Heart Rate - Daily average BPM (integer, 40-220)
2. Resting Heart Rate - Resting BPM (integer, 40-100)
3. Maximum Heart Rate - Peak BPM (integer, 60-220)
4. Heart Rate Variability - HRV in milliseconds (float, 10.0-100.0)

**SLEEP METRICS:**
1. Sleep Duration - Total sleep in hours (float, 0.0-24.0)
2. Sleep Quality - Quality score (integer, 0-100)
3. Deep Sleep - Deep sleep minutes (integer, 0-600)
4. Light Sleep - Light sleep minutes (integer, 0-600)
5. REM Sleep - REM sleep minutes (integer, 0-300)
6. Awake Time - Time awake in bed in minutes (integer, 0-300)

**ADVANCED HEALTH METRICS:**
1. Blood Oxygen - SpO2 percentage (integer, 70-100)
2. Stress Level - Stress score (integer, 0-100)
3. Body Temperature - Temperature in Celsius (float, 35.0-42.0)

**BODY COMPOSITION METRICS:**
1. Weight - Body weight in kilograms (float, 30.0-300.0)
2. Body Fat - Body fat percentage (float, 5.0-60.0)
3. Muscle Mass - Muscle mass in kilograms (float, 20.0-100.0)

**LIFESTYLE METRICS:**
1. Water Intake - Water consumed in liters (float, 0.0-10.0)
2. Mood Score - Mood rating (integer, 1-10)
3. Energy Level - Energy rating (integer, 1-10)

**EXERCISE SESSION DETAILS:**
1. Workout Type - Exercise type (string: cardio, strength, yoga, running, etc.)
2. Workout Duration - Exercise session minutes (integer, 0-480)
3. Workout Intensity - Intensity level (string: low, moderate, high)
4. Workout Calories - Calories burned during exercise (integer, 0-2000)

Return a comprehensive JSON structure like this:

{
  "extraction_summary": {
    "files_processed": number,
    "date_range": {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
    "data_quality": "excellent|good|fair|poor",
    "devices_detected": ["Samsung Health", "Apple Watch", "Fitbit", "Garmin", etc.]
  },
  "daily_health_data": [
    {
      "date": "YYYY-MM-DD",
      "activity": {
        "steps": integer,
        "distance_km": float,
        "calories_total": integer,
        "active_minutes": integer,
        "floors_climbed": integer
      },
      "heart_rate": {
        "avg": integer,
        "resting": integer,
        "max": integer,
        "variability": float
      },
      "sleep": {
        "duration_hours": float,
        "quality_score": integer,
        "deep_minutes": integer,
        "light_minutes": integer,
        "rem_minutes": integer,
        "awake_minutes": integer
      },
      "advanced_metrics": {
        "blood_oxygen_percent": integer,
        "stress_level": integer,
        "body_temperature": float
      },
      "body_composition": {
        "weight_kg": float,
        "body_fat_percent": float,
        "muscle_mass_kg": float
      },
      "lifestyle": {
        "water_intake_liters": float,
        "mood_score": integer,
        "energy_level": integer
      },
      "exercise_sessions": [
        {
          "workout_type": "string",
          "duration_minutes": integer,
          "intensity": "low|moderate|high",
          "calories": integer
        }
      ]
    }
  ],
  "health_insights": {
    "key_findings": ["string", "string", "string"],
    "recommendations": ["string", "string", "string"],
    "overall_health_score": number,
    "priority_areas": ["sleep", "exercise", "nutrition", "stress", etc.]
  },
  "optimization_suggestions": {
    "optimal_exercise_times": ["morning", "afternoon", "evening"],
    "hydration_reminders": ["09:00", "12:00", "15:00", "18:00"],
    "sleep_schedule": {"recommended_bedtime": "HH:MM", "recommended_wake_time": "HH:MM"},
    "stress_management": ["meditation", "deep_breathing", "light_exercise"]
  }
}

Focus on extracting accurate numerical values within the specified ranges and provide meaningful health insights based on wearable device data patterns.
"""
    
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
            daily_data = health_data.get('daily_health_data', [])
            for day_record in daily_data:
                try:
                    date_logged = datetime.strptime(day_record['date'], '%Y-%m-%d').date()
                    
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
                    
                    # Extract and save activity metrics
                    activity = day_record.get('activity', {})
                    daily_health_record.steps = activity.get('steps')
                    daily_health_record.distance_km = activity.get('distance_km')
                    daily_health_record.calories_total = activity.get('calories_total')
                    daily_health_record.active_minutes = activity.get('active_minutes')
                    daily_health_record.floors_climbed = activity.get('floors_climbed')
                    
                    # Extract and save heart rate metrics
                    heart_rate = day_record.get('heart_rate', {})
                    daily_health_record.heart_rate_avg = heart_rate.get('avg')
                    daily_health_record.heart_rate_resting = heart_rate.get('resting')
                    daily_health_record.heart_rate_max = heart_rate.get('max')
                    daily_health_record.heart_rate_variability = heart_rate.get('variability')
                    
                    # Extract and save sleep metrics
                    sleep = day_record.get('sleep', {})
                    daily_health_record.sleep_duration_hours = sleep.get('duration_hours')
                    daily_health_record.sleep_quality_score = sleep.get('quality_score')
                    daily_health_record.sleep_deep_minutes = sleep.get('deep_minutes')
                    daily_health_record.sleep_light_minutes = sleep.get('light_minutes')
                    daily_health_record.sleep_rem_minutes = sleep.get('rem_minutes')
                    daily_health_record.sleep_awake_minutes = sleep.get('awake_minutes')
                    
                    # Extract and save advanced metrics
                    advanced = day_record.get('advanced_metrics', {})
                    daily_health_record.blood_oxygen_percent = advanced.get('blood_oxygen_percent')
                    daily_health_record.stress_level = advanced.get('stress_level')
                    daily_health_record.body_temperature = advanced.get('body_temperature')
                    
                    # Extract and save body composition metrics
                    body = day_record.get('body_composition', {})
                    daily_health_record.weight_kg = body.get('weight_kg')
                    daily_health_record.body_fat_percent = body.get('body_fat_percent')
                    daily_health_record.muscle_mass_kg = body.get('muscle_mass_kg')
                    
                    # Extract and save lifestyle metrics
                    lifestyle = day_record.get('lifestyle', {})
                    daily_health_record.water_intake_liters = lifestyle.get('water_intake_liters')
                    daily_health_record.mood_score = lifestyle.get('mood_score')
                    daily_health_record.energy_level = lifestyle.get('energy_level')
                    
                    # Extract and save exercise session details (save first session if multiple)
                    exercise_sessions = day_record.get('exercise_sessions', [])
                    if exercise_sessions:
                        first_session = exercise_sessions[0]
                        daily_health_record.workout_type = first_session.get('workout_type')
                        daily_health_record.workout_duration_minutes = first_session.get('duration_minutes')
                        daily_health_record.workout_intensity = first_session.get('intensity')
                        daily_health_record.workout_calories = first_session.get('calories')
                    
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