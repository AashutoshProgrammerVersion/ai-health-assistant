"""
AI Services Module for Health Assistant

This module handles all AI-related functionality including:
- Google Gemini API integration for health advice
- Health pattern recognition using scikit-learn
- Natural language processing with spaCy
- Schedule optimization algorithms

Based on extensive research into health app features and AI recommendations.
"""

import os
import json
import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

# Suppress numpy warnings that appear during health data calculations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='numpy')

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import spacy
import google.generativeai as genai
from google.genai import types
from flask import current_app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthAIService:
    """
    Core AI service for health analysis and recommendations
    Implements features from research: personalized advice, pattern recognition, decision support
    """
    
    def __init__(self):
        """Initialize AI services with models and configurations"""
        self.nlp = None
        self.health_scaler = StandardScaler()
        self.setup_gemini()
        self.load_spacy_model()
    
    def setup_gemini(self):
        """Configure Google Gemini AI service"""
        try:
            api_key = current_app.config.get('GEMINI_API_KEY')
            if api_key and api_key not in ['demo_api_key', 'your-actual-gemini-api-key-here', 'test_api_key']:
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
                logger.info("Gemini AI configured successfully with JSON response")
            else:
                if api_key in ['demo_api_key', 'your-actual-gemini-api-key-here', 'test_api_key']:
                    logger.info("Using demo API key - AI features will use fallback responses")
                else:
                    logger.warning("Gemini API key not found")
                self.gemini_model = None
        except Exception as e:
            logger.error(f"Failed to configure Gemini AI: {e}")
            self.gemini_model = None
    
    def load_spacy_model(self):
        """Load spaCy model for natural language processing"""
        try:
            model_name = current_app.config.get('SPACY_MODEL', 'en_core_web_sm')
            self.nlp = spacy.load(model_name)
            logger.info(f"spaCy model '{model_name}' loaded successfully")
        except OSError:
            logger.error(f"spaCy model not found. Install with: python -m spacy download {model_name}")
            self.nlp = None
    
    def calculate_health_score(self, health_data_list: List) -> float:
        """
        Calculate comprehensive health score based on multiple metrics
        Implements research finding: personalized scoring motivates users
        """
        if not health_data_list:
            return 0.0
        
        # Get recent data for scoring (last 7 days or available data)
        recent_data = health_data_list[-7:] if len(health_data_list) >= 7 else health_data_list
        
        # Calculate individual scores for each category
        scores = {
            'activity': self._calculate_activity_score(recent_data),
            'sleep': self._calculate_sleep_quality_score(recent_data),
            'nutrition': self._calculate_nutrition_score(recent_data),
            'hydration': self._calculate_hydration_score(recent_data),
            'heart_health': self._calculate_heart_health_score(recent_data),
            'wellness': self._calculate_wellness_score(recent_data)
        }
        
        # Weight the scores based on importance
        weights = {
            'activity': 0.25,
            'sleep': 0.25,
            'nutrition': 0.15,
            'hydration': 0.15,
            'heart_health': 0.15,
            'wellness': 0.05
        }
        
        # Calculate weighted overall score
        weighted_score = sum(scores[metric] * weights[metric] for metric in scores)
        return round(weighted_score, 1)
    
    def calculate_individual_scores(self, health_data_list: List) -> Dict:
        """
        Calculate individual health metric scores for dashboard display
        Returns scores for activity, sleep, nutrition, hydration, etc.
        """
        if not health_data_list:
            return {
                'activity_score': 0,
                'sleep_score': 0,
                'nutrition_score': 0,
                'hydration_score': 0,
                'heart_health_score': 0,
                'wellness_score': 0
            }
        
        recent_data = health_data_list[-7:] if len(health_data_list) >= 7 else health_data_list
        
        return {
            'activity_score': round(self._calculate_activity_score(recent_data)),
            'sleep_score': round(self._calculate_sleep_quality_score(recent_data)),
            'nutrition_score': round(self._calculate_nutrition_score(recent_data)),
            'hydration_score': round(self._calculate_hydration_score(recent_data)),
            'heart_health_score': round(self._calculate_heart_health_score(recent_data)),
            'wellness_score': round(self._calculate_wellness_score(recent_data))
        }
    
    def _calculate_activity_score(self, recent_data: List) -> float:
        """Calculate activity score based on steps, active minutes, and exercise"""
        if not recent_data:
            return 0.0
        
        scores = []
        for data in recent_data:
            day_score = 0
            components = 0
            
            # Steps component (0-40 points)
            if data.steps is not None:
                if data.steps >= 10000:
                    day_score += 40
                elif data.steps >= 7500:
                    day_score += 30
                elif data.steps >= 5000:
                    day_score += 20
                elif data.steps >= 2500:
                    day_score += 10
                components += 1
            
            # Active minutes component (0-30 points)
            if data.active_minutes is not None:
                if data.active_minutes >= 60:
                    day_score += 30
                elif data.active_minutes >= 30:
                    day_score += 20
                elif data.active_minutes >= 15:
                    day_score += 10
                components += 1
            
            # Exercise session component (0-30 points)
            if data.workout_duration_minutes is not None:
                if data.workout_duration_minutes >= 45:
                    day_score += 30
                elif data.workout_duration_minutes >= 30:
                    day_score += 20
                elif data.workout_duration_minutes >= 15:
                    day_score += 10
                components += 1
            
            if components > 0:
                # Points already sum to 100 (steps: 40, active: 30, workout: 30)
                scores.append(day_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_sleep_quality_score(self, recent_data: List) -> float:
        """Calculate sleep quality score based on duration and quality metrics"""
        if not recent_data:
            return 0.0
        
        scores = []
        for data in recent_data:
            day_score = 0
            components = 0
            
            # Sleep duration component (0-50 points)
            if data.sleep_duration_hours is not None:
                if 7 <= data.sleep_duration_hours <= 9:
                    day_score += 50
                elif 6 <= data.sleep_duration_hours < 7 or 9 < data.sleep_duration_hours <= 10:
                    day_score += 35
                elif 5 <= data.sleep_duration_hours < 6 or 10 < data.sleep_duration_hours <= 11:
                    day_score += 20
                else:
                    day_score += 10
                components += 1
            
            # Sleep quality score component (0-50 points)
            if data.sleep_quality_score is not None:
                day_score += (data.sleep_quality_score / 100) * 50
                components += 1
            
            if components > 0:
                # Points already sum to 100 (duration: 50, quality: 50)
                scores.append(day_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_nutrition_score(self, recent_data: List) -> float:
        """Calculate nutrition score based on available dietary data"""
        if not recent_data:
            return 0.0  # Return 0 when no data available
        
        scores = []
        for data in recent_data:
            day_score = 50  # Start with neutral score for days with some data
            
            # Body composition trends (if available)
            if data.weight_kg is not None and data.body_fat_percent is not None:
                # Stable/healthy body composition gets higher score
                day_score = 70
            
            # Energy level as nutrition indicator
            if data.energy_level is not None:
                energy_score = (data.energy_level / 10) * 30
                day_score = max(day_score, energy_score + 20)
            
            scores.append(min(day_score, 100))
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_hydration_score(self, recent_data: List) -> float:
        """Calculate hydration score based on water intake"""
        if not recent_data:
            return 0.0
        
        scores = []
        for data in recent_data:
            if data.water_intake_liters is not None:
                # Optimal hydration: 2-3 liters per day
                if 2.0 <= data.water_intake_liters <= 3.0:
                    scores.append(100)
                elif 1.5 <= data.water_intake_liters < 2.0 or 3.0 < data.water_intake_liters <= 3.5:
                    scores.append(75)
                elif 1.0 <= data.water_intake_liters < 1.5 or 3.5 < data.water_intake_liters <= 4.0:
                    scores.append(50)
                else:
                    scores.append(25)
            else:
                scores.append(0)  # No data = 0 score
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_heart_health_score(self, recent_data: List) -> float:
        """Calculate heart health score based on HR metrics"""
        if not recent_data:
            return 0.0
        
        scores = []
        for data in recent_data:
            day_score = 0
            components = 0
            
            # Resting heart rate (0-40 points)
            if data.heart_rate_resting is not None:
                if 50 <= data.heart_rate_resting <= 70:
                    day_score += 40
                elif 40 <= data.heart_rate_resting < 50 or 70 < data.heart_rate_resting <= 85:
                    day_score += 30
                elif 85 < data.heart_rate_resting <= 100:
                    day_score += 20
                else:
                    day_score += 10
                components += 1
            
            # Heart rate variability (0-30 points)
            if data.heart_rate_variability is not None:
                if data.heart_rate_variability >= 35:
                    day_score += 30
                elif data.heart_rate_variability >= 25:
                    day_score += 20
                elif data.heart_rate_variability >= 15:
                    day_score += 10
                components += 1
            
            # Blood oxygen (0-30 points)
            if data.blood_oxygen_percent is not None:
                if data.blood_oxygen_percent >= 97:
                    day_score += 30
                elif data.blood_oxygen_percent >= 95:
                    day_score += 20
                elif data.blood_oxygen_percent >= 90:
                    day_score += 10
                components += 1
            
            if components > 0:
                # Points already sum to 100 (resting HR: 40, HRV: 30, blood O2: 30)
                scores.append(day_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_wellness_score(self, recent_data: List) -> float:
        """Calculate wellness score based on mood, energy, and stress"""
        if not recent_data:
            return 0.0
        
        scores = []
        for data in recent_data:
            day_score = 0
            components = 0
            
            # Mood score (0-40 points)
            if data.mood_score is not None:
                day_score += (data.mood_score / 10) * 40
                components += 1
            
            # Energy level (0-30 points)
            if data.energy_level is not None:
                day_score += (data.energy_level / 10) * 30
                components += 1
            
            # Stress level (0-30 points, inverted)
            if data.stress_level is not None:
                stress_score = (100 - data.stress_level) / 100 * 30
                day_score += stress_score
                components += 1
            
            if components > 0:
                # Points already sum to 100 (stress: 30, screen time: 40, mindfulness: 30)
                scores.append(day_score)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_sleep_score(self, sleep_hours: float) -> float:
        """Calculate sleep quality score based on optimal range"""
        if sleep_hours < 5:
            return 2.0
        elif sleep_hours < 6:
            return 4.0
        elif 7 <= sleep_hours <= 9:
            return 10.0
        elif 6 <= sleep_hours < 7 or 9 < sleep_hours <= 10:
            return 8.0
        else:
            return 5.0
    
    def _calculate_heart_rate_score(self, heart_rate: int) -> float:
        """Calculate heart rate score based on healthy ranges"""
        if 60 <= heart_rate <= 80:
            return 10.0
        elif 50 <= heart_rate < 60 or 80 < heart_rate <= 100:
            return 8.0
        elif 40 <= heart_rate < 50 or 100 < heart_rate <= 120:
            return 6.0
        else:
            return 3.0
    
    def analyze_health_patterns(self, health_data_list: List) -> Dict:
        """
        Analyze health patterns using machine learning
        Implements research: pattern recognition for personalized insights
        """
        if len(health_data_list) < 7:
            return {"error": "Insufficient data for pattern analysis"}
        
        try:
            # Convert to DataFrame for analysis
            data_rows = []
            for data in health_data_list:
                row = {
                    # Activity Metrics
                    'steps_count': data.steps or 0,
                    'activity_level': data.active_minutes or 0,
                    'distance_km': data.distance_km or 0,
                    'floors_climbed': data.floors_climbed or 0,
                    
                    # Sleep Metrics
                    'sleep_hours': data.sleep_duration_hours or 0,
                    'sleep_quality': data.sleep_quality_score or 0,
                    'deep_sleep_min': data.sleep_deep_minutes or 0,
                    'rem_sleep_min': data.sleep_rem_minutes or 0,
                    
                    # Heart Health
                    'heart_rate': data.heart_rate_avg or 0,
                    'resting_hr': data.heart_rate_resting or 0,
                    'hrv': data.heart_rate_variability or 0,
                    
                    # Nutrition
                    'water_intake': data.water_intake_liters or 0,
                    'calories_consumed': data.calories_consumed or 0,
                    'protein_grams': data.protein_grams or 0,
                    'carbs_grams': data.carbs_grams or 0,
                    'fiber_grams': data.fiber_grams or 0,
                    
                    # Wellness
                    'mood': data.mood_score or 0,
                    'energy': data.energy_level or 0,
                    'stress': data.stress_level or 0,
                    
                    # Lifestyle
                    'meditation_min': data.meditation_minutes or 0,
                    'screen_time_hrs': data.screen_time_hours or 0,
                    'social_interactions': data.social_interactions or 0,
                    
                    'date': data.date_logged
                }
                data_rows.append(row)
            
            df = pd.DataFrame(data_rows)
            
            # Identify correlations
            correlations = self._find_health_correlations(df)
            
            # Detect trends
            trends = self._detect_health_trends(df)
            
            # Generate insights
            insights = self._generate_health_insights(correlations, trends)
            
            return {
                'correlations': correlations,
                'trends': trends,
                'insights': insights,
                'data_points': len(health_data_list)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing health patterns: {e}")
            return {"error": str(e)}
    
    def _find_health_correlations(self, df: pd.DataFrame) -> Dict:
        """Find correlations between health metrics - Enhanced with nutrition & lifestyle"""
        # All numeric columns to analyze
        numeric_cols = [
            'steps_count', 'activity_level', 'distance_km', 'floors_climbed',
            'sleep_hours', 'sleep_quality', 'deep_sleep_min', 'rem_sleep_min',
            'heart_rate', 'resting_hr', 'hrv',
            'water_intake', 'calories_consumed', 'protein_grams', 'carbs_grams', 'fiber_grams',
            'mood', 'energy', 'stress',
            'meditation_min', 'screen_time_hrs', 'social_interactions'
        ]
        
        correlations = {}
        
        # Find correlations with mood (primary wellness indicator)
        if 'mood' in df.columns:
            for col in numeric_cols:
                if col in df.columns and col != 'mood':
                    corr_with_mood = df[col].corr(df['mood']) if 'mood' in df.columns else 0
                    correlations[f"{col}_mood_correlation"] = round(corr_with_mood, 3) if not np.isnan(corr_with_mood) else 0
        
        # Find correlations with energy (another key indicator)
        if 'energy' in df.columns:
            for col in numeric_cols:
                if col in df.columns and col != 'energy':
                    corr_with_energy = df[col].corr(df['energy']) if 'energy' in df.columns else 0
                    correlations[f"{col}_energy_correlation"] = round(corr_with_energy, 3) if not np.isnan(corr_with_energy) else 0
        
        # Key correlations: Sleep quality vs various factors
        if 'sleep_quality' in df.columns:
            for col in ['screen_time_hrs', 'meditation_min', 'stress', 'activity_level']:
                if col in df.columns:
                    corr = df[col].corr(df['sleep_quality'])
                    correlations[f"{col}_sleep_correlation"] = round(corr, 3) if not np.isnan(corr) else 0
        
        return correlations
    
    def _detect_health_trends(self, df: pd.DataFrame) -> Dict:
        """Detect trends in health metrics over time - Enhanced with all metrics"""
        trends = {}
        # Expanded list of metrics to analyze for trends
        numeric_cols = [
            'sleep_hours', 'sleep_quality', 'deep_sleep_min',
            'water_intake', 'calories_consumed', 'protein_grams', 'fiber_grams',
            'activity_level', 'steps_count', 'distance_km',
            'resting_hr', 'hrv',
            'mood', 'energy', 'stress',
            'meditation_min', 'screen_time_hrs', 'social_interactions'
        ]
        
        for col in numeric_cols:
            if col in df.columns and len(df) > 1:
                # Simple trend detection using linear regression
                x = np.arange(len(df)).reshape(-1, 1)
                y = df[col].values
                
                # Remove zeros for better trend analysis
                non_zero_indices = y > 0
                if np.sum(non_zero_indices) > 1:
                    x_filtered = x[non_zero_indices]
                    y_filtered = y[non_zero_indices]
                    
                    model = LinearRegression()
                    model.fit(x_filtered, y_filtered)
                    
                    slope = model.coef_[0]
                    if abs(slope) > 0.1:  # Significant trend threshold
                        trends[col] = 'increasing' if slope > 0 else 'decreasing'
                    else:
                        trends[col] = 'stable'
                else:
                    trends[col] = 'insufficient_data'
        
        return trends
    
    def _generate_health_insights(self, correlations: Dict, trends: Dict) -> List[str]:
        """Generate human-readable health insights - Enhanced with comprehensive analysis"""
        insights = []
        
        # Priority insights based on strong correlations with mood and energy
        priority_correlations = [
            ('sleep_hours_mood_correlation', 'sleep duration', 'mood'),
            ('sleep_quality_mood_correlation', 'sleep quality', 'mood'),
            ('activity_level_mood_correlation', 'physical activity', 'mood'),
            ('meditation_min_mood_correlation', 'meditation', 'mood'),
            ('screen_time_hrs_mood_correlation', 'screen time', 'mood'),
            ('water_intake_mood_correlation', 'hydration', 'mood'),
            ('protein_grams_energy_correlation', 'protein intake', 'energy levels'),
            ('fiber_grams_energy_correlation', 'fiber intake', 'energy levels'),
            ('social_interactions_mood_correlation', 'social interactions', 'mood'),
        ]
        
        for corr_key, metric_name, impact in priority_correlations:
            if corr_key in correlations:
                corr_value = correlations[corr_key]
                if abs(corr_value) > 0.5:  # Strong correlation
                    if corr_value > 0:
                        insights.append(f"Higher {metric_name} appears to improve your {impact}")
                    else:
                        insights.append(f"Higher {metric_name} may negatively affect your {impact}")
        
        # Sleep quality specific insights
        if 'screen_time_hrs_sleep_correlation' in correlations:
            corr = correlations['screen_time_hrs_sleep_correlation']
            if corr < -0.4:
                insights.append("Excessive screen time appears to reduce your sleep quality")
        
        if 'meditation_min_sleep_correlation' in correlations:
            corr = correlations['meditation_min_sleep_correlation']
            if corr > 0.4:
                insights.append("Meditation appears to improve your sleep quality")
        
        # Trend insights with context
        important_trends = {
            'sleep_hours': 'sleep duration',
            'water_intake': 'hydration',
            'activity_level': 'physical activity',
            'meditation_min': 'meditation practice',
            'screen_time_hrs': 'screen time',
            'stress': 'stress levels',
            'protein_grams': 'protein intake',
            'fiber_grams': 'fiber intake'
        }
        
        for metric, friendly_name in important_trends.items():
            if metric in trends:
                trend = trends[metric]
                if trend == 'increasing':
                    # Positive trends
                    if metric in ['sleep_hours', 'water_intake', 'activity_level', 'meditation_min', 'protein_grams', 'fiber_grams']:
                        insights.append(f"Great progress! Your {friendly_name} has been improving")
                    # Negative trends
                    elif metric in ['screen_time_hrs', 'stress']:
                        insights.append(f"Attention needed: Your {friendly_name} has been increasing")
                elif trend == 'decreasing':
                    # Concerning decreases
                    if metric in ['sleep_hours', 'water_intake', 'activity_level', 'meditation_min']:
                        insights.append(f"Your {friendly_name} has been declining - consider addressing this")
                    # Positive decreases
                    elif metric in ['screen_time_hrs', 'stress']:
                        insights.append(f"Excellent! Your {friendly_name} has been decreasing")
        
        # Limit to top 8 most actionable insights
        return insights[:8]
    
    def generate_personalized_advice(self, user_context: Dict, health_patterns: Dict) -> Dict:
        """
        Generate personalized health advice based on user data and AI analysis
        Implements research findings: evidence-based, personalized recommendations
        """
        if not user_context.get('recent_health'):
            return self._get_default_advice()
        
        try:
            # Use Gemini AI if available
            if self.gemini_model:
                return self._generate_ai_advice_with_gemini(user_context, health_patterns)
            else:
                return self._generate_fallback_advice(user_context, health_patterns)
        except Exception as e:
            logger.error(f"Error generating AI advice: {e}")
            return self._generate_fallback_advice(user_context, health_patterns)
    
    def _generate_ai_advice_with_gemini(self, user_context: Dict, health_patterns: Dict) -> Dict:
        """Generate advice using Gemini AI with comprehensive health context"""
        
        # Prepare context for AI
        health_summary = self._prepare_health_summary(user_context)
        goals = user_context.get('goals', {})
        
        prompt = f"""
You are a health and wellness assistant. Based on the user's health data, provide personalized advice.

RECENT HEALTH DATA (Last 7 days):
{health_summary}

USER GOALS:
- Water intake: {goals.get('water_liters', 2)} liters/day
- Sleep: {goals.get('sleep_hours', 8)} hours/night
- Steps: {goals.get('steps', 10000)} steps/day
- Active minutes: {goals.get('activity_minutes', 30)} minutes/day

HEALTH PATTERNS:
{json.dumps(health_patterns, indent=2) if health_patterns else 'Insufficient data for pattern analysis'}

INSTRUCTIONS:
Respond ONLY with valid JSON. No additional text before or after. The JSON must have this exact structure:

{{
  "insights": [
    "First specific observation about their health trends",
    "Second specific observation about their health trends"
  ],
  "recommendations": [
    "First actionable, evidence-based suggestion",
    "Second actionable, evidence-based suggestion",
    "Third actionable, evidence-based suggestion"
  ],
  "quick_wins": [
    "First small change they can implement today",
    "Second small change they can implement today"
  ],
  "concerns": [
    "Any area that needs attention (if applicable)"
  ],
  "motivation": "Single encouraging message based on their progress"
}}

GUIDELINES:
- Base advice on actual data trends, not assumptions
- Provide specific, actionable recommendations
- Be encouraging and avoid overwhelming the user
- Focus on evidence-based wellness practices
- Target health-conscious individuals aged 12-35
- Address information overload by providing clear, prioritized advice

Respond with JSON only:"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            
            if not response.text:
                logger.warning("Empty response from Gemini AI")
                return self._generate_fallback_advice(user_context, health_patterns)
            
            # Clean the response text
            response_text = response.text.strip()
            
            # Try to extract JSON if wrapped in markdown or other text
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
            
            # Try to parse JSON response
            try:
                advice_data = json.loads(response_text)
                
                # Validate the structure
                required_keys = ['insights', 'recommendations', 'quick_wins', 'concerns', 'motivation']
                for key in required_keys:
                    if key not in advice_data:
                        advice_data[key] = []
                
                # Ensure lists are actually lists and contain strings
                for key in ['insights', 'recommendations', 'quick_wins', 'concerns']:
                    if not isinstance(advice_data[key], list):
                        advice_data[key] = []
                    else:
                        # Filter out non-string items and ensure they're clean
                        advice_data[key] = [str(item).strip() for item in advice_data[key] if item and str(item).strip()]
                
                # Ensure motivation is a string
                if not isinstance(advice_data.get('motivation'), str):
                    advice_data['motivation'] = "Keep up the great work tracking your health data!"
                
                return {
                    'insights': advice_data['insights'],
                    'recommendations': advice_data['recommendations'],
                    'quick_wins': advice_data['quick_wins'],
                    'concerns': advice_data['concerns'],
                    'motivation': advice_data['motivation'],
                    'source': 'gemini_ai',
                    'generated_at': datetime.now().isoformat()
                }
            except json.JSONDecodeError as json_error:
                logger.error(f"JSON parsing failed: {json_error}")
                logger.error(f"Raw response: {response_text[:500]}...")
                # Try to parse the text manually
                return self._parse_text_advice(response_text)
                
        except Exception as e:
            logger.error(f"Gemini AI error: {e}")
            return self._generate_fallback_advice(user_context, health_patterns)
    
    def _generate_fallback_advice(self, user_context: Dict, health_patterns: Dict) -> Dict:
        """Generate advice using rule-based system when AI is unavailable - Enhanced"""
        
        recent_health = user_context.get('recent_health', [])
        goals = user_context.get('goals', {})
        
        if not recent_health:
            return self._get_default_advice()
        
        # Analyze recent data for patterns
        insights = []
        recommendations = []
        quick_wins = []
        concerns = []
        
        # Analyze activity levels
        steps_data = [day.get('steps') for day in recent_health if day.get('steps')]
        if steps_data:
            avg_steps = sum(steps_data) / len(steps_data)
            step_goal = goals.get('steps', 10000)
            
            if avg_steps >= step_goal:
                insights.append(f"Great job! You're averaging {avg_steps:.0f} steps/day, meeting your goal.")
            elif avg_steps >= step_goal * 0.8:
                insights.append(f"You're close to your step goal with {avg_steps:.0f} steps/day average.")
                recommendations.append("Try adding a 10-minute walk after meals to reach your daily step goal.")
            else:
                concerns.append("Your step count is below target. Consider gradually increasing daily activity.")
                quick_wins.append("Take the stairs instead of elevators when possible.")
        
        # Analyze sleep patterns
        sleep_data = [day.get('sleep_duration_hours') for day in recent_health if day.get('sleep_duration_hours')]
        if sleep_data:
            avg_sleep = sum(sleep_data) / len(sleep_data)
            sleep_goal = goals.get('sleep_hours', 8)
            
            if avg_sleep >= sleep_goal:
                insights.append(f"Your sleep duration is good at {avg_sleep:.1f} hours/night on average.")
            elif avg_sleep >= 6:
                recommendations.append("Try to establish a consistent bedtime routine to improve sleep duration.")
                quick_wins.append("Set a phone reminder 1 hour before your target bedtime.")
            else:
                concerns.append("Your sleep duration appears insufficient for optimal health.")
                recommendations.append("Prioritize sleep hygiene: dark room, cool temperature, no screens 1 hour before bed.")
        
        # Analyze sleep quality (enhanced)
        sleep_quality_data = [day.get('sleep_quality_score') for day in recent_health if day.get('sleep_quality_score')]
        if sleep_quality_data:
            avg_quality = sum(sleep_quality_data) / len(sleep_quality_data)
            if avg_quality < 60:
                concerns.append("Your sleep quality scores suggest room for improvement.")
                recommendations.append("Consider reducing caffeine intake after 2 PM and establishing a wind-down routine.")
        
        # Analyze hydration
        water_data = [day.get('water_intake_liters') for day in recent_health if day.get('water_intake_liters')]
        if water_data:
            avg_water = sum(water_data) / len(water_data)
            water_goal = goals.get('water_liters', 2)
            
            if avg_water >= water_goal:
                insights.append("Excellent hydration habits! Keep it up.")
            else:
                recommendations.append("Set hourly reminders to drink water throughout the day.")
                quick_wins.append("Keep a water bottle visible on your desk or workspace.")
        
        # Analyze nutrition (NEW)
        protein_data = [day.get('protein_grams') for day in recent_health if day.get('protein_grams')]
        if protein_data:
            avg_protein = sum(protein_data) / len(protein_data)
            if avg_protein < 50:  # Minimum recommended
                recommendations.append("Consider increasing protein intake to support muscle maintenance and satiety.")
        
        fiber_data = [day.get('fiber_grams') for day in recent_health if day.get('fiber_grams')]
        if fiber_data:
            avg_fiber = sum(fiber_data) / len(fiber_data)
            if avg_fiber < 25:  # Recommended daily intake
                recommendations.append("Boost your fiber intake with whole grains, fruits, and vegetables for better digestive health.")
        
        # Analyze stress levels (NEW)
        stress_data = [day.get('stress_level') for day in recent_health if day.get('stress_level')]
        if stress_data:
            avg_stress = sum(stress_data) / len(stress_data)
            if avg_stress > 60:
                concerns.append("Your stress levels appear elevated. This may impact sleep and overall well-being.")
                recommendations.append("Try stress-reduction techniques like deep breathing, meditation, or gentle exercise.")
                quick_wins.append("Practice 5 minutes of deep breathing before bed.")
        
        # Analyze meditation practice (NEW)
        meditation_data = [day.get('meditation_minutes') for day in recent_health if day.get('meditation_minutes')]
        if meditation_data:
            total_meditation = sum(meditation_data)
            if total_meditation > 0:
                insights.append(f"You've been practicing mindfulness - keep building this healthy habit!")
        else:
            quick_wins.append("Try a 5-minute guided meditation app to reduce stress and improve focus.")
        
        # Analyze screen time (NEW)
        screen_data = [day.get('screen_time_hours') for day in recent_health if day.get('screen_time_hours')]
        if screen_data:
            avg_screen = sum(screen_data) / len(screen_data)
            if avg_screen > 8:
                concerns.append("High screen time may impact sleep quality and eye health.")
                recommendations.append("Follow the 20-20-20 rule: Every 20 minutes, look at something 20 feet away for 20 seconds.")
                quick_wins.append("Set a screen time limit on your devices for non-work hours.")
        
        # Analyze heart rate variability (NEW)
        hrv_data = [day.get('heart_rate_variability') for day in recent_health if day.get('heart_rate_variability')]
        if hrv_data:
            avg_hrv = sum(hrv_data) / len(hrv_data)
            if avg_hrv > 50:
                insights.append("Your heart rate variability suggests good cardiovascular health and recovery.")
            elif avg_hrv < 30:
                recommendations.append("Low HRV may indicate stress or insufficient recovery. Prioritize rest and relaxation.")
        
        # Generate motivation message based on actual progress
        motivation = "Remember, small consistent changes lead to big improvements in health and well-being. You're taking positive steps by tracking your health data!"
        
        if len(insights) >= 2:
            motivation = "You're making great progress! Your consistent tracking shows commitment to your health journey."
        
        if not insights:
            insights.append("Start logging your daily health metrics to get personalized insights.")
        
        if not recommendations:
            recommendations.append("Focus on the basics: adequate sleep, regular movement, and staying hydrated.")
        
        if not quick_wins:
            quick_wins.append("Log your health data consistently to track your progress.")
        
        return {
            'insights': insights,
            'recommendations': recommendations,
            'quick_wins': quick_wins,
            'concerns': concerns,
            'motivation': motivation,
            'source': 'rule_based',
            'generated_at': datetime.now().isoformat()
        }
    
    def _prepare_health_summary(self, user_context: Dict) -> str:
        """Prepare a comprehensive formatted summary of health data for AI analysis"""
        recent_health = user_context.get('recent_health', [])
        
        if not recent_health:
            return "No recent health data available."
        
        summary_lines = []
        for day in recent_health:
            date = day.get('date', 'Unknown date')
            summary_lines.append(f"\n📅 Date: {date}")
            
            # Activity Metrics
            activity_data = []
            if day.get('steps'):
                activity_data.append(f"Steps: {day['steps']}")
            if day.get('active_minutes'):
                activity_data.append(f"Active: {day['active_minutes']} min")
            if day.get('distance_km'):
                activity_data.append(f"Distance: {day['distance_km']:.1f} km")
            if day.get('floors_climbed'):
                activity_data.append(f"Floors: {day['floors_climbed']}")
            if day.get('calories_total'):
                activity_data.append(f"Calories burned: {day['calories_total']}")
            if activity_data:
                summary_lines.append(f"  🏃 Activity: {', '.join(activity_data)}")
            
            # Sleep Metrics
            sleep_data = []
            if day.get('sleep_duration_hours'):
                sleep_data.append(f"Duration: {day['sleep_duration_hours']}h")
            if day.get('sleep_quality_score'):
                sleep_data.append(f"Quality: {day['sleep_quality_score']}/100")
            if day.get('sleep_deep_minutes'):
                sleep_data.append(f"Deep: {day['sleep_deep_minutes']}min")
            if day.get('sleep_rem_minutes'):
                sleep_data.append(f"REM: {day['sleep_rem_minutes']}min")
            if day.get('sleep_light_minutes'):
                sleep_data.append(f"Light: {day['sleep_light_minutes']}min")
            if sleep_data:
                summary_lines.append(f"  😴 Sleep: {', '.join(sleep_data)}")
            
            # Heart Health Metrics
            heart_data = []
            if day.get('heart_rate_resting'):
                heart_data.append(f"Resting: {day['heart_rate_resting']} BPM")
            if day.get('heart_rate_avg'):
                heart_data.append(f"Avg: {day['heart_rate_avg']} BPM")
            if day.get('heart_rate_max'):
                heart_data.append(f"Max: {day['heart_rate_max']} BPM")
            if day.get('heart_rate_variability'):
                heart_data.append(f"HRV: {day['heart_rate_variability']}ms")
            if day.get('blood_oxygen_percent'):
                heart_data.append(f"SpO2: {day['blood_oxygen_percent']}%")
            if heart_data:
                summary_lines.append(f"  ❤️ Heart: {', '.join(heart_data)}")
            
            # Blood Pressure
            if day.get('systolic_bp') and day.get('diastolic_bp'):
                summary_lines.append(f"  🩺 Blood Pressure: {day['systolic_bp']}/{day['diastolic_bp']} mmHg")
            
            # Nutrition Metrics
            nutrition_data = []
            if day.get('water_intake_liters'):
                nutrition_data.append(f"Water: {day['water_intake_liters']}L")
            if day.get('calories_consumed'):
                nutrition_data.append(f"Calories: {day['calories_consumed']} kcal")
            if day.get('protein_grams'):
                nutrition_data.append(f"Protein: {day['protein_grams']}g")
            if day.get('carbs_grams'):
                nutrition_data.append(f"Carbs: {day['carbs_grams']}g")
            if day.get('fat_grams'):
                nutrition_data.append(f"Fat: {day['fat_grams']}g")
            if day.get('fiber_grams'):
                nutrition_data.append(f"Fiber: {day['fiber_grams']}g")
            if nutrition_data:
                summary_lines.append(f"  🍎 Nutrition: {', '.join(nutrition_data)}")
            
            # Body Composition
            body_data = []
            if day.get('weight_kg'):
                body_data.append(f"Weight: {day['weight_kg']}kg")
            if day.get('bmi'):
                body_data.append(f"BMI: {day['bmi']:.1f}")
            if day.get('body_fat_percent'):
                body_data.append(f"Body Fat: {day['body_fat_percent']}%")
            if day.get('muscle_mass_kg'):
                body_data.append(f"Muscle: {day['muscle_mass_kg']}kg")
            if body_data:
                summary_lines.append(f"  ⚖️ Body: {', '.join(body_data)}")
            
            # Workout Details
            if day.get('workout_type'):
                workout_info = [f"Type: {day['workout_type']}"]
                if day.get('workout_duration_minutes'):
                    workout_info.append(f"{day['workout_duration_minutes']}min")
                if day.get('workout_intensity'):
                    workout_info.append(f"Intensity: {day['workout_intensity']}")
                if day.get('workout_calories'):
                    workout_info.append(f"{day['workout_calories']} cal")
                summary_lines.append(f"  💪 Workout: {', '.join(workout_info)}")
            
            # Wellness & Mental Health
            wellness_data = []
            if day.get('mood_score'):
                wellness_data.append(f"Mood: {day['mood_score']}/10")
            if day.get('energy_level'):
                wellness_data.append(f"Energy: {day['energy_level']}/10")
            if day.get('stress_level'):
                wellness_data.append(f"Stress: {day['stress_level']}/100")
            if wellness_data:
                summary_lines.append(f"  🧠 Wellness: {', '.join(wellness_data)}")
            
            # Lifestyle Metrics
            lifestyle_data = []
            if day.get('meditation_minutes'):
                lifestyle_data.append(f"Meditation: {day['meditation_minutes']}min")
            if day.get('screen_time_hours'):
                lifestyle_data.append(f"Screen: {day['screen_time_hours']}h")
            if day.get('social_interactions'):
                lifestyle_data.append(f"Social: {day['social_interactions']} interactions")
            if lifestyle_data:
                summary_lines.append(f"  🌟 Lifestyle: {', '.join(lifestyle_data)}")
            
            # Body Temperature
            if day.get('body_temperature'):
                summary_lines.append(f"  🌡️ Temperature: {day['body_temperature']}°C")
            
            # User Notes
            if day.get('notes'):
                summary_lines.append(f"  📝 Notes: {day['notes']}")
        
        return '\n'.join(summary_lines)
    
    def _parse_text_advice(self, text: str) -> Dict:
        """Parse unstructured text advice into structured format"""
        if not text or len(text.strip()) < 10:
            return self._get_default_advice()
        
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        insights = []
        recommendations = []
        quick_wins = []
        concerns = []
        motivation = "Keep up the great work tracking your health data!"
        
        # Try to extract content from different sections
        current_section = None
        
        for line in lines:
            line_lower = line.lower()
            
            # Identify sections
            if any(keyword in line_lower for keyword in ['insight', 'observation', 'notice']):
                current_section = 'insights'
                # Extract the actual insight content
                content = line
                for prefix in ['insight:', 'insights:', 'observation:', 'notice:']:
                    if prefix in line_lower:
                        content = line[line_lower.find(prefix) + len(prefix):].strip()
                        break
                if content and not any(skip in content.lower() for skip in ['[', ']', '{', '}', '"']):
                    insights.append(content)
                    
            elif any(keyword in line_lower for keyword in ['recommend', 'suggest', 'try', 'consider']):
                current_section = 'recommendations'
                content = line
                for prefix in ['recommend:', 'recommendation:', 'suggest:', 'try:', 'consider:']:
                    if prefix in line_lower:
                        content = line[line_lower.find(prefix) + len(prefix):].strip()
                        break
                if content and not any(skip in content.lower() for skip in ['[', ']', '{', '}', '"']):
                    recommendations.append(content)
                    
            elif any(keyword in line_lower for keyword in ['quick', 'easy', 'simple', 'today']):
                current_section = 'quick_wins'
                content = line
                for prefix in ['quick:', 'easy:', 'simple:', 'today:']:
                    if prefix in line_lower:
                        content = line[line_lower.find(prefix) + len(prefix):].strip()
                        break
                if content and not any(skip in content.lower() for skip in ['[', ']', '{', '}', '"']):
                    quick_wins.append(content)
                    
            elif any(keyword in line_lower for keyword in ['concern', 'attention', 'watch', 'caution']):
                current_section = 'concerns'
                content = line
                for prefix in ['concern:', 'concerns:', 'attention:', 'watch:', 'caution:']:
                    if prefix in line_lower:
                        content = line[line_lower.find(prefix) + len(prefix):].strip()
                        break
                if content and not any(skip in content.lower() for skip in ['[', ']', '{', '}', '"']):
                    concerns.append(content)
                    
            elif any(keyword in line_lower for keyword in ['motivat', 'encourag', 'keep', 'great']):
                if len(line) > 10 and not any(skip in line.lower() for skip in ['[', ']', '{', '}', '"']):
                    motivation = line
            
            # If we're in a section and this line doesn't start a new section, add it to current section
            elif current_section and line and len(line) > 5:
                if not any(skip in line.lower() for skip in ['[', ']', '{', '}', '"', 'json']):
                    if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                        line = line[2:].strip()
                    elif line[0].isdigit() and '. ' in line:
                        line = line.split('. ', 1)[1] if '. ' in line else line
                    
                    if current_section == 'insights' and line not in insights:
                        insights.append(line)
                    elif current_section == 'recommendations' and line not in recommendations:
                        recommendations.append(line)
                    elif current_section == 'quick_wins' and line not in quick_wins:
                        quick_wins.append(line)
                    elif current_section == 'concerns' and line not in concerns:
                        concerns.append(line)
        
        # If we didn't extract much, create some generic advice
        if not insights and not recommendations and not quick_wins:
            return self._get_default_advice()
        
        return {
            'insights': insights[:3],  # Limit to 3 items
            'recommendations': recommendations[:4],  # Limit to 4 items
            'quick_wins': quick_wins[:3],  # Limit to 3 items
            'concerns': concerns[:2],  # Limit to 2 items
            'motivation': motivation,
            'source': 'parsed_text',
            'generated_at': datetime.now().isoformat()
        }
    
    def _get_default_advice(self) -> Dict:
        """Return default advice when no data is available"""
        return {
            'insights': [
                "Start by logging your daily health metrics to get personalized insights.",
                "Consistent tracking helps identify patterns in your health and wellness."
            ],
            'recommendations': [
                "Begin with the basics: track your sleep, water intake, and daily activity.",
                "Set realistic goals and gradually build healthy habits.",
                "Use the file upload feature to import data from your wearable devices."
            ],
            'quick_wins': [
                "Log today's sleep hours and water intake to get started.",
                "Set up daily reminders to track your health metrics.",
                "Connect your wearable device for automatic data import."
            ],
            'concerns': [],
            'motivation': "Welcome to your health journey! Every small step counts toward better wellness.",
            'source': 'default',
            'generated_at': datetime.now().isoformat()
        }
