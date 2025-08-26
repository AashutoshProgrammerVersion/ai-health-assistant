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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import spacy
import google.generativeai as genai
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
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI configured successfully")
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
                scores.append(day_score / components * (100/100))  # Normalize to 100
        
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
                scores.append(day_score / components * (100/50))  # Normalize to 100
        
        return sum(scores) / len(scores) if scores else 0.0
    
    def _calculate_nutrition_score(self, recent_data: List) -> float:
        """Calculate nutrition score based on available dietary data"""
        if not recent_data:
            return 50.0  # Default neutral score when no data
        
        scores = []
        for data in recent_data:
            day_score = 50  # Start with neutral score
            
            # Body composition trends (if available)
            if data.weight_kg is not None and data.body_fat_percent is not None:
                # Stable/healthy body composition gets higher score
                day_score = 70
            
            # Energy level as nutrition indicator
            if data.energy_level is not None:
                energy_score = (data.energy_level / 10) * 30
                day_score = max(day_score, energy_score + 20)
            
            scores.append(min(day_score, 100))
        
        return sum(scores) / len(scores) if scores else 50.0
    
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
                scores.append(day_score / components * (100/100))  # Normalize to 100
        
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
                scores.append(day_score / components * (100/100))  # Normalize to 100
        
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
                    'sleep_hours': data.sleep_duration_hours or 0,
                    'water_intake': data.water_intake_liters or 0,
                    'activity_level': data.active_minutes or 0,
                    'heart_rate': data.heart_rate_avg or 0,
                    'steps_count': data.steps or 0,
                    'mood': data.mood_score or 0,
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
        """Find correlations between health metrics"""
        numeric_cols = ['sleep_hours', 'water_intake', 'activity_level', 'heart_rate', 'steps_count', 'mood']
        correlations = {}
        
        for col in numeric_cols:
            if col in df.columns:
                corr_with_mood = df[col].corr(df['mood']) if 'mood' in df.columns else 0
                correlations[f"{col}_mood_correlation"] = round(corr_with_mood, 3) if not np.isnan(corr_with_mood) else 0
        
        return correlations
    
    def _detect_health_trends(self, df: pd.DataFrame) -> Dict:
        """Detect trends in health metrics over time"""
        trends = {}
        numeric_cols = ['sleep_hours', 'water_intake', 'activity_level', 'steps_count']
        
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
        """Generate human-readable health insights"""
        insights = []
        
        # Correlation insights
        for corr_key, corr_value in correlations.items():
            if abs(corr_value) > 0.5:  # Strong correlation
                metric = corr_key.replace('_mood_correlation', '')
                if corr_value > 0:
                    insights.append(f"Higher {metric.replace('_', ' ')} appears to improve your mood")
                else:
                    insights.append(f"Higher {metric.replace('_', ' ')} may negatively affect your mood")
        
        # Trend insights
        for metric, trend in trends.items():
            if trend == 'increasing':
                insights.append(f"Your {metric.replace('_', ' ')} has been improving over time")
            elif trend == 'decreasing':
                insights.append(f"Your {metric.replace('_', ' ')} has been declining recently")
        
        return insights[:5]  # Limit to top 5 insights
    
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
        """Generate advice using rule-based system when AI is unavailable"""
        
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
        
        # Generate motivation message
        motivation = "Remember, small consistent changes lead to big improvements in health and well-being. You're taking positive steps by tracking your health data!"
        
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
        """Prepare a formatted summary of health data for AI analysis"""
        recent_health = user_context.get('recent_health', [])
        
        if not recent_health:
            return "No recent health data available."
        
        summary_lines = []
        for day in recent_health:
            date = day.get('date', 'Unknown date')
            summary_lines.append(f"\nDate: {date}")
            
            # Activity summary
            if day.get('steps'):
                summary_lines.append(f"  Steps: {day['steps']}")
            if day.get('active_minutes'):
                summary_lines.append(f"  Active minutes: {day['active_minutes']}")
            
            # Sleep summary
            if day.get('sleep_duration_hours'):
                summary_lines.append(f"  Sleep: {day['sleep_duration_hours']} hours")
            if day.get('sleep_quality_score'):
                summary_lines.append(f"  Sleep quality: {day['sleep_quality_score']}/100")
            
            # Health metrics
            if day.get('heart_rate_resting'):
                summary_lines.append(f"  Resting HR: {day['heart_rate_resting']} BPM")
            if day.get('water_intake_liters'):
                summary_lines.append(f"  Water: {day['water_intake_liters']} liters")
            
            # Wellness indicators
            if day.get('mood_score'):
                summary_lines.append(f"  Mood: {day['mood_score']}/10")
            if day.get('stress_level'):
                summary_lines.append(f"  Stress: {day['stress_level']}/100")
        
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
