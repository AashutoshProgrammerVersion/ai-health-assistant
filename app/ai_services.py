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
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                logger.info("Gemini AI configured successfully")
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
        
        # Get weights from configuration
        weights = current_app.config.get('HEALTH_SCORE_WEIGHTS', {
            'sleep_hours': 0.25,
            'water_intake': 0.15,
            'activity_level': 0.20,
            'heart_rate': 0.15,
            'steps_count': 0.15,
            'mood': 0.10
        })
        
        # Calculate individual scores (0-10 scale)
        scores = {}
        recent_data = health_data_list[-7:] if len(health_data_list) >= 7 else health_data_list
        
        for data in recent_data:
            # Sleep score (optimal: 7-9 hours)
            if data.sleep_hours:
                sleep_score = self._calculate_sleep_score(data.sleep_hours)
                scores.setdefault('sleep_hours', []).append(sleep_score)
            
            # Water intake score (goal: 8 glasses)
            if data.water_intake:
                water_score = min(10, (data.water_intake / 8) * 10)
                scores.setdefault('water_intake', []).append(water_score)
            
            # Activity level score (1-10 scale, as recorded)
            if data.activity_level:
                scores.setdefault('activity_level', []).append(data.activity_level)
            
            # Heart rate score (resting HR: 60-100 optimal)
            if data.heart_rate:
                hr_score = self._calculate_heart_rate_score(data.heart_rate)
                scores.setdefault('heart_rate', []).append(hr_score)
            
            # Steps score (goal: 10,000 steps)
            if data.steps_count:
                steps_score = min(10, (data.steps_count / 10000) * 10)
                scores.setdefault('steps_count', []).append(steps_score)
            
            # Mood score (1-10 scale, as recorded)
            if data.mood:
                scores.setdefault('mood', []).append(data.mood)
        
        # Calculate weighted average
        total_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in scores and scores[metric]:
                avg_score = sum(scores[metric]) / len(scores[metric])
                total_score += avg_score * weight
                total_weight += weight
        
        return round(total_score / total_weight if total_weight > 0 else 0.0, 1)
    
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
                    'sleep_hours': data.sleep_hours or 0,
                    'water_intake': data.water_intake or 0,
                    'activity_level': data.activity_level or 0,
                    'heart_rate': data.heart_rate or 0,
                    'steps_count': data.steps_count or 0,
                    'mood': data.mood or 0,
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
    
    def generate_personalized_advice(self, user_data: Dict, health_patterns: Dict) -> str:
        """
        Generate personalized health advice using Gemini AI
        Implements research: AI-powered personalized recommendations
        """
        if not self.gemini_model:
            return self._generate_fallback_advice(user_data, health_patterns)
        
        try:
            # Prepare context for AI
            context = self._prepare_health_context(user_data, health_patterns)
            
            prompt = f"""
            You are a helpful health assistant AI. Based on the following user health data, provide personalized, 
            actionable health advice. Keep advice practical and evidence-based. Do not provide medical diagnoses.
            
            User Health Context:
            {context}
            
            Please provide 3-5 specific, actionable recommendations to improve their health routine.
            Focus on small, achievable changes. Format as a friendly, encouraging message.
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            if response.text:
                return response.text
            else:
                return self._generate_fallback_advice(user_data, health_patterns)
                
        except Exception as e:
            logger.error(f"Error generating AI advice: {e}")
            return self._generate_fallback_advice(user_data, health_patterns)
    
    def _prepare_health_context(self, user_data: Dict, health_patterns: Dict) -> str:
        """Prepare health data context for AI processing"""
        context_parts = []
        
        # Recent health metrics
        if 'recent_health' in user_data:
            recent = user_data['recent_health']
            context_parts.append(f"Recent health data: {recent}")
        
        # Health score
        if 'health_score' in user_data:
            context_parts.append(f"Current health score: {user_data['health_score']}/10")
        
        # Patterns and trends
        if health_patterns.get('trends'):
            context_parts.append(f"Health trends: {health_patterns['trends']}")
        
        if health_patterns.get('insights'):
            context_parts.append(f"Key insights: {'; '.join(health_patterns['insights'])}")
        
        return '\n'.join(context_parts)
    
    def _generate_fallback_advice(self, user_data: Dict, health_patterns: Dict) -> str:
        """Generate basic advice when AI is unavailable"""
        advice_parts = []
        
        # Basic recommendations based on common health patterns
        if user_data.get('health_score', 0) < 6:
            advice_parts.append("Your health score suggests room for improvement.")
        
        advice_parts.extend([
            "• Aim for 7-9 hours of sleep each night",
            "• Drink at least 8 glasses of water daily",
            "• Include 30 minutes of physical activity",
            "• Take regular breaks to manage stress"
        ])
        
        return '\n'.join(advice_parts)

# Singleton instance
health_ai_service = HealthAIService()
