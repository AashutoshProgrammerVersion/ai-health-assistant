from .health_ai import HealthAIService
from .calendar_optimization import CalendarOptimizationService
from flask import current_app

# Services will be initialized when needed with proper Flask context
def get_health_ai_service():
    """Get or create HealthAI service instance within Flask context"""
    if not hasattr(current_app, '_health_ai_service'):
        current_app._health_ai_service = HealthAIService()
    return current_app._health_ai_service

def get_calendar_service():
    """Get or create Calendar service instance within Flask context"""
    if not hasattr(current_app, '_calendar_service'):
        current_app._calendar_service = CalendarOptimizationService()
    return current_app._calendar_service