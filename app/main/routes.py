"""
Main Application Routes for Core Functionality

This file contains the primary routes for the health assistant application.
These routes handle the home page, dashboard, health data logging, calendar management,
and AI-powered features. Routes connect URLs to Python functions that generate responses.
"""

# FLASK CORE IMPORTS - Essential Flask functionality (patterns already explained in auth/routes.py)
from flask import render_template, redirect, url_for, flash, request, jsonify, session, current_app, Response
import logging
import json

# Set up logger
logger = logging.getLogger(__name__)

# FLASK-LOGIN IMPORTS - User authentication and access control
from flask_login import login_required, current_user
"""
'from flask_login import login_required, current_user' - Authentication utilities
'login_required' - Decorator that protects routes from unauthorized access
'current_user' - Object representing the currently logged-in user (already explained)
"""

# DATE/TIME IMPORTS - For handling dates and timestamps
from datetime import datetime, date, timedelta
"""
'from datetime import datetime, date' - Date and time handling classes
'datetime' - Class for date and time combined (already explained in models.py)
'date' - Class for date only (year, month, day)
'timedelta' - Class for time differences and calculations
Used for tracking when health data was logged and comparing dates
"""

# APPLICATION IMPORTS - Blueprint, database, and models
from app.main import bp
"""
'from app.main import bp' - Import main blueprint
'bp' - Blueprint for main application routes (created in main/__init__.py)
"""

from app import db
"""
'from app import db' - Import database instance (already explained)
"""

from app.models import HealthData, CalendarEvent, UserPreferences, AIRecommendation, EventBackup
"""
'from app.models import HealthData, CalendarEvent, UserPreferences, AIRecommendation, EventBackup' - Import models
All the database models we created for comprehensive health and calendar management including event backup
"""

from app.main.forms import HealthDataForm, CalendarEventForm, UserPreferencesForm
"""
'from app.main.forms import HealthDataForm, CalendarEventForm, UserPreferencesForm' - Import forms
All the forms for user input validation and data collection
"""

# AI SERVICES IMPORTS - Custom AI and integration services
from app.ai_services import get_health_ai_service
from app.calendar_service import get_calendar_service, get_google_calendar_service
from app.health_file_processor import get_health_processor
"""
Updated imports - removed Samsung Health service, added health file processor
Uses the successful Gemini 2.5 Flash approach from the test application
"""

# HELPER FUNCTIONS
def calculate_ai_confidence(health_data_count: int, optimization_result: dict, 
                           schedule_changes: int, reminders_created: int) -> float:
    """
    Calculate dynamic AI confidence based on multiple factors
    
    Args:
        health_data_count: Number of days of health data available
        optimization_result: The optimization result dictionary
        schedule_changes: Number of events moved
        reminders_created: Number of reminders created
    
    Returns:
        Confidence score between 0.0 and 1.0
    """
    # Base confidence starts at 0.5
    confidence = 0.5
    
    # Factor 1: Data Quality (up to +0.25)
    # More days of data = higher confidence
    if health_data_count >= 30:
        confidence += 0.25
    elif health_data_count >= 14:
        confidence += 0.20
    elif health_data_count >= 7:
        confidence += 0.15
    elif health_data_count >= 3:
        confidence += 0.10
    else:
        confidence += 0.05
    
    # Factor 2: AI Source (up to +0.20)
    # Check if Gemini AI was used (look for 'gemini' in optimization tips or high-quality insights)
    optimization_tips = optimization_result.get('optimization_tips', [])
    ai_insights = optimization_result.get('ai_insights', [])
    
    gemini_indicators = ['Gemini AI', 'AI analysis', 'intelligent', 'personalized based on']
    uses_gemini = any(
        any(indicator.lower() in str(tip).lower() for indicator in gemini_indicators)
        for tip in (optimization_tips + ai_insights)
    )
    
    if uses_gemini:
        confidence += 0.20  # Gemini AI active
    elif len(ai_insights) > 0:
        confidence += 0.10  # Some AI insights present
    else:
        confidence += 0.05  # Fallback/rule-based only
    
    # Factor 3: Optimization Quality (up to +0.15)
    # Successfully created changes = higher confidence
    total_actions = schedule_changes + reminders_created
    if total_actions >= 10:
        confidence += 0.15
    elif total_actions >= 5:
        confidence += 0.12
    elif total_actions >= 3:
        confidence += 0.08
    elif total_actions >= 1:
        confidence += 0.05
    
    # Factor 4: Pattern Analysis (up to +0.10)
    # Check if health patterns were found
    schedule_changes_list = optimization_result.get('schedule_changes', [])
    if schedule_changes_list:
        # Look for variety in optimization reasons (indicates sophisticated analysis)
        reasons = set(change.get('reason', '') for change in schedule_changes_list)
        if len(reasons) >= 3:
            confidence += 0.10  # Diverse, thoughtful optimizations
        elif len(reasons) >= 2:
            confidence += 0.07
        else:
            confidence += 0.03
    
    # Ensure confidence stays within valid range [0.5, 1.0]
    confidence = max(0.5, min(1.0, confidence))
    
    return round(confidence, 2)

# HOME PAGE ROUTE - Landing page and entry point
@bp.route('/')
@bp.route('/index')
def index():
    """
    '@bp.route('/')' - Route decorator for root URL
    '@bp.route('/index')' - Alternative route for same function
    Multiple decorators allow same function to handle multiple URLs
    '/' - Root URL (domain.com/)
    '/index' - Alternative URL (domain.com/index)
    
    'def index():' - Function to handle home page requests
    'index' - Function name (commonly used for home/landing pages)
    """
    
    # REDIRECT AUTHENTICATED USERS TO DASHBOARD
    if current_user.is_authenticated:
        """
        'if current_user.is_authenticated:' - Check if user is logged in
        Same pattern as in auth routes - check authentication status
        """
        return redirect(url_for('main.dashboard'))
        """
        'return redirect(url_for('main.dashboard'))' - Send logged-in users to dashboard
        'main.dashboard' - Blueprint.route reference for dashboard route
        Logged-in users bypass home page and go directly to their dashboard
        """
    
    # SHOW HOME PAGE FOR ANONYMOUS USERS
    return render_template('index.html', title='Home')
    """
    'return render_template('index.html', title='Home')' - Render home page template
    'index.html' - Template file for home page (in templates/ directory)
    'title='Home'' - Page title variable passed to template
    """

# DASHBOARD ROUTE - Enhanced main user interface with AI insights
@bp.route('/dashboard')
@login_required
def dashboard():
    """
    Enhanced dashboard with AI-powered health insights and calendar integration
    Implements research features: health scoring, pattern analysis, personalized advice
    """
    
    # GET RECENT HEALTH DATA FOR ANALYSIS
    recent_data = current_user.health_data.order_by(HealthData.date_logged.desc()).limit(30).all()
    """
    Increased from 7 to 30 days of data for better AI pattern analysis
    More data points improve the accuracy of health insights and recommendations
    """
    
    # GET USER PREFERENCES (CREATE DEFAULT IF NONE)
    preferences = current_user.preferences
    if not preferences:
        preferences = UserPreferences(user_id=current_user.id)
        db.session.add(preferences)
        db.session.commit()
    
    # CALCULATE COMPREHENSIVE HEALTH SCORE
    health_score = get_health_ai_service().calculate_health_score(recent_data) if recent_data else 0
    
    # CALCULATE INDIVIDUAL HEALTH SCORES FOR DASHBOARD DISPLAY
    individual_scores = get_health_ai_service().calculate_individual_scores(recent_data) if recent_data else {
        'activity_score': 0,
        'sleep_score': 0,
        'nutrition_score': 0,
        'hydration_score': 0,
        'heart_health_score': 0,
        'wellness_score': 0
    }
    
    # ANALYZE HEALTH PATTERNS WITH AI
    health_patterns = get_health_ai_service().analyze_health_patterns(recent_data) if len(recent_data) >= 7 else {}
    
    # GENERATE PERSONALIZED AI ADVICE
    user_context = {
        'recent_health': [
            {
                'date': data.date_logged.isoformat(),
                # Activity Metrics
                'steps': data.steps,
                'distance_km': data.distance_km,
                'calories_total': data.calories_total,
                'active_minutes': data.active_minutes,
                'floors_climbed': data.floors_climbed,
                
                # Heart Rate Metrics
                'heart_rate_avg': data.heart_rate_avg,
                'heart_rate_resting': data.heart_rate_resting,
                'heart_rate_max': data.heart_rate_max,
                'heart_rate_variability': data.heart_rate_variability,
                
                # Sleep Metrics
                'sleep_duration_hours': data.sleep_duration_hours,
                'sleep_quality_score': data.sleep_quality_score,
                'sleep_deep_minutes': data.sleep_deep_minutes,
                'sleep_light_minutes': data.sleep_light_minutes,
                'sleep_rem_minutes': data.sleep_rem_minutes,
                'sleep_awake_minutes': data.sleep_awake_minutes,
                
                # Advanced Health Metrics
                'blood_oxygen_percent': data.blood_oxygen_percent,
                'stress_level': data.stress_level,
                'body_temperature': data.body_temperature,
                
                # Body Composition Metrics
                'weight_kg': data.weight_kg,
                'body_fat_percent': data.body_fat_percent,
                'muscle_mass_kg': data.muscle_mass_kg,
                
                # Lifestyle Metrics
                'water_intake_liters': data.water_intake_liters,
                'mood_score': data.mood_score,
                'energy_level': data.energy_level,
                
                # Exercise Session Details
                'workout_type': data.workout_type,
                'workout_duration_minutes': data.workout_duration_minutes,
                'workout_intensity': data.workout_intensity,
                'workout_calories': data.workout_calories
            } for data in recent_data[:7]  # Last 7 days for context
        ],
        'health_score': health_score,
        'goals': {
            'water_liters': preferences.daily_water_goal,
            'sleep_hours': preferences.daily_sleep_goal,
            'steps': preferences.daily_steps_goal,
            'activity_minutes': preferences.daily_activity_goal
        }
    }
    
    # LOAD EXISTING PERSONALIZED ADVICE OR GENERATE IF NONE EXISTS
    existing_advice = current_user.personalized_advice
    ai_advice = None
    
    if existing_advice:
        # User has existing advice, load it
        ai_advice = existing_advice.to_dict()
    else:
        # No existing advice, generate it automatically for first time
        if len(recent_data) >= 1:  # Only if user has some health data
            try:
                user_context = {
                    'recent_health': [
                        {
                            'date': data.date_logged.isoformat(),
                            # Activity Metrics
                            'steps': data.steps,
                            'distance_km': data.distance_km,
                            'calories_total': data.calories_total,
                            'active_minutes': data.active_minutes,
                            'floors_climbed': data.floors_climbed,
                            
                            # Heart Rate Metrics
                            'heart_rate_avg': data.heart_rate_avg,
                            'heart_rate_resting': data.heart_rate_resting,
                            'heart_rate_max': data.heart_rate_max,
                            'heart_rate_variability': data.heart_rate_variability,
                            
                            # Sleep Metrics
                            'sleep_duration_hours': data.sleep_duration_hours,
                            'sleep_quality_score': data.sleep_quality_score,
                            'sleep_deep_minutes': data.sleep_deep_minutes,
                            'sleep_light_minutes': data.sleep_light_minutes,
                            'sleep_rem_minutes': data.sleep_rem_minutes,
                            'sleep_awake_minutes': data.sleep_awake_minutes,
                            
                            # Advanced Health Metrics
                            'blood_oxygen_percent': data.blood_oxygen_percent,
                            'stress_level': data.stress_level,
                            'body_temperature': data.body_temperature,
                            
                            # Body Composition Metrics
                            'weight_kg': data.weight_kg,
                            'body_fat_percent': data.body_fat_percent,
                            'muscle_mass_kg': data.muscle_mass_kg,
                            'bmi': data.bmi,
                            
                            # Nutrition Metrics
                            'water_intake_liters': data.water_intake_liters,
                            'calories_consumed': data.calories_consumed,
                            'protein_grams': data.protein_grams,
                            'carbs_grams': data.carbs_grams,
                            'fat_grams': data.fat_grams,
                            'fiber_grams': data.fiber_grams,
                            
                            # Blood Pressure
                            'systolic_bp': data.systolic_bp,
                            'diastolic_bp': data.diastolic_bp,
                            
                            # Subjective Wellness Metrics
                            'mood_score': data.mood_score,
                            'energy_level': data.energy_level,
                            
                            # Lifestyle & Mental Wellness
                            'meditation_minutes': data.meditation_minutes,
                            'screen_time_hours': data.screen_time_hours,
                            'social_interactions': data.social_interactions,
                            
                            # Exercise Session Details
                            'workout_type': data.workout_type,
                            'workout_duration_minutes': data.workout_duration_minutes,
                            'workout_intensity': data.workout_intensity,
                            'workout_calories': data.workout_calories,
                            
                            # User Notes
                            'notes': data.notes
                        } for data in recent_data[:7]  # Last 7 days for context
                    ],
                    'health_score': health_score,
                    'goals': {
                        'water_liters': preferences.daily_water_goal,
                        'sleep_hours': preferences.daily_sleep_goal,
                        'steps': preferences.daily_steps_goal,
                        'activity_minutes': preferences.daily_activity_goal
                    }
                }
                
                # Generate new advice
                ai_advice = get_health_ai_service().generate_personalized_advice(user_context, health_patterns)
                
                # Save to database for persistence
                if ai_advice:
                    from app.models import PersonalizedHealthAdvice
                    advice_record = PersonalizedHealthAdvice.from_dict(
                        user_id=current_user.id,
                        advice_dict=ai_advice,
                        health_score=health_score
                    )
                    db.session.add(advice_record)
                    db.session.commit()
                    
            except Exception as e:
                logger.error(f"Error auto-generating initial advice: {e}")
                ai_advice = None
    
    # GET UPCOMING CALENDAR EVENTS (NEXT 7 DAYS)
    upcoming_events = current_user.calendar_events.filter(
        CalendarEvent.start_time >= datetime.now(),
        CalendarEvent.start_time <= datetime.now() + timedelta(days=7)
    ).order_by(CalendarEvent.start_time).limit(10).all()
    
    # GET RECENT AI RECOMMENDATIONS
    recent_ai_recommendations = current_user.ai_recommendations.order_by(
        AIRecommendation.created_at.desc()
    ).limit(5).all()
    
    # CALCULATE TODAY'S PROGRESS FOR DASHBOARD DISPLAY
    today = date.today()
    today_data = current_user.health_data.filter(
        HealthData.date_logged >= today,
        HealthData.date_logged < today + timedelta(days=1)
    ).first()
    
    # Extract today's metrics with fallbacks
    sleep_hours = today_data.sleep_duration_hours if today_data and today_data.sleep_duration_hours else 0
    water_consumed = today_data.water_intake_liters if today_data and today_data.water_intake_liters else 0
    steps_taken = today_data.steps if today_data and today_data.steps else 0
    
    # Get goals from preferences with fallbacks
    sleep_goal = preferences.daily_sleep_goal if preferences.daily_sleep_goal else 8
    water_goal_glasses = preferences.daily_water_goal if preferences.daily_water_goal else 8
    water_goal = water_goal_glasses * 0.25  # Convert glasses (8oz each) to liters (8 glasses ≈ 2 liters)
    steps_goal = preferences.daily_steps_goal if preferences.daily_steps_goal else 10000
    
    # Convert health data to JSON-serializable format for chart rendering
    health_data_json = []
    for data in recent_data[:7]:
        health_data_json.append({
            'date': data.date_logged.strftime('%Y-%m-%d') if data.date_logged else None,
            'steps': data.steps,
            'calories_total': data.calories_total,
            'sleep_duration_hours': data.sleep_duration_hours,
            'water_intake_liters': data.water_intake_liters,
            'heart_rate_resting': data.heart_rate_resting,
            'heart_rate_avg': data.heart_rate_avg,
            'heart_rate_max': data.heart_rate_max,
            'calories_consumed': data.calories_consumed,
            'mood_score': data.mood_score
        })
    
    # RENDER ENHANCED DASHBOARD
    return render_template('dashboard.html', 
                         title='Dashboard', 
                         health_data=recent_data[:7],  # Show last 7 days (for template iteration)
                         health_data_json=health_data_json,  # JSON-serializable for charts
                         latest=recent_data[0] if recent_data else None,  # Most recent entry for detailed display
                         health_score=health_score,
                         activity_score=individual_scores['activity_score'],
                         sleep_score=individual_scores['sleep_score'],
                         nutrition_score=individual_scores['nutrition_score'],
                         hydration_score=individual_scores['hydration_score'],
                         heart_health_score=individual_scores['heart_health_score'],
                         wellness_score=individual_scores['wellness_score'],
                         # Today's actual values
                         sleep_hours=sleep_hours,
                         water_consumed=water_consumed,
                         steps_taken=steps_taken,
                         # Goals from preferences
                         sleep_goal=sleep_goal,
                         water_goal=water_goal,
                         steps_goal=steps_goal,
                         health_patterns=health_patterns,
                         ai_advice=ai_advice,
                         upcoming_events=upcoming_events,
                         ai_recommendations=recent_ai_recommendations,
                         preferences=preferences)

# HEALTH DATA LOGGING ROUTE - Form for entering daily health metrics
@bp.route('/log_data', methods=['GET', 'POST'])
@login_required
def log_data():
    """
    '@bp.route('/log_data', methods=['GET', 'POST'])' - Health data logging route
    '/log_data' - URL for health data entry
    'methods=['GET', 'POST']' - Accept both GET (show form) and POST (process form)
    '@login_required' - Only logged-in users can log health data
    """
    
    # CREATE HEALTH DATA FORM INSTANCE
    form = HealthDataForm()
    """
    'form = HealthDataForm()' - Create health data form
    'HealthDataForm()' - Form class from forms.py for health metrics
    """
    
    # PROCESS FORM SUBMISSION
    if form.validate_on_submit():
        """
        'if form.validate_on_submit():' - Check if form was submitted and is valid
        Same validation pattern as in auth routes
        """
        
        logger.info("Form validation passed, processing health data submission")
        
        # CHECK FOR EXISTING DATA TODAY
        today = date.today()
        # Use form date if provided, otherwise use today's date
        log_date = form.date_logged.data if form.date_logged.data else today
        """
        'today = date.today()' - Get current date
        'date.today()' - Returns today's date (year, month, day)
        Used to check if user already logged health data today
        """
        
        existing_data = HealthData.query.filter_by(
            user_id=current_user.id, 
            date_logged=log_date
        ).first()
        """
        'existing_data = HealthData.query.filter_by(...).first()' - Look for today's data
        'HealthData.query' - Query the health data table
        'filter_by(user_id=current_user.id, date_logged=today)' - Filter conditions
        'user_id=current_user.id' - Records belonging to current user
        'date_logged=today' - Records from today's date
        'first()' - Get first matching record, or None if no match
        This prevents duplicate entries for the same day
        """
        
        # UPDATE EXISTING DATA OR CREATE NEW ENTRY
        if existing_data:
            """
            'if existing_data:' - Check if data for today already exists
            If user already logged data today, update it instead of creating duplicate
            """
            
            # UPDATE EXISTING RECORD - Only update fields that exist in the simple template
            existing_data.sleep_duration_hours = form.sleep_duration_hours.data
            existing_data.water_intake_liters = form.water_intake_liters.data
            existing_data.active_minutes = form.active_minutes.data
            existing_data.mood_score = form.mood_score.data
            existing_data.updated_at = datetime.utcnow()
            existing_data.data_source = 'manual'
            
            flash('Your health data has been updated!', 'success')
            
        else:
            # CREATE NEW HEALTH DATA RECORD - Only fields from the simple template
            health_data = HealthData(
                user_id=current_user.id,
                date_logged=log_date,  # Use the determined date (today or form date)
                data_source='manual',
                
                # Only the 4 fields from the template
                sleep_duration_hours=form.sleep_duration_hours.data,
                water_intake_liters=form.water_intake_liters.data,
                active_minutes=form.active_minutes.data,
                mood_score=form.mood_score.data
            )
            
            db.session.add(health_data)
            """
            'db.session.add(health_data)' - Add new record to database session
            'db.session.add()' - Stage new record for database insertion
            """
            
            flash('Your health data has been logged!', 'success')
            """
            'flash('Your health data has been logged!', 'success')' - Show success message
            Different message for new entries
            """
        
        # SAVE CHANGES TO DATABASE
        try:
            db.session.commit()
            """
            'db.session.commit()' - Save all changes to database
            Commits both updates to existing records and new record insertions
            """
            logger.info("Health data saved successfully to database")
            
            return redirect(url_for('main.dashboard'))
            """
            'return redirect(url_for('main.dashboard'))' - Redirect to dashboard
            After successful data logging, show user their updated dashboard
            """
        except Exception as e:
            db.session.rollback()
            logger.error(f"Database error when saving health data: {e}")
            flash(f'Error saving health data: {str(e)}', 'error')
    
    else:
        # Log form validation errors for debugging
        if request.method == 'POST':
            logger.error(f"Form validation failed. Errors: {form.errors}")
            for field, errors in form.errors.items():
                for error in errors:
                    flash(f'{field}: {error}', 'error')
    
    # SHOW HEALTH DATA FORM (for GET requests or invalid submissions)
    return render_template('log_data.html', title='Log Health Data', form=form)
    """
    'return render_template('log_data.html', title='Log Health Data', form=form)' - Render data entry form
    'log_data.html' - Template for health data entry form
    'form=form' - Pass form object to template for HTML generation
    """

# CALENDAR MANAGEMENT ROUTES - AI-powered calendar with optimization

@bp.route('/calendar')
@login_required
def calendar():
    """
    Calendar view showing user events with AI optimization insights
    Implements research requirement for calendar integration
    """
    # GET ALL USER EVENTS
    events = current_user.calendar_events.order_by(CalendarEvent.start_time).all()
    
    # Convert events to dictionaries for JSON serialization
    events_data = [event.to_dict() for event in events]
    
    # GET USER PREFERENCES FOR AI SETTINGS
    preferences = current_user.preferences
    if not preferences:
        preferences = UserPreferences(user_id=current_user.id)
        db.session.add(preferences)
        db.session.commit()
    
    return render_template('calendar.html', title='My Calendar', 
                         events=events, events_data=events_data, preferences=preferences)

@bp.route('/calendar/add', methods=['GET', 'POST'])
@login_required
def add_calendar_event():
    """
    Add new calendar event with AI optimization controls
    Implements user control over AI modifications based on research
    """
    form = CalendarEventForm()
    
    # Pre-fill date if provided in URL
    if request.method == 'GET':
        date_param = request.args.get('date')
        if date_param:
            try:
                # Parse the date and set default start time to 9 AM
                from datetime import datetime
                date_obj = datetime.strptime(date_param, '%Y-%m-%d')
                form.start_time.data = date_obj.replace(hour=9, minute=0)
                form.end_time.data = date_obj.replace(hour=10, minute=0)
            except ValueError:
                pass  # Ignore invalid date format
    
    if form.validate_on_submit():
        try:
            # ANALYZE EVENT TEXT FOR SMART DEFAULTS
            event_analysis = get_calendar_service().analyze_event_text(form.title.data + ' ' + (form.description.data or ''))
            
            # CREATE NEW CALENDAR EVENT
            event = CalendarEvent(
                user_id=current_user.id,
                title=form.title.data,
                description=form.description.data,
                start_time=form.start_time.data,
                end_time=form.end_time.data,
                is_ai_modifiable=form.is_ai_modifiable.data,
                is_fixed_time=form.is_fixed_time.data,
                priority_level=form.priority_level.data,
                event_type=form.event_type.data or event_analysis.get('event_type', 'personal')
            )
            
            db.session.add(event)
            db.session.commit()
            
            flash(f'Event "{event.title}" has been added to your calendar!', 'success')
            
            # SUGGEST AI OPTIMIZATION IF ENABLED
            preferences = current_user.preferences
            if preferences and preferences.ai_optimization_enabled:
                flash('AI optimization is enabled. Check your dashboard for schedule suggestions!', 'info')
            
            return redirect(url_for('main.calendar'))
            
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding event: {str(e)}', 'error')
            logger.error(f"Error adding calendar event: {e}")
    
    return render_template('add_calendar_event.html', title='Add Event', form=form)

@bp.route('/calendar/edit/<int:event_id>', methods=['GET', 'POST'])
@login_required
def edit_calendar_event(event_id):
    """
    Edit existing calendar event
    Maintains user control over AI modification settings
    """
    event = CalendarEvent.query.filter_by(id=event_id, user_id=current_user.id).first_or_404()
    form = CalendarEventForm(obj=event)
    
    if form.validate_on_submit():
        event.title = form.title.data
        event.description = form.description.data
        event.start_time = form.start_time.data
        event.end_time = form.end_time.data
        event.is_ai_modifiable = form.is_ai_modifiable.data
        event.is_fixed_time = form.is_fixed_time.data
        event.priority_level = form.priority_level.data
        event.event_type = form.event_type.data
        event.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash(f'Event "{event.title}" has been updated!', 'success')
        return redirect(url_for('main.calendar'))
    
    return render_template('edit_calendar_event.html', title='Edit Event', form=form, event=event)

@bp.route('/calendar/delete/<int:event_id>', methods=['POST'])
@login_required
def delete_calendar_event(event_id):
    """Delete calendar event"""
    event = CalendarEvent.query.filter_by(id=event_id, user_id=current_user.id).first_or_404()
    event_title = event.title
    
    db.session.delete(event)
    db.session.commit()
    
    flash(f'Event "{event_title}" has been deleted.', 'success')
    return redirect(url_for('main.calendar'))

# AI OPTIMIZATION ROUTES - Smart scheduling and recommendations

@bp.route('/ai/optimize_schedule', methods=['POST'])
@login_required
def optimize_schedule():
    """
    AI-powered schedule optimization
    Implements research requirement for AI schedule optimization with user control
    """
    try:
        # GET USER EVENTS AND PREFERENCES - ONLY TODAY'S EVENTS
        today = datetime.now().date()
        today_start = datetime.combine(today, datetime.min.time())
        today_end = datetime.combine(today + timedelta(days=1), datetime.min.time())
        
        user_events_objects = current_user.calendar_events.filter(
            CalendarEvent.start_time >= today_start,
            CalendarEvent.start_time < today_end
        ).all()
        
        logger.info(f"Filtering to today's events only: {len(user_events_objects)} events found for {today}")
        
        # Convert SQLAlchemy objects to dictionaries for JSON serialization
        user_events = [event.to_dict() for event in user_events_objects]
        
        preferences = current_user.preferences
        if not preferences:
            # Create default preferences for user if they don't exist
            preferences = UserPreferences(
                user_id=current_user.id,
                ai_optimization_enabled=True,
                reminder_water=True,
                reminder_exercise=True,
                reminder_sleep=True,
                reminder_medication=False,
                reminder_meal=False,
                reminder_mindfulness=True  # Added mindfulness default
            )
            db.session.add(preferences)
            db.session.commit()
            
        if not (preferences.ai_optimization_enabled or preferences.smart_reminders_enabled):
            return jsonify({'error': 'AI features are disabled. Enable AI Schedule Optimization or Smart Adaptive Reminders in settings.'}), 400
        
        # GET RECENT HEALTH DATA FOR CONTEXT
        recent_health = current_user.health_data.order_by(
            HealthData.date_logged.desc()
        ).limit(7).all()
        
        health_context = {
            'recent_activity_level': sum(d.active_minutes or 0 for d in recent_health) / len(recent_health) if recent_health else 0,
            'recent_sleep_average': sum(d.sleep_duration_hours or 0 for d in recent_health) / len(recent_health) if recent_health else 8,
            'recent_steps': sum(d.steps or 0 for d in recent_health) / len(recent_health) if recent_health else 0
        }
        
        # Convert preferences to dictionary
        preferences_dict = {
            'ai_optimization_enabled': preferences.ai_optimization_enabled,
            'smart_reminders_enabled': preferences.smart_reminders_enabled,
            'daily_water_goal': preferences.daily_water_goal,
            'daily_sleep_goal': preferences.daily_sleep_goal,
            'daily_steps_goal': preferences.daily_steps_goal,
            'daily_activity_goal': preferences.daily_activity_goal,
            'reminder_water': preferences.reminder_water,
            'reminder_exercise': preferences.reminder_exercise,
            'reminder_sleep': preferences.reminder_sleep,
            'reminder_meal': preferences.reminder_meal,
            'reminder_nutrition': preferences.reminder_meal,  # Use meal preference for nutrition
            'reminder_mindfulness': preferences.reminder_mindfulness,
            'quiet_hours_start': preferences.quiet_hours_start.strftime('%H:%M') if preferences.quiet_hours_start else '22:00',
            'quiet_hours_end': preferences.quiet_hours_end.strftime('%H:%M') if preferences.quiet_hours_end else '08:00'
        }
        
        # RUN AI OPTIMIZATION
        # Check if optimization has already been completed today by looking for optimization records
        today = datetime.now().date()
        existing_optimization = AIRecommendation.query.filter(
            AIRecommendation.user_id == current_user.id,
            AIRecommendation.recommendation_type == 'schedule',
            AIRecommendation.title == 'AI Schedule Optimization',
            AIRecommendation.created_at >= datetime.combine(today, datetime.min.time()),
            AIRecommendation.created_at < datetime.combine(today + timedelta(days=1), datetime.min.time())
        ).first()
        
        if existing_optimization:
            return jsonify({
                'success': False,
                'error': f'AI optimization already completed today at {existing_optimization.created_at.strftime("%H:%M")}.',
                'message': 'AI optimization can only be run once per day. Use "Reset Today" to allow re-optimization.',
                'last_optimization': existing_optimization.created_at.isoformat()
            }), 400
        
        optimization_result = get_calendar_service().optimize_schedule(user_events, health_context, preferences_dict)
        
        # APPLY SCHEDULE CHANGES TO DATABASE WITH BACKUP
        events_updated = 0
        today = datetime.now().date()
        
        if optimization_result.get('schedule_changes'):
            for change in optimization_result['schedule_changes']:
                try:
                    event_id = change.get('event_id')
                    if event_id:
                        # Find the event in the database
                        event_to_update = CalendarEvent.query.filter_by(
                            id=event_id, user_id=current_user.id
                        ).first()
                        
                        if event_to_update and event_to_update.is_ai_modifiable:
                            # CREATE BACKUP BEFORE MODIFYING - Check if backup already exists for today
                            existing_backup = EventBackup.query.filter_by(
                                user_id=current_user.id,
                                event_id=event_id,
                                optimization_date=today
                            ).first()
                            
                            if not existing_backup:
                                backup = EventBackup(
                                    user_id=current_user.id,
                                    event_id=event_id,
                                    original_start_time=event_to_update.start_time,
                                    original_end_time=event_to_update.end_time,
                                    optimization_date=today,
                                    backup_reason='ai_optimization'
                                )
                                db.session.add(backup)
                                logger.info(f"Created backup for event '{event_to_update.title}' - Original time: {event_to_update.start_time}")
                            
                            # Parse the new optimized time
                            new_start_time = datetime.strptime(change['optimized_start'], '%H:%M').time()
                            original_date = event_to_update.start_time.date()
                            duration = event_to_update.end_time - event_to_update.start_time
                            
                            # Update the event times
                            event_to_update.start_time = datetime.combine(original_date, new_start_time)
                            event_to_update.end_time = event_to_update.start_time + duration
                            
                            events_updated += 1
                            logger.info(f"Updated event '{event_to_update.title}' to new time: {new_start_time}")
                            
                except Exception as e:
                    logger.warning(f"Failed to apply schedule change: {e}")
                    continue
        
        # CREATE CALENDAR EVENTS FOR NEW REMINDERS
        reminders_created = 0
        reminder_types_created = []
        
        if optimization_result.get('new_reminders'):
            for reminder in optimization_result['new_reminders']:
                try:
                    # Parse reminder time more carefully
                    reminder_time_str = reminder['time']
                    if reminder_time_str.endswith('Z'):
                        reminder_time_str = reminder_time_str[:-1] + '+00:00'
                    
                    reminder_time = datetime.fromisoformat(reminder_time_str)
                    if reminder_time.tzinfo:
                        reminder_time = reminder_time.replace(tzinfo=None)
                    
                    # Get duration from reminder data
                    duration_minutes = reminder.get('duration_minutes', 5)
                    
                    # Create appropriate emoji and title based on type
                    reminder_type = reminder['type']
                    emoji_map = {
                        'water': '💧',
                        'hydration': '💧',  # Support both water and hydration types
                        'exercise': '🏃',
                        'sleep': '😴',
                        'meal': '🍽️',
                        'nutrition': '🍽️',
                        'mindfulness': '🧘‍♀️',
                        'meditation': '🧘‍♀️'
                    }
                    
                    emoji = emoji_map.get(reminder_type, '⏰')
                    title = f"{emoji} {reminder['message']}"
                    
                    # Create calendar event for the reminder
                    reminder_event = CalendarEvent(
                        user_id=current_user.id,
                        title=title,
                        description=f"AI-generated {reminder_type} reminder - Auto-created by your health assistant",
                        start_time=reminder_time,
                        end_time=reminder_time + timedelta(minutes=duration_minutes),
                        event_type='personal',
                        priority_level=reminder.get('priority', 3),
                        is_ai_modifiable=True,
                        is_fixed_time=False
                    )
                    db.session.add(reminder_event)
                    reminders_created += 1
                    reminder_types_created.append(reminder_type)
                    
                    logger.info(f"Created {reminder_type} reminder: {title} at {reminder_time}")
                    
                except Exception as reminder_error:
                    logger.warning(f"Could not create reminder event: {reminder_error}")
                    continue
        
        # Log reminder preferences vs what was actually created
        enabled_reminders = [k.replace('reminder_', '') for k, v in preferences_dict.items() if k.startswith('reminder_') and v]
        logger.info(f"Reminder preferences enabled: {enabled_reminders}")
        logger.info(f"Reminder types actually created: {list(set(reminder_types_created))}")
        
        # CALCULATE DYNAMIC AI CONFIDENCE based on multiple factors
        health_data_count = len(recent_health)
        calculated_confidence = calculate_ai_confidence(
            health_data_count=health_data_count,
            optimization_result=optimization_result,
            schedule_changes=events_updated,
            reminders_created=reminders_created
        )
        
        logger.info(f"AI Confidence calculated: {calculated_confidence:.0%} (based on {health_data_count} days of data, {events_updated} events moved, {reminders_created} reminders)")
        
        # SAVE AI RECOMMENDATIONS with calculated confidence
        if optimization_result.get('schedule_changes') or optimization_result.get('new_reminders'):
            recommendation = AIRecommendation(
                user_id=current_user.id,
                recommendation_type='schedule',
                title='AI Schedule Optimization',
                description=f"Moved {events_updated} events to optimal times and created {reminders_created} health reminders for today",
                ai_confidence=calculated_confidence  # Using dynamic confidence calculation
            )
            db.session.add(recommendation)
        
        # Commit all changes
        db.session.commit()
        
        # Update the result message with comprehensive info
        optimization_result['message'] = f"AI optimization completed: Moved {events_updated} events to optimal times and created {reminders_created} health reminders for today"
        optimization_result['reminders_created'] = reminders_created
        optimization_result['events_moved'] = events_updated
        optimization_result['optimization_date'] = datetime.now().date().isoformat()
        
        logger.info(f"AI optimization completed: Moved {events_updated} events, Created {reminders_created} reminders")
        
        return jsonify(optimization_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/ai/reset_today_optimization', methods=['POST'])
@login_required  
def reset_today_optimization():
    """
    Reset today's AI optimization - restores events to original times and removes AI-generated reminders
    Comprehensive reset that undoes all optimization changes for today
    """
    try:
        today = datetime.now().date()
        
        # STEP 1: RESTORE EVENTS TO ORIGINAL TIMES FROM BACKUPS
        backups_today = EventBackup.query.filter_by(
            user_id=current_user.id,
            optimization_date=today
        ).all()
        
        events_restored = 0
        for backup in backups_today:
            try:
                # Find the event that was modified
                event_to_restore = CalendarEvent.query.filter_by(
                    id=backup.event_id,
                    user_id=current_user.id
                ).first()
                
                if event_to_restore:
                    # Restore original times
                    event_to_restore.start_time = backup.original_start_time
                    event_to_restore.end_time = backup.original_end_time
                    events_restored += 1
                    logger.info(f"Restored event '{event_to_restore.title}' to original time: {backup.original_start_time}")
                
                # Delete the backup record
                db.session.delete(backup)
                
            except Exception as e:
                logger.warning(f"Failed to restore event from backup {backup.id}: {e}")
                continue
        
        # STEP 2: REMOVE AI-GENERATED REMINDERS FOR TODAY
        ai_reminders_today = current_user.calendar_events.filter(
            CalendarEvent.start_time >= datetime.combine(today, datetime.min.time()),
            CalendarEvent.start_time < datetime.combine(today + timedelta(days=1), datetime.min.time()),
            CalendarEvent.description.like('%AI-generated%')
        ).all()
        
        reminders_removed = len(ai_reminders_today)
        for reminder in ai_reminders_today:
            db.session.delete(reminder)
            logger.info(f"Removed AI reminder: {reminder.title}")
        
        # STEP 3: REMOVE AI OPTIMIZATION RECORDS FOR TODAY
        optimization_records_today = AIRecommendation.query.filter(
            AIRecommendation.user_id == current_user.id,
            AIRecommendation.recommendation_type == 'schedule',
            AIRecommendation.title == 'AI Schedule Optimization',
            AIRecommendation.created_at >= datetime.combine(today, datetime.min.time()),
            AIRecommendation.created_at < datetime.combine(today + timedelta(days=1), datetime.min.time())
        ).all()
        
        records_removed = len(optimization_records_today)
        for record in optimization_records_today:
            db.session.delete(record)
        
        # Commit all changes
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Reset complete: Restored {events_restored} events to original times, removed {reminders_removed} AI reminders, and deleted {records_removed} optimization records',
            'events_restored': events_restored,
            'reminders_removed': reminders_removed,
            'records_removed': records_removed
        })
        
    except Exception as e:
        logger.error(f"Error resetting today's optimization: {e}")
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

@bp.route('/ai/generate_reminders', methods=['POST'])
@login_required
def generate_smart_reminders():
    """
    Generate smart reminders based on health patterns and schedule
    Implements adaptive reminders from research
    """
    try:
        user_events_objects = current_user.calendar_events.filter(
            CalendarEvent.start_time >= datetime.now(),
            CalendarEvent.start_time <= datetime.now() + timedelta(days=1)
        ).all()
        
        # Convert SQLAlchemy objects to dictionaries for JSON serialization
        user_events = [event.to_dict() for event in user_events_objects]
        
        preferences = current_user.preferences
        if not preferences:
            return jsonify({'error': 'User preferences not found'}), 400
        
        recent_health = current_user.health_data.order_by(
            HealthData.date_logged.desc()
        ).limit(7).all()
        
        health_context = {
            'recent_water_avg': sum(d.water_intake_liters or 0 for d in recent_health) / len(recent_health) if recent_health else 0,
            'recent_activity_avg': sum(d.active_minutes or 0 for d in recent_health) / len(recent_health) if recent_health else 0,
            'recent_steps': sum(d.steps or 0 for d in recent_health) / len(recent_health) if recent_health else 0
        }
        
        # Convert preferences to dictionary
        preferences_dict = {
            'reminder_water': preferences.reminder_water,
            'reminder_exercise': preferences.reminder_exercise,
            'reminder_sleep': preferences.reminder_sleep,
            'daily_water_goal': preferences.daily_water_goal,
            'daily_activity_goal': preferences.daily_activity_goal
        }
        
        # GENERATE SMART REMINDERS
        reminders = get_calendar_service().generate_smart_reminders(user_events, health_context, preferences_dict)
        
        return jsonify({'reminders': reminders})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# AI ADVICE GENERATION ROUTE - Generate advice on-demand
@bp.route('/ai/generate_advice', methods=['POST'])
@login_required
def generate_personalized_advice():
    """
    Generate personalized health advice on-demand
    Only called when user explicitly requests new advice
    """
    try:
        # GET RECENT HEALTH DATA FOR ANALYSIS
        recent_data = current_user.health_data.order_by(HealthData.date_logged.desc()).limit(30).all()
        
        if not recent_data:
            return jsonify({'error': 'No health data available for analysis'}), 400
            
        # GET USER PREFERENCES
        preferences = current_user.preferences
        if not preferences:
            preferences = UserPreferences(user_id=current_user.id)
            db.session.add(preferences)
            db.session.commit()
        
        # ANALYZE HEALTH PATTERNS WITH AI
        health_patterns = get_health_ai_service().analyze_health_patterns(recent_data) if len(recent_data) >= 7 else {}
        
        # GENERATE PERSONALIZED AI ADVICE
        user_context = {
            'recent_health': [
                {
                    'date': data.date_logged.isoformat(),
                    # Activity Metrics
                    'steps': data.steps,
                    'distance_km': data.distance_km,
                    'calories_total': data.calories_total,
                    'active_minutes': data.active_minutes,
                    'floors_climbed': data.floors_climbed,
                    
                    # Heart Rate Metrics
                    'heart_rate_avg': data.heart_rate_avg,
                    'heart_rate_resting': data.heart_rate_resting,
                    'heart_rate_max': data.heart_rate_max,
                    'heart_rate_variability': data.heart_rate_variability,
                    
                    # Sleep Metrics
                    'sleep_duration_hours': data.sleep_duration_hours,
                    'sleep_quality_score': data.sleep_quality_score,
                    'sleep_deep_minutes': data.sleep_deep_minutes,
                    'sleep_light_minutes': data.sleep_light_minutes,
                    'sleep_rem_minutes': data.sleep_rem_minutes,
                    'sleep_awake_minutes': data.sleep_awake_minutes,
                    
                    # Advanced Health Metrics
                    'blood_oxygen_percent': data.blood_oxygen_percent,
                    'stress_level': data.stress_level,
                    'body_temperature': data.body_temperature,
                    
                    # Body Composition Metrics
                    'weight_kg': data.weight_kg,
                    'body_fat_percent': data.body_fat_percent,
                    'muscle_mass_kg': data.muscle_mass_kg,
                    
                    # Lifestyle Metrics
                    'water_intake_liters': data.water_intake_liters,
                    'mood_score': data.mood_score,
                    'energy_level': data.energy_level,
                    
                    # Exercise Session Details
                    'workout_type': data.workout_type,
                    'workout_duration_minutes': data.workout_duration_minutes,
                    'workout_intensity': data.workout_intensity,
                    'workout_calories': data.workout_calories
                } for data in recent_data[:7]  # Last 7 days for context
            ],
            'health_score': get_health_ai_service().calculate_health_score(recent_data),
            'goals': {
                'water_liters': preferences.daily_water_goal,
                'sleep_hours': preferences.daily_sleep_goal,
                'steps': preferences.daily_steps_goal,
                'activity_minutes': preferences.daily_activity_goal
            }
        }
        
        ai_advice = get_health_ai_service().generate_personalized_advice(user_context, health_patterns)
        
        # Save or update advice in database for persistence
        if ai_advice:
            from app.models import PersonalizedHealthAdvice
            
            # Check if user already has advice
            existing_advice = current_user.personalized_advice
            
            if existing_advice:
                # Update existing advice
                existing_advice.insights = json.dumps(ai_advice.get('insights', []))
                existing_advice.recommendations = json.dumps(ai_advice.get('recommendations', []))
                existing_advice.quick_wins = json.dumps(ai_advice.get('quick_wins', []))
                existing_advice.concerns = json.dumps(ai_advice.get('concerns', []))
                existing_advice.motivation = ai_advice.get('motivation', '')
                existing_advice.source = ai_advice.get('source', 'gemini_ai')
                existing_advice.health_score_at_generation = get_health_ai_service().calculate_health_score(recent_data)
                existing_advice.generated_at = datetime.utcnow()
            else:
                # Create new advice record
                advice_record = PersonalizedHealthAdvice.from_dict(
                    user_id=current_user.id,
                    advice_dict=ai_advice,
                    health_score=get_health_ai_service().calculate_health_score(recent_data)
                )
                db.session.add(advice_record)
            
            db.session.commit()
        
        # Add timestamp to advice for frontend
        if isinstance(ai_advice, dict):
            ai_advice['generated_at'] = datetime.utcnow().isoformat()
        
        return jsonify({'success': True, 'ai_advice': ai_advice})
        
    except Exception as e:
        logger.error(f"Error generating AI advice: {e}")
        return jsonify({'error': str(e)}), 500

# USER PREFERENCES ROUTES - App and AI behavior settings

@bp.route('/preferences', methods=['GET', 'POST'])
@login_required
def user_preferences():
    """
    User preferences for AI behavior and app settings
    Implements research emphasis on user control
    """
    preferences = current_user.preferences
    if not preferences:
        preferences = UserPreferences(user_id=current_user.id)
        db.session.add(preferences)
        db.session.commit()
    
    form = UserPreferencesForm(obj=preferences)
    
    if form.validate_on_submit():
        preferences.ai_optimization_enabled = form.ai_optimization_enabled.data
        preferences.reminder_frequency = form.reminder_frequency.data
        preferences.daily_water_goal = form.daily_water_goal.data
        preferences.daily_sleep_goal = form.daily_sleep_goal.data
        preferences.daily_steps_goal = form.daily_steps_goal.data
        preferences.daily_activity_goal = form.daily_activity_goal.data
        preferences.reminder_water = form.reminder_water.data
        preferences.reminder_exercise = form.reminder_exercise.data
        preferences.reminder_sleep = form.reminder_sleep.data
        preferences.reminder_meal = form.reminder_meal.data
        preferences.reminder_mindfulness = form.reminder_mindfulness.data
        preferences.smart_reminders_enabled = form.smart_reminders_enabled.data
        preferences.updated_at = datetime.utcnow()
        
        db.session.commit()
        flash('Your preferences have been updated!', 'success')
        return redirect(url_for('main.dashboard'))
    
    # Check if user has uploaded health data files
    health_data_uploaded = preferences.health_data_uploaded if preferences else False
    
    # Get AI insights if health data is available
    ai_insights = []
    if health_data_uploaded:
        recent_health_data = current_user.health_data.order_by(HealthData.date_logged.desc()).limit(7).all()
        if recent_health_data:
            health_patterns = get_health_ai_service().analyze_health_patterns(recent_health_data)
            ai_insights = health_patterns.get('insights', [])
    
    return render_template('preferences.html', 
                         title='Preferences', 
                         form=form, 
                         preferences=preferences,
                         ai_insights=ai_insights,
                         google_calendar_connected=preferences.google_calendar_connected if preferences else False)

# HEALTH DATA FILE UPLOAD ROUTES - Replaced Samsung Health API integration

@bp.route('/health_data/upload')
@login_required
def health_data_upload():
    """
    Health data file upload page
    Implements research requirement for multi-device health data support
    Supports Samsung Health, Apple Health, Fitbit, Garmin file uploads
    """
    return render_template('health_data_upload.html', 
                         title='Upload Health Data',
                         supported_formats=current_app.config.get('ALLOWED_HEALTH_FILE_EXTENSIONS', 
                                                                {'csv', 'json', 'txt', 'xml', 'pdf'}))

@bp.route('/health_data/progress')
@login_required
def get_progress():
    """
    Get current processing progress for the user (legacy endpoint for compatibility)
    """
    from app.health_file_processor import PROGRESS_STORE
    
    user_id = current_user.id
    progress_key = f'health_data_progress_{user_id}'
    
    # Try global store first, fallback to session
    progress = PROGRESS_STORE.get(progress_key)
    if not progress:
        progress = session.get(progress_key, {
            'step': 0,
            'total_steps': 4,
            'message': 'Initializing...',
            'percentage': 0,
            'estimated_time': None,
            'start_time': None
        })
    
    return jsonify(progress)

@bp.route('/health_data/test_progress')
@login_required
def test_progress():
    """
    Test endpoint to simulate progress updates (for debugging)
    """
    import time
    import json
    
    user_id = current_user.id  # Get user ID before entering generator
    
    def generate():
        # Send test progress updates
        for step in range(1, 5):
            progress_data = {
                'step': step,
                'total_steps': 4,
                'message': f'Test step {step}: {"File selection" if step == 1 else "Chunking" if step == 2 else "Processing" if step == 3 else "Merging"}...',
                'percentage': (step / 4) * 100,
                'estimated_time': (4 - step) * 10,
                'type': 'progress'
            }
            
            # Update session with user ID
            session[f'health_data_progress_{user_id}'] = progress_data
            
            yield f"data: {json.dumps(progress_data)}\n\n"
            time.sleep(2)  # Wait 2 seconds between updates
        
        # Send completion
        yield f"data: {json.dumps({'type': 'complete', 'message': 'Test complete'})}\n\n"
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@bp.route('/health_data/progress_stream')
@login_required
def progress_stream():
    """
    Server-Sent Events stream for real-time progress updates
    """
    user_id = current_user.id  # Get user ID before entering generator
    
    def generate():
        import time
        import json
        from app.health_file_processor import PROGRESS_STORE
        
        # Send initial connection event
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Progress stream connected'})}\n\n"
        
        last_progress = None
        max_iterations = 300  # 5 minutes max (300 seconds)
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Get current progress from global store
                current_progress = PROGRESS_STORE.get(user_id)
                
                if current_progress:
                    # Always send progress update if it exists
                    if current_progress != last_progress:
                        current_progress['type'] = 'progress'
                        yield f"data: {json.dumps(current_progress)}\n\n"
                        last_progress = current_progress.copy()
                    
                    # If processing is complete, send completion event and close
                    if current_progress.get('step', 0) >= current_progress.get('total_steps', 4):
                        yield f"data: {json.dumps({'type': 'complete', 'message': 'Processing complete'})}\n\n"
                        break
                
                time.sleep(1)  # Check every second
                iteration += 1
                
            except Exception as e:
                logger.error(f"SSE stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                break
    
    response = Response(generate(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@bp.route('/health_data/process', methods=['POST'])
@login_required  
def process_health_files():
    """
    Process uploaded health data files using Gemini 2.5 Flash
    Main replacement for Samsung Health API integration
    """
    try:
        # Check if files were uploaded
        if 'health_files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('health_files')
        
        if not files or all(f.filename == '' for f in files):
            return jsonify({'error': 'No files selected'}), 400
        
        # Validate file types
        health_processor = get_health_processor()
        validation = health_processor.validate_health_files(files)
        if not validation['valid']:
            return jsonify({
                'error': 'File validation failed',
                'details': validation['errors']
            }), 400
        
        # Process files with Gemini 2.5 Flash
        result = health_processor.process_health_files(files, current_user.id)
        
        if result['success']:
            flash(f'Successfully processed {result["files_processed"]} health data files!', 'success')
            
            # Update user preferences to show they have health data
            preferences = current_user.preferences
            if preferences:
                preferences.health_data_uploaded = True
                db.session.commit()
            
            return jsonify({
                'success': True,
                'message': result['message'],
                'files_processed': result['files_processed'],
                'data_summary': result['data'].get('extraction_summary', {}),
                'health_score': result['data'].get('insights', {}).get('health_score', 0),
                'recommendations': result['data'].get('insights', {}).get('recommendations', [])
            })
        else:
            return jsonify({
                'error': result['error'],
                'details': result.get('validation_errors', [])
            }), 400
            
    except Exception as e:
        logger.error(f"Error in health file processing: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@bp.route('/health_data/history')
@login_required
def health_data_history():
    """
    View processed health data history
    Shows data extracted from uploaded files
    """
    # Get processed health data
    processed_data = current_user.health_data.filter(
        HealthData.data_source == 'file_upload'
    ).order_by(HealthData.extraction_date.desc()).limit(10).all()
    
    history = []
    for data in processed_data:
        try:
            processed_json = json.loads(data.processed_data) if data.processed_data else {}
            history.append({
                'id': data.id,
                'extraction_date': data.extraction_date,
                'health_score': data.health_score,
                'device_type': data.device_type,
                'summary': processed_json.get('extraction_summary', {}),
                'insights': processed_json.get('insights', {})
            })
        except json.JSONDecodeError:
            # Skip malformed data
            continue
    
    return render_template('health_data_history.html',
                         title='Health Data History',
                         history=history,
                         now=datetime.utcnow())

@bp.route('/health_data/manage')
@login_required
def manage_health_data():
    """
    View and manage all health data entries
    Shows both manual entries and file uploads with edit/delete options
    """
    # Get all health data for the user
    page = request.args.get('page', 1, type=int)
    per_page = 20  # Show 20 entries per page
    
    health_data = current_user.health_data.order_by(
        HealthData.date_logged.desc(), 
        HealthData.created_at.desc()
    ).paginate(
        page=page, 
        per_page=per_page, 
        error_out=False
    )
    
    return render_template('manage_health_data.html',
                         title='Manage Health Data',
                         health_data=health_data)

@bp.route('/health_data/edit/<int:data_id>', methods=['GET', 'POST'])
@login_required
def edit_health_data(data_id):
    """
    Edit existing health data entry
    Allows users to modify their health data entries
    """
    # Get the health data entry (ensure it belongs to current user)
    health_data = HealthData.query.filter_by(id=data_id, user_id=current_user.id).first_or_404()
    
    # Create form and manually populate with existing data due to field name mismatches
    form = HealthDataForm()
    
    # On GET request, populate form with existing data
    if request.method == 'GET':
        # Date and source
        form.date_logged.data = health_data.date_logged
        form.data_source.data = health_data.data_source or 'manual'
        
        # Activity Metrics
        form.steps.data = health_data.steps
        form.distance_km.data = health_data.distance_km
        form.calories_burned.data = health_data.calories_total  # Map DB field to form field
        form.active_minutes.data = health_data.active_minutes
        
        # Heart Rate Metrics
        form.heart_rate_avg.data = health_data.heart_rate_avg
        form.heart_rate_resting.data = health_data.heart_rate_resting
        form.heart_rate_max.data = health_data.heart_rate_max
        form.heart_rate_variability.data = health_data.heart_rate_variability
        
        # Sleep Metrics
        form.sleep_duration_hours.data = health_data.sleep_duration_hours
        form.sleep_quality_score.data = health_data.sleep_quality_score
        # Convert minutes to hours for display
        form.deep_sleep_hours.data = health_data.sleep_deep_minutes / 60 if health_data.sleep_deep_minutes else None
        form.rem_sleep_hours.data = health_data.sleep_rem_minutes / 60 if health_data.sleep_rem_minutes else None
        form.sleep_awake_minutes.data = health_data.sleep_awake_minutes
        
        # Advanced Health Metrics
        form.oxygen_saturation.data = health_data.blood_oxygen_percent  # Map DB field to form field
        form.systolic_bp.data = health_data.systolic_bp
        form.diastolic_bp.data = health_data.diastolic_bp
        form.stress_level.data = health_data.stress_level
        form.body_temperature.data = health_data.body_temperature
        
        # Body Composition Metrics
        form.weight_kg.data = health_data.weight_kg
        form.body_fat_percentage.data = health_data.body_fat_percent  # Map DB field to form field
        form.muscle_mass_kg.data = health_data.muscle_mass_kg
        form.bmi.data = health_data.bmi
        
        # Nutrition Metrics
        form.calories_consumed.data = health_data.calories_consumed
        form.protein_grams.data = health_data.protein_grams
        form.carbs_grams.data = health_data.carbs_grams
        form.fat_grams.data = health_data.fat_grams
        form.fiber_grams.data = health_data.fiber_grams
        
        # Lifestyle Metrics
        form.water_intake_liters.data = health_data.water_intake_liters
        form.mood_score.data = health_data.mood_score
        form.energy_level.data = health_data.energy_level
        form.meditation_minutes.data = health_data.meditation_minutes
        form.screen_time_hours.data = health_data.screen_time_hours
        form.social_interactions.data = health_data.social_interactions
        
        # Exercise Session Details
        form.exercise_type.data = health_data.workout_type  # Map DB field to form field
        form.exercise_duration_minutes.data = health_data.workout_duration_minutes  # Map DB field to form field
        
        # Notes
        form.notes.data = health_data.notes
    
    if form.validate_on_submit():
        # Update date and source
        health_data.date_logged = form.date_logged.data
        
        # Activity Metrics
        health_data.steps = form.steps.data
        health_data.distance_km = form.distance_km.data
        health_data.calories_total = form.calories_burned.data  # Form field is calories_burned
        health_data.active_minutes = form.active_minutes.data
        # Note: floors_climbed exists in DB but not in form yet
        
        # Heart Rate Metrics
        health_data.heart_rate_avg = form.heart_rate_avg.data
        health_data.heart_rate_resting = form.heart_rate_resting.data
        health_data.heart_rate_max = form.heart_rate_max.data
        health_data.heart_rate_variability = form.heart_rate_variability.data
        
        # Sleep Metrics
        health_data.sleep_duration_hours = form.sleep_duration_hours.data
        health_data.sleep_quality_score = form.sleep_quality_score.data
        # Convert hours to minutes for deep and REM sleep
        if form.deep_sleep_hours.data:
            health_data.sleep_deep_minutes = int(form.deep_sleep_hours.data * 60)
        if form.rem_sleep_hours.data:
            health_data.sleep_rem_minutes = int(form.rem_sleep_hours.data * 60)
        health_data.sleep_awake_minutes = form.sleep_awake_minutes.data
        # Note: sleep_light_minutes exists in DB but not in form
        
        # Advanced Health Metrics (mapping form names to DB names)
        health_data.blood_oxygen_percent = form.oxygen_saturation.data  # Form field is oxygen_saturation
        health_data.systolic_bp = form.systolic_bp.data
        health_data.diastolic_bp = form.diastolic_bp.data
        health_data.stress_level = form.stress_level.data
        health_data.body_temperature = form.body_temperature.data
        
        # Body Composition Metrics
        health_data.weight_kg = form.weight_kg.data
        health_data.body_fat_percent = form.body_fat_percentage.data  # Form field is body_fat_percentage
        health_data.muscle_mass_kg = form.muscle_mass_kg.data
        health_data.bmi = form.bmi.data
        
        # Nutrition Metrics
        health_data.calories_consumed = form.calories_consumed.data
        health_data.protein_grams = form.protein_grams.data
        health_data.carbs_grams = form.carbs_grams.data
        health_data.fat_grams = form.fat_grams.data
        health_data.fiber_grams = form.fiber_grams.data
        
        # Lifestyle Metrics
        health_data.water_intake_liters = form.water_intake_liters.data
        health_data.mood_score = form.mood_score.data
        health_data.energy_level = form.energy_level.data
        health_data.meditation_minutes = form.meditation_minutes.data
        health_data.screen_time_hours = form.screen_time_hours.data
        health_data.social_interactions = form.social_interactions.data
        
        # Exercise Session Details (mapping form names to DB names)
        health_data.workout_type = form.exercise_type.data  # Form field is exercise_type
        health_data.workout_duration_minutes = form.exercise_duration_minutes.data  # Form field is exercise_duration_minutes
        # Note: workout_intensity and workout_calories exist in DB but not in form yet
        
        # Notes
        health_data.notes = form.notes.data
        
        # Update metadata
        health_data.updated_at = datetime.utcnow()
        health_data.data_source = form.data_source.data or 'manual'
        
        try:
            db.session.commit()
            flash(f'Health data for {health_data.date_logged.strftime("%Y-%m-%d")} has been updated!', 'success')
            return redirect(url_for('main.manage_health_data'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating health data: {str(e)}', 'error')
            logger.error(f"Error updating health data: {e}")
    
    # Log form errors for debugging
    if form.errors:
        logger.error(f"Form validation errors: {form.errors}")
        for field, errors in form.errors.items():
            for error in errors:
                flash(f'{field}: {error}', 'error')
    
    return render_template('edit_health_data.html', 
                         title='Edit Health Data', 
                         form=form, 
                         health_data=health_data)

@bp.route('/health_data/delete/<int:data_id>', methods=['POST'])
@login_required
def delete_health_data(data_id):
    """
    Delete health data entry
    Allows users to remove their health data entries
    """
    # Get the health data entry (ensure it belongs to current user)
    health_data = HealthData.query.filter_by(id=data_id, user_id=current_user.id).first_or_404()
    date_logged = health_data.date_logged.strftime('%Y-%m-%d')
    
    db.session.delete(health_data)
    db.session.commit()
    
    flash(f'Health data for {date_logged} has been deleted.', 'success')
    return redirect(url_for('main.manage_health_data'))

@bp.route('/health_data/delete_upload/<int:data_id>', methods=['POST'])
@login_required
def delete_health_upload(data_id):
    """
    Delete health data upload entry (file upload)
    Allows users to remove their uploaded health data
    """
    # Get the health data entry (ensure it belongs to current user)
    health_data = HealthData.query.filter_by(id=data_id, user_id=current_user.id).first_or_404()
    
    # Only allow deletion of file uploads, not manual entries
    if health_data.data_source != 'file_upload':
        flash('Only uploaded health data can be deleted from this page.', 'error')
        return redirect(url_for('main.health_data_history'))
    
    extraction_date = health_data.extraction_date.strftime('%Y-%m-%d %H:%M') if health_data.extraction_date else 'Unknown date'
    
    db.session.delete(health_data)
    db.session.commit()
    
    flash(f'Health data upload from {extraction_date} has been deleted.', 'success')
    return redirect(url_for('main.health_data_history'))

# HELPER FUNCTION - Calculate overall health score from recent data
def calculate_health_score(health_data_list):
    """
    'def calculate_health_score(health_data_list):' - Define health score calculation function
    'calculate_health_score' - Function name describing its purpose
    'health_data_list' - Parameter containing list of HealthData objects
    This function analyzes recent health data to calculate an overall wellness score
    """
    """Calculate a simple health score based on recent data"""
    
    # HANDLE EMPTY DATA
    if not health_data_list:
        """
        'if not health_data_list:' - Check if list is empty
        'not' - Python operator that inverts boolean value
        True if list is empty or None
        """
        return 0
        """
        'return 0' - Return zero score if no data available
        """
    
    # INITIALIZE SCORING VARIABLES
    total_score = 0
    count = 0
    """
    'total_score = 0' - Initialize accumulator for total score
    'count = 0' - Initialize counter for number of days with data
    These track cumulative scores across multiple days
    """
    
    # PROCESS EACH DAY'S HEALTH DATA
    for data in health_data_list:
        """
        'for data in health_data_list:' - Iterate through each health data record
        'for' - Python loop keyword
        'data' - Variable holding current HealthData object
        'in' - Python membership operator
        'health_data_list' - List of health records to process
        Loop processes each day's health data to calculate daily scores
        """
        
        # INITIALIZE DAILY SCORING
        daily_score = 0
        metrics_count = 0
        """
        'daily_score = 0' - Initialize score for this day
        'metrics_count = 0' - Count how many metrics have data for this day
        """
        
        # SLEEP SCORE CALCULATION - Optimal range scoring
        if data.sleep_duration_hours:
            """
            'if data.sleep_duration_hours:' - Check if sleep data exists
            'data.sleep_duration_hours' - Sleep hours value from database
            Only calculate sleep score if user logged sleep data
            """
            if 7 <= data.sleep_duration_hours <= 9:
                """
                'if 7 <= data.sleep_duration_hours <= 9:' - Check if sleep is in optimal range
                '7 <= data.sleep_duration_hours <= 9' - Chained comparison for range check
                Optimal sleep range is 7-9 hours for most adults
                """
                sleep_score = 10
                """
                'sleep_score = 10' - Perfect score for optimal sleep
                """
            elif 6 <= data.sleep_duration_hours < 7 or 9 < data.sleep_duration_hours <= 10:
                """
                'elif 6 <= data.sleep_duration_hours < 7 or 9 < data.sleep_duration_hours <= 10:' - Check near-optimal range
                'elif' - Python "else if" for additional conditions
                'or' - Logical OR operator
                Covers slightly less than optimal (6-7 hours) or slightly more (9-10 hours)
                """
                sleep_score = 8
            elif 5 <= data.sleep_duration_hours < 6 or 10 < data.sleep_duration_hours <= 11:
                sleep_score = 6
            else:
                sleep_score = 4
                """
                Lower scores for sleep outside healthy ranges
                """
            daily_score += sleep_score
            """
            'daily_score += sleep_score' - Add sleep score to daily total
            '+=' - Python addition assignment operator (same as daily_score = daily_score + sleep_score)
            """
            metrics_count += 1
            """
            'metrics_count += 1' - Increment count of metrics with data
            """
        
        # WATER SCORE CALCULATION - Linear scoring with limits
        if data.water_intake_liters:
            # Convert liters to approximate glasses (1 liter = ~4 glasses)
            glasses = data.water_intake_liters * 4
            water_score = min(10, max(1, glasses))
            """
            'water_score = min(10, max(1, glasses))' - Calculate water score
            'max(1, glasses)' - Ensure minimum score of 1 (at least 1 glass gets 1 point)
            'min(10, ...)' - Cap maximum score at 10 (more than 10 glasses still gets 10 points)
            This creates a score that increases with water intake but has reasonable bounds
            """
            daily_score += water_score
            metrics_count += 1
        
        # ACTIVITY SCORE - Based on active minutes (converted to 1-10 scale)
        if data.active_minutes:
            # Convert active minutes to 1-10 scale (30+ minutes = 10)
            activity_score = min(10, max(1, data.active_minutes / 3))
            daily_score += activity_score
            """
            'daily_score += activity_score' - Add activity score
            Activity minutes converted to 1-10 scale where 30+ minutes = 10 points
            """
            metrics_count += 1
        
        # MOOD SCORE - Direct mapping (already 1-10 scale)
        if data.mood_score:
            daily_score += data.mood_score
            """
            'daily_score += data.mood_score' - Add mood score directly
            Mood is on 1-10 scale
            """
            metrics_count += 1
        
        # CALCULATE AVERAGE DAILY SCORE
        if metrics_count > 0:
            """
            'if metrics_count > 0:' - Check if any metrics had data
            Prevents division by zero error
            """
            total_score += daily_score / metrics_count
            """
            'total_score += daily_score / metrics_count' - Add daily average to total
            'daily_score / metrics_count' - Calculate average score for this day
            '/' - Division operator
            This ensures days with more data don't artificially inflate the score
            """
            count += 1
            """
            'count += 1' - Increment count of days with health data
            """
    
    # RETURN FINAL AVERAGED SCORE
    return round(total_score / count, 1) if count > 0 else 0
    """
    'return round(total_score / count, 1) if count > 0 else 0' - Calculate final score
    'total_score / count' - Average score across all days with data
    'round(..., 1)' - Round to 1 decimal place for clean display
    'if count > 0 else 0' - Conditional expression to handle no data case
    Returns average health score across recent days, or 0 if no data
    """

# GOOGLE CALENDAR INTEGRATION ROUTES

@bp.route('/google_calendar/connect')
@login_required
def connect_google_calendar():
    """
    Initiate Google Calendar connection
    Implements Google Calendar integration for schedule optimization
    """
    try:
        redirect_uri = url_for('main.google_calendar_callback', _external=True)
        auth_url = get_google_calendar_service().get_authorization_url(redirect_uri)
        return redirect(auth_url)
    except ValueError as e:
        # Handle credential errors with helpful message
        if "not configured" in str(e).lower():
            flash('Google Calendar API is not configured. Please add your API credentials to enable calendar integration!', 'info')
        else:
            flash(f'Error connecting to Google Calendar: {str(e)}', 'error')
        return redirect(url_for('main.user_preferences'))
    except Exception as e:
        flash(f'Error connecting to Google Calendar: {str(e)}', 'error')
        return redirect(url_for('main.user_preferences'))

@bp.route('/google_calendar/callback')
@login_required
def google_calendar_callback():
    """
    Handle Google Calendar OAuth callback
    Processes authorization and syncs calendar events
    """
    try:
        authorization_code = request.args.get('code')
        if not authorization_code:
            flash('Google Calendar authorization was cancelled.', 'warning')
            return redirect(url_for('main.user_preferences'))
        
        redirect_uri = url_for('main.google_calendar_callback', _external=True)
        credentials = get_google_calendar_service().exchange_code_for_token(authorization_code, redirect_uri)
        
        if credentials and 'access_token' in credentials:
            # Store credentials securely (in production, encrypt this)
            session['google_calendar_credentials'] = credentials
            
            # OAuth authentication succeeded, so mark as connected
            # We'll test the actual API connection during sync
            connection_successful = True
            flash('Google Calendar connected successfully!', 'success')
            flash('Use the "Sync Calendar" button below to import your events.', 'info')
            
            # Update user preferences only if we have valid credentials
            preferences = current_user.preferences
            if preferences:
                preferences.google_calendar_connected = connection_successful
                db.session.commit()
        else:
            flash('Failed to connect to Google Calendar. Please try again.', 'error')
        
    except Exception as e:
        flash(f'Error processing Google Calendar connection: {str(e)}', 'error')
    
    return redirect(url_for('main.user_preferences'))

@bp.route('/google_calendar/sync', methods=['POST'])
@login_required
def sync_google_calendar():
    """
    Manual Google Calendar sync
    Allows users to refresh their calendar events
    """
    try:
        logger.info("Google Calendar sync requested by user")
        
        credentials = session.get('google_calendar_credentials')
        if not credentials:
            logger.error("No Google Calendar credentials found in session")
            return jsonify({'success': False, 'error': 'Google Calendar not connected'}), 400
        
        logger.info("Starting calendar sync with Google Calendar service")
        sync_result = get_google_calendar_service().sync_events_to_database(
            current_user.id, 
            credentials, 
            days_ahead=30
        )
        
        logger.info(f"Sync result: {sync_result}")
        return jsonify(sync_result)
        
    except Exception as e:
        logger.error(f"Error in Google Calendar sync: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@bp.route('/google_calendar/disconnect', methods=['POST'])
@login_required
def disconnect_google_calendar():
    """
    Disconnect Google Calendar integration
    """
    try:
        # Clear session credentials
        session.pop('google_calendar_credentials', None)
        
        # Update user preferences
        preferences = current_user.preferences
        if preferences:
            preferences.google_calendar_connected = False
            db.session.commit()
        
        flash('Google Calendar disconnected successfully.', 'success')
        
    except Exception as e:
        flash(f'Error disconnecting Google Calendar: {str(e)}', 'error')
    
    return redirect(url_for('main.user_preferences'))

@bp.route('/health_data/delete_processing/<int:processing_id>', methods=['DELETE'])
@login_required
def delete_processing_record(processing_id):
    """
    Delete a processing record and all associated health data from that upload session
    """
    try:
        # Get the processing record (summary record from file upload)
        processing_record = HealthData.query.filter_by(
            id=processing_id, 
            user_id=current_user.id, 
            data_source='file_upload'
        ).first()
        
        if not processing_record:
            return jsonify({'success': False, 'error': 'Processing record not found'}), 404
        
        # Get the upload session ID to find all related health data
        # Handle both old records (without upload_session_id) and new records (with upload_session_id)
        upload_session_id = getattr(processing_record, 'upload_session_id', None)
        
        if upload_session_id:
            # New records with session ID - delete all records from this session
            related_records = HealthData.query.filter_by(
                user_id=current_user.id,
                upload_session_id=upload_session_id
            ).all()
        else:
            # Old records without session ID - use extraction_date to find related records
            extraction_date = processing_record.extraction_date
            if extraction_date:
                related_records = HealthData.query.filter_by(
                    user_id=current_user.id,
                    extraction_date=extraction_date
                ).all()
            else:
                # If no extraction date either, just delete this record
                related_records = [processing_record]
        
        # Count records for response
        deleted_count = len(related_records)
        
        # Delete all related records
        for record in related_records:
            db.session.delete(record)
        
        db.session.commit()
        
        logger.info(f"Deleted processing record {processing_id} and {deleted_count} related health data entries for user {current_user.id}")
        
        return jsonify({
            'success': True, 
            'deleted_count': deleted_count,
            'message': f'Successfully deleted processing record and {deleted_count} related health data entries'
        })
        
    except Exception as e:
        logger.error(f"Error deleting processing record {processing_id}: {e}")
        db.session.rollback()
        return jsonify({'success': False, 'error': 'Failed to delete processing record'}), 500
