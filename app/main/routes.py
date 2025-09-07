"""
Main Application Routes for Core Functionality

This file contains the primary routes for the health assistant application.
These routes handle the home page, dashboard, health data logging, calendar management,
and AI-powered features. Routes connect URLs to Python functions that generate responses.
"""

# FLASK CORE IMPORTS - Essential Flask functionality (patterns already explained in auth/routes.py)
from flask import render_template, redirect, url_for, flash, request, jsonify, session, current_app
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

from app.models import HealthData, CalendarEvent, UserPreferences, AIRecommendation
"""
'from app.models import HealthData, CalendarEvent, UserPreferences, AIRecommendation' - Import models
All the database models we created for comprehensive health and calendar management
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
    today_data = None
    if recent_data:
        today_data = recent_data[0]  # Most recent entry
    
    sleep_hours = today_data.sleep_duration_hours if today_data and today_data.sleep_duration_hours else 0
    sleep_goal = preferences.daily_sleep_goal if preferences.daily_sleep_goal else 8
    
    # RENDER ENHANCED DASHBOARD
    return render_template('dashboard.html', 
                         title='Dashboard', 
                         health_data=recent_data[:7],  # Show last 7 days
                         health_score=health_score,
                         activity_score=individual_scores['activity_score'],
                         sleep_score=individual_scores['sleep_score'],
                         nutrition_score=individual_scores['nutrition_score'],
                         hydration_score=individual_scores['hydration_score'],
                         heart_health_score=individual_scores['heart_health_score'],
                         wellness_score=individual_scores['wellness_score'],
                         sleep_hours=sleep_hours,
                         sleep_goal=sleep_goal,
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
        
        # CHECK FOR EXISTING DATA TODAY
        today = date.today()
        """
        'today = date.today()' - Get current date
        'date.today()' - Returns today's date (year, month, day)
        Used to check if user already logged health data today
        """
        
        existing_data = HealthData.query.filter_by(
            user_id=current_user.id, 
            date_logged=today
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
            
            # UPDATE EXISTING RECORD WITH COMPREHENSIVE HEALTH METRICS
            # Activity Metrics
            existing_data.steps = form.steps.data
            existing_data.distance_km = form.distance_km.data
            existing_data.calories_total = form.calories_total.data
            existing_data.active_minutes = form.active_minutes.data
            existing_data.floors_climbed = form.floors_climbed.data
            
            # Heart Rate Metrics
            existing_data.heart_rate_avg = form.heart_rate_avg.data
            existing_data.heart_rate_resting = form.heart_rate_resting.data
            existing_data.heart_rate_max = form.heart_rate_max.data
            existing_data.heart_rate_variability = form.heart_rate_variability.data
            
            # Sleep Metrics
            existing_data.sleep_duration_hours = form.sleep_duration_hours.data
            existing_data.sleep_quality_score = form.sleep_quality_score.data
            existing_data.sleep_deep_minutes = form.sleep_deep_minutes.data
            existing_data.sleep_light_minutes = form.sleep_light_minutes.data
            existing_data.sleep_rem_minutes = form.sleep_rem_minutes.data
            existing_data.sleep_awake_minutes = form.sleep_awake_minutes.data
            
            # Advanced Health Metrics
            existing_data.blood_oxygen_percent = form.blood_oxygen_percent.data
            existing_data.stress_level = form.stress_level.data
            existing_data.body_temperature = form.body_temperature.data
            
            # Body Composition Metrics
            existing_data.weight_kg = form.weight_kg.data
            existing_data.body_fat_percent = form.body_fat_percent.data
            existing_data.muscle_mass_kg = form.muscle_mass_kg.data
            
            # Lifestyle Metrics
            existing_data.water_intake_liters = form.water_intake_liters.data
            existing_data.mood_score = form.mood_score.data
            existing_data.energy_level = form.energy_level.data
            
            # Exercise Session Details
            existing_data.workout_type = form.workout_type.data
            existing_data.workout_duration_minutes = form.workout_duration_minutes.data
            existing_data.workout_intensity = form.workout_intensity.data
            existing_data.workout_calories = form.workout_calories.data
            
            existing_data.updated_at = datetime.utcnow()
            existing_data.data_source = 'manual'
            
            flash('Your comprehensive health data has been updated!', 'success')
            
        else:
            # CREATE NEW COMPREHENSIVE HEALTH DATA RECORD
            health_data = HealthData(
                user_id=current_user.id,
                date_logged=today,
                data_source='manual',
                
                # Activity Metrics
                steps=form.steps.data,
                distance_km=form.distance_km.data,
                calories_total=form.calories_total.data,
                active_minutes=form.active_minutes.data,
                floors_climbed=form.floors_climbed.data,
                
                # Heart Rate Metrics
                heart_rate_avg=form.heart_rate_avg.data,
                heart_rate_resting=form.heart_rate_resting.data,
                heart_rate_max=form.heart_rate_max.data,
                heart_rate_variability=form.heart_rate_variability.data,
                
                # Sleep Metrics
                sleep_duration_hours=form.sleep_duration_hours.data,
                sleep_quality_score=form.sleep_quality_score.data,
                sleep_deep_minutes=form.sleep_deep_minutes.data,
                sleep_light_minutes=form.sleep_light_minutes.data,
                sleep_rem_minutes=form.sleep_rem_minutes.data,
                sleep_awake_minutes=form.sleep_awake_minutes.data,
                
                # Advanced Health Metrics
                blood_oxygen_percent=form.blood_oxygen_percent.data,
                stress_level=form.stress_level.data,
                body_temperature=form.body_temperature.data,
                
                # Body Composition Metrics
                weight_kg=form.weight_kg.data,
                body_fat_percent=form.body_fat_percent.data,
                muscle_mass_kg=form.muscle_mass_kg.data,
                
                # Lifestyle Metrics
                water_intake_liters=form.water_intake_liters.data,
                mood_score=form.mood_score.data,
                energy_level=form.energy_level.data,
                
                # Exercise Session Details
                workout_type=form.workout_type.data,
                workout_duration_minutes=form.workout_duration_minutes.data,
                workout_intensity=form.workout_intensity.data,
                workout_calories=form.workout_calories.data
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
        db.session.commit()
        """
        'db.session.commit()' - Save all changes to database
        Commits both updates to existing records and new record insertions
        """
        
        return redirect(url_for('main.dashboard'))
        """
        'return redirect(url_for('main.dashboard'))' - Redirect to dashboard
        After successful data logging, show user their updated dashboard
        """
    
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
        # GET USER EVENTS AND PREFERENCES
        user_events = current_user.calendar_events.filter(
            CalendarEvent.start_time >= datetime.now()
        ).all()
        
        preferences = current_user.preferences
        if not preferences or not preferences.ai_optimization_enabled:
            return jsonify({'error': 'AI optimization is disabled'}), 400
        
        # GET RECENT HEALTH DATA FOR CONTEXT
        recent_health = current_user.health_data.order_by(
            HealthData.date_logged.desc()
        ).limit(7).all()
        
        health_context = {
            'recent_activity_level': sum(d.active_minutes or 0 for d in recent_health) / len(recent_health) if recent_health else 0,
            'recent_sleep_average': sum(d.sleep_duration_hours or 0 for d in recent_health) / len(recent_health) if recent_health else 8,
            'recent_steps': sum(d.steps or 0 for d in recent_health) / len(recent_health) if recent_health else 0
        }
        
        # RUN AI OPTIMIZATION
        optimization_result = get_calendar_service().optimize_schedule(user_events, health_context, preferences)
        
        # CREATE CALENDAR EVENTS FOR NEW REMINDERS
        if optimization_result.get('new_reminders'):
            for reminder in optimization_result['new_reminders']:
                try:
                    # Parse reminder time
                    reminder_time = datetime.fromisoformat(reminder['time'].replace('Z', '+00:00'))
                    if reminder_time.tzinfo:
                        reminder_time = reminder_time.replace(tzinfo=None)
                    
                    # Create calendar event for the reminder
                    reminder_event = CalendarEvent(
                        user_id=current_user.id,
                        title=f"💧 {reminder['message']}" if reminder['type'] == 'water' else f"🏃 {reminder['message']}" if reminder['type'] == 'exercise' else f"😴 {reminder['message']}" if reminder['type'] == 'sleep' else reminder['message'],
                        description=f"AI-generated {reminder['type']} reminder",
                        start_time=reminder_time,
                        end_time=reminder_time + timedelta(minutes=5),  # 5-minute reminder events
                        event_type='personal',
                        priority_level=reminder.get('priority', 2),
                        is_ai_modifiable=True,
                        is_fixed_time=False
                    )
                    db.session.add(reminder_event)
                except Exception as reminder_error:
                    logger.warning(f"Could not create reminder event: {reminder_error}")
                    continue
        
        # SAVE AI RECOMMENDATIONS
        if optimization_result.get('schedule_changes') or optimization_result.get('new_reminders'):
            recommendation = AIRecommendation(
                user_id=current_user.id,
                recommendation_type='schedule',
                title='AI Schedule Optimization',
                description=f"Generated {len(optimization_result.get('schedule_changes', []))} schedule suggestions and {len(optimization_result.get('new_reminders', []))} reminders",
                ai_confidence=0.8
            )
            db.session.add(recommendation)
            db.session.commit()
        
        return jsonify(optimization_result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/ai/generate_reminders', methods=['POST'])
@login_required
def generate_smart_reminders():
    """
    Generate smart reminders based on health patterns and schedule
    Implements adaptive reminders from research
    """
    try:
        user_events = current_user.calendar_events.filter(
            CalendarEvent.start_time >= datetime.now(),
            CalendarEvent.start_time <= datetime.now() + timedelta(days=1)
        ).all()
        
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
        
        # GENERATE SMART REMINDERS
        reminders = get_calendar_service().generate_smart_reminders(user_events, health_context, preferences)
        
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
        preferences.reminder_medication = form.reminder_medication.data
        preferences.reminder_meal = form.reminder_meal.data
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
    
    # Create form with existing data
    form = HealthDataForm(obj=health_data)
    
    if form.validate_on_submit():
        # Update all the fields from the form
        health_data.steps = form.steps.data
        health_data.distance_km = form.distance_km.data
        health_data.calories_total = form.calories_total.data
        health_data.active_minutes = form.active_minutes.data
        health_data.floors_climbed = form.floors_climbed.data
        
        # Heart Rate Metrics
        health_data.heart_rate_avg = form.heart_rate_avg.data
        health_data.heart_rate_resting = form.heart_rate_resting.data
        health_data.heart_rate_max = form.heart_rate_max.data
        health_data.heart_rate_variability = form.heart_rate_variability.data
        
        # Sleep Metrics
        health_data.sleep_duration_hours = form.sleep_duration_hours.data
        health_data.sleep_quality_score = form.sleep_quality_score.data
        health_data.sleep_deep_minutes = form.sleep_deep_minutes.data
        health_data.sleep_light_minutes = form.sleep_light_minutes.data
        health_data.sleep_rem_minutes = form.sleep_rem_minutes.data
        health_data.sleep_awake_minutes = form.sleep_awake_minutes.data
        
        # Advanced Health Metrics
        health_data.blood_oxygen_percent = form.blood_oxygen_percent.data
        health_data.stress_level = form.stress_level.data
        health_data.body_temperature = form.body_temperature.data
        
        # Body Composition Metrics
        health_data.weight_kg = form.weight_kg.data
        health_data.body_fat_percent = form.body_fat_percent.data
        health_data.muscle_mass_kg = form.muscle_mass_kg.data
        
        # Lifestyle Metrics
        health_data.water_intake_liters = form.water_intake_liters.data
        health_data.mood_score = form.mood_score.data
        health_data.energy_level = form.energy_level.data
        
        # Exercise Session Details
        health_data.workout_type = form.workout_type.data
        health_data.workout_duration_minutes = form.workout_duration_minutes.data
        health_data.workout_intensity = form.workout_intensity.data
        health_data.workout_calories = form.workout_calories.data
        
        # Update metadata
        health_data.updated_at = datetime.utcnow()
        health_data.data_source = 'manual'  # Mark as manually edited
        
        db.session.commit()
        flash(f'Health data for {health_data.date_logged.strftime("%Y-%m-%d")} has been updated!', 'success')
        return redirect(url_for('main.manage_health_data'))
    
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
