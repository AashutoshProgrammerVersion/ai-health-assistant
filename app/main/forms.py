"""
Health Data Forms for Tracking Wellness Metrics

This file defines forms for logging and managing health data.
These forms handle validation of health metrics and provide
user-friendly interfaces for data entry with proper constraints.
"""

# FLASK-WTF IMPORT - Secure form handling (pattern already explained in auth/forms.py)
from flask_wtf import FlaskForm

# WTFORMS FIELD IMPORTS - Form input field types
from wtforms import FloatField, IntegerField, SelectField, SubmitField, StringField, TextAreaField, BooleanField, DateTimeField, DateField
"""
'from wtforms import FloatField, IntegerField, SelectField, SubmitField' - Form field types
'FloatField' - Input field for decimal numbers (7.5 hours, 2.3 liters)
'IntegerField' - Input field for whole numbers (steps, heart rate)
'SelectField' - Dropdown menu field for predefined choices
'SubmitField' - Button field for form submission (already explained)
'StringField' - Text input field for strings
'TextAreaField' - Multi-line text input field
'BooleanField' - Checkbox field for True/False values
'DateTimeField' - Date and time picker field
"""

# WTFORMS VALIDATOR IMPORTS - Input validation functions
from wtforms.validators import DataRequired, NumberRange, Optional, Length
"""
'from wtforms.validators import DataRequired, NumberRange, Optional' - Validation functions
'DataRequired' - Ensures field is not empty (already explained)
'NumberRange' - Validates numbers are within specified minimum/maximum range
'Optional' - Allows field to be empty (opposite of DataRequired)
'Length' - Validates string length is within specified range
"""

# COMPREHENSIVE HEALTH DATA FORM - Enhanced form for detailed health tracking
class HealthDataForm(FlaskForm):
    """
    Comprehensive health data entry form supporting all major wearable device metrics
    Based on health metrics blueprint for consistent data collection
    """
    
    # DATE AND SOURCE - When and how data was logged
    date_logged = DateField('Date', validators=[DataRequired()], 
                           render_kw={"type": "date"})
    
    data_source = SelectField('Data Source', 
                             choices=[('manual', 'Manual Entry'),
                                     ('file_upload', 'File Upload'),
                                     ('fitbit', 'Fitbit'),
                                     ('apple_health', 'Apple Health'),
                                     ('samsung_health', 'Samsung Health'),
                                     ('garmin', 'Garmin'),
                                     ('other', 'Other')],
                             default='manual')
    
    # ACTIVITY METRICS - Movement and exercise tracking
    steps = IntegerField('Daily Steps', 
                        validators=[Optional(), NumberRange(min=0, max=50000)],
                        render_kw={"placeholder": "e.g., 10000"})
    
    distance_km = FloatField('Distance (km)', 
                            validators=[Optional(), NumberRange(min=0, max=100)],
                            render_kw={"placeholder": "e.g., 6.5"})
    
    calories_burned = IntegerField('Calories Burned', 
                                 validators=[Optional(), NumberRange(min=0, max=8000)],
                                 render_kw={"placeholder": "e.g., 2200"})
    
    active_minutes = IntegerField('Active Minutes', 
                                 validators=[Optional(), NumberRange(min=0, max=1440)],
                                 render_kw={"placeholder": "e.g., 45"})
    
    exercise_type = StringField('Exercise Type', 
                               validators=[Optional(), Length(max=100)],
                               render_kw={"placeholder": "e.g., Running, Yoga"})
    
    exercise_duration_minutes = IntegerField('Exercise Duration (minutes)', 
                                           validators=[Optional(), NumberRange(min=0, max=480)],
                                           render_kw={"placeholder": "e.g., 60"})
    
    # HEART RATE METRICS - Cardiovascular health indicators
    heart_rate_avg = IntegerField('Average Heart Rate (BPM)', 
                                 validators=[Optional(), NumberRange(min=40, max=220)],
                                 render_kw={"placeholder": "e.g., 72"})
    
    heart_rate_resting = IntegerField('Resting Heart Rate (BPM)', 
                                     validators=[Optional(), NumberRange(min=40, max=100)],
                                     render_kw={"placeholder": "e.g., 60"})
    
    heart_rate_max = IntegerField('Maximum Heart Rate (BPM)', 
                                 validators=[Optional(), NumberRange(min=60, max=220)],
                                 render_kw={"placeholder": "e.g., 180"})
    
    heart_rate_variability = FloatField('Heart Rate Variability (ms)', 
                                       validators=[Optional(), NumberRange(min=10, max=100)],
                                       render_kw={"placeholder": "e.g., 32.5"})
    
    # SLEEP METRICS - Sleep quality and duration tracking
    sleep_duration_hours = FloatField('Sleep Duration (hours)', 
                                     validators=[Optional(), NumberRange(min=0, max=24)],
                                     render_kw={"placeholder": "e.g., 7.5"})
    
    sleep_quality_score = IntegerField('Sleep Quality Score (1-10)', 
                                      validators=[Optional(), NumberRange(min=1, max=10)],
                                      render_kw={"placeholder": "e.g., 8"})
    
    deep_sleep_hours = FloatField('Deep Sleep (hours)', 
                                 validators=[Optional(), NumberRange(min=0, max=12)],
                                 render_kw={"placeholder": "e.g., 1.5"})
    
    rem_sleep_hours = FloatField('REM Sleep (hours)', 
                                validators=[Optional(), NumberRange(min=0, max=8)],
                                render_kw={"placeholder": "e.g., 1.75"})
    
    sleep_awake_minutes = IntegerField('Time Awake in Bed (minutes)', 
                                      validators=[Optional(), NumberRange(min=0, max=300)],
                                      render_kw={"placeholder": "e.g., 15"})
    
    # ADVANCED HEALTH METRICS - Premium device metrics
    oxygen_saturation = IntegerField('Blood Oxygen SpO2 (%)', 
                                   validators=[Optional(), NumberRange(min=70, max=100)],
                                   render_kw={"placeholder": "e.g., 98"})
    
    systolic_bp = IntegerField('Systolic Blood Pressure (mmHg)', 
                              validators=[Optional(), NumberRange(min=60, max=250)],
                              render_kw={"placeholder": "e.g., 120"})
    
    diastolic_bp = IntegerField('Diastolic Blood Pressure (mmHg)', 
                               validators=[Optional(), NumberRange(min=40, max=150)],
                               render_kw={"placeholder": "e.g., 80"})
    
    stress_level = IntegerField('Stress Level (1-10)', 
                               validators=[Optional(), NumberRange(min=1, max=10)],
                               render_kw={"placeholder": "e.g., 3"})
    
    body_temperature = FloatField('Body Temperature (°C)', 
                                 validators=[Optional(), NumberRange(min=35, max=42)],
                                 render_kw={"placeholder": "e.g., 36.8"})
    
    # NUTRITION METRICS - Daily intake tracking
    calories_consumed = IntegerField('Calories Consumed', 
                                    validators=[Optional(), NumberRange(min=0, max=10000)],
                                    render_kw={"placeholder": "e.g., 2000"})
    
    protein_grams = FloatField('Protein (grams)', 
                              validators=[Optional(), NumberRange(min=0, max=500)],
                              render_kw={"placeholder": "e.g., 80.5"})
    
    carbs_grams = FloatField('Carbohydrates (grams)', 
                            validators=[Optional(), NumberRange(min=0, max=1000)],
                            render_kw={"placeholder": "e.g., 250.0"})
    
    fat_grams = FloatField('Fat (grams)', 
                          validators=[Optional(), NumberRange(min=0, max=300)],
                          render_kw={"placeholder": "e.g., 65.5"})
    
    fiber_grams = FloatField('Fiber (grams)', 
                            validators=[Optional(), NumberRange(min=0, max=100)],
                            render_kw={"placeholder": "e.g., 25.0"})
    
    # BODY COMPOSITION METRICS - Smart scale metrics
    weight_kg = FloatField('Weight (kg)', 
                          validators=[Optional(), NumberRange(min=30, max=300)],
                          render_kw={"placeholder": "e.g., 70.5"})
    
    body_fat_percentage = FloatField('Body Fat (%)', 
                                   validators=[Optional(), NumberRange(min=5, max=60)],
                                   render_kw={"placeholder": "e.g., 18.5"})
    
    muscle_mass_kg = FloatField('Muscle Mass (kg)', 
                               validators=[Optional(), NumberRange(min=20, max=100)],
                               render_kw={"placeholder": "e.g., 45.2"})
    
    bmi = FloatField('BMI', 
                    validators=[Optional(), NumberRange(min=10, max=50)],
                    render_kw={"placeholder": "e.g., 22.5"})
    
    # LIFESTYLE METRICS - Daily wellness tracking
    water_intake_liters = FloatField('Water Intake (liters)', 
                                    validators=[Optional(), NumberRange(min=0, max=10)],
                                    render_kw={"placeholder": "e.g., 2.5"})
    
    mood_score = SelectField('Mood (1-10)', 
                            choices=[(None, 'Select...'),
                                    (1, '1 - Very Poor'), (2, '2 - Poor'),
                                    (3, '3 - Below Average'), (4, '4 - Slightly Below Average'),
                                    (5, '5 - Average'), (6, '6 - Slightly Above Average'),
                                    (7, '7 - Good'), (8, '8 - Very Good'),
                                    (9, '9 - Excellent'), (10, '10 - Outstanding')],
                            coerce=lambda x: int(x) if x else None,
                            validators=[Optional()])
    
    energy_level = SelectField('Energy Level (1-10)', 
                              choices=[(None, 'Select...'),
                                      (1, '1 - Very Low'), (2, '2 - Low'),
                                      (3, '3 - Below Average'), (4, '4 - Slightly Below Average'),
                                      (5, '5 - Average'), (6, '6 - Slightly Above Average'),
                                      (7, '7 - Good'), (8, '8 - Very Good'),
                                      (9, '9 - Excellent'), (10, '10 - Outstanding')],
                              coerce=lambda x: int(x) if x else None,
                              validators=[Optional()])
    
    meditation_minutes = IntegerField('Meditation (minutes)', 
                                     validators=[Optional(), NumberRange(min=0, max=300)],
                                     render_kw={"placeholder": "e.g., 20"})
    
    screen_time_hours = FloatField('Screen Time (hours)', 
                                  validators=[Optional(), NumberRange(min=0, max=24)],
                                  render_kw={"placeholder": "e.g., 6.5"})
    
    social_interactions = IntegerField('Social Interactions', 
                                      validators=[Optional(), NumberRange(min=0, max=50)],
                                      render_kw={"placeholder": "e.g., 5"})
    
    # ADDITIONAL NOTES
    notes = TextAreaField('Notes', 
                         validators=[Optional(), Length(max=1000)],
                         render_kw={"placeholder": "Optional notes about this health data entry", "rows": 3})
    
    # FORM SUBMISSION BUTTON
    submit = SubmitField('Update Health Data')

# CALENDAR EVENT FORM - Form for creating and editing calendar events
class CalendarEventForm(FlaskForm):
    """
    Form for creating and editing calendar events
    Implements research requirement for calendar management with AI optimization controls
    """
    
    # EVENT BASIC INFORMATION
    title = StringField('Event Title', validators=[
        DataRequired(message="Event title is required"),
        Length(min=1, max=200, message="Title must be between 1 and 200 characters")
    ])
    
    description = TextAreaField('Description', validators=[
        Optional(),
        Length(max=1000, message="Description cannot exceed 1000 characters")
    ])
    
    # EVENT TIMING
    start_time = DateTimeField('Start Time', validators=[
        DataRequired(message="Start time is required")
    ], format='%Y-%m-%d %H:%M')
    
    end_time = DateTimeField('End Time', validators=[
        DataRequired(message="End time is required")
    ], format='%Y-%m-%d %H:%M')
    
    # AI OPTIMIZATION CONTROLS - Based on research requirements
    is_ai_modifiable = BooleanField('Allow AI to reschedule this event', default=True)
    is_fixed_time = BooleanField('This is a fixed-time event (work, appointment, etc.)', default=False)
    
    priority_level = SelectField('Priority Level', choices=[
        (1, 'Very Low'),
        (2, 'Low'),
        (3, 'Normal'),
        (4, 'High'),
        (5, 'Very High')
    ], coerce=int, default=3)
    
    event_type = SelectField('Event Type', choices=[
        ('work', 'Work'),
        ('exercise', 'Exercise'),
        ('meal', 'Meal'),
        ('health', 'Health/Medical'),
        ('personal', 'Personal'),
        ('sleep', 'Sleep/Rest')
    ], default='personal')
    
    submit = SubmitField('Save Event')

# USER PREFERENCES FORM - Form for app and AI behavior settings
class UserPreferencesForm(FlaskForm):
    """
    Form for user preferences and AI behavior settings
    Implements research emphasis on user control and personalization
    """
    
    # AI BEHAVIOR SETTINGS
    ai_optimization_enabled = BooleanField('Enable AI schedule optimization', default=True)
    reminder_frequency = SelectField('Reminder Frequency', choices=[
        ('low', 'Low (few reminders)'),
        ('normal', 'Normal'),
        ('high', 'High (frequent reminders)')
    ], default='normal')
    
    # HEALTH GOALS - Personalized targets
    daily_water_goal = IntegerField('Daily Water Goal (glasses)', validators=[
        NumberRange(min=1, max=20, message="Water goal must be between 1 and 20 glasses")
    ], default=8)
    
    daily_sleep_goal = FloatField('Daily Sleep Goal (hours)', validators=[
        NumberRange(min=4, max=12, message="Sleep goal must be between 4 and 12 hours")
    ], default=8.0)
    
    daily_steps_goal = IntegerField('Daily Steps Goal', validators=[
        NumberRange(min=1000, max=50000, message="Steps goal must be between 1,000 and 50,000")
    ], default=10000)
    
    daily_activity_goal = IntegerField('Daily Active Minutes Goal', validators=[
        NumberRange(min=10, max=300, message="Activity goal must be between 10 and 300 minutes")
    ], default=30)
    
    # NOTIFICATION PREFERENCES
    reminder_water = BooleanField('Water intake reminders', default=True)
    reminder_exercise = BooleanField('Exercise reminders', default=True)
    reminder_sleep = BooleanField('Sleep reminders', default=True)
    reminder_medication = BooleanField('Medication reminders', default=False)
    reminder_meal = BooleanField('Meal reminders', default=False)
    smart_reminders_enabled = BooleanField('Enable smart adaptive reminders', default=True)
    
    submit = SubmitField('Save Preferences')
