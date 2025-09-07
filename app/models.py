"""
Database Models for Health Assistant Application

This file defines the structure of our database tables using SQLAlchemy ORM.
Each class represents a table, and each attribute represents a column.
These models define how user accounts and health data are stored in the database.
"""

# DATE AND TIME HANDLING - Import for timestamp functionality
from datetime import datetime
"""
'from datetime import datetime' - Import datetime class for handling dates and times
'datetime' - Python module for working with dates and times
'datetime' - Specific class within the module for timestamp operations
Used for automatically setting creation and update timestamps on database records
"""

# DATABASE ORM IMPORTS - SQLAlchemy for database operations (already explained in __init__.py)
from flask_sqlalchemy import SQLAlchemy

# USER AUTHENTICATION IMPORTS - Flask-Login for user session management
from flask_login import UserMixin
"""
'from flask_login import UserMixin' - Import user authentication mixin class
'flask_login' - Flask extension for handling user authentication
'UserMixin' - Mixin class that adds required methods for user authentication
Provides methods like is_authenticated(), is_active(), get_id() for login functionality
"""

# PASSWORD SECURITY IMPORTS - Werkzeug for secure password handling
from werkzeug.security import generate_password_hash, check_password_hash
"""
'from werkzeug.security import generate_password_hash, check_password_hash' - Password utilities
'werkzeug.security' - Security utilities from Werkzeug (Flask's core library)
'generate_password_hash' - Function to securely hash passwords before storage
'check_password_hash' - Function to verify passwords against stored hashes
These functions use secure hashing algorithms to protect user passwords
"""

# APPLICATION IMPORTS - Database and login manager from our app
from app import db, login
"""
'from app import db, login' - Import database and login manager from our app package
'db' - SQLAlchemy database instance for creating models and queries
'login' - LoginManager instance for user authentication functionality
These were created in app/__init__.py and are shared across the application
"""

# USER MODEL - Database table for user accounts and authentication
class User(UserMixin, db.Model):
    """
    'class User(UserMixin, db.Model):' - Define User model class
    'class' - Python keyword to define a new class
    'User' - Class name representing user accounts (capitalized by convention)
    'UserMixin' - Inherits authentication methods from Flask-Login
    'db.Model' - Inherits database functionality from SQLAlchemy
    'UserMixin, db.Model' - Multiple inheritance (gets features from both classes)
    This class becomes the 'users' table in the database
    """
    
    # PRIMARY KEY - Unique identifier for each user record
    id = db.Column(db.Integer, primary_key=True)
    """
    'id = db.Column(db.Integer, primary_key=True)' - Define primary key column
    'id' - Column name in the database table
    'db.Column()' - SQLAlchemy method to define a database column
    'db.Integer' - Data type for whole numbers (1, 2, 3, etc.)
    'primary_key=True' - Makes this column the unique identifier for each row
    SQLite automatically generates sequential values (1, 2, 3...) for primary keys
    """
    
    # USERNAME COLUMN - Unique username for login
    username = db.Column(db.String(80), index=True, unique=True, nullable=False)
    """
    'username = db.Column(db.String(80), index=True, unique=True, nullable=False)' - Username field
    'username' - Column name for storing user's chosen username
    'db.String(80)' - Text field with maximum 80 characters
    'index=True' - Creates database index for faster username searches
    'unique=True' - Ensures no two users can have the same username
    'nullable=False' - This field is required (cannot be empty)
    """
    
    # EMAIL COLUMN - Unique email address for contact and login
    email = db.Column(db.String(120), index=True, unique=True, nullable=False)
    """
    'email = db.Column(db.String(120), index=True, unique=True, nullable=False)' - Email field
    'email' - Column name for user's email address
    'db.String(120)' - Text field with maximum 120 characters (enough for email addresses)
    'index=True' - Database index for fast email lookups during login
    'unique=True' - Each user must have a different email address
    'nullable=False' - Email is required for account creation
    """
    
    # PASSWORD STORAGE - Encrypted password hash (never store plain passwords)
    password_hash = db.Column(db.String(255))
    """
    'password_hash = db.Column(db.String(255))' - Encrypted password storage
    'password_hash' - Column name (not 'password' because it's encrypted)
    'db.String(255)' - Text field with 255 characters (enough for password hashes)
    No 'nullable=False' because password is set after user creation
    Stores encrypted version of password, never the actual password text
    """
    
    # ACCOUNT CREATION TIMESTAMP - When user account was created
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    """
    'created_at = db.Column(db.DateTime, default=datetime.utcnow)' - Account creation timestamp
    'created_at' - Column name for account creation time
    'db.DateTime' - Data type for date and time values
    'default=datetime.utcnow' - Automatically set to current time when record is created
    'datetime.utcnow' - Function reference (no parentheses) called when needed
    Records when each user account was first created
    """
    
    # DATABASE RELATIONSHIPS - Define connections to other tables
    health_data = db.relationship('HealthData', backref='user', lazy='dynamic')
    """
    'health_data = db.relationship('HealthData', backref='user', lazy='dynamic')' - One-to-many relationship
    'health_data' - Attribute name to access user's health data records
    'db.relationship()' - SQLAlchemy method to define table relationships
    'HealthData' - Name of related model class (as string)
    'backref='user'' - Creates reverse relationship (health_data.user)
    'lazy='dynamic'' - Loads related data only when explicitly requested (performance optimization)
    This allows: user.health_data.all() to get all health records for a user
    """
    
    # NEW RELATIONSHIPS - Calendar events, preferences, and AI recommendations
    calendar_events = db.relationship('CalendarEvent', backref='user', lazy='dynamic')
    preferences = db.relationship('UserPreferences', backref='user', uselist=False)
    ai_recommendations = db.relationship('AIRecommendation', backref='user', lazy='dynamic')
    personalized_advice = db.relationship('PersonalizedHealthAdvice', backref='user', uselist=False)
    """
    Additional relationships for calendar integration and AI features:
    - calendar_events: User's calendar events for AI optimization
    - preferences: User's app and AI behavior settings (one-to-one relationship)
    - ai_recommendations: AI-generated suggestions for the user
    - personalized_advice: User's current personalized health advice (one-to-one relationship)
    """
    
    # USER METHODS - Functions that belong to the User model
    def set_password(self, password):
        """
        'def set_password(self, password):' - Method to securely store user password
        'def' - Python keyword to define a method/function
        'set_password' - Method name describing what it does
        'self' - Reference to the specific user instance
        'password' - Parameter containing the plain text password
        This method encrypts the password before storing it
        """
        self.password_hash = generate_password_hash(password)
        """
        'self.password_hash = generate_password_hash(password)' - Encrypt and store password
        'self.password_hash' - Set the password_hash attribute of this user
        'generate_password_hash(password)' - Function that encrypts the plain password
        'password' - The plain text password from the parameter
        This replaces the plain password with a secure hash that can't be reversed
        """
    
    def check_password(self, password):
        """
        'def check_password(self, password):' - Method to verify user's password
        'check_password' - Method name for password verification
        'password' - Plain text password to check against stored hash
        Returns True if password matches, False if it doesn't
        """
        return check_password_hash(self.password_hash, password)
        """
        'return check_password_hash(self.password_hash, password)' - Verify password
        'return' - Send back the result of the password check
        'check_password_hash()' - Function that compares password against hash
        'self.password_hash' - The stored encrypted password for this user
        'password' - The plain text password to verify
        Returns True/False indicating whether password is correct
        """
    
    def __repr__(self):
        """
        'def __repr__(self):' - Define string representation of User objects
        '__repr__' - Special Python method for object representation
        'self' - Reference to the specific user instance
        This method defines what gets printed when you print a User object
        """
        return f'<User {self.username}>'
        """
        'return f'<User {self.username}>'' - Return formatted string representation
        'f' - Python f-string for formatted string literals
        '<User {self.username}>' - Template with user's username inserted
        'self.username' - Access the username attribute of this user
        Results in strings like '<User john>' or '<User mary>'
        """

# USER LOADER FUNCTION - Required by Flask-Login to reload users from sessions
@login.user_loader
def load_user(id):
    """
    Function to load user by ID for Flask-Login
    
    '@login.user_loader' - Decorator that registers this function with Flask-Login
    '@' - Python decorator syntax for applying functions to other functions
    'login' - Our LoginManager instance from app/__init__.py
    'user_loader' - Decorator that marks this as the user loading function
    Flask-Login calls this function to reload user objects from session data
    
    'def load_user(id):' - Function to load user by ID for Flask-Login
    'load_user' - Function name (required by Flask-Login)
    'id' - User ID parameter from the session
    Flask-Login calls this function to get user objects from stored user IDs
    """
    return User.query.get(int(id))
    """
    'return User.query.get(int(id))' - Load user from database by ID
    'User.query' - SQLAlchemy query object for the User table
    'get()' - SQLAlchemy method to find record by primary key
    'int(id)' - Convert ID to integer (session stores it as string)
    Returns User object if found, None if not found
    """

# HEALTH DATA MODEL - Database table for storing health metrics
class HealthData(db.Model):
    """
    'class HealthData(db.Model):' - Define HealthData model class
    'HealthData' - Class name for health data records
    'db.Model' - Inherits SQLAlchemy database functionality
    This class becomes the 'health_data' table in the database
    """
    
    # PRIMARY KEY - Unique identifier for each health data record
    id = db.Column(db.Integer, primary_key=True)
    """
    'id = db.Column(db.Integer, primary_key=True)' - Primary key (same pattern as User.id)
    Each health data entry gets a unique ID number for identification
    """
    # FOREIGN KEY - Links health data to specific user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    """
    'user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)' - Foreign key
    'user_id' - Column name storing which user this health data belongs to
    'db.ForeignKey('user.id')' - Links to the id column in the users table
    'user.id' - References the User model's id column
    'nullable=False' - Every health record must belong to a user
    This creates the relationship between users and their health data
    """

    # DATE TRACKING - When this health data was recorded
    date_logged = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)
    """
    'date_logged = db.Column(db.Date, nullable=False, default=datetime.utcnow().date)' - Date field
    'date_logged' - Column name for the date this data was recorded
    'db.Date' - Data type for dates only (no time component)
    'nullable=False' - Date is required for every health record
    'default=datetime.utcnow().date' - Automatically set to today's date
    'datetime.utcnow().date' - Get current date (parentheses execute the function)
    """

    # ESSENTIAL HEALTH METRICS - Core metrics available from all major wearable devices
    
    # ACTIVITY METRICS - Movement and exercise tracking
    steps = db.Column(db.Integer)                    # Daily step count (0-50,000)
    distance_km = db.Column(db.Float)                # Distance traveled in kilometers (0-100)
    calories_total = db.Column(db.Integer)           # Total calories burned (0-8,000)
    active_minutes = db.Column(db.Integer)           # Minutes of active movement (0-1440)
    floors_climbed = db.Column(db.Integer)           # Floors/flights climbed (0-200)
    
    # HEART RATE METRICS - Cardiovascular health indicators  
    heart_rate_avg = db.Column(db.Integer)           # Average heart rate BPM (40-220)
    heart_rate_resting = db.Column(db.Integer)       # Resting heart rate BPM (40-100)
    heart_rate_max = db.Column(db.Integer)           # Maximum heart rate BPM (60-220)
    heart_rate_variability = db.Column(db.Float)     # HRV in milliseconds (10-100)
    
    # SLEEP METRICS - Sleep quality and duration tracking
    sleep_duration_hours = db.Column(db.Float)       # Total sleep time in hours (0-24)
    sleep_quality_score = db.Column(db.Integer)      # Sleep quality rating (0-100)
    sleep_deep_minutes = db.Column(db.Integer)       # Deep sleep in minutes (0-600)
    sleep_light_minutes = db.Column(db.Integer)      # Light sleep in minutes (0-600)
    sleep_rem_minutes = db.Column(db.Integer)        # REM sleep in minutes (0-300)
    sleep_awake_minutes = db.Column(db.Integer)      # Time awake in minutes (0-300)
    
    # ADVANCED HEALTH METRICS - Available on premium devices
    blood_oxygen_percent = db.Column(db.Integer)     # SpO2 percentage (70-100)
    stress_level = db.Column(db.Integer)             # Stress score (0-100)
    body_temperature = db.Column(db.Float)           # Body temperature in Celsius (35-42)
    
    # BODY COMPOSITION METRICS - For devices with smart scales
    weight_kg = db.Column(db.Float)                  # Body weight in kilograms (30-300)
    body_fat_percent = db.Column(db.Float)           # Body fat percentage (5-60)
    muscle_mass_kg = db.Column(db.Float)             # Muscle mass in kilograms (20-100)
    
    # HYDRATION AND NUTRITION - Lifestyle tracking
    water_intake_liters = db.Column(db.Float)        # Water intake in liters (0-10)
    
    # SUBJECTIVE METRICS - User-reported wellness indicators
    mood_score = db.Column(db.Integer)               # Mood rating (1-10)
    energy_level = db.Column(db.Integer)             # Energy level (1-10)
    
    # EXERCISE SESSION DETAILS - Workout-specific metrics
    workout_type = db.Column(db.String(50))          # Type of exercise (cardio, strength, etc.)
    workout_duration_minutes = db.Column(db.Integer)  # Exercise session length (0-480)
    workout_intensity = db.Column(db.String(20))     # Intensity level (low, moderate, high)
    workout_calories = db.Column(db.Integer)         # Calories burned during workout (0-2000)

    # FILE-BASED HEALTH DATA PROCESSING - Enhanced for multi-device support
    data_source = db.Column(db.String(100), default='manual')  # 'manual', 'file_upload', 'api'
    processed_data = db.Column(db.Text)       # JSON data from Gemini 2.5 Flash processing
    extraction_date = db.Column(db.DateTime)  # When data was extracted from files
    health_score = db.Column(db.Float)        # Overall health score from AI analysis
    device_type = db.Column(db.String(100))   # Samsung Health, Apple Health, Fitbit, etc.
    """
    Enhanced health data model for file-based processing:
    - data_source: How the data was collected (manual entry, file upload, API)
    - processed_data: Complete JSON output from Gemini 2.5 Flash analysis
    - extraction_date: When files were processed by AI
    - health_score: Calculated health score from comprehensive data
    - device_type: Which wearable device/platform the data came from
    """

    # RECORD TIMESTAMPS - Track when records are created and updated
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    """
    'created_at = db.Column(db.DateTime, default=datetime.utcnow)' - Creation timestamp
    Same pattern as User.created_at - automatically set when record is created
    """
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    """
    'updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)' - Update timestamp
    'updated_at' - Column name for last modification time
    'default=datetime.utcnow' - Set to current time when record is created
    'onupdate=datetime.utcnow' - Automatically update to current time when record is modified
    This tracks both creation and any subsequent modifications
    """

    # STRING REPRESENTATION - Define how HealthData objects appear when printed
    def __repr__(self):
        """
        'def __repr__(self):' - String representation method (same pattern as User.__repr__)
        """
        return f'<HealthData {self.user_id} - {self.date_logged}>'
        """
        'return f'<HealthData {self.user_id} - {self.date_logged}>'' - Formatted string
        'f' - F-string for formatted literals
        '{self.user_id}' - Insert the user ID this health data belongs to
        '{self.date_logged}' - Insert the date when this data was logged
        Results in strings like '<HealthData 1 - 2025-08-05>'
        """
    
    @property 
    def insights(self):
        """Parse processed_data JSON for insights"""
        if self.processed_data:
            import json
            try:
                data = json.loads(self.processed_data)
                return data.get('insights', {})
            except:
                return {}
        return {}
    
    @property
    def summary(self):
        """Parse processed_data JSON for summary"""
        if self.processed_data:
            import json
            try:
                data = json.loads(self.processed_data)
                return data.get('summary', {})
            except:
                return {}
        return {}

# CALENDAR EVENT MODEL - Database table for storing user calendar events
class CalendarEvent(db.Model):
    """
    Calendar Event model for storing user events that can be optimized by AI
    Based on research requirement for calendar integration with AI optimization
    """
    
    # PRIMARY KEY - Unique identifier for each calendar event
    id = db.Column(db.Integer, primary_key=True)
    
    # FOREIGN KEY - Links calendar events to specific user
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # EVENT DETAILS - Core event information
    title = db.Column(db.String(200), nullable=False)  # Event title/name
    description = db.Column(db.Text)  # Optional event description
    
    # EVENT TIMING - When the event occurs
    start_time = db.Column(db.DateTime, nullable=False)  # Event start time
    end_time = db.Column(db.DateTime, nullable=False)    # Event end time
    
    # AI OPTIMIZATION SETTINGS - User control over AI modifications
    is_ai_modifiable = db.Column(db.Boolean, default=True)  # Can AI move this event?
    is_fixed_time = db.Column(db.Boolean, default=False)   # Is this a fixed time event (work, etc.)?
    priority_level = db.Column(db.Integer, default=3)      # Priority 1-5 (5 = highest)
    
    # EVENT CATEGORIZATION - Type of event for AI decision making
    event_type = db.Column(db.String(50), default='personal')  # work, exercise, meal, sleep, personal
    
    # GOOGLE CALENDAR INTEGRATION - External calendar sync
    google_calendar_id = db.Column(db.String(255))  # Google Calendar event ID for sync
    
    # RECORD TIMESTAMPS
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert CalendarEvent to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'title': self.title,
            'description': self.description,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'event_type': self.event_type,
            'priority_level': self.priority_level,
            'is_ai_modifiable': self.is_ai_modifiable,
            'is_fixed_time': self.is_fixed_time,
            'google_calendar_id': self.google_calendar_id,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
    
    def __repr__(self):
        return f'<CalendarEvent {self.title} - {self.start_time.strftime("%Y-%m-%d %H:%M")}>'

# USER PREFERENCES MODEL - AI behavior and app settings
class UserPreferences(db.Model):
    """
    User preferences for AI behavior and app customization
    Based on research emphasis on user control and personalization
    """
    
    # PRIMARY KEY AND FOREIGN KEY
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    # AI BEHAVIOR SETTINGS
    ai_optimization_enabled = db.Column(db.Boolean, default=True)  # Enable AI schedule optimization
    reminder_frequency = db.Column(db.String(20), default='normal')  # low, normal, high
    
    # HEALTH GOALS AND TARGETS - Personalized health targets
    daily_water_goal = db.Column(db.Integer, default=8)      # Glasses of water
    daily_sleep_goal = db.Column(db.Float, default=8.0)     # Hours of sleep
    daily_steps_goal = db.Column(db.Integer, default=10000) # Steps per day
    daily_activity_goal = db.Column(db.Integer, default=30) # Active minutes
    
    # NOTIFICATION PREFERENCES - When and how to remind users
    reminder_water = db.Column(db.Boolean, default=True)
    reminder_exercise = db.Column(db.Boolean, default=True)
    reminder_sleep = db.Column(db.Boolean, default=True)
    reminder_medication = db.Column(db.Boolean, default=False)
    reminder_meal = db.Column(db.Boolean, default=False)
    smart_reminders_enabled = db.Column(db.Boolean, default=True)  # Enable adaptive reminders
    
    # QUIET HOURS - When not to send reminders
    quiet_hours_start = db.Column(db.Time, default=datetime.strptime('22:00', '%H:%M').time())
    quiet_hours_end = db.Column(db.Time, default=datetime.strptime('08:00', '%H:%M').time())
    
    # INTEGRATION SETTINGS
    google_calendar_connected = db.Column(db.Boolean, default=False)
    health_data_uploaded = db.Column(db.Boolean, default=False)  # Has user uploaded health data files
    
    # RECORD TIMESTAMPS
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<UserPreferences for User {self.user_id}>'

# AI RECOMMENDATIONS MODEL - Store AI-generated suggestions
class AIRecommendation(db.Model):
    """
    Store AI-generated health recommendations and schedule optimizations
    Tracks what AI suggests and user acceptance for learning
    """
    
    # PRIMARY KEY AND FOREIGN KEY
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # RECOMMENDATION DETAILS
    recommendation_type = db.Column(db.String(50), nullable=False)  # schedule, health, reminder
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    
    # AI CONFIDENCE AND USER RESPONSE
    ai_confidence = db.Column(db.Float, default=0.8)  # AI confidence in recommendation (0-1)
    user_accepted = db.Column(db.Boolean)  # Did user accept the recommendation?
    user_feedback = db.Column(db.Text)     # Optional user feedback
    
    # IMPLEMENTATION STATUS
    is_implemented = db.Column(db.Boolean, default=False)
    implementation_date = db.Column(db.DateTime)
    
    # RECORD TIMESTAMPS
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<AIRecommendation {self.title} for User {self.user_id}>'

# PERSONALIZED HEALTH ADVICE MODEL - Store persistent health advice
class PersonalizedHealthAdvice(db.Model):
    """
    Store AI-generated personalized health advice that persists across sessions
    Each user has one current advice that stays until manually refreshed
    """
    
    # PRIMARY KEY AND FOREIGN KEY
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    # ADVICE CONTENT (stored as JSON)
    insights = db.Column(db.Text)  # JSON array of insights
    recommendations = db.Column(db.Text)  # JSON array of recommendations
    quick_wins = db.Column(db.Text)  # JSON array of quick wins
    concerns = db.Column(db.Text)  # JSON array of concerns
    motivation = db.Column(db.Text)  # Motivation message
    
    # METADATA
    source = db.Column(db.String(50), default='gemini_ai')  # Source of advice (gemini_ai, rule_based, etc.)
    health_score_at_generation = db.Column(db.Float)  # Health score when advice was generated
    
    # TIMESTAMPS
    generated_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert advice to dictionary format for frontend"""
        import json
        
        try:
            return {
                'insights': json.loads(self.insights) if self.insights else [],
                'recommendations': json.loads(self.recommendations) if self.recommendations else [],
                'quick_wins': json.loads(self.quick_wins) if self.quick_wins else [],
                'concerns': json.loads(self.concerns) if self.concerns else [],
                'motivation': self.motivation or '',
                'source': self.source,
                'generated_at': self.generated_at.isoformat() if self.generated_at else None,
                'health_score': self.health_score_at_generation
            }
        except json.JSONDecodeError:
            # Return empty structure if JSON is corrupted
            return {
                'insights': [],
                'recommendations': [],
                'quick_wins': [],
                'concerns': [],
                'motivation': '',
                'source': self.source,
                'generated_at': self.generated_at.isoformat() if self.generated_at else None,
                'health_score': self.health_score_at_generation
            }
    
    @staticmethod
    def from_dict(user_id, advice_dict, health_score=None):
        """Create advice from dictionary format"""
        import json
        
        return PersonalizedHealthAdvice(
            user_id=user_id,
            insights=json.dumps(advice_dict.get('insights', [])),
            recommendations=json.dumps(advice_dict.get('recommendations', [])),
            quick_wins=json.dumps(advice_dict.get('quick_wins', [])),
            concerns=json.dumps(advice_dict.get('concerns', [])),
            motivation=advice_dict.get('motivation', ''),
            source=advice_dict.get('source', 'gemini_ai'),
            health_score_at_generation=health_score
        )
    
    def __repr__(self):
        return f'<PersonalizedHealthAdvice for User {self.user_id}>'
