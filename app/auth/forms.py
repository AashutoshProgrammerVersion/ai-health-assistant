"""
Authentication Forms for User Registration and Login

This file defines WTForms classes for user authentication functionality.
Forms handle user input validation, security, and HTML form generation.
These forms ensure user data is properly validated before being processed.
"""

# FLASK-WTF IMPORT - Flask extension for secure form handling
from flask_wtf import FlaskForm
"""
'from flask_wtf import FlaskForm' - Import Flask-WTF form base class
'flask_wtf' - Flask extension that adds security features to WTForms
'FlaskForm' - Base class for all forms with CSRF protection built-in
CSRF (Cross-Site Request Forgery) protection prevents malicious form submissions
"""

# WTFORMS FIELD IMPORTS - Different types of form input fields
from wtforms import StringField, PasswordField, BooleanField, SubmitField
"""
'from wtforms import StringField, PasswordField, BooleanField, SubmitField' - Form field types
'wtforms' - Python library for form creation and validation
'StringField' - Text input field for usernames, emails, etc.
'PasswordField' - Password input field (hides text as user types)
'BooleanField' - Checkbox field for true/false options
'SubmitField' - Button field for form submission
"""

# WTFORMS VALIDATOR IMPORTS - Functions that validate form input
from wtforms.validators import ValidationError, DataRequired, Email, EqualTo, Length
"""
'from wtforms.validators import ...' - Import validation functions
'ValidationError' - Exception class for custom validation errors
'DataRequired' - Validator ensuring field is not empty
'Email' - Validator ensuring field contains valid email format
'EqualTo' - Validator ensuring two fields have the same value
'Length' - Validator ensuring field meets minimum/maximum length requirements
"""

# MODEL IMPORT - User model for database uniqueness validation
from app.models import User
"""
'from app.models import User' - Import User model for validation
'User' - Our User database model class
Used to check if usernames and emails are already taken during registration
"""

# LOGIN FORM - Form for existing users to sign in
class LoginForm(FlaskForm):
    """
    'class LoginForm(FlaskForm):' - Define login form class
    'LoginForm' - Class name for the login form
    'FlaskForm' - Parent class providing form functionality and CSRF protection
    This form handles user authentication with username and password
    """
    
    # USERNAME FIELD - Text input for username
    username = StringField('Username', validators=[DataRequired()])
    """
    'username = StringField('Username', validators=[DataRequired()])' - Username input field
    'username' - Field attribute name (used in templates and processing)
    'StringField()' - Creates a text input field
    'Username' - Field label displayed to users
    'validators=[DataRequired()]' - List of validation rules
    'DataRequired()' - Ensures this field is not empty when submitted
    """
    
    # PASSWORD FIELD - Hidden input for password
    password = PasswordField('Password', validators=[DataRequired()])
    """
    'password = PasswordField('Password', validators=[DataRequired()])' - Password input field
    'password' - Field attribute name
    'PasswordField()' - Creates password input (hides characters as user types)
    'Password' - Field label
    'validators=[DataRequired()]' - Password is required for login
    """
    
    # REMEMBER ME CHECKBOX - Optional persistent login
    remember_me = BooleanField('Remember Me')
    """
    'remember_me = BooleanField('Remember Me')' - Checkbox for persistent login
    'remember_me' - Field attribute name
    'BooleanField()' - Creates checkbox input (True when checked, False when unchecked)
    'Remember Me' - Field label
    No validators needed since checkbox is optional
    """
    
    # SUBMIT BUTTON - Button to send form data
    submit = SubmitField('Sign In')
    """
    'submit = SubmitField('Sign In')' - Form submission button
    'submit' - Field attribute name
    'SubmitField()' - Creates form submit button
    'Sign In' - Button text displayed to users
    """

# REGISTRATION FORM - Form for new users to create accounts
class RegistrationForm(FlaskForm):
    """
    'class RegistrationForm(FlaskForm):' - Define registration form class
    'RegistrationForm' - Class name for user registration
    Inherits from FlaskForm for security and validation features
    """
    
    # USERNAME FIELD - Text input with length validation
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    """
    'username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])' - Username field
    'StringField()' - Text input field type
    'validators=[DataRequired(), Length(min=4, max=20)]' - Multiple validation rules
    'DataRequired()' - Field cannot be empty
    'Length(min=4, max=20)' - Username must be 4-20 characters long
    """
    
    # EMAIL FIELD - Text input with email validation
    email = StringField('Email', validators=[DataRequired(), Email()])
    """
    'email = StringField('Email', validators=[DataRequired(), Email()])' - Email input field
    'validators=[DataRequired(), Email()]' - Multiple validators
    'Email()' - Validates proper email format (contains @, valid domain, etc.)
    """
    
    # PASSWORD FIELD - Hidden input with length validation
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    """
    'password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])' - Password field
    'PasswordField()' - Hidden text input for passwords
    'Length(min=6)' - Password must be at least 6 characters for basic security
    """
    
    # PASSWORD CONFIRMATION FIELD - Ensures passwords match
    password2 = PasswordField('Repeat Password', 
                             validators=[DataRequired(), EqualTo('password')])
    """
    'password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])' - Password confirmation
    'password2' - Field attribute name (different from 'password')
    'Repeat Password' - Label asking user to type password again
    'EqualTo('password')' - Validates this field matches the 'password' field
    This prevents typos in password entry
    """
    
    # SUBMIT BUTTON - Button to create account
    submit = SubmitField('Register')
    """
    'submit = SubmitField('Register')' - Registration button
    'Register' - Button text for account creation
    """

    # CUSTOM VALIDATION METHODS - Additional validation logic
    def validate_username(self, username):
        """
        'def validate_username(self, username):' - Custom username validation method
        'validate_username' - Method name (WTForms automatically calls validate_<fieldname>)
        'self' - Reference to this form instance
        'username' - The username field being validated
        WTForms automatically calls this method when validating the username field
        """
        user = User.query.filter_by(username=username.data).first()
        """
        'user = User.query.filter_by(username=username.data).first()' - Check if username exists
        'User.query' - SQLAlchemy query object for User table
        'filter_by(username=username.data)' - Find records where username matches input
        'username.data' - The actual value entered in the username field
        'first()' - Get the first matching record, or None if no matches
        """
        if user is not None:
            """
            'if user is not None:' - Check if username was found in database
            'user is not None' - True if a user with this username already exists
            """
            raise ValidationError('Please use a different username.')
            """
            'raise ValidationError('Please use a different username.')' - Raise validation error
            'raise' - Python keyword to throw an exception
            'ValidationError()' - WTForms exception class for validation failures
            'Please use a different username.' - Error message shown to user
            This prevents duplicate usernames in the database
            """

    def validate_email(self, email):
        """
        'def validate_email(self, email):' - Custom email validation method
        'validate_email' - Method name for email validation
        Same pattern as validate_username but for email field
        """
        user = User.query.filter_by(email=email.data).first()
        """
        'user = User.query.filter_by(email=email.data).first()' - Check if email exists
        'filter_by(email=email.data)' - Find records where email matches input
        'email.data' - The actual email address entered in the form
        """
        if user is not None:
            raise ValidationError('Please use a different email address.')
            """
            'raise ValidationError('Please use a different email address.')' - Email uniqueness error
            Ensures each user has a unique email address for account recovery
            """
