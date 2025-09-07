#!/usr/bin/env python3
"""
Calendar Debug Script

This script helps debug Google Calendar API issues by testing the connection
and providing detailed error information.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone, timedelta

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from calendar_service import CalendarService
from flask import Flask

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calendar_debug.log')
    ]
)

def test_calendar_connection():
    """Test Google Calendar connection with detailed logging"""
    
    print("=" * 60)
    print("GOOGLE CALENDAR DEBUG TEST")
    print("=" * 60)
    
    # Create a minimal Flask app for context
    app = Flask(__name__)
    app.config['GOOGLE_CALENDAR_CLIENT_ID'] = os.environ.get('GOOGLE_CALENDAR_CLIENT_ID')
    app.config['GOOGLE_CALENDAR_CLIENT_SECRET'] = os.environ.get('GOOGLE_CALENDAR_CLIENT_SECRET')
    
    with app.app_context():
        # Check environment variables
        print(f"Current time (local): {datetime.now()}")
        print(f"Current time (UTC): {datetime.now(timezone.utc)}")
        print(f"Client ID configured: {'Yes' if app.config.get('GOOGLE_CALENDAR_CLIENT_ID') else 'No'}")
        print(f"Client Secret configured: {'Yes' if app.config.get('GOOGLE_CALENDAR_CLIENT_SECRET') else 'No'}")
        print()
        
        # Initialize calendar service
        try:
            calendar_service = CalendarService()
            print("✓ Calendar service initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize calendar service: {e}")
            return
        
        # Test with sample credentials (you'll need to replace this with real credentials)
        sample_credentials = {
            'access_token': 'test_token',
            'refresh_token': 'test_refresh',
            'token_uri': 'https://oauth2.googleapis.com/token',
            'client_id': app.config.get('GOOGLE_CALENDAR_CLIENT_ID'),
            'client_secret': app.config.get('GOOGLE_CALENDAR_CLIENT_SECRET'),
            'scopes': ['https://www.googleapis.com/auth/calendar']
        }
        
        print("Testing calendar sync...")
        print("Note: This will fail with test credentials, but we can see the error details")
        print()
        
        # Test sync with different times of day
        test_times = [
            ("Morning", 9),
            ("Afternoon", 14), 
            ("Evening", 18),
            ("Night", 22)
        ]
        
        for time_name, hour in test_times:
            print(f"--- Testing {time_name} ({hour}:00) ---")
            
            # Simulate different times by adjusting the test
            test_time = datetime.now(timezone.utc).replace(hour=hour, minute=0, second=0, microsecond=0)
            print(f"Simulating time: {test_time}")
            
            try:
                result = calendar_service.sync_with_google_calendar(
                    user_id=1,
                    credentials_dict=sample_credentials,
                    days_ahead=7
                )
                
                print(f"Result: {result}")
                
            except Exception as e:
                print(f"Error during {time_name} test: {e}")
                print(f"Error type: {type(e).__name__}")
            
            print()

def check_credentials_file():
    """Check if there's a credentials file we can examine"""
    
    print("=" * 60)
    print("CHECKING FOR EXISTING CREDENTIALS")
    print("=" * 60)
    
    possible_locations = [
        'instance/google_credentials.json',
        'google_credentials.json',
        '.env',
        'config.py'
    ]
    
    for location in possible_locations:
        full_path = os.path.join(os.path.dirname(__file__), location)
        if os.path.exists(full_path):
            print(f"✓ Found: {location}")
            if location.endswith('.json'):
                try:
                    with open(full_path, 'r') as f:
                        data = json.load(f)
                        print(f"  Keys in file: {list(data.keys())}")
                except:
                    print(f"  Could not read JSON file")
        else:
            print(f"✗ Not found: {location}")
    
    print()

def main():
    """Main debug function"""
    
    print("Google Calendar Debug Tool")
    print(f"Started at: {datetime.now()}")
    print()
    
    check_credentials_file()
    test_calendar_connection()
    
    print("=" * 60)
    print("DEBUG COMPLETE")
    print("=" * 60)
    print("Check calendar_debug.log for detailed logs")

if __name__ == "__main__":
    main()
