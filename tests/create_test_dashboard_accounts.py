"""
Comprehensive Dashboard Visualization Test Account Creator

This script creates multiple user accounts with different health data scenarios
to test all dashboard visualization components thoroughly.

Test Cases:
1. Optimal Health User - Perfect scores across all metrics
2. Sleep Deprived User - Poor sleep, good activity
3. Sedentary User - Low activity, good sleep
4. Dehydrated User - Poor hydration, otherwise good
5. Heart Health Concern - High resting heart rate, needs monitoring
6. Inconsistent Data User - Sporadic logging, missing data
7. New User - Minimal data (1-2 days)
8. Empty User - No health data logged

Each test case will validate different chart behaviors and edge cases.
"""

import os
import sys
from datetime import datetime, timedelta, date, time

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app, db
from app.models import User, HealthData, UserPreferences, PersonalizedHealthAdvice
from werkzeug.security import generate_password_hash

def create_test_accounts():
    """Create comprehensive test accounts for dashboard visualization testing"""
    
    app = create_app()
    
    with app.app_context():
        print("=" * 80)
        print("DASHBOARD VISUALIZATION TEST ACCOUNT CREATOR")
        print("=" * 80)
        print("\nThis script creates 8 test accounts with different health data scenarios")
        print("to validate all dashboard charts, graphs, and infographics.\n")
        
        # List of test accounts to create
        test_accounts = [
            {
                'username': 'optimal_health',
                'email': 'optimal@test.com',
                'password': 'Test123!',
                'description': 'Perfect health scores across all metrics',
                'generator': create_optimal_health_data
            },
            {
                'username': 'sleep_deprived',
                'email': 'sleepy@test.com',
                'password': 'Test123!',
                'description': 'Poor sleep quality, good activity levels',
                'generator': create_sleep_deprived_data
            },
            {
                'username': 'sedentary_user',
                'email': 'sedentary@test.com',
                'password': 'Test123!',
                'description': 'Low activity, mostly sitting, good sleep',
                'generator': create_sedentary_data
            },
            {
                'username': 'dehydrated_user',
                'email': 'dehydrated@test.com',
                'password': 'Test123!',
                'description': 'Poor hydration habits, otherwise healthy',
                'generator': create_dehydrated_data
            },
            {
                'username': 'heart_concern',
                'email': 'heart@test.com',
                'password': 'Test123!',
                'description': 'Elevated heart rates, needs monitoring',
                'generator': create_heart_concern_data
            },
            {
                'username': 'inconsistent_user',
                'email': 'inconsistent@test.com',
                'password': 'Test123!',
                'description': 'Sporadic data logging with gaps',
                'generator': create_inconsistent_data
            },
            {
                'username': 'new_user',
                'email': 'newuser@test.com',
                'password': 'Test123!',
                'description': 'Just started using app (1-2 days of data)',
                'generator': create_new_user_data
            },
            {
                'username': 'empty_user',
                'email': 'empty@test.com',
                'password': 'Test123!',
                'description': 'No health data logged yet (edge case)',
                'generator': None  # No data generator
            }
        ]
        
        print("CREATING TEST ACCOUNTS:")
        print("-" * 80)
        
        created_accounts = []
        
        for account_info in test_accounts:
            # Check if user already exists
            existing_user = User.query.filter_by(username=account_info['username']).first()
            if existing_user:
                print(f"⚠️  User '{account_info['username']}' already exists - DELETING...")
                # Delete associated data
                HealthData.query.filter_by(user_id=existing_user.id).delete()
                UserPreferences.query.filter_by(user_id=existing_user.id).delete()
                PersonalizedHealthAdvice.query.filter_by(user_id=existing_user.id).delete()
                db.session.delete(existing_user)
                db.session.commit()
                print(f"   Deleted old account and data")
            
            # Create new user
            user = User(
                username=account_info['username'],
                email=account_info['email']
            )
            user.set_password(account_info['password'])
            db.session.add(user)
            db.session.commit()
            
            # Create user preferences
            preferences = UserPreferences(
                user_id=user.id,
                daily_water_goal=8,
                daily_sleep_goal=8.0,
                daily_steps_goal=10000,
                daily_activity_goal=30
            )
            db.session.add(preferences)
            db.session.commit()
            
            # Generate health data if generator provided
            if account_info['generator']:
                days_created = account_info['generator'](user.id)
                print(f"✅ Created: {account_info['username']:20} | {days_created} days of data | {account_info['description']}")
            else:
                print(f"✅ Created: {account_info['username']:20} | 0 days of data  | {account_info['description']}")
            
            created_accounts.append({
                'username': account_info['username'],
                'password': account_info['password'],
                'email': account_info['email'],
                'description': account_info['description']
            })
        
        print("\n" + "=" * 80)
        print("ACCOUNT CREDENTIALS FOR TESTING")
        print("=" * 80)
        print(f"\n{'Username':<20} {'Password':<12} {'Email':<25} {'Description'}")
        print("-" * 80)
        for account in created_accounts:
            print(f"{account['username']:<20} {account['password']:<12} {account['email']:<25} {account['description'][:30]}")
        
        print("\n" + "=" * 80)
        print("TESTING INSTRUCTIONS")
        print("=" * 80)
        print("""
1. Start Flask server: python run.py
2. Open browser to http://127.0.0.1:5000
3. Log in with each account above
4. Take screenshot of dashboard for each account
5. Verify the following visualizations:

   FOR EACH ACCOUNT CHECK:
   ✓ Weekly Activity Chart (steps + calories bars)
   ✓ Sleep Quality Timeline (line chart)
   ✓ Hydration Progress (line chart + circle goal)
   ✓ Heart Health Metrics (multi-line + zone cards)
   ✓ Wellness Correlation Matrix (4 cards)
   ✓ Health Score Radar Chart (6-axis breakdown)
   ✓ Weekly Summary Cards (steps, calories, sleep)
   ✓ Upcoming Events sidebar position (between Quick Actions & System Status)
   ✓ All data displays correctly without errors
   ✓ Charts handle missing/null data gracefully

EXPECTED BEHAVIOR BY ACCOUNT:
- optimal_health:     All charts full, high scores, green indicators
- sleep_deprived:     Sleep chart shows <6 hours, other metrics good
- sedentary_user:     Activity chart low (<3000 steps), sleep normal
- dehydrated_user:    Hydration chart/circle low (<1.5L), other metrics good
- heart_concern:      Heart rate zones show elevated BPM (>90 resting)
- inconsistent_user:  Charts have gaps, some null values
- new_user:           Only 1-2 data points, partial week display
- empty_user:         No visualization section shown, "No data" messages
        """)
        
        print("\n✅ All test accounts created successfully!")
        print(f"   Total accounts: {len(created_accounts)}")
        print(f"   Database: instance/app.db")
        print("\n" + "=" * 80)


# ============================================================================
# DATA GENERATORS FOR EACH TEST CASE
# ============================================================================

def create_optimal_health_data(user_id):
    """Test Case 1: Optimal Health User - Perfect scores across all metrics"""
    days_created = 0
    
    for i in range(7):  # Last 7 days
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Perfect activity metrics
            steps=12000 + (i * 500),  # Well above 10k goal
            distance_km=9.5 + (i * 0.3),
            calories_total=2800 + (i * 100),
            active_minutes=60 + (i * 5),
            floors_climbed=15 + i,
            # Optimal heart health
            heart_rate_avg=75 - i,
            heart_rate_resting=60 + i,
            heart_rate_max=150 + (i * 2),
            heart_rate_variability=55.0 + (i * 2),
            # Perfect sleep
            sleep_duration_hours=8.0 + (i * 0.2),
            sleep_quality_score=90 + i,
            sleep_deep_minutes=120 + (i * 5),
            sleep_light_minutes=240 + (i * 5),
            sleep_rem_minutes=90 + (i * 3),
            sleep_awake_minutes=10 - i if i < 5 else 5,
            # Great hydration
            water_intake_liters=3.0 + (i * 0.2),
            # Good nutrition
            calories_consumed=2200 + (i * 50),
            protein_grams=120 + (i * 5),
            carbs_grams=250 + (i * 10),
            fat_grams=70 + (i * 3),
            fiber_grams=30 + i,
            # Optimal vitals
            blood_oxygen_percent=98 + (i % 2),
            stress_level=20 - (i * 2) if i < 10 else 5,
            body_temperature=36.6 + (i * 0.1),
            # Body composition
            weight_kg=75.0 - (i * 0.1),
            body_fat_percent=18.0 - (i * 0.2),
            muscle_mass_kg=58.0 + (i * 0.1),
            bmi=22.5,
            # Wellness metrics
            mood_score=9 - (i % 2),  # 8-9 range
            energy_level=9,
            meditation_minutes=20 + (i * 2),
            screen_time_hours=4.0 - (i * 0.2),
            social_interactions=8 + i,
            # Exercise
            workout_type='cardio' if i % 2 == 0 else 'strength',
            workout_duration_minutes=45 + (i * 5),
            workout_intensity='high',
            workout_calories=500 + (i * 20),
            # Vitals
            systolic_bp=115 + i,
            diastolic_bp=75 + (i % 3),
            data_source='manual',
            device_type='Samsung Health'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_sleep_deprived_data(user_id):
    """Test Case 2: Sleep Deprived User - Poor sleep, good activity"""
    days_created = 0
    
    for i in range(7):
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Good activity
            steps=11000 + (i * 300),
            distance_km=8.5,
            calories_total=2600,
            active_minutes=50,
            # Poor sleep - MAIN ISSUE
            sleep_duration_hours=5.0 + (i * 0.3),  # Only 5-6 hours!
            sleep_quality_score=45 + (i * 3),  # Low quality
            sleep_deep_minutes=45,  # Not enough deep sleep
            sleep_light_minutes=180,
            sleep_rem_minutes=50,
            sleep_awake_minutes=45 + (i * 5),  # Waking up a lot
            # Affected by poor sleep
            heart_rate_resting=75 + (i * 2),  # Slightly elevated
            heart_rate_avg=85,
            heart_rate_max=165,
            stress_level=65 + (i * 3),  # High stress from lack of sleep
            mood_score=5 - (i % 2),  # Lower mood
            energy_level=4,  # Low energy
            # Other metrics normal
            water_intake_liters=2.5,
            calories_consumed=2000,
            blood_oxygen_percent=97,
            data_source='manual',
            device_type='Apple Watch'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_sedentary_data(user_id):
    """Test Case 3: Sedentary User - Low activity, good sleep"""
    days_created = 0
    
    for i in range(7):
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Very low activity - MAIN ISSUE
            steps=2500 + (i * 200),  # Well below 10k goal
            distance_km=1.8,
            calories_total=1800,  # Low calorie burn
            active_minutes=10 + i,  # Barely moving
            floors_climbed=2,
            # Good sleep
            sleep_duration_hours=8.5,
            sleep_quality_score=85,
            sleep_deep_minutes=110,
            sleep_light_minutes=230,
            sleep_rem_minutes=80,
            # Heart health affected by inactivity
            heart_rate_resting=72,
            heart_rate_avg=78,
            heart_rate_max=130,  # Low max (not exercising)
            # Other metrics
            water_intake_liters=2.0,
            calories_consumed=2100,
            mood_score=6,
            energy_level=5,
            screen_time_hours=9.5,  # Lots of screen time
            social_interactions=3,  # Low social activity
            weight_kg=82.0,  # Slightly overweight
            body_fat_percent=26.0,
            bmi=26.5,
            data_source='manual',
            device_type='Fitbit'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_dehydrated_data(user_id):
    """Test Case 4: Dehydrated User - Poor hydration, otherwise healthy"""
    days_created = 0
    
    for i in range(7):
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Good activity
            steps=10500,
            distance_km=8.0,
            calories_total=2500,
            active_minutes=45,
            # Poor hydration - MAIN ISSUE
            water_intake_liters=1.0 + (i * 0.15),  # Only 1-2 liters, goal is 3L
            # Good sleep
            sleep_duration_hours=7.5,
            sleep_quality_score=80,
            sleep_deep_minutes=100,
            sleep_light_minutes=220,
            sleep_rem_minutes=75,
            # Heart health
            heart_rate_resting=68,
            heart_rate_avg=80,
            heart_rate_max=155,
            # Affected by dehydration
            mood_score=6,  # Slightly lower
            energy_level=6,
            # Other metrics
            calories_consumed=2000,
            blood_oxygen_percent=96,
            data_source='manual',
            device_type='Samsung Health'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_heart_concern_data(user_id):
    """Test Case 5: Heart Health Concern - Elevated heart rates"""
    days_created = 0
    
    for i in range(7):
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Moderate activity
            steps=8500,
            distance_km=6.5,
            calories_total=2300,
            active_minutes=35,
            # Elevated heart rates - MAIN CONCERN
            heart_rate_resting=88 + (i * 2),  # High resting HR (should be 60-70)
            heart_rate_avg=105 + (i * 3),  # Elevated average
            heart_rate_max=175 + (i * 2),  # High max HR
            heart_rate_variability=28.0,  # Low HRV (not good)
            # Blood pressure also high
            systolic_bp=140 + i,  # Pre-hypertension
            diastolic_bp=90 + (i % 4),
            # Other metrics
            sleep_duration_hours=7.0,
            sleep_quality_score=70,
            water_intake_liters=2.3,
            calories_consumed=2100,
            stress_level=70 + (i * 2),  # High stress
            mood_score=5,
            energy_level=5,
            blood_oxygen_percent=95,
            # Body composition
            weight_kg=95.0,
            body_fat_percent=30.0,
            bmi=29.5,  # Overweight
            data_source='manual',
            device_type='Apple Watch'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_inconsistent_data(user_id):
    """Test Case 6: Inconsistent User - Sporadic logging with gaps"""
    days_created = 0
    
    # Only log data for days 0, 1, 3, 6 (leaving gaps on days 2, 4, 5)
    log_days = [0, 1, 3, 6]
    
    for i in log_days:
        day_date = date.today() - timedelta(days=i)
        
        # Vary the completeness of data
        if i == 0:  # Most recent - complete data
            health_data = HealthData(
                user_id=user_id,
                date_logged=day_date,
                steps=9500,
                calories_total=2400,
                sleep_duration_hours=7.5,
                water_intake_liters=2.5,
                heart_rate_resting=70,
                heart_rate_avg=82,
                heart_rate_max=150,
                calories_consumed=2000,
                mood_score=7,
                data_source='manual'
            )
        elif i == 1:  # Missing some metrics
            health_data = HealthData(
                user_id=user_id,
                date_logged=day_date,
                steps=8000,
                sleep_duration_hours=6.5,
                water_intake_liters=2.0,
                # Missing heart rate, calories, etc.
                data_source='manual'
            )
        elif i == 3:  # Only basic metrics
            health_data = HealthData(
                user_id=user_id,
                date_logged=day_date,
                steps=7500,
                water_intake_liters=1.8,
                # Missing most data
                data_source='manual'
            )
        else:  # i == 6 - Minimal data
            health_data = HealthData(
                user_id=user_id,
                date_logged=day_date,
                steps=5000,
                data_source='manual'
            )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


def create_new_user_data(user_id):
    """Test Case 7: New User - Just started (1-2 days of data)"""
    days_created = 0
    
    for i in range(2):  # Only 2 days
        day_date = date.today() - timedelta(days=i)
        
        health_data = HealthData(
            user_id=user_id,
            date_logged=day_date,
            # Basic logging
            steps=9000 + (i * 500),
            distance_km=7.0,
            calories_total=2300,
            sleep_duration_hours=7.0,
            water_intake_liters=2.2,
            heart_rate_resting=68,
            heart_rate_avg=78,
            heart_rate_max=145,
            calories_consumed=1900,
            mood_score=7,
            energy_level=7,
            data_source='manual',
            device_type='Samsung Health'
        )
        
        db.session.add(health_data)
        days_created += 1
    
    db.session.commit()
    return days_created


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    try:
        create_test_accounts()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
