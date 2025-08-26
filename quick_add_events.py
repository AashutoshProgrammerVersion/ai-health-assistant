#!/usr/bin/env python3
"""
Quick Calendar Event Creator

A simple script to quickly add specific calendar events for testing
AI optimization scenarios. Useful for creating test cases or specific
scheduling conflicts.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User, CalendarEvent


def create_test_scenario(user_id, scenario_name):
    """Create specific test scenarios for AI optimization testing"""
    
    scenarios = {
        'conflict': [
            # Create scheduling conflicts to test AI resolution
            {
                'title': 'Important Meeting',
                'description': 'High priority meeting that cannot be moved',
                'start_time': datetime.now().replace(hour=10, minute=0, second=0, microsecond=0),
                'duration_minutes': 60,
                'event_type': 'work',
                'priority_level': 5,
                'is_fixed_time': True,
                'is_ai_modifiable': False
            },
            {
                'title': 'Gym Session',
                'description': 'Workout session that can be rescheduled',
                'start_time': datetime.now().replace(hour=10, minute=30, second=0, microsecond=0),
                'duration_minutes': 60,
                'event_type': 'exercise',
                'priority_level': 3,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Lunch with Client',
                'description': 'Business lunch that overlaps with gym',
                'start_time': datetime.now().replace(hour=11, minute=0, second=0, microsecond=0),
                'duration_minutes': 90,
                'event_type': 'meal',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            }
        ],
        
        'busy_day': [
            # Create a very busy day to test optimization
            {
                'title': 'Morning Standup',
                'description': 'Daily team meeting',
                'start_time': datetime.now().replace(hour=9, minute=0, second=0, microsecond=0),
                'duration_minutes': 30,
                'event_type': 'work',
                'priority_level': 4,
                'is_fixed_time': True,
                'is_ai_modifiable': False
            },
            {
                'title': 'Project Planning',
                'description': 'Plan next sprint activities',
                'start_time': datetime.now().replace(hour=10, minute=0, second=0, microsecond=0),
                'duration_minutes': 90,
                'event_type': 'work',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Client Presentation',
                'description': 'Present quarterly results',
                'start_time': datetime.now().replace(hour=14, minute=0, second=0, microsecond=0),
                'duration_minutes': 60,
                'event_type': 'work',
                'priority_level': 5,
                'is_fixed_time': True,
                'is_ai_modifiable': False
            },
            {
                'title': 'Workout',
                'description': 'Cardio and strength training',
                'start_time': datetime.now().replace(hour=17, minute=0, second=0, microsecond=0),
                'duration_minutes': 60,
                'event_type': 'exercise',
                'priority_level': 3,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Dinner Prep',
                'description': 'Prepare healthy dinner',
                'start_time': datetime.now().replace(hour=18, minute=30, second=0, microsecond=0),
                'duration_minutes': 45,
                'event_type': 'meal',
                'priority_level': 3,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Family Time',
                'description': 'Spend time with family',
                'start_time': datetime.now().replace(hour=19, minute=30, second=0, microsecond=0),
                'duration_minutes': 90,
                'event_type': 'personal',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            }
        ],
        
        'health_focus': [
            # Create health-focused events to test health optimization
            {
                'title': 'Morning Meditation',
                'description': 'Mindfulness meditation session',
                'start_time': datetime.now().replace(hour=6, minute=30, second=0, microsecond=0),
                'duration_minutes': 20,
                'event_type': 'personal',
                'priority_level': 3,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Healthy Breakfast',
                'description': 'Nutritious breakfast with protein and fiber',
                'start_time': datetime.now().replace(hour=7, minute=0, second=0, microsecond=0),
                'duration_minutes': 30,
                'event_type': 'meal',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Mid-Morning Walk',
                'description': 'Short walk for mental clarity',
                'start_time': datetime.now().replace(hour=10, minute=30, second=0, microsecond=0),
                'duration_minutes': 15,
                'event_type': 'exercise',
                'priority_level': 2,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Healthy Lunch',
                'description': 'Balanced lunch with vegetables and lean protein',
                'start_time': datetime.now().replace(hour=12, minute=30, second=0, microsecond=0),
                'duration_minutes': 45,
                'event_type': 'meal',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Afternoon Exercise',
                'description': 'High-intensity interval training',
                'start_time': datetime.now().replace(hour=16, minute=0, second=0, microsecond=0),
                'duration_minutes': 45,
                'event_type': 'exercise',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            },
            {
                'title': 'Evening Wind-down',
                'description': 'Relaxation routine before bed',
                'start_time': datetime.now().replace(hour=21, minute=0, second=0, microsecond=0),
                'duration_minutes': 30,
                'event_type': 'sleep',
                'priority_level': 4,
                'is_fixed_time': False,
                'is_ai_modifiable': True
            }
        ]
    }
    
    if scenario_name not in scenarios:
        print(f"❌ Unknown scenario: {scenario_name}")
        print(f"Available scenarios: {', '.join(scenarios.keys())}")
        return []
    
    events = scenarios[scenario_name]
    created_events = []
    
    print(f"🎬 Creating '{scenario_name}' test scenario...")
    
    for event_data in events:
        start_time = event_data['start_time']
        end_time = start_time + timedelta(minutes=event_data['duration_minutes'])
        
        event = CalendarEvent(
            user_id=user_id,
            title=event_data['title'],
            description=event_data['description'],
            start_time=start_time,
            end_time=end_time,
            event_type=event_data['event_type'],
            priority_level=event_data['priority_level'],
            is_ai_modifiable=event_data['is_ai_modifiable'],
            is_fixed_time=event_data['is_fixed_time']
        )
        
        db.session.add(event)
        created_events.append(event)
        
        modifiable_indicator = "🤖" if event.is_ai_modifiable else "🔒"
        print(f"  {modifiable_indicator} {event.title} - {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')} (Priority: {event.priority_level})")
    
    try:
        db.session.commit()
        print(f"✅ Created {len(created_events)} events for '{scenario_name}' scenario!")
        return created_events
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating events: {e}")
        return []


def add_custom_event(user_id):
    """Interactively create a custom event"""
    
    print("\n📝 Create Custom Event")
    print("-" * 30)
    
    # Get event details
    title = input("Event title: ").strip()
    if not title:
        print("❌ Title is required")
        return None
    
    description = input("Event description (optional): ").strip()
    
    # Event type
    event_types = ['work', 'exercise', 'health', 'meal', 'personal', 'sleep']
    print(f"Event types: {', '.join(event_types)}")
    event_type = input("Event type: ").strip().lower()
    if event_type not in event_types:
        print(f"❌ Invalid event type. Must be one of: {', '.join(event_types)}")
        return None
    
    # Priority level
    try:
        priority = int(input("Priority level (1-5, where 5 is highest): ").strip())
        if priority < 1 or priority > 5:
            print("❌ Priority must be between 1 and 5")
            return None
    except ValueError:
        print("❌ Invalid priority level")
        return None
    
    # Time
    try:
        hour = int(input("Start hour (0-23): ").strip())
        minute = int(input("Start minute (0-59): ").strip())
        duration = int(input("Duration in minutes: ").strip())
        
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and duration > 0):
            print("❌ Invalid time values")
            return None
            
    except ValueError:
        print("❌ Invalid time format")
        return None
    
    # AI settings
    is_ai_modifiable = input("Can AI modify this event? (y/N): ").strip().lower() in ['y', 'yes']
    is_fixed_time = input("Is this a fixed-time event? (y/N): ").strip().lower() in ['y', 'yes']
    
    # Create the event
    start_time = datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0)
    end_time = start_time + timedelta(minutes=duration)
    
    event = CalendarEvent(
        user_id=user_id,
        title=title,
        description=description,
        start_time=start_time,
        end_time=end_time,
        event_type=event_type,
        priority_level=priority,
        is_ai_modifiable=is_ai_modifiable,
        is_fixed_time=is_fixed_time
    )
    
    try:
        db.session.add(event)
        db.session.commit()
        
        modifiable_indicator = "🤖" if event.is_ai_modifiable else "🔒"
        print(f"\n✅ Created event: {modifiable_indicator} {event.title}")
        print(f"   Time: {start_time.strftime('%H:%M')} to {end_time.strftime('%H:%M')}")
        print(f"   Type: {event_type}, Priority: {priority}")
        
        return event
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating event: {e}")
        return None


def main():
    """Main function for quick event creation"""
    
    print("⚡ Quick Calendar Event Creator")
    print("=" * 40)
    
    # Create app context
    app = create_app()
    with app.app_context():
        
        # Get user
        username = input("Enter your username: ").strip()
        if not username:
            print("❌ Username is required")
            return
        
        user = User.query.filter_by(username=username).first()
        if not user:
            print(f"❌ User '{username}' not found")
            return
        
        print(f"👤 User: {user.username}")
        
        while True:
            print("\n🎯 What would you like to do?")
            print("1. Create test scenario")
            print("2. Add custom event")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ").strip()
            
            if choice == '1':
                print("\n📋 Available test scenarios:")
                print("• conflict - Create scheduling conflicts to test AI resolution")
                print("• busy_day - Create a very busy day to test optimization")
                print("• health_focus - Create health-focused events")
                
                scenario = input("\nEnter scenario name: ").strip().lower()
                if scenario:
                    create_test_scenario(user.id, scenario)
                
            elif choice == '2':
                add_custom_event(user.id)
                
            elif choice == '3':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
