#!/usr/bin/env python3
"""
Mock Calendar Events Generator

This script adds realistic mock calendar events to your account for testing
the AI optimization feature. It creates a variety of events with different
types, priorities, and AI modification settings.
"""

import sys
import os
from datetime import datetime, timedelta
import random

# Add the parent directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import create_app, db
from app.models import User, CalendarEvent


def create_mock_events(user_id, num_days=7):
    """
    Create mock calendar events for testing AI optimization
    
    Args:
        user_id: The ID of the user to create events for
        num_days: Number of days to create events for (default: 7)
    """
    
    # Define event templates with realistic scenarios
    event_templates = [
        # Work Events
        {
            'title': 'Team Standup Meeting',
            'description': 'Daily standup with development team',
            'event_type': 'work',
            'priority_level': 4,
            'duration_minutes': 30,
            'is_fixed_time': True,
            'is_ai_modifiable': False,
            'preferred_times': [(9, 0), (9, 30), (10, 0)]
        },
        {
            'title': 'Project Review Meeting',
            'description': 'Weekly project progress review with stakeholders',
            'event_type': 'work',
            'priority_level': 5,
            'duration_minutes': 60,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(14, 0), (15, 0), (16, 0)]
        },
        {
            'title': 'Code Review Session',
            'description': 'Review recent code changes and provide feedback',
            'event_type': 'work',
            'priority_level': 3,
            'duration_minutes': 45,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(10, 30), (11, 0), (14, 30)]
        },
        {
            'title': 'Client Presentation',
            'description': 'Present quarterly results to key client',
            'event_type': 'work',
            'priority_level': 5,
            'duration_minutes': 90,
            'is_fixed_time': True,
            'is_ai_modifiable': False,
            'preferred_times': [(13, 0), (14, 0)]
        },
        
        # Exercise Events
        {
            'title': 'Morning Gym Session',
            'description': 'Strength training and cardio workout',
            'event_type': 'exercise',
            'priority_level': 4,
            'duration_minutes': 60,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(6, 30), (7, 0), (7, 30)]
        },
        {
            'title': 'Yoga Class',
            'description': 'Relaxing yoga session for flexibility and mindfulness',
            'event_type': 'exercise',
            'priority_level': 3,
            'duration_minutes': 75,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(18, 0), (18, 30), (19, 0)]
        },
        {
            'title': 'Evening Run',
            'description': '5K run around the neighborhood',
            'event_type': 'exercise',
            'priority_level': 3,
            'duration_minutes': 30,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(17, 30), (18, 0), (18, 30)]
        },
        
        # Health Events
        {
            'title': 'Annual Health Checkup',
            'description': 'Comprehensive health examination with family doctor',
            'event_type': 'health',
            'priority_level': 5,
            'duration_minutes': 60,
            'is_fixed_time': True,
            'is_ai_modifiable': False,
            'preferred_times': [(10, 0), (11, 0), (14, 0)]
        },
        {
            'title': 'Dental Cleaning',
            'description': 'Routine dental cleaning and checkup',
            'event_type': 'health',
            'priority_level': 4,
            'duration_minutes': 45,
            'is_fixed_time': True,
            'is_ai_modifiable': False,
            'preferred_times': [(9, 0), (10, 30), (15, 0)]
        },
        {
            'title': 'Physical Therapy Session',
            'description': 'Physical therapy for lower back pain',
            'event_type': 'health',
            'priority_level': 4,
            'duration_minutes': 50,
            'is_fixed_time': True,
            'is_ai_modifiable': False,
            'preferred_times': [(16, 0), (17, 0)]
        },
        
        # Meal Events
        {
            'title': 'Lunch with Sarah',
            'description': 'Catch up lunch with college friend Sarah',
            'event_type': 'meal',
            'priority_level': 2,
            'duration_minutes': 90,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(12, 0), (12, 30), (13, 0)]
        },
        {
            'title': 'Team Lunch Meeting',
            'description': 'Working lunch to discuss upcoming project milestones',
            'event_type': 'meal',
            'priority_level': 3,
            'duration_minutes': 60,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(12, 0), (12, 30)]
        },
        {
            'title': 'Dinner with Family',
            'description': 'Family dinner at home',
            'event_type': 'meal',
            'priority_level': 4,
            'duration_minutes': 60,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(18, 30), (19, 0), (19, 30)]
        },
        
        # Personal Events
        {
            'title': 'Grocery Shopping',
            'description': 'Weekly grocery shopping at the supermarket',
            'event_type': 'personal',
            'priority_level': 2,
            'duration_minutes': 45,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(10, 0), (15, 0), (16, 0)]
        },
        {
            'title': 'Call Mom',
            'description': 'Weekly catch-up call with mother',
            'event_type': 'personal',
            'priority_level': 3,
            'duration_minutes': 30,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(19, 0), (19, 30), (20, 0)]
        },
        {
            'title': 'House Cleaning',
            'description': 'Deep clean the house and organize',
            'event_type': 'personal',
            'priority_level': 2,
            'duration_minutes': 120,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(10, 0), (14, 0)]
        },
        {
            'title': 'Book Reading Time',
            'description': 'Personal reading time for self-improvement',
            'event_type': 'personal',
            'priority_level': 2,
            'duration_minutes': 45,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(20, 0), (20, 30), (21, 0)]
        },
        
        # Sleep Events
        {
            'title': 'Bedtime Routine',
            'description': 'Wind down routine before sleep',
            'event_type': 'sleep',
            'priority_level': 4,
            'duration_minutes': 30,
            'is_fixed_time': False,
            'is_ai_modifiable': True,
            'preferred_times': [(21, 30), (22, 0), (22, 30)]
        }
    ]
    
    created_events = []
    start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    
    print(f"Creating mock events for {num_days} days starting from {start_date.strftime('%Y-%m-%d')}...")
    
    for day in range(num_days):
        current_date = start_date + timedelta(days=day)
        
        # Skip weekends for work events, but include some personal events
        is_weekend = current_date.weekday() >= 5
        
        # Select random events for this day
        if is_weekend:
            # Weekend: more personal, exercise, and meal events
            day_templates = [t for t in event_templates if t['event_type'] in ['exercise', 'personal', 'meal', 'sleep']]
            events_per_day = random.randint(2, 4)
        else:
            # Weekday: mix of work and personal events
            day_templates = event_templates
            events_per_day = random.randint(3, 6)
        
        # Randomly select events for this day
        selected_templates = random.sample(day_templates, min(events_per_day, len(day_templates)))
        
        for template in selected_templates:
            # Choose random preferred time
            preferred_time = random.choice(template['preferred_times'])
            start_time = current_date.replace(hour=preferred_time[0], minute=preferred_time[1])
            end_time = start_time + timedelta(minutes=template['duration_minutes'])
            
            # Add some random variation to make it more realistic
            if not template['is_fixed_time']:
                # Add random variation of ±30 minutes
                variation = random.randint(-30, 30)
                start_time += timedelta(minutes=variation)
                end_time += timedelta(minutes=variation)
            
            # Create the event
            event = CalendarEvent(
                user_id=user_id,
                title=template['title'],
                description=template['description'],
                start_time=start_time,
                end_time=end_time,
                event_type=template['event_type'],
                priority_level=template['priority_level'],
                is_ai_modifiable=template['is_ai_modifiable'],
                is_fixed_time=template['is_fixed_time']
            )
            
            db.session.add(event)
            created_events.append(event)
            
            print(f"  📅 {event.title} - {start_time.strftime('%m/%d %H:%M')} to {end_time.strftime('%H:%M')} ({event.event_type}, priority: {event.priority_level})")
    
    try:
        db.session.commit()
        print(f"\n✅ Successfully created {len(created_events)} mock calendar events!")
        
        # Show summary by type
        event_summary = {}
        for event in created_events:
            event_type = event.event_type
            if event_type not in event_summary:
                event_summary[event_type] = {'count': 0, 'ai_modifiable': 0}
            event_summary[event_type]['count'] += 1
            if event.is_ai_modifiable:
                event_summary[event_type]['ai_modifiable'] += 1
        
        print("\n📊 Event Summary by Type:")
        for event_type, stats in event_summary.items():
            print(f"  {event_type.title()}: {stats['count']} events ({stats['ai_modifiable']} AI-modifiable)")
        
        ai_modifiable_count = sum(1 for e in created_events if e.is_ai_modifiable)
        print(f"\n🤖 AI Optimization Ready: {ai_modifiable_count} out of {len(created_events)} events can be optimized by AI")
        
        return created_events
        
    except Exception as e:
        db.session.rollback()
        print(f"❌ Error creating events: {e}")
        return []


def get_user_by_username(username):
    """Get user by username"""
    return User.query.filter_by(username=username).first()


def clear_existing_events(user_id):
    """Clear existing calendar events for the user"""
    existing_events = CalendarEvent.query.filter_by(user_id=user_id).all()
    if existing_events:
        print(f"🗑️  Removing {len(existing_events)} existing events...")
        for event in existing_events:
            db.session.delete(event)
        db.session.commit()
        print("✅ Existing events cleared")


def main():
    """Main function to run the mock data generator"""
    
    print("🎭 Mock Calendar Events Generator")
    print("=" * 50)
    
    # Create app context
    app = create_app()
    with app.app_context():
        
        # Get user input
        username = input("Enter your username: ").strip()
        if not username:
            print("❌ Username is required")
            return
        
        # Find user
        user = get_user_by_username(username)
        if not user:
            print(f"❌ User '{username}' not found")
            return
        
        print(f"👤 Found user: {user.username} (ID: {user.id})")
        
        # Ask about clearing existing events
        clear_existing = input("\nClear existing calendar events? (y/N): ").strip().lower()
        if clear_existing in ['y', 'yes']:
            clear_existing_events(user.id)
        
        # Ask for number of days
        try:
            days_input = input("\nNumber of days to create events for (default: 7): ").strip()
            num_days = int(days_input) if days_input else 7
            if num_days < 1 or num_days > 30:
                print("❌ Number of days must be between 1 and 30")
                return
        except ValueError:
            print("❌ Invalid number of days")
            return
        
        # Create mock events
        print(f"\n🎯 Creating mock events for {num_days} days...")
        events = create_mock_events(user.id, num_days)
        
        if events:
            print(f"\n🎉 Done! You can now:")
            print(f"   1. Visit your dashboard to see upcoming events")
            print(f"   2. Go to the calendar page to view all events")
            print(f"   3. Click 'AI Optimize Schedule' to test the AI optimization feature")
            print(f"   4. Try adding your own events and see how AI handles them")
            
            print(f"\n💡 Pro Tips:")
            print(f"   • Events marked as 'AI-modifiable' can be moved by the AI optimizer")
            print(f"   • Fixed-time events (meetings, appointments) won't be moved")
            print(f"   • Higher priority events get preference in optimization")
            print(f"   • The AI considers your health data when optimizing")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled by user")
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        import traceback
        traceback.print_exc()
