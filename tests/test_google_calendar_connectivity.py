#!/usr/bin/env python3
"""
Simple Google Calendar API connectivity test
This script tests if we can reach Google Calendar API without any Flask dependencies
"""

import requests
import socket
import time

def test_basic_connectivity():
    """Test basic internet connectivity"""
    print("=== Testing Basic Connectivity ===")
    
    endpoints = [
        "https://www.google.com",
        "https://www.googleapis.com", 
        "https://calendar-json.googleapis.com"
    ]
    
    for endpoint in endpoints:
        try:
            start_time = time.time()
            response = requests.get(endpoint, timeout=30)
            elapsed = time.time() - start_time
            print(f"✅ {endpoint}: {response.status_code} ({elapsed:.2f}s)")
        except requests.exceptions.Timeout:
            print(f"❌ {endpoint}: TIMEOUT after 30 seconds")
        except Exception as e:
            print(f"❌ {endpoint}: ERROR - {e}")

def test_socket_timeout():
    """Test socket-level connectivity"""
    print("\n=== Testing Socket Connectivity ===")
    
    try:
        # Test direct socket connection to Google
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        
        start_time = time.time()
        result = sock.connect_ex(("www.googleapis.com", 443))
        elapsed = time.time() - start_time
        
        if result == 0:
            print(f"✅ Socket connection to googleapis.com:443 successful ({elapsed:.2f}s)")
        else:
            print(f"❌ Socket connection failed with code: {result}")
        sock.close()
        
    except Exception as e:
        print(f"❌ Socket test failed: {e}")

def test_google_calendar_api():
    """Test Google Calendar API with minimal authentication"""
    print("\n=== Testing Google Calendar API Discovery ===")
    
    try:
        # Test the discovery document
        discovery_url = "https://www.googleapis.com/discovery/v1/apis/calendar/v3/rest"
        
        start_time = time.time()
        response = requests.get(discovery_url, timeout=30)
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"✅ Calendar API discovery successful ({elapsed:.2f}s)")
            data = response.json()
            print(f"   API Name: {data.get('name', 'Unknown')}")
            print(f"   API Version: {data.get('version', 'Unknown')}")
        else:
            print(f"❌ Calendar API discovery failed: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Calendar API test failed: {e}")

def test_proxy_settings():
    """Check for proxy configuration"""
    print("\n=== Checking Proxy Settings ===")
    
    import os
    import urllib.request
    
    proxies = urllib.request.getproxies()
    if proxies:
        print("🔍 Proxy settings detected:")
        for protocol, proxy in proxies.items():
            print(f"   {protocol}: {proxy}")
    else:
        print("✅ No proxy settings detected")
    
    # Check environment variables
    proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_env_vars:
        value = os.environ.get(var)
        if value:
            print(f"🔍 Environment variable {var}: {value}")

if __name__ == "__main__":
    print("Google Calendar Connectivity Diagnostic Tool")
    print("=" * 50)
    
    test_basic_connectivity()
    test_socket_timeout()
    test_google_calendar_api()
    test_proxy_settings()
    
    print("\n" + "=" * 50)
    print("Diagnostic complete. Check the results above.")
    print("If all tests pass but calendar sync still fails,")
    print("the issue may be with OAuth credentials or API quotas.")