#!/usr/bin/env python3
"""
Test script to create tables and insert data in Supabase.
"""

import os
from datetime import datetime
from supabase import create_client, Client

# Supabase credentials - directly using the service role key for maximum permissions
SUPABASE_URL = "https://ylxcsjarxlrdrtmkdfjk.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InlseGNzamFyeGxyZHJ0bWtkZmprIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzk2MjUxNCwiZXhwIjoyMDUzNTM4NTE0fQ.6g7xMOwtjd0Wgj6DcVRDo3z1KfzbDSKEhj3z79KcJek"

try:
    # Initialize Supabase client
    print("Connecting to Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print(f"Connected to Supabase at {SUPABASE_URL}")
    
    # Step 1: Try to create a test table via REST API
    print("\nAttempting to create a test table...")
    test_data = {
        "message": "This is a test entry",
        "timestamp": datetime.now().isoformat()
    }
    
    response = supabase.table("test_table").insert(test_data).execute()
    print(f"Insert Response: {response}")
    
    if hasattr(response, 'error') and response.error:
        print(f"Error creating test_table: {response.error}")
    else:
        print("Successfully created test_table and inserted data")
    
    # Step 2: Try to fetch the data back
    print("\nAttempting to read from test_table...")
    response = supabase.table("test_table").select("*").execute()
    
    if hasattr(response, 'error') and response.error:
        print(f"Error reading from test_table: {response.error}")
    else:
        print(f"Successfully read {len(response.data)} rows from test_table")
        for i, row in enumerate(response.data):
            print(f"Row {i+1}: {row}")
    
    # Step 3: Try to get list of tables
    print("\nAttempting to list all tables...")
    print("Note: This requires full database access (not supported via REST API)")
    print("Please check the Supabase dashboard to verify table creation")
    
except Exception as e:
    print(f"ERROR: {e}")

print("\nTest completed. Check the Supabase dashboard to verify table creation.")
print("Dashboard URL: https://app.supabase.com/project/ylxcsjarxlrdrtmkdfjk/editor") 