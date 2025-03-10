#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Database setup script for MLB Betting AI Agent.
This script initializes the database with the schema defined in schema.sql.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_db_connection():
    """
    Create a connection to the PostgreSQL database.
    
    Returns:
        connection: A PostgreSQL database connection
    """
    try:
        connection = psycopg2.connect(
            host=os.getenv("DB_HOST", "localhost"),
            database=os.getenv("DB_NAME", "mlbetr"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            port=os.getenv("DB_PORT", "5432")
        )
        connection.autocommit = True
        return connection
    except Exception as e:
        logger.error(f"Error connecting to the database: {e}")
        sys.exit(1)

def create_database(connection, db_name):
    """
    Create the database if it doesn't exist.
    
    Args:
        connection: PostgreSQL connection
        db_name: Name of the database to create
    """
    try:
        # Create a cursor with the connection
        cursor = connection.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create database
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))
            logger.info(f"Database '{db_name}' created successfully")
        else:
            logger.info(f"Database '{db_name}' already exists")
            
        cursor.close()
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        sys.exit(1)

def execute_schema_script(connection, schema_file):
    """
    Execute the SQL schema script to create tables.
    
    Args:
        connection: PostgreSQL connection
        schema_file: Path to the schema SQL file
    """
    try:
        cursor = connection.cursor()
        
        # Read the schema file
        with open(schema_file, 'r') as f:
            schema_script = f.read()
        
        # Execute the schema script
        cursor.execute(schema_script)
        
        logger.info("Schema created successfully")
        cursor.close()
    except Exception as e:
        logger.error(f"Error executing schema script: {e}")
        sys.exit(1)

def main():
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description='Set up the MLB Betting AI Agent database')
    parser.add_argument('--reset', action='store_true', help='Drop and recreate all tables')
    args = parser.parse_args()
    
    # Get the path to the schema file
    current_dir = Path(__file__).parent.absolute()
    schema_file = current_dir / "schema.sql"
    
    if not schema_file.exists():
        logger.error(f"Schema file not found: {schema_file}")
        sys.exit(1)
    
    # Connect to PostgreSQL server
    connection = get_db_connection()
    
    # Get database name from environment or use default
    db_name = os.getenv("DB_NAME", "mlbetr")
    
    # Create database if it doesn't exist
    create_database(connection, db_name)
    
    # Close the initial connection
    connection.close()
    
    # Connect to the newly created database
    connection = get_db_connection()
    
    # If reset flag is set, drop all tables
    if args.reset:
        try:
            cursor = connection.cursor()
            cursor.execute("""
                DO $$ DECLARE
                    r RECORD;
                BEGIN
                    FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                        EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                    END LOOP;
                END $$;
            """)
            logger.info("All tables dropped successfully")
            cursor.close()
        except Exception as e:
            logger.error(f"Error dropping tables: {e}")
            sys.exit(1)
    
    # Execute the schema script
    execute_schema_script(connection, schema_file)
    
    # Close the connection
    connection.close()
    
    logger.info("Database setup completed successfully")

if __name__ == "__main__":
    main() 