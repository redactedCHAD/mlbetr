#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data acquisition module for MLB Betting AI Agent.
This module handles fetching data from various sources including MLB APIs,
statistics websites, and betting odds providers.
"""

import os
import sys
import logging
import json
import time
from datetime import datetime, timedelta
import requests
import pandas as pd
from bs4 import BeautifulSoup
import pybaseball as pb
from sqlalchemy import create_engine
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

class MLBDataAcquisition:
    """Class for acquiring MLB data from various sources."""
    
    def __init__(self):
        """Initialize the data acquisition module."""
        self.mlb_api_key = os.getenv("MLB_API_KEY")
        self.odds_api_key = os.getenv("ODDS_API_KEY")
        self.weather_api_key = os.getenv("WEATHER_API_KEY")
        
        # Database connection
        self.db_engine = create_engine(
            f"postgresql://{os.getenv('DB_USER', 'postgres')}:"
            f"{os.getenv('DB_PASSWORD', '')}@"
            f"{os.getenv('DB_HOST', 'localhost')}:"
            f"{os.getenv('DB_PORT', '5432')}/"
            f"{os.getenv('DB_NAME', 'mlbetr')}"
        )
        
        # API endpoints
        self.mlb_api_base_url = "https://statsapi.mlb.com/api/v1"
        self.odds_api_base_url = "https://api.the-odds-api.com/v4"
        self.weather_api_base_url = "https://api.weatherapi.com/v1"
        
        # Rate limiting parameters
        self.request_delay = 1.0  # seconds between requests
    
    def fetch_mlb_schedule(self, start_date, end_date):
        """
        Fetch MLB game schedule for a specified date range.
        
        Args:
            start_date (str): Start date in format 'YYYY-MM-DD'
            end_date (str): End date in format 'YYYY-MM-DD'
            
        Returns:
            list: List of game data dictionaries
        """
        logger.info(f"Fetching MLB schedule from {start_date} to {end_date}")
        
        url = f"{self.mlb_api_base_url}/schedule"
        params = {
            "startDate": start_date,
            "endDate": end_date,
            "sportId": 1,  # MLB
            "hydrate": "team,venue,game(content(summary)),linescore"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            games = []
            for date in data.get("dates", []):
                for game in date.get("games", []):
                    games.append(game)
            
            logger.info(f"Retrieved {len(games)} games")
            return games
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching MLB schedule: {e}")
            return []
    
    def fetch_team_data(self):
        """
        Fetch data for all MLB teams.
        
        Returns:
            list: List of team data dictionaries
        """
        logger.info("Fetching MLB team data")
        
        url = f"{self.mlb_api_base_url}/teams"
        params = {
            "sportId": 1,  # MLB
            "hydrate": "venue"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            teams = data.get("teams", [])
            logger.info(f"Retrieved data for {len(teams)} teams")
            return teams
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching team data: {e}")
            return []
    
    def fetch_player_data(self, team_id=None):
        """
        Fetch player data, optionally filtered by team.
        
        Args:
            team_id (str, optional): Team ID to filter players
            
        Returns:
            list: List of player data dictionaries
        """
        logger.info(f"Fetching player data{' for team ' + team_id if team_id else ''}")
        
        if team_id:
            url = f"{self.mlb_api_base_url}/teams/{team_id}/roster"
            params = {
                "hydrate": "person(stats(type=season))"
            }
        else:
            url = f"{self.mlb_api_base_url}/sports/1/players"
            params = {
                "hydrate": "currentTeam,stats(type=season)"
            }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if team_id:
                players = data.get("roster", [])
            else:
                players = data.get("people", [])
            
            logger.info(f"Retrieved data for {len(players)} players")
            return players
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching player data: {e}")
            return []
    
    def fetch_game_stats(self, game_id):
        """
        Fetch detailed statistics for a specific game.
        
        Args:
            game_id (str): MLB game ID
            
        Returns:
            dict: Game statistics data
        """
        logger.info(f"Fetching statistics for game {game_id}")
        
        url = f"{self.mlb_api_base_url}/game/{game_id}/boxscore"
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Retrieved statistics for game {game_id}")
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching game statistics: {e}")
            return {}
    
    def fetch_player_stats(self, player_id, season=None):
        """
        Fetch statistics for a specific player.
        
        Args:
            player_id (str): MLB player ID
            season (str, optional): Season year (e.g., '2023')
            
        Returns:
            dict: Player statistics data
        """
        season = season or datetime.now().year
        logger.info(f"Fetching statistics for player {player_id} for season {season}")
        
        url = f"{self.mlb_api_base_url}/people/{player_id}/stats"
        params = {
            "stats": "season",
            "season": season,
            "group": "hitting,pitching,fielding"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Retrieved statistics for player {player_id}")
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching player statistics: {e}")
            return {}
    
    def fetch_betting_odds(self, sport="baseball_mlb", regions="us", markets="h2h,spreads,totals", date=None):
        """
        Fetch betting odds from various sportsbooks.
        
        Args:
            sport (str): Sport key
            regions (str): Regions for sportsbooks (comma-separated)
            markets (str): Betting markets (comma-separated)
            date (str, optional): Date in format 'YYYY-MM-DD'
            
        Returns:
            list: List of odds data dictionaries
        """
        if not self.odds_api_key:
            logger.error("Odds API key not configured")
            return []
        
        date_str = f" for {date}" if date else ""
        logger.info(f"Fetching {markets} odds for {sport}{date_str}")
        
        url = f"{self.odds_api_base_url}/sports/{sport}/odds"
        params = {
            "apiKey": self.odds_api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american"
        }
        
        if date:
            params["date"] = date
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Retrieved odds for {len(data)} events")
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching betting odds: {e}")
            return []
    
    def fetch_weather_forecast(self, lat, lon, date=None):
        """
        Fetch weather forecast for a specific location and date.
        
        Args:
            lat (float): Latitude
            lon (float): Longitude
            date (str, optional): Date in format 'YYYY-MM-DD'
            
        Returns:
            dict: Weather forecast data
        """
        if not self.weather_api_key:
            logger.error("Weather API key not configured")
            return {}
        
        date_str = f" for {date}" if date else ""
        logger.info(f"Fetching weather forecast for location ({lat}, {lon}){date_str}")
        
        url = f"{self.weather_api_base_url}/forecast.json"
        params = {
            "key": self.weather_api_key,
            "q": f"{lat},{lon}",
            "days": 10,
            "aqi": "no",
            "alerts": "no"
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # If a specific date is requested, filter the forecast
            if date:
                for forecast_day in data.get("forecast", {}).get("forecastday", []):
                    if forecast_day.get("date") == date:
                        return forecast_day
                
                logger.warning(f"No forecast found for date {date}")
                return {}
            
            logger.info(f"Retrieved weather forecast")
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching weather forecast: {e}")
            return {}
    
    def fetch_advanced_stats_pybaseball(self, season=None, team=None):
        """
        Fetch advanced statistics using pybaseball library.
        
        Args:
            season (int, optional): Season year
            team (str, optional): Team abbreviation
            
        Returns:
            dict: Dictionary containing various advanced statistics DataFrames
        """
        season = season or datetime.now().year
        logger.info(f"Fetching advanced statistics for season {season}")
        
        stats = {}
        
        try:
            # Batting statistics
            logger.info("Fetching batting statistics")
            batting_stats = pb.batting_stats(season)
            stats["batting"] = batting_stats
            
            # Pitching statistics
            logger.info("Fetching pitching statistics")
            pitching_stats = pb.pitching_stats(season)
            stats["pitching"] = pitching_stats
            
            # Team batting statistics
            logger.info("Fetching team batting statistics")
            team_batting = pb.team_batting(season)
            stats["team_batting"] = team_batting
            
            # Team pitching statistics
            logger.info("Fetching team pitching statistics")
            team_pitching = pb.team_pitching(season)
            stats["team_pitching"] = team_pitching
            
            # If team is specified, filter the data
            if team:
                logger.info(f"Filtering statistics for team {team}")
                stats["batting"] = stats["batting"][stats["batting"]["Team"] == team]
                stats["pitching"] = stats["pitching"][stats["pitching"]["Team"] == team]
            
            logger.info("Advanced statistics fetched successfully")
            return stats
        
        except Exception as e:
            logger.error(f"Error fetching advanced statistics: {e}")
            return {}
    
    def save_to_database(self, data, table_name, if_exists="append"):
        """
        Save data to the database.
        
        Args:
            data (DataFrame or dict): Data to save
            table_name (str): Target table name
            if_exists (str): How to behave if the table exists
            
        Returns:
            bool: Success status
        """
        logger.info(f"Saving data to table {table_name}")
        
        try:
            # Convert dict to DataFrame if necessary
            if isinstance(data, dict):
                df = pd.DataFrame([data])
            elif isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data
            
            # Save to database
            df.to_sql(table_name, self.db_engine, if_exists=if_exists, index=False)
            
            logger.info(f"Successfully saved {len(df)} records to {table_name}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving data to database: {e}")
            return False
    
    def run_daily_data_collection(self, days_ahead=7):
        """
        Run a complete daily data collection process.
        
        Args:
            days_ahead (int): Number of days ahead to fetch schedule
            
        Returns:
            bool: Success status
        """
        logger.info("Starting daily data collection process")
        
        today = datetime.now().strftime("%Y-%m-%d")
        end_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        try:
            # 1. Fetch and save team data
            teams = self.fetch_team_data()
            if teams:
                self.save_to_database(teams, "teams", if_exists="replace")
            
            # 2. Fetch and save upcoming schedule
            games = self.fetch_mlb_schedule(today, end_date)
            if games:
                self.save_to_database(games, "games", if_exists="append")
            
            # 3. Fetch and save player data
            players = self.fetch_player_data()
            if players:
                self.save_to_database(players, "players", if_exists="replace")
            
            # 4. Fetch and save betting odds
            odds = self.fetch_betting_odds(date=today)
            if odds:
                self.save_to_database(odds, "betting_odds", if_exists="append")
            
            # 5. Fetch advanced statistics
            current_season = datetime.now().year
            advanced_stats = self.fetch_advanced_stats_pybaseball(current_season)
            if advanced_stats:
                for stat_type, stat_data in advanced_stats.items():
                    self.save_to_database(stat_data, f"advanced_{stat_type}", if_exists="replace")
            
            # 6. For each upcoming game, fetch weather forecast
            for game in games:
                if "venue" in game and "location" in game["venue"]:
                    venue = game["venue"]
                    if "defaultCoordinates" in venue:
                        coords = venue["defaultCoordinates"]
                        lat = coords.get("latitude")
                        lon = coords.get("longitude")
                        game_date = game.get("gameDate", "").split("T")[0]
                        
                        if lat and lon and game_date:
                            weather = self.fetch_weather_forecast(lat, lon, game_date)
                            if weather:
                                # Add weather data to game record
                                game_id = game.get("gamePk")
                                weather_data = {
                                    "game_id": game_id,
                                    "weather_data": json.dumps(weather)
                                }
                                self.save_to_database(weather_data, "game_weather", if_exists="append")
                            
                            # Respect API rate limits
                            time.sleep(self.request_delay)
            
            logger.info("Daily data collection completed successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error in daily data collection process: {e}")
            return False

if __name__ == "__main__":
    data_acquisition = MLBDataAcquisition()
    data_acquisition.run_daily_data_collection() 