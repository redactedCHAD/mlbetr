# Example code for a basic ETL pipeline
def fetch_mlb_game_data(start_date, end_date):
    """
    Fetches game data from MLB API for a specified date range.
    Returns structured data ready for preprocessing.
    """
    # API connection and authentication
    mlb_api = MLBStatsAPI(api_key=config.MLB_API_KEY)
    
    # Fetch raw game data
    raw_games = mlb_api.get_games(start_date=start_date, end_date=end_date)
    
    # Transform data into standardized format
    structured_games = []
    for game in raw_games:
        structured_game = {
            'game_id': game['gamePk'],
            'date': game['gameDate'],
            'home_team': game['teams']['home']['team']['name'],
            'away_team': game['teams']['away']['team']['name'],
            'home_score': game['teams']['home'].get('score', 0),
            'away_score': game['teams']['away'].get('score', 0),
            'status': game['status']['detailedState'],
            'weather': game.get('weather', {})
        }
        structured_games.append(structured_game)
    
    return structured_games
