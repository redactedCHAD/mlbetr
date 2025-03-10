-- MLB Betting AI Agent Database Schema

-- Teams table
CREATE TABLE teams (
    team_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(10) NOT NULL,
    league VARCHAR(5) NOT NULL,
    division VARCHAR(10) NOT NULL,
    home_stadium VARCHAR(100) NOT NULL,
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50),
    country VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Players table
CREATE TABLE players (
    player_id VARCHAR(20) PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    position VARCHAR(10) NOT NULL,
    bats VARCHAR(5),
    throws VARCHAR(5),
    height VARCHAR(10),
    weight INT,
    birth_date DATE,
    debut_date DATE,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Stadiums table
CREATE TABLE stadiums (
    stadium_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    team_id VARCHAR(10) REFERENCES teams(team_id),
    city VARCHAR(100) NOT NULL,
    state VARCHAR(50),
    country VARCHAR(50) NOT NULL,
    capacity INT,
    surface_type VARCHAR(50),
    roof_type VARCHAR(50),
    dimensions JSON,
    elevation INT,
    weather_factors JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Games table
CREATE TABLE games (
    game_id VARCHAR(20) PRIMARY KEY,
    season INT NOT NULL,
    game_date DATE NOT NULL,
    start_time TIMESTAMP NOT NULL,
    home_team_id VARCHAR(10) REFERENCES teams(team_id),
    away_team_id VARCHAR(10) REFERENCES teams(team_id),
    stadium_id INT REFERENCES stadiums(stadium_id),
    weather_conditions JSON,
    temperature DECIMAL(5,2),
    wind_speed DECIMAL(5,2),
    wind_direction VARCHAR(20),
    humidity DECIMAL(5,2),
    precipitation_chance DECIMAL(5,2),
    game_status VARCHAR(20) NOT NULL,
    final_home_score INT,
    final_away_score INT,
    home_starting_pitcher VARCHAR(20) REFERENCES players(player_id),
    away_starting_pitcher VARCHAR(20) REFERENCES players(player_id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player game stats table
CREATE TABLE player_game_stats (
    stat_id UUID PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    player_id VARCHAR(20) REFERENCES players(player_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    position VARCHAR(10) NOT NULL,
    
    -- Batting stats
    at_bats INT,
    runs INT,
    hits INT,
    doubles INT,
    triples INT,
    home_runs INT,
    rbi INT,
    stolen_bases INT,
    caught_stealing INT,
    walks INT,
    strikeouts INT,
    batting_avg DECIMAL(5,3),
    on_base_pct DECIMAL(5,3),
    slugging_pct DECIMAL(5,3),
    ops DECIMAL(5,3),
    
    -- Pitching stats
    innings_pitched DECIMAL(5,1),
    hits_allowed INT,
    runs_allowed INT,
    earned_runs INT,
    walks_allowed INT,
    strikeouts_pitched INT,
    home_runs_allowed INT,
    pitches_thrown INT,
    strikes_thrown INT,
    era DECIMAL(6,2),
    whip DECIMAL(6,3),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Team game stats table
CREATE TABLE team_game_stats (
    stat_id UUID PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    team_id VARCHAR(10) REFERENCES teams(team_id),
    is_home BOOLEAN NOT NULL,
    
    -- Offensive stats
    runs INT,
    hits INT,
    doubles INT,
    triples INT,
    home_runs INT,
    rbi INT,
    stolen_bases INT,
    walks INT,
    strikeouts INT,
    left_on_base INT,
    batting_avg DECIMAL(5,3),
    
    -- Pitching stats
    pitchers_used INT,
    total_innings_pitched DECIMAL(5,1),
    hits_allowed INT,
    runs_allowed INT,
    earned_runs INT,
    walks_allowed INT,
    strikeouts_pitched INT,
    home_runs_allowed INT,
    team_era DECIMAL(6,2),
    
    -- Defensive stats
    errors INT,
    double_plays INT,
    passed_balls INT,
    wild_pitches INT,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Betting odds table
CREATE TABLE betting_odds (
    odds_id UUID PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    sportsbook VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Moneyline odds
    home_moneyline INT,
    away_moneyline INT,
    
    -- Spread/runline
    home_runline DECIMAL(4,1),
    home_runline_odds INT,
    away_runline DECIMAL(4,1),
    away_runline_odds INT,
    
    -- Totals
    over_under DECIMAL(4,1),
    over_odds INT,
    under_odds INT,
    
    -- Other props
    additional_props JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Predictions table
CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    model_version VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    
    -- Win probability
    home_win_probability DECIMAL(5,4) NOT NULL,
    away_win_probability DECIMAL(5,4) NOT NULL,
    
    -- Run predictions
    predicted_home_runs DECIMAL(5,2),
    predicted_away_runs DECIMAL(5,2),
    predicted_total_runs DECIMAL(5,2),
    
    -- Confidence metrics
    confidence_score DECIMAL(5,4) NOT NULL,
    key_factors JSON NOT NULL,
    feature_importance JSON,
    
    -- Betting recommendations
    recommended_bets JSON,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Users table
CREATE TABLE users (
    user_id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    is_verified BOOLEAN DEFAULT FALSE
);

-- User preferences table
CREATE TABLE user_preferences (
    preference_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    risk_tolerance DECIMAL(3,2) NOT NULL,
    bankroll DECIMAL(10,2),
    bet_size_strategy VARCHAR(50),
    favorite_teams JSON,
    notification_preferences JSON,
    dashboard_layout JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User bets table
CREATE TABLE user_bets (
    bet_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    prediction_id UUID REFERENCES predictions(prediction_id),
    game_id VARCHAR(20) REFERENCES games(game_id),
    bet_type VARCHAR(20) NOT NULL,
    bet_amount DECIMAL(10,2) NOT NULL,
    odds INT NOT NULL,
    potential_payout DECIMAL(10,2) NOT NULL,
    sportsbook VARCHAR(50),
    placed_at TIMESTAMP NOT NULL,
    outcome VARCHAR(10),
    profit_loss DECIMAL(10,2),
    settled_at TIMESTAMP,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Subscriptions table
CREATE TABLE subscriptions (
    subscription_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    plan_type VARCHAR(50) NOT NULL,
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    payment_status VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    payment_method VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance tracking
CREATE TABLE model_performance (
    performance_id UUID PRIMARY KEY,
    model_version VARCHAR(20) NOT NULL,
    evaluation_date TIMESTAMP NOT NULL,
    evaluation_period VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    log_loss DECIMAL(8,6),
    betting_roi DECIMAL(6,4),
    sample_size INT NOT NULL,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance optimization
CREATE INDEX idx_games_date ON games(game_date);
CREATE INDEX idx_games_teams ON games(home_team_id, away_team_id);
CREATE INDEX idx_player_game_stats_game ON player_game_stats(game_id);
CREATE INDEX idx_player_game_stats_player ON player_game_stats(player_id);
CREATE INDEX idx_team_game_stats_game ON team_game_stats(game_id);
CREATE INDEX idx_team_game_stats_team ON team_game_stats(team_id);
CREATE INDEX idx_betting_odds_game ON betting_odds(game_id);
CREATE INDEX idx_predictions_game ON predictions(game_id);
CREATE INDEX idx_user_bets_user ON user_bets(user_id);
CREATE INDEX idx_user_bets_game ON user_bets(game_id);