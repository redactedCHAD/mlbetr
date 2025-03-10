-- Example schema for core game prediction tables
CREATE TABLE teams (
    team_id VARCHAR(10) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    league VARCHAR(5) NOT NULL,
    division VARCHAR(10) NOT NULL,
    home_stadium VARCHAR(100) NOT NULL
);

CREATE TABLE games (
    game_id VARCHAR(20) PRIMARY KEY,
    start_time TIMESTAMP NOT NULL,
    home_team_id VARCHAR(10) REFERENCES teams(team_id),
    away_team_id VARCHAR(10) REFERENCES teams(team_id),
    stadium VARCHAR(100) NOT NULL,
    weather_conditions JSON,
    game_status VARCHAR(20) NOT NULL,
    final_home_score INT,
    final_away_score INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE predictions (
    prediction_id UUID PRIMARY KEY,
    game_id VARCHAR(20) REFERENCES games(game_id),
    model_version VARCHAR(20) NOT NULL,
    predicted_home_win_prob DECIMAL(5,4) NOT NULL,
    predicted_total_runs DECIMAL(6,2),
    confidence_score DECIMAL(5,4) NOT NULL,
    key_factors JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE user_bets (
    bet_id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(user_id),
    prediction_id UUID REFERENCES predictions(prediction_id),
    bet_type VARCHAR(20) NOT NULL,
    bet_amount DECIMAL(10,2) NOT NULL,
    odds DECIMAL(7,2) NOT NULL,
    potential_payout DECIMAL(10,2) NOT NULL,
    outcome VARCHAR(10),
    settled_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
