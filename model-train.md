# Example of a basic ML pipeline for game prediction
def train_game_prediction_model(training_data, model_params):
    """
    Trains a machine learning model to predict game outcomes.
    """
    # Feature extraction
    X = extract_features(training_data)
    y = extract_targets(training_data)  # Win/loss or point spread
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create model pipeline with preprocessing
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection', SelectKBest(k=20)),
        ('model', XGBoostClassifier(**model_params))
    ])
    
    # Train with cross-validation
    cv_results = cross_validate(
        pipeline, X_train, y_train, 
        cv=5, 
        scoring=['accuracy', 'roc_auc', 'precision', 'recall']
    )
    
    # Train final model
    pipeline.fit(X_train, y_train)
    
    # Evaluate on test set
    test_score = pipeline.score(X_test, y_test)
    
    return pipeline, cv_results, test_score
