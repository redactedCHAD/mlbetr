// Example API Gateway route with authentication
app.post('/api/v1/predictions', authenticateJWT, async (req, res) => {
  try {
    const userId = req.user.id;
    const { gameId, predictionParams } = req.body;
    
    // Validate user subscription
    const userSubscription = await SubscriptionService.getUserSubscription(userId);
    if (!userSubscription.isActive) {
      return res.status(403).json({ error: 'Active subscription required' });
    }
    
    // Rate limiting check
    const rateLimit = await RateLimitService.checkLimit(userId, 'predictions');
    if (!rateLimit.allowed) {
      return res.status(429).json({ 
        error: 'Rate limit exceeded', 
        retryAfter: rateLimit.retryAfter 
      });
    }
    
    // Forward to prediction service
    const prediction = await PredictionService.generatePrediction(gameId, predictionParams);
    
    // Log usage for analytics
    await AnalyticsService.logPredictionRequest(userId, gameId, prediction.id);
    
    return res.json(prediction);
  } catch (error) {
    console.error('Prediction error:', error);
    return res.status(500).json({ error: 'Prediction service unavailable' });
  }
});
