# Stock Price Prediction Using Multi-Source Data and Hybrid Modeling Approaches

## Project Proposal

### a. Project Team
**Zetao Huang (huan2635)** - Individual project

### b. Dataset Description

This project will utilize three complementary datasets to create a comprehensive stock prediction system:

1. **Stock Price Data**: Historical stock price data will be obtained from Yahoo Finance via the `yfinance` Python library. This dataset provides essential market information including daily open, high, low, close prices, adjusted close prices, and trading volumes for multiple stocks. The project will focus on 10 major NASDAQ stocks: ADBE, CMCSA, QCOM, GOOG, PEP, SBUX, COST, AMD, INTC, and PYPL. This data will serve as the foundation for price prediction and provide the target variables for our models.

2. **Economic Indicators**: Macroeconomic data will be sourced from the Federal Reserve Economic Data (FRED) using the `fredapi` Python library. The specific economic indicators include GDP, Real GDP, Unemployment Rate, Consumer Price Index (CPI), Federal Funds Rate, and S&P 500 index. These indicators provide crucial context about the broader economic environment that influences stock performance. The API key '5634d0081e84d747c4413186eb2c19cb' will be used to access this data. Economic indicators with different reporting frequencies (daily, monthly, quarterly) will be appropriately processed to align with daily stock data.

3. **Financial News Articles**: News sentiment data will be obtained from the Hugging Face dataset "benstaf/FNSPID-filtered-nasdaq-100" (https://huggingface.co/datasets/benstaf/FNSPID-filtered-nasdaq-100). This dataset contains financial news articles specifically related to NASDAQ-100 companies. Instead of using the pre-trained FinBERT model for sentiment analysis, I will implement a dictionary-based approach using VADER (Valence Aware Dictionary and sEntiment Reasoner) to extract sentiment scores from news articles, ensuring compliance with project requirements to avoid pre-trained neural networks.

The integration of these three datasets creates a rich feature set that captures market behavior (stock prices), economic context (FRED indicators), and market sentiment (news articles), providing a holistic view for stock prediction.

### c. Research Questions and Tasks

The primary research questions this project aims to address are:

1. How effectively can stock price movements be predicted by combining technical indicators, economic data, and news sentiment?
2. Which features (technical, economic, or sentiment-based) contribute most significantly to prediction accuracy?
3. How do different model architectures (neural networks vs. tree-based models) compare in stock prediction performance?
4. Can attention mechanisms in LSTM networks improve prediction accuracy by identifying the most relevant time steps?
5. How does prediction performance vary across different market conditions (bull vs. bear markets)?

The specific tasks include:
- Implementing a data pipeline that integrates all three data sources
- Developing a feature engineering process to create meaningful predictors
- Building and training two distinct model architectures (LSTM with attention and XGBoost)
- Evaluating and comparing model performance using multiple metrics
- Analyzing feature importance across different model types

### d. Prior Work

Stock prediction using machine learning has been extensively studied. The GitHub repository https://github.com/Durpfish/stock-prediction provides a comprehensive implementation of a stock prediction pipeline using Apache Airflow. This implementation demonstrates the integration of multiple data sources and the use of linear regression for prediction. However, my project will extend beyond this work in several key ways:

1. Implementing more sophisticated models (LSTM with attention mechanism and XGBoost)
2. Conducting a rigorous comparison between neural network and tree-based approaches
3. Incorporating a custom sentiment analysis approach instead of using pre-trained models
4. Analyzing feature importance and model performance across different market conditions

Other notable prior work includes papers by Sezer et al. (2020) who reviewed deep learning methods for financial time series forecasting, and Fischer & Krauss (2018) who demonstrated the effectiveness of LSTM networks for stock market prediction. My project will build upon these foundations while focusing on the unique aspects of multi-source data integration and model comparison.

### e. Model Architectures and Training Strategy

I will implement and compare two distinct model architectures:

1. **LSTM with Attention Mechanism (PyTorch)**:
   - Architecture: A multi-layer LSTM network enhanced with an attention mechanism
   - Input: Sequences of multivariate time series data (20-day windows)
   - Hidden layers: 2 LSTM layers with 128 hidden units each
   - Attention layer: Self-attention mechanism to focus on important time steps
   - Output: Single value prediction (next-day adjusted close price)
   - Hyperparameters to tune:
     - Sequence length (10, 15, 20, 30 days)
     - Hidden layer size (64, 128, 256 units)
     - Dropout rate (0.1, 0.2, 0.3)
     - Learning rate (0.0001, 0.001, 0.01)
   - Training strategy: Mini-batch gradient descent with Adam optimizer, early stopping based on validation loss, learning rate reduction on plateau

2. **XGBoost**:
   - Architecture: Gradient boosting decision tree ensemble
   - Input: Tabular data with engineered features
   - Output: Next-day adjusted close price
   - Hyperparameters to tune:
     - Max depth (3, 5, 7, 9)
     - Learning rate (0.01, 0.05, 0.1)
     - Number of estimators (100, 200, 300)
     - Subsample ratio (0.8, 0.9, 1.0)
     - Colsample bytree (0.8, 0.9, 1.0)
     - Regularization parameters (alpha, lambda)
   - Training strategy: Cross-validation with randomized search for hyperparameter optimization

### f. Evaluation Metrics

I will use the following metrics to evaluate and compare model performance:

1. **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors, giving higher weight to larger errors. This is relevant for assessing the overall prediction accuracy.

2. **Mean Absolute Error (MAE)**: Measures the average absolute difference between predicted and actual values. This provides a more interpretable measure of prediction error in the original price scale.

3. **R-squared (R²)**: Indicates the proportion of variance in the dependent variable explained by the model. This helps assess how well the model captures the underlying patterns.

4. **Direction Accuracy**: Measures the percentage of correct predictions of price movement direction (up or down). This is particularly important for trading strategies where the direction of movement matters more than the exact price.

5. **Feature Importance Analysis**: For XGBoost, I will analyze feature importance scores; for LSTM, I will examine attention weights to understand which features and time periods contribute most to predictions.

### g. Planned Visualizations

The final report will include the following visualizations:

1. **Model Architecture Diagram**: A detailed schematic representation of the LSTM with attention architecture, showing the input layer, LSTM layers, attention mechanism, and output layer. This will be created using a neural network visualization tool like TensorBoard or a custom diagram.

2. **Price Prediction Comparison Plot**: A time series plot showing actual stock prices alongside predictions from both models (LSTM and XGBoost). This will include confidence intervals for predictions and highlight periods of significant divergence.

3. **Prediction Error Analysis**: Histograms and box plots of prediction errors for both models, allowing for visual comparison of error distributions.

4. **Feature Importance Visualization**: 
   - For XGBoost: A horizontal bar chart showing the top 15 features ranked by importance score
   - For LSTM: A heatmap visualization of attention weights across time steps, highlighting which historical days had the most influence on predictions

5. **Performance Across Market Conditions**: Line charts comparing model performance metrics (RMSE, Direction Accuracy) across different market volatility regimes.

6. **Learning Curves**: Plots showing training and validation loss over epochs for the LSTM model to illustrate the learning process and potential overfitting.

7. **Correlation Matrix Heatmap**: Visualization of correlations between different features (stock, economic, and sentiment) to understand relationships in the data.

### h. Timeline

As I am working individually, I will adhere to the following timeline based on the course deadlines:

- **April 24**: Project Proposal submission deadline (4pm)

- **April 25-30**: 
  - Data collection and preprocessing
  - Implementation of VADER sentiment analysis
  - Feature engineering

- **May 1-5**:
  - LSTM model implementation and training
  - XGBoost model implementation and training
  - Initial evaluation and hyperparameter tuning

- **May 6-7**:
  - Finalize model comparison
  - Complete visualizations
  - Prepare poster for presentation

- **May 8**: Poster Presentation in Walter 402 (4pm)

- **May 9**: Final Report submission deadline (11:59pm)

I will dedicate approximately 15-20 hours per week to this project, with more intensive work during the final days before deadlines to ensure timely completion.

### i. Backup Plan

If the proposed approach encounters significant challenges, I have prepared the following fallback options:

1. **Model Simplification**: If the LSTM with attention proves too complex to implement effectively, I will fall back to a simpler RNN or vanilla LSTM architecture.

2. **Data Scope Reduction**: If processing all three data sources becomes unwieldy, I will focus on stock price data and one additional source (either economic indicators or sentiment).

3. **Alternative Sentiment Analysis**: If VADER sentiment analysis proves insufficient, I will implement a basic TF-IDF approach with logistic regression for sentiment classification.

4. **Prediction Horizon Adjustment**: If next-day prediction proves too challenging, I will shift to predicting weekly price movements, which may exhibit more predictable patterns.

5. **Feature Subset Selection**: If feature engineering becomes too complex, I will use a more limited set of established technical indicators and fundamental features.
