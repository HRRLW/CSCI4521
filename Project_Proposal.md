# Stock Price Prediction Using Multi-Source Data and Hybrid Modeling Approaches

## Project Proposal

### Project Team
**Zetao Huang (huan2635)** - Individual project

### Dataset Description

This project will utilize three complementary datasets to create a focused stock prediction system:

1. **Stock Price Data**: Historical stock price data from Yahoo Finance via the `yfinance` Python library. To keep the project manageable, I will focus on 5 major tech stocks (AAPL, MSFT, GOOG, AMZN, META) over a 3-year period (2020-2023). This data includes daily open, high, low, close prices, adjusted close prices, and trading volumes.

2. **Economic Indicators**: Key macroeconomic data from the Federal Reserve Economic Data (FRED) using the 'fredapi' Python library. I will select a focused set of indicators: Unemployment Rate, Consumer Price Index (CPI), Federal Funds Rate, and S&P 500 index. These indicators will be processed to align with daily stock data.

3. **Financial News Headlines**: News sentiment data from the Hugging Face dataset "benstaf/FNSPID-filtered-nasdaq-100". To simplify the analysis, I will focus only on article headlines rather than full content, and implement a dictionary-based approach using VADER for sentiment analysis, avoiding pre-trained neural networks.

By limiting the scope to fewer stocks and key indicators, the project remains feasible within the timeline while still providing meaningful insights into stock prediction.

### Research Questions and Tasks

This project focuses on two primary research questions:

1. How do neural network models (LSTM) compare with ensemble tree-based models (Random Forest) in predicting next-day stock price movements?

2. Which data source (technical indicators, economic data, or news sentiment) contributes most significantly to prediction accuracy?

The specific tasks include:
- Creating a streamlined data pipeline that integrates the three data sources
- Implementing basic feature engineering focused on proven technical indicators
- Building and training two models: a PyTorch LSTM network and Random Forest Regressor
- Comparing model performance using standard evaluation metrics
- Analyzing which features each model finds most important

### Prior Work

Stock prediction using machine learning has been extensively studied. The GitHub repository https://github.com/Durpfish/stock-prediction provides an implementation of a stock prediction pipeline using Apache Airflow with linear regression models. My project builds upon this foundation with two key differences:

1. Implementing more sophisticated models (LSTM and Random Forest)
2. Conducting a direct comparison between neural network and ensemble tree-based approaches

Recent research by Fischer & Krauss (2018) demonstrated that LSTM networks can achieve directional accuracies of approximately 53.5% on out-of-sample stock movements. Meanwhile, Khaidem et al. (2016) reported that Random Forest models achieved next-day directional accuracies ranging from 44.5% to 58.2%. My project will provide a controlled comparison between these approaches using identical data inputs.

### Model Architectures and Training Strategy

I will implement and compare two distinct model architectures:

1. **LSTM Network (PyTorch)**:
   - Architecture: A single-layer LSTM network
   - Input: Sequences of multivariate time series data (10-day windows)
   - Hidden layer: 64 hidden units
   - Output: Single value prediction (next-day adjusted close price)
   - Hyperparameters to tune:
     - Sequence length (5, 10, 15 days)
     - Hidden layer size (32, 64, 128 units)
     - Dropout rate (0.1, 0.2)
     - Learning rate (0.001, 0.01)
   - Training strategy: Adam optimizer with early stopping

2. **Random Forest Regressor**:
   - Architecture: Ensemble of decision trees using bootstrap aggregation
   - Input: Tabular data with engineered features
   - Output: Next-day adjusted close price
   - Hyperparameters to tune:
     - Number of trees (100, 200, 500)
     - Max depth (None, 10, 20)
     - Min samples split (2, 5, 10)
     - Min samples leaf (1, 2, 4)
   - Training strategy: 5-fold cross-validation

This focused approach allows for thorough implementation and comparison within the project timeline.

### Evaluation Metrics

I will use the following metrics to evaluate and compare model performance:

1. **Root Mean Squared Error (RMSE)**: Measures the average magnitude of prediction errors, giving higher weight to larger errors. This is the primary metric for assessing prediction accuracy.

2. **Mean Absolute Error (MAE)**: Provides a more interpretable measure of prediction error in the original price scale.

3. **Direction Accuracy**: Measures the percentage of correct predictions of price movement direction (up or down). This is particularly important for trading strategies.

4. **Feature Importance Analysis**: For Random Forest, I will analyze the built-in feature importance scores; for LSTM, I will use a permutation-based approach to determine which features contribute most to predictions.

### Planned Visualizations

The final report will include the following visualizations:

1. **Model Architecture Diagram**: A schematic representation of the LSTM network architecture, showing the input layer, LSTM layer, and output layer. This will be created using a simple diagram tool.

2. **Price Prediction Comparison Plot**: A time series plot showing actual stock prices alongside predictions from both models for each of the 5 stocks.

3. **Prediction Error Analysis**: Box plots comparing prediction errors for both models across different stocks.

4. **Feature Importance Visualization**: 
   - For Random Forest: A horizontal bar chart showing features ranked by importance score
   - For LSTM: A bar chart showing feature importance based on permutation importance

5. **Learning Curves**: Plots showing training and validation loss over epochs for the LSTM model.

These visualizations will provide clear insights into model performance and feature importance while remaining feasible to implement within the project timeline.

### Timeline

As I am working individually, I will adhere to the following timeline based on the course deadlines:

- **April 24**: Project Proposal submission deadline (4pm)

- **April 25-27**: 
  - Data collection (stock prices, economic indicators)
  - Implementation of VADER sentiment analysis for news headlines

- **April 28-30**: 
  - Data preprocessing and integration
  - Basic feature engineering

- **May 1-3**:
  - LSTM model implementation and training
  - Random Forest model implementation and training

- **May 4-5**:
  - Model evaluation and comparison
  - Generate visualizations

- **May 6-7**:
  - Prepare poster for presentation
  - Final adjustments to models if needed

- **May 8**: Poster Presentation in Walter 402 (4pm)

- **May 9**: Final Report submission deadline (11:59pm)

This timeline allows for focused work on each component while ensuring adequate time for testing and refinement.

### Backup Plan

If the proposed approach encounters challenges, I have prepared the following fallback options:

1. **Model Simplification**: If the LSTM implementation proves challenging, I will fall back to a simpler feed-forward neural network using PyTorch.

2. **Data Scope Reduction**: If processing all three data sources becomes unwieldy, I will focus only on stock price data with technical indicators.

3. **Stock Selection Adjustment**: If the selected stocks show too much correlation, I will switch to stocks from different sectors to provide more diverse patterns.

4. **Prediction Task Modification**: Instead of predicting exact prices, I could simplify to a classification task (predicting whether the price will go up or down).

5. **Time Period Reduction**: If data processing is too time-consuming, I will reduce the time period to the most recent year (2022-2023).
