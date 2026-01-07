"""
Egypt Monthly Inflation Forecaster
Creates 1-6 month ahead forecasts based on sentiment with lag structure from dissertation
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# ============================================================================
# MONTHLY INFLATION DATA (Update manually from CBE/Trading Economics)
# ============================================================================
MONTHLY_INFLATION = [
    {'date': '2025-06-01', 'inflation': 13.9},
    {'date': '2025-07-01', 'inflation': 13.1},
    {'date': '2025-08-01', 'inflation': 12.0},
    {'date': '2025-09-01', 'inflation': 11.7},
    {'date': '2025-10-01', 'inflation': 12.5},
    {'date': '2025-11-01', 'inflation': 12.3},
    {'date': '2025-12-01', 'inflation': 12.3},
]

inflation_df = pd.DataFrame(MONTHLY_INFLATION)
inflation_df['date'] = pd.to_datetime(inflation_df['date'])
inflation_df['type'] = 'actual'

print(f"Loaded {len(inflation_df)} months of inflation data")
print(f"Latest: {inflation_df.iloc[-1]['date'].strftime('%b %Y')} = {inflation_df.iloc[-1]['inflation']:.1f}%")

# ============================================================================
# FETCH SENTIMENT FROM GITHUB AND AGGREGATE TO MONTHLY
# ============================================================================

arabic_url = "https://raw.githubusercontent.com/Its-Riad/Its-Riad.github.io/main/data/arabic_news.csv"

try:
    arabic_df = pd.read_csv(arabic_url)
    arabic_df = arabic_df[arabic_df['sentiment_label'] != 'neutral'].copy()
    arabic_df['date'] = pd.to_datetime(arabic_df['date_published'])
    
    # Calculate daily net sentiment
    def daily_net(df):
        return df.groupby('date').apply(
            lambda x: (x['sentiment_label'] == 'positive').sum() - (x['sentiment_label'] == 'negative').sum(),
            include_groups=False
        ).reset_index(name='net')
    
    arabic_daily = daily_net(arabic_df)
    arabic_daily['ma'] = arabic_daily['net'].rolling(7, min_periods=1).mean()
    
    # AGGREGATE TO MONTHLY
    arabic_daily['year_month'] = arabic_daily['date'].dt.to_period('M')
    arabic_monthly = arabic_daily.groupby('year_month')['ma'].mean().reset_index()
    arabic_monthly.columns = ['year_month', 'sentiment']
    arabic_monthly['date'] = arabic_monthly['year_month'].dt.to_timestamp()
    
    print(f"Loaded {len(arabic_daily)} days → {len(arabic_monthly)} months of sentiment")
    print(f"Latest sentiment: {arabic_monthly.iloc[-1]['date'].strftime('%b %Y')} = {arabic_monthly.iloc[-1]['sentiment']:.2f}")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    arabic_monthly = pd.DataFrame({
        'date': pd.date_range(start='2025-06-01', periods=7, freq='MS'),
        'sentiment': [5.0] * 7
    })

# ============================================================================
# MERGE INFLATION AND SENTIMENT ON SAME DATES
# ============================================================================
print("\n Merging data...")

merged = inflation_df.merge(arabic_monthly[['date', 'sentiment']], on='date', how='left')
merged['sentiment'] = merged['sentiment'].ffill().bfill()

print(f"Merged dataset has {len(merged)} months")
print(merged[['date', 'inflation', 'sentiment']].tail())

# ============================================================================
# CALCULATE FORECASTS WITH PROPER LAG STRUCTURE
# ============================================================================
print("\n Calculating forecasts...")

SENTIMENT_WEIGHTS = [1.00, 0.70, 0.49, 0.34, 0.24, 0.17, 0.12, 0.08, 0.06]
INFLATION_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
SCALE_FACTOR = 0.03
INTERCEPT = 0.5

def get_lags(series, n_lags):
    """Get last n_lags values, padding if needed"""
    if len(series) >= n_lags:
        return series.tail(n_lags).tolist()[::-1]
    else:
        last_val = series.iloc[-1] if len(series) > 0 else 0
        return [last_val] * n_lags

def calculate_forecast(sentiment_lags, inflation_lags):
    """Calculate forecast given lags"""
    sentiment_effect = sum(w * s for w, s in zip(SENTIMENT_WEIGHTS, sentiment_lags)) * SCALE_FACTOR
    inflation_momentum = sum(w * i for w, i in zip(INFLATION_WEIGHTS, inflation_lags))
    return INTERCEPT + sentiment_effect + inflation_momentum

# Get initial lags
sentiment_lags = get_lags(merged['sentiment'], 9)
inflation_lags = get_lags(merged['inflation'], 4)

print(f"\nInitial lags:")
print(f"  Sentiment (9 lags): {[f'{s:.2f}' for s in sentiment_lags]}")
print(f"  Inflation (4 lags): {[f'{i:.1f}' for i in inflation_lags]}")

# ============================================================================
# GENERATE MULTI-STEP FORECASTS (1-6 months ahead)
# ============================================================================
print("\n Generating multi-step forecasts...")

last_date = merged['date'].max()
forecasts = []

# Keep track for recursive forecasting
forecasted_inflation = inflation_lags.copy()
forecasted_sentiment = sentiment_lags.copy()

for month_ahead in range(1, 7):
    forecast_date = last_date + relativedelta(months=month_ahead)
    
    # Assume sentiment persists at last known value
    current_sentiment = sentiment_lags[0]
    
    # Update sentiment lags (shift and add current)
    forecasted_sentiment = [current_sentiment] + forecasted_sentiment[:-1]
    
    # Calculate forecast
    forecast_value = calculate_forecast(forecasted_sentiment, forecasted_inflation)
    
    # Update inflation lags for next iteration
    forecasted_inflation = [forecast_value] + forecasted_inflation[:-1]
    
    # Add uncertainty (grows with horizon)
    uncertainty = 0.3 * np.sqrt(month_ahead)
    
    forecasts.append({
        'date': forecast_date,
        'inflation': forecast_value,
        'lower_bound': forecast_value - 1.96 * uncertainty,
        'upper_bound': forecast_value + 1.96 * uncertainty,
        'type': 'forecast',
        'horizon': month_ahead
    })
    
    print(f"  {forecast_date.strftime('%b %Y')} (h={month_ahead}): {forecast_value:.1f}% [{forecast_value - 1.96*uncertainty:.1f}%, {forecast_value + 1.96*uncertainty:.1f}%]")

forecast_df = pd.DataFrame(forecasts)

# ============================================================================
# OUTPUT
# ============================================================================

inflation_df['lower_bound'] = np.nan
inflation_df['upper_bound'] = np.nan
inflation_df['horizon'] = 0

combined = pd.concat([inflation_df, forecast_df], ignore_index=True)
combined = combined.sort_values('date')

os.makedirs('data', exist_ok=True)
combined.to_csv('data/egypt_inflation_forecast.csv', index=False)

print(f"\n Saved to data/egypt_inflation_forecast.csv")
print(f"   Historical: {len(inflation_df)} months")
print(f"   Forecast: {len(forecast_df)} months")
print(f"\n Forecast range: {forecast_df['inflation'].min():.1f}% to {forecast_df['inflation'].max():.1f}%")
