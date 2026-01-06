"""
Egypt Monthly Inflation Forecaster
Creates 1-6 month ahead forecasts based on sentiment
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
from dateutil.relativedelta import relativedelta
import os

# ============================================================================
# MONTHLY INFLATION DATA (Update this manually from CBE/Trading Economics)
# ============================================================================
MONTHLY_INFLATION = [
    {'date': '2025-06-01', 'inflation': 13.9},
    {'date': '2025-07-01', 'inflation': 13.1},
    {'date': '2025-08-01', 'inflation': 12.0},
    {'date': '2025-09-01', 'inflation': 11.7},
    {'date': '2025-10-01', 'inflation': 12.5},
    {'date': '2025-11-01', 'inflation': 12.3},
    {'date': '2025-12-01', 'inflation': 12.3},  # Estimate
]

inflation_df = pd.DataFrame(MONTHLY_INFLATION)
inflation_df['date'] = pd.to_datetime(inflation_df['date'])
inflation_df['type'] = 'actual'

print(f"📊 Loaded {len(inflation_df)} months of inflation data")
print(f"Latest: {inflation_df.iloc[-1]['date'].strftime('%b %Y')} = {inflation_df.iloc[-1]['inflation']:.1f}%")

# ============================================================================
# FETCH SENTIMENT FROM GITHUB
# ============================================================================
print("\n📰 Fetching sentiment...")

arabic_url = "https://raw.githubusercontent.com/Its-Riad/Its-Riad.github.io/main/data/arabic_news.csv"

try:
    arabic_df = pd.read_csv(arabic_url)
    arabic_df = arabic_df[arabic_df['sentiment_label'] != 'neutral'].copy()
    arabic_df['date'] = pd.to_datetime(arabic_df['date_published'])
    
    def daily_net(df):
        return df.groupby('date').apply(
            lambda x: (x['sentiment_label'] == 'positive').sum() - (x['sentiment_label'] == 'negative').sum(),
            include_groups=False
        ).reset_index(name='net')
    
    arabic_daily = daily_net(arabic_df)
    arabic_daily['ma'] = arabic_daily['net'].rolling(7, min_periods=1).mean()
    
    print(f"✅ Loaded {len(arabic_daily)} days of sentiment")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    arabic_daily = pd.DataFrame({'ma': [5.0]})

# ============================================================================
# CALCULATE FORECASTS FOR 1-6 MONTHS AHEAD
# ============================================================================
print("\n🧮 Calculating forecasts...")

# Weights
SENTIMENT_WEIGHTS = [1.00, 0.70, 0.49, 0.34, 0.24, 0.17, 0.12, 0.08, 0.06]
INFLATION_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
SCALE_FACTOR = 0.03
INTERCEPT = 0.5

# Get sentiment lags (9 values)
if len(arabic_daily) >= 9:
    sentiment_history = arabic_daily['ma'].tail(9).tolist()[::-1]
else:
    sentiment_history = [arabic_daily['ma'].iloc[-1]] * 9

# Get inflation lags (4 values)
inflation_history = inflation_df['inflation'].tail(4).tolist()[::-1]

# Calculate effects
sentiment_effect = sum(w * s for w, s in zip(SENTIMENT_WEIGHTS, sentiment_history)) * SCALE_FACTOR
inflation_momentum = sum(w * i for w, i in zip(INFLATION_WEIGHTS, inflation_history))
base_forecast = INTERCEPT + sentiment_effect + inflation_momentum

print(f"\nBase forecast calculation:")
print(f"  Sentiment effect: {sentiment_effect:.2f} pp")
print(f"  Inflation momentum: {inflation_momentum:.2f} pp")
print(f"  Intercept: {INTERCEPT:.2f} pp")
print(f"  = {base_forecast:.2f}%")

# Generate forecasts for next 6 months
last_date = inflation_df['date'].max()
forecasts = []

for month_ahead in range(1, 7):
    forecast_date = last_date + relativedelta(months=month_ahead)
    
    # Forecast value (simple approach - same for all months, can add decay)
    forecast_value = base_forecast
    
    forecasts.append({
        'date': forecast_date,
        'inflation': forecast_value,
        'type': 'forecast'
    })

forecast_df = pd.DataFrame(forecasts)

print(f"\n📅 Created forecasts for {len(forecast_df)} months:")
for _, row in forecast_df.iterrows():
    print(f"  {row['date'].strftime('%b %Y')}: {row['inflation']:.1f}%")

# ============================================================================
# OUTPUT
# ============================================================================

combined = pd.concat([inflation_df, forecast_df], ignore_index=True)
combined = combined.sort_values('date')

os.makedirs('data', exist_ok=True)
combined.to_csv('data/egypt_inflation_forecast.csv', index=False)

print(f"\n✅ Saved to data/egypt_inflation_forecast.csv")
print(f"   Historical: {len(inflation_df)} months")
print(f"   Forecast: {len(forecast_df)} months")
