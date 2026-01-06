"""
Egypt Inflation Forecaster
Combines IMF API + GitHub sentiment data scraped from two Egyptian websites. Generates OLS forecast to a CSV file, whcich is then used in VegaLite. I chose IMF inflation data and not world 

Auto-runs monthly via GitHub Actions
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import sys
import os

# ============================================================================
# STEP 1: FETCH IMF INFLATION DATA
# ============================================================================
print("Fetching IMF inflation data.")

try:
    imf_url = "https://www.imf.org/external/datamapper/api/v1/PCPIPCH/EGY"
    response = requests.get(imf_url, timeout=10)
    imf_data = response.json()
    
    inflation_data = []
    for year, value in imf_data['values']['PCPIPCH']['EGY'].items():
        if value and int(year) >= 2020:
            inflation_data.append({
                'date': f'{year}-01-01',
                'inflation': float(value),
                'type': 'actual'
            })
    
    inflation_df = pd.DataFrame(inflation_data)
    inflation_df['date'] = pd.to_datetime(inflation_df['date'])
    inflation_df = inflation_df.sort_values('date')
    
    print(f"✅ Loaded {len(inflation_df)} years of IMF data")
    
except Exception as e:
    print(f"⚠️  IMF API failed: {e}")
    print("Using fallback data...")
    inflation_data = [
        {'date': '2020-01-01', 'inflation': 5.7},
        {'date': '2021-01-01', 'inflation': 5.2},
        {'date': '2022-01-01', 'inflation': 8.5},
        {'date': '2023-01-01', 'inflation': 24.0},
        {'date': '2024-01-01', 'inflation': 28.3},
        {'date': '2025-01-01', 'inflation': 14.0}
    ]
    inflation_df = pd.DataFrame(inflation_data)
    inflation_df['date'] = pd.to_datetime(inflation_df['date'])
    inflation_df['type'] = 'actual'

# ============================================================================
# STEP 2: FETCH SENTIMENT DATA FROM GITHUB
# ============================================================================
print("\n📰 Fetching sentiment data from GitHub...")

arabic_url = "https://raw.githubusercontent.com/Its-Riad/Its-Riad.github.io/main/data/arabic_news.csv"

try:
    arabic_df = pd.read_csv(arabic_url)
    
    # Process sentiment
    arabic_df = arabic_df[arabic_df['sentiment_label'] != 'neutral'].copy()
    arabic_df['date'] = pd.to_datetime(arabic_df['date_published'])
    
    # Daily net sentiment
    def daily_net(df):
        return df.groupby('date').apply(
            lambda x: (x['sentiment_label'] == 'positive').sum() - (x['sentiment_label'] == 'negative').sum(),
            include_groups=False
        ).reset_index(name='net')
    
    arabic_daily = daily_net(arabic_df)
    arabic_daily['ma'] = arabic_daily['net'].rolling(7, min_periods=1).mean()
    
    print(f"✅ Loaded {len(arabic_daily)} days of sentiment data")
    
except Exception as e:
    print(f"❌ Failed to load sentiment: {e}")
    print("Using fallback neutral sentiment...")
    arabic_daily = pd.DataFrame({'ma': [5.0]})

# ============================================================================
# STEP 3: CALCULATE OLS FORECAST
# ============================================================================
print("\n🧮 Calculating OLS forecast...")

# OLS PARAMETERS (from dissertation - exponential decay)
SENTIMENT_WEIGHTS = [1.00, 0.70, 0.49, 0.34, 0.24, 0.17, 0.12, 0.08, 0.06]
INFLATION_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
SCALE_FACTOR = 0.03  # 20% total weight for sentiment
INTERCEPT = 0.5

# Get last 9 sentiment values (current + 8 lags)
if len(arabic_daily) >= 9:
    sentiment_history = arabic_daily['ma'].tail(9).tolist()[::-1]
else:
    sentiment_history = [arabic_daily['ma'].iloc[-1]] * 9

# Get last 4 inflation values
inflation_history = inflation_df['inflation'].tail(4).tolist()[::-1]
while len(inflation_history) < 4:
    inflation_history.append(inflation_history[0] if inflation_history else 14.0)

# Calculate sentiment effect
sentiment_weighted = sum(w * s for w, s in zip(SENTIMENT_WEIGHTS, sentiment_history))
sentiment_effect = sentiment_weighted * SCALE_FACTOR

# Calculate inflation momentum
inflation_momentum = sum(w * i for w, i in zip(INFLATION_WEIGHTS, inflation_history))

# Final forecast
forecast_value = INTERCEPT + sentiment_effect + inflation_momentum

print(f"\n📈 FORECAST BREAKDOWN:")
print(f"  Sentiment effect: {sentiment_effect:.2f} pp")
print(f"  Inflation momentum: {inflation_momentum:.2f} pp")
print(f"  Intercept: {INTERCEPT:.2f} pp")
print(f"  ━━━━━━━━━━━━━━━━━━━━━━")
print(f"  2026 Forecast: {forecast_value:.2f}%")

# ============================================================================
# STEP 4: CREATE OUTPUT CSV
# ============================================================================
print("\n💾 Creating output CSV...")

# Add forecast point
forecast_df = pd.DataFrame([{
    'date': '2026-01-01',
    'inflation': forecast_value,
    'type': 'forecast'
}])
forecast_df['date'] = pd.to_datetime(forecast_df['date'])

# Combine
combined = pd.concat([inflation_df, forecast_df], ignore_index=True)
combined = combined.sort_values('date')

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save
output_path = 'data/egypt_inflation_forecast.csv'
combined.to_csv(output_path, index=False)

print(f"✅ Saved to: {output_path}")
print(f"\n📊 Final dataset:")
print(combined)
print(f"\n🎉 Done! Forecast updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
