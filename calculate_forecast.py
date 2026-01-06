"""
Egypt Inflation Forecaster - CORRECTED
Only uses historical data, creates ONE forecast year
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

# ============================================================================
# STEP 1: FETCH ONLY HISTORICAL IMF DATA
# ============================================================================
print("📊 Fetching IMF inflation data...")

current_year = 2026  # We are in 2026
last_historical_year = 2025  # Last year with actual data

try:
    imf_url = "https://www.imf.org/external/datamapper/api/v1/PCPIPCH/EGY"
    response = requests.get(imf_url, timeout=10)
    imf_data = response.json()
    
    inflation_data = []
    for year, value in imf_data['values']['PCPIPCH']['EGY'].items():
        year_int = int(year)
        # ONLY historical data (up to 2025)
        if value and year_int >= 2020 and year_int <= last_historical_year:
            inflation_data.append({
                'date': f'{year}-01-01',
                'inflation': float(value),
                'type': 'actual'
            })
    
    inflation_df = pd.DataFrame(inflation_data)
    inflation_df['date'] = pd.to_datetime(inflation_df['date'])
    inflation_df = inflation_df.sort_values('date')
    
    print(f"✅ Loaded {len(inflation_df)} years (2020-2025)")
    
except Exception as e:
    print(f"⚠️  IMF API failed: {e}")
    inflation_data = [
        {'date': '2020-01-01', 'inflation': 5.7},
        {'date': '2021-01-01', 'inflation': 5.2},
        {'date': '2022-01-01', 'inflation': 8.5},
        {'date': '2023-01-01', 'inflation': 24.0},
        {'date': '2024-01-01', 'inflation': 28.3},
        {'date': '2025-01-01', 'inflation': 20.4}
    ]
    inflation_df = pd.DataFrame(inflation_data)
    inflation_df['date'] = pd.to_datetime(inflation_df['date'])
    inflation_df['type'] = 'actual'

# ============================================================================
# STEP 2: FETCH SENTIMENT
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
    
    print(f"✅ Loaded {len(arabic_daily)} days")
    
except Exception as e:
    print(f"❌ Failed: {e}")
    arabic_daily = pd.DataFrame({'ma': [5.0]})

# ============================================================================
# STEP 3: OLS FORECAST FOR 2026 ONLY
# ============================================================================
print("\n🧮 Calculating forecast...")

SENTIMENT_WEIGHTS = [1.00, 0.70, 0.49, 0.34, 0.24, 0.17, 0.12, 0.08, 0.06]
INFLATION_WEIGHTS = [0.60, 0.25, 0.10, 0.05]
SCALE_FACTOR = 0.03
INTERCEPT = 0.5

if len(arabic_daily) >= 9:
    sentiment_history = arabic_daily['ma'].tail(9).tolist()[::-1]
else:
    sentiment_history = [arabic_daily['ma'].iloc[-1]] * 9

inflation_history = inflation_df['inflation'].tail(4).tolist()[::-1]
while len(inflation_history) < 4:
    inflation_history.append(inflation_history[0])

sentiment_effect = sum(w * s for w, s in zip(SENTIMENT_WEIGHTS, sentiment_history)) * SCALE_FACTOR
inflation_momentum = sum(w * i for w, i in zip(INFLATION_WEIGHTS, inflation_history))
forecast_2026 = INTERCEPT + sentiment_effect + inflation_momentum

print(f"2026 Forecast: {forecast_2026:.2f}%")

# ============================================================================
# STEP 4: OUTPUT
# ============================================================================

forecast_df = pd.DataFrame([{
    'date': '2026-01-01',
    'inflation': forecast_2026,
    'type': 'forecast'
}])
forecast_df['date'] = pd.to_datetime(forecast_df['date'])

combined = pd.concat([inflation_df, forecast_df], ignore_index=True)
combined = combined.sort_values('date')

os.makedirs('data', exist_ok=True)
combined.to_csv('data/egypt_inflation_forecast.csv', index=False)

print(f"\n✅ Output: {len(inflation_df)} historical + 1 forecast")
print(combined)
