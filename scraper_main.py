"""
Main Scraper Orchestrator
Runs both English and Arabic scrapers
"""

import sys
from datetime import datetime

print("="*80)
print("🚀 EGYPTIAN ECONOMIC SENTIMENT SCRAPER")
print("="*80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("="*80)
print()

# Import and run English scraper
print("=" * 80)
print("PART 1: ENGLISH NEWS")
print("=" * 80)
try:
    from scraper_english import scrape_english_news
    scrape_english_news(days_back=14)
except Exception as e:
    print(f"❌ English scraper failed: {e}")
    print("Continuing to Arabic scraper...\n")

# Import and run Arabic scraper
print("=" * 80)
print("PART 2: ARABIC NEWS")
print("=" * 80)
try:
    from scraper_arabic import scrape_arabic_news
    scrape_arabic_news(days_back=14)
except Exception as e:
    print(f"❌ Arabic scraper failed: {e}\n")

# Final summary
print("=" * 80)
print("✅ ALL SCRAPERS COMPLETE")
print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
print("=" * 80)
