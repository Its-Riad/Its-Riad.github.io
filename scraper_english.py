"""
English News Scraper - Daily News Egypt
Scrapes economic articles in English and analyzes sentiment with FinBERT
"""

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import pandas as pd
import os
from transformers import pipeline

# ============================================================================
# CONFIGURATION
# ============================================================================

ENGLISH_KEYWORDS = [
    'inflation', 'deflation', 'price increase', 'price rise', 'cost of living',
    'purchasing power', 'consumer prices', 'cpi', 'core inflation',
    'headline inflation', 'price pressure', 'monetary policy', 'interest rate',
    'central bank', 'cbe', 'price index', 'economic growth', 'gdp',
    'economic performance', 'unemployment', 'employment', 'currency',
    'exchange rate', 'dollar', 'pound', 'egp'
]

# ============================================================================
# LOAD MODEL ONCE (Part 10!)
# ============================================================================

print("📚 Loading FinBERT model for English sentiment analysis...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")
print("✅ FinBERT loaded successfully!\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_economic_article(title, content):
    """
    Check if article is about economics using our keywords
    
    Why: Filter out non-economic articles early to save processing time
    """
    text = (title + ' ' + content).lower()
    return any(keyword in text for keyword in ENGLISH_KEYWORDS)


def analyze_sentiment_english(text, max_length=512):
    """
    Analyze sentiment using FinBERT (already loaded globally)
    
    Args:
        text: Article text to analyze
        max_length: Max tokens (FinBERT limit is 512)
    
    Returns:
        tuple: (sentiment_label, confidence_score)
    
    Why max_length? FinBERT can only process 512 tokens at a time.
    We truncate to avoid crashes.
    """
    if not text or len(str(text).strip()) == 0:
        return ('neutral', 0.0)
    
    # Truncate to rough character limit (512 tokens ≈ 2048 chars)
    text = str(text)[:2048]
    
    try:
        result = finbert(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        return (label, score)
        
    except Exception as e:
        print(f"  ⚠️ Sentiment analysis error: {e}")
        return ('error', 0.0)


def scrape_article_details(url):
    """
    Scrape full article content from Daily News Egypt
    
    Returns:
        dict: Article data or None if failed
    """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Extract title
        title_tag = soup.find('h1', class_='entry-title') or soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        
        # Extract content
        content_div = soup.find('div', class_='entry-content')
        content = content_div.get_text(separator='\n', strip=True) if content_div else ""
        
        # Extract date (multiple methods for robustness)
        date_published = None
        
        # Method 1: Meta tag
        meta_published = soup.find('meta', property='article:published_time')
        if meta_published and meta_published.get('content'):
            date_published = meta_published.get('content')[:10]
        
        # Method 2: Time tag
        if not date_published:
            time_tag = soup.find('time', class_='updated-date')
            if time_tag and time_tag.get('datetime'):
                date_published = time_tag.get('datetime')[:10]
        
        # Method 3: URL pattern (e.g., /2025/12/08/article)
        if not date_published:
            import re
            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
            if date_match:
                year, month, day = date_match.groups()
                date_published = f"{year}-{month}-{day}"
        
        return {
            'date_published': date_published or 'Unknown',
            'title': title,
            'url': url,
            'text': content,
            'word_count': len(content.split()),
            'source': 'daily_news_egypt'
        }
        
    except Exception as e:
        print(f"  ❌ Error scraping {url}: {e}")
        return None


def get_articles_in_date_range(search_term, start_date, end_date, max_pages=10):
    """
    Get articles from Daily News Egypt within date range
    
    Args:
        search_term: What to search for
        start_date: datetime object (earliest date)
        end_date: datetime object (latest date)
        max_pages: Maximum pages to scrape
    
    Returns:
        list: Article URLs within date range
    """
    all_urls = []
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    print(f"🔍 Searching '{search_term}' from {start_date_str} to {end_date_str}")
    
    for page in range(1, max_pages + 1):
        if page == 1:
            url = f'https://www.dailynewsegypt.com/?s={search_term}'
        else:
            url = f'https://www.dailynewsegypt.com/page/{page}/?s={search_term}'
        
        print(f"  📄 Page {page}...", end=' ')
        
        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'lxml')
            
            titles = soup.find_all(class_='entry-title')
            
            if not titles:
                print("No more articles. Stopping.")
                break
            
            page_urls = []
            for title in titles:
                link = title.find('a')
                if link and link.get('href'):
                    page_urls.append(link['href'])
            
            all_urls.extend(page_urls)
            print(f"Found {len(page_urls)} articles")
            
            time.sleep(2)  # Be polite to the server
            
        except Exception as e:
            print(f"Error: {e}")
            break
    
    return all_urls


def scrape_english_news(days_back=14, csv_file='data/english_news.csv'):
    """
    Main function: Scrape English economic news
    
    Args:
        days_back: How many days of history to scrape
        csv_file: Where to save data
    """
    print(f"\n{'='*80}")
    print(f"📰 ENGLISH NEWS SCRAPER - Daily News Egypt")
    print(f"{'='*80}\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    # Load existing URLs to avoid duplicates
    existing_urls = set()
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            existing_urls = set(df['url'].tolist())
            print(f"📂 Loaded {len(existing_urls)} existing articles\n")
        except Exception as e:
            print(f"⚠️ Could not load existing CSV: {e}\n")
    
    # Get article URLs
    search_terms = ['inflation', 'economy', 'economic']
    all_urls = []
    
    for term in search_terms:
        urls = get_articles_in_date_range(term, start_date, end_date, max_pages=5)
        all_urls.extend(urls)
    
    # Remove duplicates
    all_urls = list(set(all_urls))
    print(f"\n✓ Found {len(all_urls)} total unique articles")
    
    # Filter new URLs
    new_urls = [url for url in all_urls if url not in existing_urls]
    print(f"✓ {len(new_urls)} new articles to process\n")
    
    if not new_urls:
        print("✨ No new articles found!\n")
        return
    
    # Scrape and analyze
    print(f"{'='*80}")
    print(f"🤖 Processing {len(new_urls)} articles...")
    print(f"{'='*80}\n")
    
    new_articles = []
    filtered_count = 0
    
    for i, url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] Processing...")
        
        article = scrape_article_details(url)
        
        if article:
            # Check if economic
            if is_economic_article(article['title'], article['text']):
                # Analyze sentiment
                sentiment_label, sentiment_score = analyze_sentiment_english(article['text'])
                
                article['sentiment_label'] = sentiment_label
                article['sentiment_score'] = sentiment_score
                article['scrape_time'] = datetime.now().isoformat()
                
                new_articles.append(article)
                
                print(f"  ✓ {article['title'][:60]}...")
                print(f"    📅 {article['date_published']} | 📊 {sentiment_label} ({sentiment_score:.2f})")
            else:
                filtered_count += 1
                print(f"  ⊘ Filtered (not economic)")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n⚠️ Filtered out {filtered_count} non-economic articles")
    
    if not new_articles:
        print("\n⚠️ No relevant articles found!\n")
        return
    
    # Save to CSV
    df_new = pd.DataFrame(new_articles)
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ ENGLISH SCRAPER COMPLETE")
    print(f"   Added: {len(new_articles)} articles")
    print(f"   Total in database: {len(df_combined)}")
    
    if new_articles:
        avg_sentiment = sum(a['sentiment_score'] for a in new_articles) / len(new_articles)
        print(f"   Average sentiment: {avg_sentiment:.3f}")
    
    print(f"   Saved to: {csv_file}")
    print(f"{'='*80}\n")


# ============================================================================
# RUN WHEN CALLED DIRECTLY
# ============================================================================

if __name__ == "__main__":
    scrape_english_news(days_back=14)
