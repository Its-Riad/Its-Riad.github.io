"""
Arabic News Scraper - Youm7 & Al-Masry Al-Youm
Scrapes economic articles in Arabic and analyzes sentiment with CAMeL-BERT
"""

import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime, timedelta
import pandas as pd
import os
from transformers import pipeline
import re

# ============================================================================
# CONFIGURATION
# ============================================================================

ARABIC_KEYWORDS = [
    # Inflation & Prices
    "التضخم",           # inflation
    "الأسعار",          # prices
    "غلاء",             # high cost
    "ارتفاع الأسعار",   # price increases
    "أسعار المستهلك",   # consumer prices
    
    # Economic Growth
    "النمو الاقتصادي",  # economic growth
    "الاقتصاد",         # economy
    "الناتج المحلي",    # GDP
    "الأداء الاقتصادي", # economic performance
    
    # Currency & Exchange
    "الدولار",          # dollar
    "سعر الصرف",        # exchange rate
    "الجنيه",           # pound
    "العملة",           # currency
    
    # Monetary Policy
    "البنك المركزي",    # Central Bank
    "الفائدة",          # interest rate
    "السياسة النقدية",  # monetary policy
    
    # Employment
    "البطالة",          # unemployment
    "التوظيف",          # employment
]

# ============================================================================
# LOAD MODEL ONCE
# ============================================================================

print("📚 Loading CAMeL-BERT model for Arabic sentiment analysis...")
camelbert = pipeline(
    "sentiment-analysis",
    model="CAMeL-Lab/bert-base-arabic-camelbert-msa-sentiment"
)
print("✅ CAMeL-BERT loaded successfully!\n")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def is_economic_article_arabic(title, content):
    """Check if article is about economics using Arabic keywords"""
    text = title + ' ' + content
    return any(keyword in text for keyword in ARABIC_KEYWORDS)


def analyze_sentiment_arabic(text, max_length=512):
    """
    Analyze sentiment using CAMeL-BERT
    
    Returns:
        tuple: (sentiment_label, confidence_score)
    """
    if not text or len(str(text).strip()) == 0:
        return ('neutral', 0.0)
    
    # Truncate (Arabic characters are larger, so we're more conservative)
    text = str(text)[:1500]
    
    try:
        result = camelbert(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        return (label, score)
        
    except Exception as e:
        print(f"  ⚠️ Sentiment analysis error: {e}")
        return ('error', 0.0)


def extract_date_from_arabic_url(url):
    """
    Extract date from URL patterns like:
    /2025/12/08/article-name
    /story/2025/12/8/article
    """
    date_match = re.search(r'/(\d{4})/(\d{1,2})/(\d{1,2})/', url)
    if date_match:
        year, month, day = date_match.groups()
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return None


# ============================================================================
# YOUM7 SCRAPER - FIXED BASED ON YOUR INSPECTION
# ============================================================================

def scrape_youm7_economy_page(page_num=1):
    """Scrape Youm7 economy section page - FIXED VERSION"""
    url = f"https://www.youm7.com/Section/اقتصاد-وبورصة/297/{page_num}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        articles = []
        
        # Find all links with /story/ in href
        story_links = soup.find_all('a', href=lambda x: x and '/story/' in str(x))
        
        for link in story_links:
            href = link.get('href')
            title = link.get_text(strip=True)
            
            # Skip if no text or too short
            if not title or len(title) < 10:
                continue
            
            # Only keep if parent is <h3> (filters out sidebar/related links)
            if link.parent and link.parent.name == 'h3':
                # Make full URL
                if not href.startswith('http'):
                    href = 'https://www.youm7.com' + href
                
                articles.append({
                    'url': href,
                    'title': title,
                    'source': 'youm7'
                })
        
        # Remove duplicates
        seen = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen:
                seen.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles
        
    except Exception as e:
        print(f"  ❌ Error scraping Youm7 page {page_num}: {e}")
        return []


def scrape_youm7_article(url):
    """Scrape full article from Youm7 - FIXED VERSION"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Title
        title_tag = soup.find('h1')
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        
        # Content - Get ALL <p> tags (simple approach)
        all_paragraphs = soup.find_all('p')
        content_parts = []
        
        for p in all_paragraphs:
            text = p.get_text(strip=True)
            # Filter out very short paragraphs (likely navigation/ads)
            if text and len(text) > 20:
                content_parts.append(text)
        
        content = '\n'.join(content_parts)
        
        # Date from URL
        date_published = extract_date_from_arabic_url(url)
        
        return {
            'date_published': date_published or 'Unknown',
            'title': title,
            'url': url,
            'text': content,
            'word_count': len(content.split()),
            'source': 'youm7'
        }
        
    except Exception as e:
        print(f"  ❌ Error scraping Youm7 article: {e}")
        return None


# ============================================================================
# AL-MASRY AL-YOUM SCRAPER - FIXED BASED ON YOUR INSPECTION
# ============================================================================

def scrape_almasry_economy_page(page_num=1):
    """Scrape Al-Masry Al-Youm economy section - FIXED VERSION"""
    url = f"https://www.almasryalyoum.com/section/index/4?page={page_num}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        articles = []
        
        # Find all links with class "article-title"
        article_links = soup.find_all('a', class_='article-title')
        
        for link in article_links:
            href = link.get('href')
            title = link.get_text(strip=True)
            
            if href and title and len(title) > 10:
                # URL is already full, no need to prepend domain
                articles.append({
                    'url': href,
                    'title': title,
                    'source': 'almasry_alyoum'
                })
        
        # Remove duplicates
        seen = set()
        unique_articles = []
        for article in articles:
            if article['url'] not in seen:
                seen.add(article['url'])
                unique_articles.append(article)
        
        return unique_articles
        
    except Exception as e:
        print(f"  ❌ Error scraping Al-Masry page {page_num}: {e}")
        return []


def scrape_almasry_article(url):
    """Scrape full article from Al-Masry Al-Youm - FIXED VERSION"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')
        
        # Title
        title_tag = soup.find('h1') or soup.find('h2')
        title = title_tag.get_text(strip=True) if title_tag else "No Title"
        
        # Content - Get ALL <p> tags (simple approach)
        all_paragraphs = soup.find_all('p')
        content_parts = []
        
        for p in all_paragraphs:
            text = p.get_text(strip=True)
            # Filter out very short paragraphs (likely navigation/ads)
            if text and len(text) > 20:
                content_parts.append(text)
        
        content = '\n'.join(content_parts)
        
        # Date from URL
        date_published = extract_date_from_arabic_url(url)
        
        return {
            'date_published': date_published or 'Unknown',
            'title': title,
            'url': url,
            'text': content,
            'word_count': len(content.split()),
            'source': 'almasry_alyoum'
        }
        
    except Exception as e:
        print(f"  ❌ Error scraping Al-Masry article: {e}")
        return None


# ============================================================================
# MAIN ARABIC SCRAPER
# ============================================================================

def scrape_arabic_news(days_back=14, csv_file='data/arabic_news.csv'):
    """
    Main function: Scrape Arabic economic news from multiple sources
    """
    print(f"\n{'='*80}")
    print(f"📰 ARABIC NEWS SCRAPER - Youm7 & Al-Masry Al-Youm")
    print(f"{'='*80}\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load existing URLs
    existing_urls = set()
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            existing_urls = set(df['url'].tolist())
            print(f"📂 Loaded {len(existing_urls)} existing articles\n")
        except Exception as e:
            print(f"⚠️ Could not load existing CSV: {e}\n")
    
    # Scrape article URLs from both sources
    all_articles = []
    
    # Youm7
    print("🔍 Scraping Youm7 economy section...")
    for page in range(1, 6):  # 5 pages
        print(f"  Page {page}...", end=' ')
        articles = scrape_youm7_economy_page(page)
        all_articles.extend(articles)
        print(f"Found {len(articles)} articles")
        time.sleep(2)
    
    # Al-Masry Al-Youm
    print("\n🔍 Scraping Al-Masry Al-Youm economy section...")
    for page in range(1, 6):  # 5 pages
        print(f"  Page {page}...", end=' ')
        articles = scrape_almasry_economy_page(page)
        all_articles.extend(articles)
        print(f"Found {len(articles)} articles")
        time.sleep(2)
    
    print(f"\n✓ Found {len(all_articles)} total articles across both sites")
    
    # Filter new URLs
    new_articles = [a for a in all_articles if a['url'] not in existing_urls]
    print(f"✓ {len(new_articles)} new articles to process\n")
    
    if not new_articles:
        print("✨ No new articles found!\n")
        return
    
    # Scrape full content and analyze
    print(f"{'='*80}")
    print(f"🤖 Processing {len(new_articles)} articles...")
    print(f"{'='*80}\n")
    
    processed_articles = []
    filtered_count = 0
    
    for i, article_preview in enumerate(new_articles, 1):
        print(f"[{i}/{len(new_articles)}] Processing...")
        
        # Determine source and scrape accordingly
        url = article_preview['url']
        
        if 'youm7.com' in url:
            article = scrape_youm7_article(url)
        elif 'almasryalyoum.com' in url:
            article = scrape_almasry_article(url)
        else:
            print(f"  ⚠️ Unknown source: {url}")
            continue
        
        if article and article['text']:
            # Check if economic
            if is_economic_article_arabic(article['title'], article['text']):
                # Analyze sentiment
                sentiment_label, sentiment_score = analyze_sentiment_arabic(article['text'])
                
                article['sentiment_label'] = sentiment_label
                article['sentiment_score'] = sentiment_score
                article['scrape_time'] = datetime.now().isoformat()
                
                processed_articles.append(article)
                
                print(f"  ✓ {article['title'][:60]}...")
                print(f"    📅 {article['date_published']} | 📊 {sentiment_label} ({sentiment_score:.2f})")
            else:
                filtered_count += 1
                print(f"  ⊘ Filtered (not economic)")
        
        time.sleep(2)  # Rate limiting
    
    print(f"\n⚠️ Filtered out {filtered_count} non-economic articles")
    
    if not processed_articles:
        print("\n⚠️ No relevant articles found!\n")
        return
    
    # Save to CSV
    df_new = pd.DataFrame(processed_articles)
    
    if os.path.exists(csv_file):
        df_existing = pd.read_csv(csv_file)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(csv_file, index=False, encoding='utf-8')
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ ARABIC SCRAPER COMPLETE")
    print(f"   Added: {len(processed_articles)} articles")
    print(f"   Total in database: {len(df_combined)}")
    
    if processed_articles:
        avg_sentiment = sum(a['sentiment_score'] for a in processed_articles) / len(processed_articles)
        print(f"   Average sentiment: {avg_sentiment:.3f}")
    
    print(f"   Saved to: {csv_file}")
    print(f"{'='*80}\n")


# ============================================================================
# RUN WHEN CALLED DIRECTLY
# ============================================================================

if __name__ == "__main__":
    scrape_arabic_news(days_back=14)
