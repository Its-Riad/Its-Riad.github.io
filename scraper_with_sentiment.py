import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import csv
import os
import re
from transformers import pipeline

# Load FinBERT once at startup
print("Loading FinBERT model...")
finbert = pipeline("sentiment-analysis", model="ProsusAI/finbert")

def is_inflation_related(title, content):
    """Check if article is actually about inflation"""
    inflation_keywords = [
        'inflation', 'inflat', 'price increase', 'price rise', 'cost of living',
        'purchasing power', 'consumer prices', 'cpi', 'core inflation',
        'headline inflation', 'price pressure', 'monetary policy', 'interest rate',
        'central bank', 'cbe', 'price index'
    ]
    
    text = (title + ' ' + content).lower()
    
    for keyword in inflation_keywords:
        if keyword in text:
            return True
    
    return False

def analyze_sentiment(text, max_length=512):
    """Analyze financial sentiment using FinBERT"""
    if not text or len(str(text).strip()) == 0:
        return 0.0
    
    text = str(text)[:max_length * 4]
    
    try:
        result = finbert(text)[0]
        label = result['label'].lower()
        score = result['score']
        
        if label == 'positive':
            return score
        elif label == 'negative':
            return -score
        else:
            return 0.0
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0.0

def scrape_article_details(url):
    """Scrape detailed information from a single article"""
    try:
        response = requests.get(url, timeout=30)
        soup = BeautifulSoup(response.text, 'lxml')
        
        content_div = soup.find('div', class_='entry-content')
        content = content_div.get_text(separator='\n', strip=True) if content_div else ""
        
        article_data = {
            'title': '',
            'url': url,
            'content': content,
            'author': '',
            'date_published': '',
            'categories': [],
            'word_count': len(content.split()),
            'sentiment_score': 0.0,
            'scraped_at': datetime.now().isoformat()
        }
        
        # Get title
        title_tag = soup.find('h1', class_='entry-title') or soup.find('h1')
        if title_tag:
            article_data['title'] = title_tag.get_text(strip=True)
        
        # Get author
        author_tag = soup.find('a', class_='author') or soup.find('span', class_='author')
        if author_tag:
            article_data['author'] = author_tag.get_text(strip=True)
        
        # FIXED DATE EXTRACTION
        date_published = None
        
        # Method 1: Meta tag (most reliable)
        meta_published = soup.find('meta', property='article:published_time')
        if meta_published and meta_published.get('content'):
            date_published = meta_published.get('content')[:10]
        
        # Method 2: Time tag with updated-date class
        if not date_published:
            updated_time = soup.find('time', class_='updated-date')
            if updated_time and updated_time.get('datetime'):
                date_published = updated_time.get('datetime')[:10]
        
        # Method 3: Extract from URL
        if not date_published:
            date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', url)
            if date_match:
                year, month, day = date_match.groups()
                date_published = f"{year}-{month}-{day}"
        
        article_data['date_published'] = date_published or 'Unknown'
        
        # Get categories
        category_tags = soup.find_all('a', rel='category tag')
        article_data['categories'] = [cat.get_text(strip=True) for cat in category_tags]
        
        # Analyze sentiment
        article_data['sentiment_score'] = analyze_sentiment(content)
        
        return article_data
        
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def load_existing_urls(csv_file):
    """Load URLs from existing CSV to avoid duplicates"""
    existing_urls = set()
    if os.path.exists(csv_file):
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_urls = {row['url'] for row in reader}
            print(f"📂 Loaded {len(existing_urls)} existing articles")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print("📂 No existing CSV found. Starting fresh.")
    return existing_urls

def get_article_urls(search_term, max_pages=3):
    """Get article URLs from multiple search pages"""
    all_urls = []
    
    for page in range(1, max_pages + 1):
        if page == 1:
            url = f'https://www.dailynewsegypt.com/?s={search_term}'
        else:
            url = f'https://www.dailynewsegypt.com/page/{page}/?s={search_term}'
        
        print(f"📄 Fetching page {page}...")
        
        try:
            response = requests.get(url, timeout=30)
            soup = BeautifulSoup(response.text, 'lxml')
            
            titles = soup.find_all(class_='entry-title')
            
            if not titles:
                print(f"   No articles found. Stopping.")
                break
            
            for title in titles:
                link = title.find('a')
                if link and link.get('href'):
                    all_urls.append(link['href'])
            
            print(f"   Found {len(titles)} articles")
            time.sleep(2)
            
        except Exception as e:
            print(f"   Error: {e}")
            break
    
    return all_urls

def scrape_and_update(search_term='inflation', csv_file='data/egyptian_news.csv', max_pages=3):
    """Main function: scrape new articles with sentiment and append to CSV"""
    
    print(f"\n{'='*80}")
    print(f"🚀 Starting scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Load existing URLs
    existing_urls = load_existing_urls(csv_file)
    
    # Get all article URLs
    print(f"\n🔍 Searching for '{search_term}' articles...")
    all_urls = get_article_urls(search_term, max_pages)
    print(f"\n✓ Found {len(all_urls)} total articles")
    
    # Filter out already scraped
    new_urls = [url for url in all_urls if url not in existing_urls]
    print(f"✓ {len(new_urls)} new articles to process")
    
    if not new_urls:
        print("\n✨ No new articles found!")
        return
    
    print(f"\n{'='*80}")
    print(f"📝 Scraping and analyzing {len(new_urls)} articles...")
    print(f"{'='*80}\n")
    
    # Scrape new articles
    new_articles = []
    filtered_out = 0
    
    for i, url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] Processing...")
        article_data = scrape_article_details(url)
        
        if article_data:
            # Check if inflation-related
            if is_inflation_related(article_data['title'], article_data['content']):
                new_articles.append(article_data)
                print(f"  ✓ {article_data['title'][:60]}...")
                print(f"    📅 Date: {article_data['date_published']} | 📊 Sentiment: {article_data['sentiment_score']:.3f}")
            else:
                filtered_out += 1
                print(f"  ⊗ FILTERED (not inflation-related)")
        
        time.sleep(2)
    
    print(f"\n⚠️  Filtered out {filtered_out} non-inflation articles")
    
    if not new_articles:
        print("\n⚠️  No relevant inflation articles found!")
        return
    
    # Save to CSV
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['title', 'url', 'author', 'date_published', 'categories', 
                      'word_count', 'sentiment_score', 'content', 'scraped_at']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for article in new_articles:
            article_copy = article.copy()
            article_copy['categories'] = ', '.join(article['categories'])
            writer.writerow(article_copy)
    
    print(f"\n{'='*80}")
    print(f"✅ SUCCESS!")
    print(f"   Added: {len(new_articles)} articles")
    print(f"   Filtered: {filtered_out} articles")
    print(f"   Total in database: {len(existing_urls) + len(new_articles)}")
    
    # Calculate average sentiment
    if new_articles:
        avg_sentiment = sum(a['sentiment_score'] for a in new_articles) / len(new_articles)
        print(f"   Average sentiment: {avg_sentiment:.3f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    scrape_and_update(
        search_term='inflation',
        csv_file='data/egyptian_news.csv',
        max_pages=3
    )
