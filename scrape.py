import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import csv
import os

def scrape_article_details(url):
    """Scrape detailed information from a single article"""
    try:
        response = requests.get(url)
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
            'scraped_at': datetime.now().isoformat()
        }
        
        title_tag = soup.find('h1', class_='entry-title') or soup.find('h1')
        if title_tag:
            article_data['title'] = title_tag.get_text(strip=True)
        
        author_tag = soup.find('a', class_='author') or soup.find('span', class_='author')
        if author_tag:
            article_data['author'] = author_tag.get_text(strip=True)
        
        time_tag = soup.find('time')
        if time_tag:
            article_data['date_published'] = time_tag.get('datetime') or time_tag.get_text(strip=True)
        
        category_tags = soup.find_all('a', rel='category tag')
        article_data['categories'] = [cat.get_text(strip=True) for cat in category_tags]
        
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
            print(f"Loaded {len(existing_urls)} existing articles")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    return existing_urls

def get_article_urls(search_term, max_pages=3):
    """Get article URLs from multiple search pages"""
    all_urls = []
    
    for page in range(1, max_pages + 1):
        if page == 1:
            url = f'https://www.dailynewsegypt.com/?s={search_term}'
        else:
            url = f'https://www.dailynewsegypt.com/page/{page}/?s={search_term}'
        
        print(f"Fetching page {page}...")
        
        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'lxml')
            
            titles = soup.find_all(class_='entry-title')
            
            if not titles:
                break
            
            for title in titles:
                link = title.find('a')
                if link and link.get('href'):
                    all_urls.append(link['href'])
            
            print(f"  Found {len(titles)} articles")
            time.sleep(1)
            
        except Exception as e:
            print(f"  Error: {e}")
            break
    
    return all_urls

def scrape_and_update(search_term='inflation', csv_file='data/egyptian_news.csv', max_pages=3):
    """Main function: scrape new articles and append to CSV"""
    
    print(f"\nStarting scraper at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Load existing URLs
    existing_urls = load_existing_urls(csv_file)
    
    # Get all article URLs
    print(f"\nSearching for '{search_term}' articles...")
    all_urls = get_article_urls(search_term, max_pages)
    print(f"Found {len(all_urls)} total articles")
    
    # Filter out already scraped
    new_urls = [url for url in all_urls if url not in existing_urls]
    print(f"{len(new_urls)} new articles to scrape\n")
    
    if not new_urls:
        print("No new articles found!")
        return
    
    # Scrape new articles
    new_articles = []
    for i, url in enumerate(new_urls, 1):
        print(f"[{i}/{len(new_urls)}] Scraping {url[:60]}...")
        article_data = scrape_article_details(url)
        
        if article_data:
            new_articles.append(article_data)
        
        time.sleep(2)  # Be extra polite for automated scraping
    
    # Save to CSV
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', encoding='utf-8', newline='') as f:
        fieldnames = ['title', 'url', 'author', 'date_published', 'categories', 'word_count', 'content', 'scraped_at']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        for article in new_articles:
            article_copy = article.copy()
            article_copy['categories'] = ', '.join(article['categories'])
            writer.writerow(article_copy)
    
    print(f"\n{'='*80}")
    print(f"SUCCESS! Added {len(new_articles)} new articles")
    print(f"Total articles: {len(existing_urls) + len(new_articles)}")
    print(f"{'='*80}")

if __name__ == "__main__":
    scrape_and_update(search_term='inflation', csv_file='data/egyptian_news.csv', max_pages=3)
