import requests
from bs4 import BeautifulSoup
from newspaper import Article
import time

def google_news_search(query, start_year=2022, num_articles=10):
    search_url = f"https://www.google.com/search?q=site:news.google.com+{query}+after:{start_year}-01-01&hl=en&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    links = []
    for a in soup.select("a"):
        href = a.get("href")
        if href and "url?q=" in href:
            url = href.split("url?q=")[-1].split("&")[0]
            if url not in links:
                links.append(url)
        if len(links) >= num_articles:
            break
    return links

def scrape_articles(urls):
    articles = []
    for url in urls:
        try:
            article = Article(url)
            article.download()
            article.parse()
            articles.append({
                "title": article.title,
                "url": url,
                "text": article.text
            })
            time.sleep(1)  # Be polite
        except Exception as e:
            print(f"Failed to scrape {url}: {e}")
    return articles

def search_and_scrape(query="Coursera")
    urls = google_news_search(query, start_year=2022, num_articles=10)
    articles = scrape_articles(urls)
    # view an article
    print(f"\n📰 {article[0]['title']}\n🔗 {article[0]['url']}\n📄 {article[0]['text'][:500]}...\n")
    return articles
