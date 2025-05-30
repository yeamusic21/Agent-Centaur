import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
from newspaper import Article
import time

def google_news_search(query, start_year=2022, num_articles=10):
    search_url = f"https://www.google.com/search?q={query}+after:{start_year}-01-01&hl=en&tbm=nws"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }
    response = requests.get(search_url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    # get links from parsed google results 
    links = []
    for a_tag in soup.find_all('a'):
        # get href from anchor tag
        href = a_tag.get('href')
        print("href: ", href)
        # only add quality hrefs to the 'links' list
        if href and href.startswith('/url?q='):
            real_url = href.split('/url?q=')[1].split('&')[0] # cutting at &, cuts off any dynamic ids that have been added
            real_url = unquote(real_url)
            print("real_url: ", real_url)
            # Basic filtering
            if real_url.startswith('http') and 'google.com' not in real_url:
                links.append(real_url)
        # stop if num_articles achieved
        if len(links) >= num_articles:
            break
    # return links
    return links

def scrape_articles(urls):
    # given a news article, returns text and other metadata
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


def search_and_scrape(query="Coursera"):
    urls = google_news_search(query, start_year=2022, num_articles=10)
    articles = scrape_articles(urls)
    # view an article
    print(f"\n📰 {articles[0]['title']}\n🔗 {articles[0]['url']}\n📄 {articles[0]['text'][:500]}...\n")
    return articles

if __name__=="__main__":
    articles = search_and_scrape()
    print("==========================================================")
    for article in articles:
        print(f"\n📰 {article['title']}\n🔗 {article['url']}\n📄 {article['text'][:500]}...\n")
