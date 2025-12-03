import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def get_soup(url):
    response = requests.get(url)
    response.raise_for_status()
    return BeautifulSoup(response.content, 'html.parser')

def extract_article_links(index_url, link_selector):
    soup = get_soup(index_url)

    links = []
    for article in soup.select(link_selector):
        href = article.get('href')
        if href:
            links.append(urljoin(index_url, href))

    return list(set(links))

def parse_article(article_url, text_selector, title_selector=None, date_selector=None):
    soup = get_soup(article_url)

    title = None
    if title_selector:
        tag = soup.select_one(title_selector)
        if tag:
            title = tag.get_text(strip=True)

    pub_date = None
    if date_selector:
        tag = soup.select_one(date_selector)
        if tag:
            pub_date = tag.get_text(strip=True)

    text_elements = soup.select(text_selector)
    text = '\n'.join(el.get_text(strip=True) for el in text_elements if el)

    media = {
        'img': [],
        'video': [],
        'audio': []
    }

    image_urls = set()

    for img in soup.find_all("img"):
        src = (
                img.get("src")
                or img.get("data-src")
                or img.get("data-original")
        )

        if not src:
            continue

        if "_next/image" in src:
            continue

        full_url = urljoin(article_url, src)

        if full_url.startswith("http"):
            image_urls.add(full_url)

    media["img"] = list(image_urls)

    video_urls = set()

    for vid in soup.find_all("video"):
        src = vid.get("src")
        if src and "sprite" not in src and not src.startswith("blob:") and "_next/" not in src:
            video_urls.add(urljoin(article_url, src))

        for source in vid.find_all("source"):
            src2 = source.get("src")

            if not src2:
                continue

            if "_next/" in src2 or src2.startswith("blob:"):
                continue

            video_urls.add(urljoin(article_url, src2))

    media["video"] = list(video_urls)

    audio_urls = set()

    for aud in soup.find_all("audio"):

        src = aud.get("src")
        if src and not src.startswith("blob:"):
            audio_urls.add(urljoin(article_url, src))

        for source in aud.find_all("source"):
            src2 = source.get("src")

            if not src2:
                continue

            if src2.startswith("blob:"):
                continue

            audio_urls.add(urljoin(article_url, src2))

    media["audio"] = list(audio_urls)

    return {
        "url": article_url,
        "title": title,
        "date": pub_date,
        "text": text,
        "media": media
    }

def scrap_site(base_url, link_selector, text_selector,
               title_selector=None, date_selector=None,
               max_pages=2):

    all_links = set()
    page = 1

    while page <= max_pages:
        index_url = f"{base_url}/page/{page}"

        try:
            soup = get_soup(index_url)

            page_links = []

            for article in soup.select(link_selector):
                href = article.get('href')
                if href:
                    page_links.append(urljoin(index_url, href))

            if not page_links:
                break

            new_links = set(page_links) - all_links
            print(f"✅ Found {len(page_links)} links ({len(new_links)} new)")

            if not new_links:
                break

            all_links.update(new_links)
            page += 1

        except Exception as e:
            print(f"❌ Error on page {page}: {e}")
            break

    articles = []

    for url in all_links:
        try:
            print(f"Scraping: {url}")
            article_data = parse_article(
                url,
                text_selector=text_selector,
                title_selector=title_selector,
                date_selector=date_selector
            )
            articles.append(article_data)
        except Exception as e:
            print(f"Error scraping {url}: {e}")

    return articles


