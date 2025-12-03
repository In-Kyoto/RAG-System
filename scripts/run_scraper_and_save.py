from app.ingestion.scraper import scrap_site
from app.ingestion.save_to_db import save_articles_to_db
import time

if __name__ == "__main__":
    index_url = "https://www.deeplearning.ai/the-batch"
    link_selector = "article[data-sentry-component=\'PostCard\'] a[href^=\'/the-batch/issue\']"

    text_selector = """
        div.postContent.wgtc p.overflow, 
        div.postContent.wgtc h2, 
        div.postContent.wgtc h3, 
        div.postContent.wgtc li, 
        article p
    """.replace("\n", "").strip()
    title_selector = "h1, h2:first-of-type"
    date_selector = "time, .date, span[datetime]"

    articles = scrap_site(
        base_url=index_url,
        link_selector=link_selector,
        text_selector=text_selector,
        title_selector=title_selector,
        date_selector=date_selector,
        max_pages=23
    )

    articles = articles[:50]

    save_articles_to_db(articles, source_name="The Batch (deeplearning.ai)")

    time.sleep(3)