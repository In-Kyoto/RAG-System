from app.database.db import init_db, get_db
from app.database.models import Article
from sqlalchemy.exc import IntegrityError


def save_articles_to_db(articles, source_name="unknown"):
    init_db()

    db_gen = get_db()
    db = next(db_gen)

    saved = 0
    skipped = 0

    for art in articles:
        exists = db.query(Article).filter(Article.url == art["url"]).first()
        if exists:
            skipped += 1
            continue

        article = Article(
            url=art["url"],
            title=art.get("title"),
            date=art.get("date"),
            text=art.get("text", ""),
            media=art.get("media", {}),
            source=source_name
        )
        db.add(article)
        saved += 1

    db.commit()
    print(f"Збережено нових статей: {saved}, пропущено дублікатів: {skipped}")