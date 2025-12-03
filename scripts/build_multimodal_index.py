import lancedb
from sentence_transformers import SentenceTransformer
from app.database.db import get_db
from app.database.models import Article
from tqdm import tqdm

db = lancedb.connect("../data/lancedb")
model = SentenceTransformer("all-MiniLM-L6-v2")

table_name = "the_batch"
if table_name in db.table_names():
    db.drop_table(table_name)

data = []
articles = get_db().__next__().query(Article).all()

for art in tqdm(articles, desc="Індексуємо"):
    text = f"{art.title or ''}\n{art.text}"[:30000]
    vector = model.encode(text).tolist()

    data.append({
        "id": art.id,
        "vector": vector,
        "title": art.title or "Без назви",
        "date": art.date or "",
        "url": art.url,
        "text": art.text[:5000],
        "images": art.media.get("img", [])[:6]
    })

table = db.create_table(table_name, data=data)
print(f"Готово! LanceDB таблиця '{table_name}' створена з {len(data)} статтями")