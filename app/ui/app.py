import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import requests
from PIL import Image
from io import BytesIO

API_KEY = st.secrets["GROQ_API_KEY"]
got_result = True

st.set_page_config(page_title="Multimodal RAG", layout="wide")
st.title("Multimodal RAG System")


def get_answer_from_llama(query, context):
    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {"role": "system", "content": "Дай коротку зрозумілу відповідь українською мовою, максимум 250 слів."},
            {"role": "user", "content": f"Контекст:\n{context}\n\nПитання: {query}"}
        ],


        "temperature": 0.7,
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=payload, timeout=25)

    return response.json()["choices"][0]["message"]["content"]


@st.cache_resource
def get_db_and_model():
    db = lancedb.connect("./data/lancedb")
    table = db.open_table("the_batch")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    return table, model


table, model = get_db_and_model()

query = st.chat_input("Введіть питання...")

if query:
    with st.spinner("🔍 Шукаю релевантну статтю..."):
        query_vec = model.encode(query).tolist()

        results = table.search(query_vec).limit(8).to_list()
        # results = [r for r in results if r["_distance"] < 0.85]
        # results = sorted(results, key=lambda x: x["_distance"])[:5]
        results = results[:5]

        if not results:
            st.error("Нічого не знайдено")
            st.stop()

        context_parts = [f"Питання: {query}\n"]
        all_images = []

        for i, r in enumerate(results, 1):
            title = r.get("title", "Без назви")
            date = r.get("date", "невідомо")
            text = r.get("text", "")[:1100] + ("..." if len(r.get("text", "")) > 1100 else "")
            images = r.get("images", [])[:2]

            context_parts.append(f"Стаття {i}: {title} ({date})\n{text}\n")
            all_images.extend(images)

        context_parts.append(f"\nПитання ще раз: {query}")
        context = "\n".join(context_parts)

        all_images = all_images[:3]

        answer = get_answer_from_llama(query, context)

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer)


    if got_result:
        if all_images:
            st.subheader("Зображення з релевантних статей")
            cols = st.columns(3)
            for idx, img_url in enumerate(all_images):
                col = cols[idx % 3]
                try:
                    response = requests.get(img_url, timeout=8)
                    if response.status_code == 200:
                        img = Image.open(BytesIO(response.content))
                        col.image(img, use_container_width=True)
                    else:
                        col.image("https://via.placeholder.com/400x300?text=Зображення+недоступне",
                                  use_container_width=True)
                except:
                    col.image("https://via.placeholder.com/400x300?text=Помилка+завантаження", use_container_width=True)

        st.subheader("Джерела")
        for idx, r in enumerate(results, 1):
            title = r.get("title", "Без назви")
            url = r.get("url", "#")
            date = r.get("date", "невідомо")
            score = round(1 - r["_distance"], 3)

            with st.expander(f"{idx}. {title} — {date} (релевантність: {score:.1%})"):
                st.markdown(f"**Посилання:** [{title}]({url})")
                st.caption(f"Релевантність: {score:.1%} | Дата: {date}")
                preview = r.get("text", "")[:600]
                if len(r.get("text", "")) > 600:
                    preview += "..."
                st.write(preview)

    got_result = True
