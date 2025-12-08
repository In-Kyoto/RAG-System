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

        for i, r in enumerate(results, 1):
            title = r.get("title", "Без назви")
            date = r.get("date", "невідомо")
            text = r.get("text", "")[:1100] + ("..." if len(r.get("text", "")) > 1100 else "")

            context_parts.append(f"Стаття {i}: {title} ({date})\n{text}\n")

        context_parts.append(f"\nПитання ще раз: {query}")
        context = "\n".join(context_parts)

        answer = get_answer_from_llama(query, context)

    st.chat_message("user").write(query)
    st.chat_message("assistant").write(answer)


    if got_result:
        if images:
            st.subheader("🖼 Зображення зі статті")
            cols = st.columns(len(images))

            for i, img_url in enumerate(images):
                try:
                    response = requests.get(img_url, timeout=5)
                    img = Image.open(BytesIO(response.content))
                    cols[i].image(img, use_container_width=True, caption=title[:40])
                except:
                    cols[i].image(img_url, use_container_width=True)

        st.subheader("📄 Джерело відповіді")
        st.markdown(f"**[{title}]({url})**")

        with st.expander("📖 Читати фрагмент статті"):
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

    got_result = True
