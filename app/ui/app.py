import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import ollama
from PIL import Image
import requests
from io import BytesIO

API_KEY = st.secrets["GROK_API_KEY"]


st.set_page_config(page_title="RAG System", layout="wide")
st.title("Multimodal RAG")

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
    db = lancedb.connect('./data/lancedb')

    print(db.table_names())
    table = db.open_table("the_batch")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return table, model

table, model = get_db_and_model()

query = st.chat_input("Твоє питання", key="query")

if query:
    with st.spinner("Шукаємо...") and st.chat_message('user'):
        query_vec = model.encode(query).tolist()
        results = table.search(query_vec).limit(4).to_list()

        filtered_results = [r for r in results if r.get('_distance', 1.0) < 0.7]

        if not filtered_results:
            filtered_results = results[:3]
        results = filtered_results

        context = ""
        all_images = []
        for r in results:
            context += f"### {r['title']}\n{r['text']}\n\n"
            all_images.extend(r['images'][:2])

        answer = get_answer_from_llama(query, context)

        st.write(answer)

        st.write("**Джерела та зображення**")
        cols = st.columns(3)
        for i, img_url in enumerate(all_images):
            if i >= 6: break
            try:
                response = requests.get(img_url, timeout=5)
                img = Image.open(BytesIO(response.content))

                result_idx = i // 2
                cols[i % 3].image(img, use_container_width=True, caption=f"з '{results[result_idx]['title'][:50]}...'")
            except:
                cols[i % 3].image(img_url, use_column_width=True)

        st.write("**Статті**")
        for r in results[:3]:
            st.markdown(f"**[{r['title']}]({r['url']})** — {r['date']}")
            with st.expander("Читати фрагмент"):
                st.write(r['text'][:500] + "..." if len(r['text']) > 500 else r['text'])