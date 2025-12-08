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
    model = "llama-3.1-8b-instant"
    max_tokens = 512

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": """
                Ти — корисний асистент-експерт з українських новин.
                Відповідай ТІЛЬКИ українською мовою.
                
                Правила:
                1. Дай коротку, зрозумілу і змістовну відповідь (максимум 150–200 слів).
                2. Використовуй тільки інформацію з наданого нижче контексту.
                3. Якщо в контексті немає прямої відповіді на питання або інформація відсутня/нечітка — обов’язково напиши:
                   «У статтях немає інформації про це.»
                   і більше нічого не додавай.
                4. Ніколи не вигадуй і не домислюй факти.
                """
            },
            {"role": "user", "content": f"Контекст:\n{context}\n\nПитання: {query}"}

        ],
        "temperature": 0.6,
        "max_tokens": max_tokens
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers, timeout=20)

        if response.status_code != 200:
            return f"Groq помилка {response.status_code}: {response.text[:200]}"

        data = response.json()

        if "choices" in data and data["choices"]:
            return data["choices"][0]["message"]["content"].strip()

        return "Невідомий формат відповіді"

    except Exception as e:
        return f"Помилка: {str(e)}"


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

        results = table.search(query_vec).limit(5).to_list()

        if not results:
            st.error("Нічого не знайдено")
            st.stop()


        best_article = min(results, key=lambda x: x.get("_distance", 1))

        title = best_article.get("title", "Без назви")
        text = best_article.get("text", "")
        images = best_article.get("images", [])[:3]
        url = best_article.get("url", "#")
        date = best_article.get("date", "Невідома дата")
        score = round(1 - best_article.get("_distance", 1), 3)

        context = f"""
            Назва: {title}
            Дата: {date}

            Текст статті:
            {text}
            """

        answer = get_answer_from_llama(query, context)

        if "немає інформації" in answer.lower():
            got_result = False

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
        st.write(f"Дата: {date}")
        st.write(f"Релевантність: {score * 100}%")

        with st.expander("📖 Читати фрагмент статті"):
            st.write(text[:1000] + "..." if len(text) > 1000 else text)

    images = []
    got_result = True
    context = ""
    best_article = None
    results = []
