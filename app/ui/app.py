import streamlit as st
import lancedb
from sentence_transformers import SentenceTransformer
import ollama
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(page_title="The Batch RAG", layout="wide")
st.title("Multimodal RAG — The Batch (Andrew Ng)")
st.markdown("### 330+ випусків • Пошук по тексту + картинки • Llama 3.1 офлайн")

@st.cache_resource
def get_db_and_model():
    db = lancedb.connect('/Users/in_kyoto/Documents/Self Harm/RagSys/data/lancedb')

    print(db.table_names())
    table = db.open_table("the_batch")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return table, model

table, model = get_db_and_model()

query = st.text_input("Твоє питання (укр/англ)", placeholder="Grok-1.5 Vision, Llama 3.1, Claude 3.5", key="query")

if st.button("Пошук") and query:
    with st.spinner("Шукаємо..."):
        query_vec = model.encode(query).tolist()
        results = table.search(query_vec).limit(8).to_list()

        context = ""
        all_images = []
        for r in results:
            context += f"### {r['title']}\n{r['text']}\n\n"
            all_images.extend(r['images'])

        response = ollama.generate(
            model="llama3.1:8b",
            prompt=f"Дай коротку зрозумілу відповідь українською, тільки за цим контекстом:\n\n{context}\n\nПитання: {query}"
        )
        answer = response["response"]

        st.success("**Відповідь від Llama 3.1**")
        st.write(answer)

        st.write("**Джерела та зображення**")
        cols = st.columns(3)
        for i, img_url in enumerate(all_images):
            if i >= 9: break
            try:
                response = requests.get(img_url, timeout=10)
                img = Image.open(BytesIO(response.content))
                cols[i % 3].image(img, use_container_width=True, caption=f"з {results[i//3]['title']}")
            except:
                cols[i % 3].image(img_url, use_column_width=True)

        st.write("**Статті**")
        for r in results[:5]:
            st.markdown(f"**[{r['title']}]({r['url']})** — {r['date']}")
            with st.expander("Читати фрагмент"):
                st.write(r['text'][:1000] + "...")