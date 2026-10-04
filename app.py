import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import streamlit as st
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from loader import load_all_documents
from chunker import chunk_text

st.set_page_config(page_title="公司知识库问答机器人", page_icon="🤖")

@st.cache_resource    # 缓存：模型和数据库只准备一次
def init():
    embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
    rerank_model = CrossEncoder("BAAI/bge-reranker-base")

    docs = load_all_documents()
    ids, texts, metas = [], [], []
    for doc in docs:
        for i, piece in enumerate(chunk_text(doc["content"])):
            ids.append(f"{doc['source']}-{i}")
            texts.append(piece)
            metas.append({"source": doc["source"], "chunk_id": i})
    vectors = embed_model.encode(texts)

    client_db = chromadb.PersistentClient(path="./chroma_db")
    names = [c.name for c in client_db.list_collections()]
    if "company_docs" in names:
        client_db.delete_collection("company_docs")
    collection = client_db.create_collection("company_docs")
    collection.add(ids=ids, embeddings=vectors.tolist(), documents=texts, metadatas=metas)

    llm = OpenAI(api_key="sk-0a015e865c06404cb7ef64bd4d4e2687" \
    "", base_url="https://api.deepseek.com")
    return embed_model, rerank_model, collection, llm

embed_model, rerank_model, collection, llm = init()

def ask(question):
    q_vector = embed_model.encode([question])
    results = collection.query(query_embeddings=q_vector.tolist(), n_results=10)
    candidates = results["documents"][0]
    candidate_metas = results["metadatas"][0]

    pairs = [(question, doc) for doc in candidates]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(scores, candidates, candidate_metas), reverse=True)
    top3 = ranked[:3]

    context = "\n\n".join([doc for _, doc, _ in top3])
    prompt = f"""你是公司知识库助手。请严格根据下面的资料回答问题。
如果资料里没有相关信息，请回答"知识库中没有相关信息"，不要编造。

【资料】
{context}

【问题】
{question}"""

    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content, [m["source"] for _, _, m in top3]

# ===== 网页部分 =====
st.title("🤖 公司知识库问答机器人")

if "history" not in st.session_state:          # 第一次打开：给机器人"开记忆"
    st.session_state.history = []

for role, content in st.session_state.history:  # 把历史记录画到屏幕上
    with st.chat_message(role):
        st.markdown(content)

question = st.chat_input("请输入你的问题……")    # 屏幕下方的输入框
if question:
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并思考……"):  # 转圈等待动画
            answer, sources = ask(question)
        st.markdown(answer)
        st.caption("📚 参考：" + "、".join(sources))   # caption = 灰色小字

    st.session_state.history.append(("assistant", answer))
