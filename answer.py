import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from loader import load_all_documents
from chunker import chunk_text

# ===== 第 1 部分：准备 =====
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")   # 海选裁判
rerank_model = CrossEncoder("BAAI/bge-reranker-base")          # 终面裁判

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

llm = OpenAI(
    api_key="sk-0a015e865c06404cb7ef64bd4d4e2687",
    base_url="https://api.deepseek.com",
)

# ===== 第 2 部分：定义"提问"功能 =====
def ask(question):
    # 1. 海选：向量检索 10 个
    q_vector = embed_model.encode([question])
    results = collection.query(query_embeddings=q_vector.tolist(), n_results=10)
    candidates = results["documents"][0]
    candidate_metas = results["metadatas"][0]

    # 2. 终面：重排序，取前 3
    pairs = [(question, doc) for doc in candidates]
    scores = rerank_model.predict(pairs)
    ranked = sorted(zip(scores, candidates, candidate_metas), reverse=True)
    top3 = ranked[:3]

    # 3. 组装提示词
    context = "\n\n".join([doc for _, doc, _ in top3])   # context = 资料上下文（3 块拼起来）
    prompt = f"""你是公司知识库助手。请严格根据下面的资料回答问题。
如果资料里没有相关信息，请回答"知识库中没有相关信息"，不要编造。

【资料】
{context}

【问题】
{question}"""

    # 4. 发给大模型
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.choices[0].message.content

    # 5. 打印答案 + 来源
    sources = "、".join([meta["source"] for _, _, meta in top3])
    print(f"❓ 问题：{question}")
    print(f"🤖 回答：{answer}")
    print(f"📚 参考：{sources}\n")

# ===== 第 3 部分：测试两个问题 =====
ask("保温杯能保温多久")
ask("你们公司食堂在哪里")
