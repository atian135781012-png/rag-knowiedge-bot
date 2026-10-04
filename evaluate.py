import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder
from loader import load_all_documents
from chunker import chunk_text

# —— 准备（和之前一样）——
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

llm = OpenAI(api_key="sk-0a015e865c06404cb7ef64bd4d4e2687", base_url="https://api.deepseek.com")

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

# —— 批量测试 ——
with open("eval_questions.txt", "r", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

with open("eval_results.txt", "w", encoding="utf-8") as out:
    for i, q in enumerate(questions, 1):
        answer, sources = ask(q)
        out.write(f"===== 第 {i} 题 =====\n")
        out.write(f"问题：{q}\n")
        out.write(f"回答：{answer}\n")
        out.write(f"参考：{'、'.join(sources)}\n\n")
        print(f"✅ 第 {i}/{len(questions)} 题完成")

print("\n全部完成！打开 eval_results.txt 查看所有回答")
