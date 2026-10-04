import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from loader import load_all_documents
from chunker import chunk_text

# 两个模型分工不同：
embed_model = SentenceTransformer("BAAI/bge-small-zh-v1.5")   # 嵌入模型：快，负责海选
rerank_model = CrossEncoder("BAAI/bge-reranker-base")          # 重排序模型：慢，负责精选

# —— 建库（和第 4 关一样）——
docs = load_all_documents()
ids, texts, metas = [], [], []
for doc in docs:
    for i, piece in enumerate(chunk_text(doc["content"])):
        ids.append(f"{doc['source']}-{i}")
        texts.append(piece)
        metas.append({"source": doc["source"], "chunk_id": i})
vectors = embed_model.encode(texts)

client = chromadb.PersistentClient(path="./chroma_db")
names = [c.name for c in client.list_collections()]
if "company_docs" in names:
    client.delete_collection("company_docs")
collection = client.create_collection("company_docs")
collection.add(ids=ids, embeddings=vectors.tolist(), documents=texts, metadatas=metas)

# —— 用户提问 ——
question = "保温杯能保温多久"
q_vector = embed_model.encode([question])

# 第 1 步：向量检索，海选 10 个（我们只有 6 块，相当于全选，流程一样）
results = collection.query(query_embeddings=q_vector.tolist(), n_results=10)
candidates = results["documents"][0]          # 候选 = 海选出来的文档们
candidate_metas = results["metadatas"][0]

# 第 2 步：重排序，把"问题+每块"配成对，重新打分
pairs = [(question, doc) for doc in candidates]   # pairs = 配对们
scores = rerank_model.predict(pairs)              # predict = 预测，给每对打一个分数

# 第 3 步：按新分数从大到小排序
ranked = sorted(zip(scores, candidates, candidate_metas), reverse=True)

print(f"❓ 问题：{question}\n")
print("重排序后的前 3 名：")
for i, (score, doc, meta) in enumerate(ranked[:3]):
    print(f"\n🏅 第 {i+1} 名：{meta['source']}（相关分数 {score:.4f}）")
    print(doc[:100], "……")
