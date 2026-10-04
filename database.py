import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb                                    # chromadb = 向量数据库
from sentence_transformers import SentenceTransformer
from loader import load_all_documents
from chunker import chunk_text

model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

# 第 1 步：读文档、切块、编号
docs = load_all_documents()
ids, texts, metas = [], [], []
for doc in docs:
    for i, piece in enumerate(chunk_text(doc["content"])):
        ids.append(f"{doc['source']}-{i}")        # 唯一编号：文件名-第几块
        texts.append(piece)
        metas.append({"source": doc["source"], "chunk_id": i})

# 第 2 步：全部变成向量
vectors = model.encode(texts)

# 第 3 步：存进 Chroma
client = chromadb.PersistentClient(path="./chroma_db")   # 仓库建在项目文件夹里

names = [c.name for c in client.list_collections()]      # 看看仓库里已有哪些书架
if "company_docs" in names:
    client.delete_collection("company_docs")             # 删掉旧书架（防止重复存报错）

collection = client.create_collection("company_docs")    # 建新书架

collection.add(
    ids=ids,                        # 每块的编号
    embeddings=vectors.tolist(),    # 向量（转成列表格式才能存）
    documents=texts,                # 每块的原始文字（以后取答案要用）
    metadatas=metas                 # 来源信息（哪个文件、第几块）
)
print(f"✅ 已存入 {collection.count()} 块到数据库")

# 第 4 步：测试——随便问一个问题
question = "保温杯能保温多久"
q_vector = model.encode([question])                 # 把问题也变成向量

results = collection.query(
    query_embeddings=q_vector.tolist(),             # 带着问题向量去查
    n_results=2                                      # 找最接近的前 2 名
)

print(f"\n❓ 问题：{question}")
for i in range(len(results["ids"][0])):
    print(f"\n🏅 第 {i+1} 名：{results['ids'][0][i]}（距离 {results['distances'][0][i]:.4f}）")
    print(results["documents"][0][i][:100], "……")
