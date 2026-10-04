import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"   # 模型下载走国内镜像，否则会非常慢

from sentence_transformers import SentenceTransformer  # SentenceTransformer = 句子转换器（文字变数字的工具）
from loader import load_all_documents
from chunker import chunk_text

# 加载模型（第一次运行要下载约 100MB，耐心等）
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

docs = load_all_documents()

texts = []   # texts = 所有块的文字
metas = []   # metas = 所有块的来源信息
for doc in docs:
    for i, piece in enumerate(chunk_text(doc["content"])):
        texts.append(piece)
        metas.append({"source": doc["source"], "chunk_id": i})

print(f"共有 {len(texts)} 块，开始变成向量……")

vectors = model.encode(texts)   # encode = 编码，把 6 段文字变成 6 串数字

print(f"向量的形状：{vectors.shape}")       # shape = 形状，(6, 512) 表示 6 块、每块 512 个数字
print(f"第 1 块向量开头的 5 个数字：{vectors[0][:5]}")
