from loader import load_all_documents   # 从 loader.py 借用读取功能

def chunk_text(text, chunk_size=500, overlap=50):
    # chunk_text = 文字切块工具
    # chunk_size = 每块多大（500 字）
    # overlap    = 重叠多少字（50 字）
    chunks = []          # chunks = 切好的块们
    start = 0            # start = 当前从第几个字开始切

    while start < len(text):
        end = start + chunk_size              # end = 切到哪
        piece = text[start:end]               # piece = 这一小块
        chunks.append(piece)                  # 放进盒子里

        if end >= len(text):                  # 已经切到文章末尾了
            break
        start = end - overlap                 # 下一块回退 50 字（这就是"重叠"）

    return chunks

if __name__ == "__main__":
    docs = load_all_documents()               # 先读取所有文档
    all_chunks = []                           # all_chunks = 所有块的大盒子

    for doc in docs:
        pieces = chunk_text(doc["content"])   # 把这篇文档切块
        for i, piece in enumerate(pieces):    # enumerate = 顺便给每块编号
            all_chunks.append({
                "source": doc["source"],      # 这块来自哪个文件
                "chunk_id": i,                # 是这篇文档的第几块
                "content": piece              # 块的内容
            })
        print(f"📄 {doc['source']} → 切成 {len(pieces)} 块")

    print(f"\n总共得到 {len(all_chunks)} 块")

    if all_chunks:
        print("\n----- 第 1 块的内容 -----")
        print(all_chunks[0]["content"])
