from pathlib import Path
from pypdf import PdfReader
from docx import Document

def read_txt(file_path):
    """读取 TXT 文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path):
    """读取 PDF：逐页提取文字，再拼成一整段"""
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    return "\n".join(pages)

def read_docx(file_path):
    """读取 Word：逐个段落提取文字"""
    doc = Document(file_path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

def load_all_documents(folder="docs"):
    """遍历文件夹，按后缀名调用对应的读取函数"""
    folder = Path(folder)
    documents = []

    for file_path in folder.iterdir():
        suffix = file_path.suffix.lower()

        if suffix == ".txt":
            text = read_txt(file_path)
        elif suffix == ".pdf":
            text = read_pdf(file_path)
        elif suffix == ".docx":
            text = read_docx(file_path)
        else:
            print(f"跳过不支持的文件：{file_path.name}")
            continue

        documents.append({
            "source": file_path.name,   # 记录这段文字来自哪个文件
            "content": text
        })
        print(f"✅ 已读取：{file_path.name}（{len(text)} 个字符）")

    return documents

if __name__ == "__main__":
    docs = load_all_documents()
    print(f"\n一共读取了 {len(docs)} 个文件")

    # 打印第一个文件的前 200 个字，检查读取是否正常
    if docs:
        print("\n----- 第一个文件的开头 -----")
        print(docs[0]["content"][:200])
