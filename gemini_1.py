# rag_with_gemini.py
from google import genai
import json
import numpy as np
import faiss
from typing import List
import time
from dotenv import load_dotenv
import os

load_dotenv()
MAX_CHARS = 2000
api_key = os.getenv("GENAI_API_KEY")
def truncate_prompt(prompt: str):
    if len(prompt) > MAX_CHARS:
        return prompt[:MAX_CHARS] + "..."
    return prompt

#Client
client = genai.Client(api_key = api_key)  

# Data
with open("data.json", "r", encoding="utf-8") as f:
    docs = json.load(f)  # list of {id, dialect, text, norm, ...}

with open("law_data.json", "r", encoding="utf-8") as f:
    laws = json.load(f)  # list of {Source, text}

# padding
for law in laws:
    law["dialect"] = law["Source"]  
    law["meaning"] = ""             
    law["is_law"] = True

# padding
for d in docs:
    d["is_law"] = False

# 
all_data = docs + laws

texts = []
for d in all_data:
    if d.get("is_law", False):
        # Embed
        texts.append(f"Law Source: {d['Source']} ||| Text: {d['text']} ||| Type: {d.get('Type','')}")
    else:
        # Embed
        texts.append(f"Dialect: {d['dialect']} ||| Text: {d['text']} ||| Meaning: {d.get('meaning','')}")


# 3) embeddings (batch)
def embed_texts(text_list: List[str], batch_size=100):
    embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i+batch_size]
        res = client.models.embed_content(
            model="models/embedding-001",
            contents=batch
        )
        embeddings.extend([e.values for e in res.embeddings])
    return embeddings

embs = embed_texts(texts)

# 4) build FAISS index (L2 / cosine -> normalize for cosine)
d = len(embs[0])
arr = np.array(embs).astype("float32")
# normalize for cosine similarity
faiss.normalize_L2(arr)
index = faiss.IndexFlatIP(d)  # inner product on normalized vectors = cosine
index.add(arr)

# keep mapping id -> original doc
id_map = {i: all_data[i] for i in range(len(all_data))}

# 5) query + retrieval
def retrieve(query: str, k=3):
    q_emb = embed_texts([query])[0]
    q = np.array([q_emb]).astype("float32")
    faiss.normalize_L2(q)
    D, I = index.search(q, k)
    hits = [id_map[idx] for idx in I[0] if idx != -1]
    return hits

def print_typing_effect(text, delay=0.03):
    for ch in text:
        print(ch, end="", flush=True)
        time.sleep(delay)

# 6) call LLM with retrieved context
def answer_with_context(user_q: str):
    try:
        hits = retrieve(user_q, k=10)
        if not hits:
            print("Xin lỗi, mình chưa có dữ liệu để trả lời câu hỏi này.")
            return
        
        context = "\n\n".join(
            [
                (
                    f"Source: {h['Source']}\nText: {h['text']}\nType: {h.get('Type','')}"
                    if h.get("is_law")
                    else f"Dialect: {h['dialect']}\nText: {h['text']}\nMeaning: {h.get('meaning','')}"
                )
                for h in hits
            ]
        )

        user_prompt = f"""
        Bạn là trợ lý AI chuyên về pháp luật Việt Nam.
        Người dùng hỏi: {user_q}

        Thông tin được cung cấp (lấy từ cơ sở dữ liệu, đây là nguồn thông tin giúp bạn trả lời câu hỏi):
        {context}

        Yêu cầu:
        
        2. Nếu người dùng hỏi về tín đúng đắn hay sai trái của luật thì đưa ra các cách kiểm tra cho người dùng. Không trả lời 1 cách chắc chắn
        3. Nếu người dùng hỏi về nội dung luật pháp về vấn đề gì thì sử dụng dữ liệu trong Ngữ cảnh. Không sửa đổi hay bổ sung thêm
        4. Không thêm bất kì lưu ý hay nhắc nhờ về nội dung trong Ngữ cảnh
        5. Nếu không thấy dữ liệu trong Thông tin được cung cấp thì trả lời là "Mặc dù dữ liệu chưa được cung cấp theo tôi biết..." rồi thêm ý của bạn vào
        6. Đừng nhắc đến "Thông tin được cung cấp" mà tôi đã gửi cho bạn trong câu trả lời
        "
        """
        user_prompt = truncate_prompt(user_prompt)

        # print(user_prompt)

        stream = client.models.generate_content_stream(
            model="gemini-2.5-flash",
            contents=user_prompt
        )
        response = ""

        for chunk in stream:
            response += chunk.text

        return response
    except Exception as e:
        print(f"Lỗi khi gọi API: {e}")
        return "Xin lỗi, mình gặp lỗi khi xử lý câu hỏi của bạn."

# run example
# if __name__ == "__main__":
#     while True:
#         q = input("Bạn: ")
#         if q.strip().lower() in ("exit","quit"):
#             break
#         time.sleep(1)
#         print("Bot: ", end="")
#         print_typing_effect(answer_with_context(q))
#         print("\n")

# with gr.Blocks() as demo:
#     gr.Markdown("# 🤖")
#     chatbot = gr.Chatbot()
#     msg = gr.Textbox(placeholder="Nhập câu hỏi về pháp luật...")
#     clear = gr.Button("Xóa hội thoại")

#     def user_message(user_input, chat_history):
#         response = answer_with_context(user_input)
#         chat_history.append((user_input, response))
#         return "", chat_history

#     msg.submit(user_message, [msg, chatbot], [msg, chatbot])
#     clear.click(lambda: [], None, chatbot, queue=False)

# demo.launch()