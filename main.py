import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
groq_api_key = os.getenv("GROQ_API_KEY")

def rag(question):
    vectordb = FAISS.load_local("vectordb", OllamaEmbeddings(base_url=ollama_base_url, model="nomic-embed-text", show_progress=False),
    allow_dangerous_deserialization=True)

    retriever = vectordb.similarity_search(question, k=5)
    print("Hasil pencarian retriever:", retriever)

    prompt = f"""
    Anda adalah AiDA asisten pilkada yang memberikan informasi tentang Pilkada dalam format yang terstruktur. 
    Setiap jawaban yang diberikan harus dipisahkan dengan poin-poin atau penomoran yang jelas.
    - Gunakan bahasa indonesia.
    - Jawab sesuai apa yang ditanyakan saja.
    - Jangan mengarang informasi yang tidak sesuai konteks.
    - Jangan berkata kasar, menghina, sarkas, satir, atau merendahkan pihak lain.
    - Berikan jawaban yang lengkap, rapi, dan penomoran jika diperlukan sesuai konteks.
    Konteks: {retriever}
    """

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=question)
    ]

    response = ChatGroq(
        model="gemma2-9b-it",
        temperature=0,
        max_tokens=None,
        timeout=None,
    )

    result = response.invoke(messages).content
    print("Hasil respons model:", result)  # Periksa hasil yang dikembalikan oleh model

    # Jangan lakukan modifikasi jika hasilnya null
    if not result:
        print("Hasil null, pastikan model merespons dengan benar!")
        return "Maaf, saya tidak dapat memberikan jawaban."

    # Jika hasil ada, lanjutkan dengan penggantian tanda bintang
    result_with_bold = result.replace("**", "<strong>").replace("**", "</strong>")  # Ganti ** menjadi <strong>

    return result_with_bold