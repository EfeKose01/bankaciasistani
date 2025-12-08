# rag_indexer.py
import os
import pickle
import PyPDF2
import faiss
from sentence_transformers import SentenceTransformer

# Data klasörünü tara
DATA_DIR = "data"
INDEX_FILE = "rag_index.faiss"
CONTENT_FILE = "rag_content.pkl"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


def create_index():
    if not os.path.exists(DATA_DIR):
        print(f"⚠️ '{DATA_DIR}' klasörü bulunamadı.")
        return

    print("🧠 Model yükleniyor (Bu işlem biraz sürebilir)...")
    model = SentenceTransformer(MODEL_NAME)

    documents = []

    # Klasördeki tüm dosyaları gez
    for filename in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, filename)

        # --- DURUM 1: PDF DOSYALARI ---
        if filename.lower().endswith(".pdf"):
            print(f"📄 Okunuyor (PDF): {filename}")
            try:
                reader = PyPDF2.PdfReader(path)
                text = ""
                for page in reader.pages:
                    extract = page.extract_text()
                    if extract: text += extract + "\n"

                # Chunking (Parçalama)
                chunk_size = 1000
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) > 50:  # Çok kısa parçaları alma
                        documents.append({
                            "source": filename,
                            "text": chunk
                        })
            except Exception as e:
                print(f"❌ Hata ({filename}): {e}")

        # --- DURUM 2: HABER (TXT) DOSYALARI (YENİ EKLENEN KISIM) ---
        elif filename.lower().endswith(".txt"):
            print(f"📰 Okunuyor (HABER): {filename}")
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()

                # Haberler genelde daha kısa olur ama yine de bölelim
                chunk_size = 1000
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i + chunk_size]
                    if len(chunk) > 50:
                        documents.append({
                            "source": filename,
                            "text": chunk
                        })
            except Exception as e:
                print(f"❌ Hata ({filename}): {e}")

    if not documents:
        print("⚠️ İndekslenecek belge bulunamadı.")
        return

    print(f"🔄 {len(documents)} parça metin vektörleştiriliyor...")
    embeddings = model.encode([d["text"] for d in documents])

    # Faiss Index Oluştur
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Kaydet
    faiss.write_index(index, INDEX_FILE)
    with open(CONTENT_FILE, "wb") as f:
        pickle.dump(documents, f)

    print("✅ BEYİN GÜNCELLENDİ! (PDF'ler ve Haberler Hafızaya Alındı)")


if __name__ == "__main__":
    create_index()