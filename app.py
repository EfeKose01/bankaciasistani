import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
import faiss
import numpy as np
import datetime
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# Kendi modüllerimiz
import banking_tools
import scraper_service

# --- AYARLAR ---
st.set_page_config(page_title="Bankacı Asistanı", page_icon="🏦", layout="wide")
load_dotenv()


# --- YARDIMCI FONKSİYONLAR ---
def get_last_update_time():
    file_path = "rag_index.faiss"
    if os.path.exists(file_path):
        timestamp = os.path.getmtime(file_path)
        return datetime.datetime.fromtimestamp(timestamp).strftime('%d.%m.%Y %H:%M')
    return "Henüz güncelleme yapılmadı"


# --- RAG MODELİNİ YÜKLE (CACHE) ---
@st.cache_resource
def load_rag_engine():
    if not os.path.exists("rag_index.faiss"):
        return None, None, None
    model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    index = faiss.read_index("rag_index.faiss")
    with open("rag_content.pkl", "rb") as f:
        content = pickle.load(f)
    return model, index, content


rag_model, rag_index, rag_content = load_rag_engine()

# --- ANTHROPIC CLIENT ---
api_key = os.getenv("ANTHROPIC_API_KEY")
client = Anthropic(api_key=api_key) if api_key else None


def ask_llm(context, question):
    if not client:
        return "⚠️ API Key eksik. Lütfen .env dosyasını kontrol edin."

    # EN ÜST DÜZEY ANALİZ İÇİN ZORLAYICI PROMPT
    system_prompt = """Sen dünyanın en iyi bankacılık mevzuat danışmanısın. 
    Haiku modeli olmana rağmen, bir uzman gibi derinlemesine, kapsamlı ve çok detaylı yanıtlar üretmen gerekiyor.

    CEVAP YAPISI ŞÖYLE OLMALIDIR:
    1. **Yönetici Özeti:** Konunun 1-2 cümlelik en kritik özeti.
    2. **Mevzuat Dayanakları:** CONTEXT içindeki bilgileri kullanarak maddeler halinde (Bullet points) teknik detaylar.
    3. **Operasyonel Analiz:** Bu bilgilerin banka personeli için ne anlama geldiği, nelere dikkat edilmesi gerektiği.
    4. **Sınırlamalar ve İstisnalar:** Varsa limitler, yasaklar veya özel durumlar.
    5. **Pratik Uygulama Örneği:** Konunun daha iyi anlaşılması için bir müşteri senaryosu.

    KURALLAR:
    - Asla 3-4 cümlede kesme. Her zaman en az 4-5 başlık altında detay ver.
    - Metin içerisinde geçen sayısal verileri (oranlar, tutarlar) **kalın** yaz.
    - Eğer context'te bilgi kısıtlıysa, "Mevcut kaynaklarda şu kısımlar yer almaktadır..." diyerek elindekini sonuna kadar kullan.
    - Profesyonel, ciddi ve yol gösterici bir ton kullan.
    """

    prompt = f"Aşağıdaki kapsamlı dökümanları analiz et ve soruyu bir rapor titizliğinde yanıtla:\n\nCONTEXT:\n{context}\n\nSORU: {question}"

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",  # 404 hatasını önlemek için stabil model
            max_tokens=3000,  # Çok daha uzun yazması için limit artırıldı
            temperature=0.3,  # Odaklı ama yorum yapabilen bir seviye
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"LLM Hatası: {str(e)}"


# --- ARAYÜZ ---
st.title("🏦 Bankacı Asistanı")

# --- SIDEBAR ---
st.sidebar.title("Menü")
mode = st.sidebar.radio("Çalışma Modu", ["Mevzuat Asistanı (RAG)", "Kredi Hesaplayıcı", "Mevduat & Getiri"])
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Veri Yönetimi")

if st.sidebar.button("🔄 Verileri Güncelle (BDDK & Haberler)"):
    with st.spinner("Yeni veriler sisteme entegre ediliyor..."):
        try:
            status_msg = scraper_service.run_daily_update()
            st.cache_resource.clear()
            st.sidebar.success(status_msg)
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Güncelleme Hatası: {e}")

st.sidebar.info(f"📅 Son Güncelleme: {get_last_update_time()}")

# --- MOD 1: MEVZUAT ASISTANI ---
if mode == "Mevzuat Asistanı (RAG)":
    st.header("⚖️ Mevzuat & Operasyon Analiz Laboratuvarı")

    if rag_index is None:
        st.warning("⚠️ Veritabanı bulunamadı. Lütfen güncellemeyi başlatın.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Analiz edilmesini istediğiniz konuyu yazın..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Mevzuat dökümanları derinlemesine taranıyor..."):
                query_vector = rag_model.encode([prompt])
                # Modele daha fazla malzeme vermek için k=12 yapıldı
                distances, indices = rag_index.search(np.array(query_vector).astype("float32"), k=12)

                context_texts = []
                sources = set()

                for i in indices[0]:
                    if i < len(rag_content):
                        text_segment = rag_content[i]['text']
                        source_file = rag_content[i].get('source', 'Bilinmeyen Kaynak')
                        context_texts.append(f"KAYNAK: {source_file}\nİÇERİK: {text_segment}")
                        sources.add(source_file)

                full_context = "\n---\n".join(context_texts)

            with st.chat_message("assistant"):
                with st.spinner("Rapor hazırlanıyor..."):
                    response = ask_llm(full_context, prompt)
                    st.markdown(response)

                    if sources:
                        with st.expander("📚 Analiz Dayanağı Olan Belgeler"):
                            for s in sources:
                                st.write(f"- {s}")

            st.session_state.messages.append({"role": "assistant", "content": response})

# --- DİĞER MODLAR (HESAPLAYICILAR) ---
elif mode == "Kredi Hesaplayıcı":
    # (Kredi hesaplama kodları burada - Değişmedi)
    st.header("💳 Detaylı Kredi Simülasyonu")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Kredi Tutarı (TL)", min_value=1000, value=100000, step=1000)
    with col2:
        rate = st.number_input("Aylık Faiz Oranı (%)", min_value=0.01, value=3.50, step=0.01)
    with col3:
        term = st.number_input("Vade (Ay)", min_value=1, value=12, step=1)
    tax_option = st.checkbox("Vergiler Dahil", value=True)
    if st.button("Hesapla"):
        df_plan, summary = banking_tools.calculate_loan_schedule(amount, rate, term, tax_option)
        st.metric("Aylık Taksit", f"{summary['Aylık Taksit']:,.2f} TL")
        st.dataframe(df_plan)

elif mode == "Mevduat & Getiri":
    # (Mevduat kodları burada - Değişmedi)
    st.header("💰 Mevduat Getiri Hesaplama")
    c1, c2, c3 = st.columns(3)
    with c1:
        m_amount = st.number_input("Anapara (TL)", value=500000)
    with c2:
        m_days = st.number_input("Gün Sayısı", value=32)
    with c3:
        m_rate = st.number_input("Faiz Oranı (%)", value=45.0)
    if st.button("Getiri Hesapla"):
        res = banking_tools.calculate_deposit_return(m_amount, m_days, m_rate, 0.075)
        st.success(f"Net Getiri: {res['Net Ele Geçen (Stopaj Düşülmüş)']:,.2f} TL")