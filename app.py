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
    # Türkçe performansı için bu model oldukça iyidir
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

    # DETAYLI CEVAP İÇİN GÜNCELLENMİŞ PROMPT
    system_prompt = """Sen üst düzey bir Kıdemli Bankacılık Mevzuat Analistisin. 
    Görevin, sana sunulan kaynakları (CONTEXT) kullanarak kullanıcının sorusuna DERİNLEMESİNE analiz sunmaktır.

    TALİMATLAR:
    1. ASLA KISA CEVAP VERME. Konuyu her yönüyle ele al.
    2. Mevzuat bilgilerini maddeler halinde ve başlıklar kullanarak açıkla.
    3. Kaynaklardaki verileri kullanarak "Operasyonel Etki" yorumu yap (Örn: Bu kural banka için ne anlama geliyor?).
    4. Varsa oranları, limitleri ve tarihleri tablo veya kalın harflerle vurgula.
    5. "Senaryo Analizi" başlığı altında, bu kuralın gerçek hayatta nasıl uygulanabileceğine dair örnek bir durum uydur.
    6. Eğer context içinde yeterli bilgi yoksa, eldeki kısıtlı bilgiyi ver ve eksik kısımlar için 'Resmi Gazete veya BDDK duyurularının orijinal metninin tamamı taranmalıdır' notunu düş.
    7. Profesyonel, eğitici ve güven verici bir ton kullan.
    """

    prompt = f"Aşağıdaki kaynaklara dayanarak detaylı bir analiz yap:\n\nCONTEXT:\n{context}\n\nSORU: {question}"

    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",  # Haiku'dan Sonnet'e geçiş (Zeka ve detay artışı)
            max_tokens=2500,  # Daha uzun cevaplar için limit artırıldı
            temperature=0.4,  # Daha iyi yorum yapabilmesi için biraz artırıldı
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
    with st.spinner("Veriler taranıyor ve analiz ediliyor..."):
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
    st.header("⚖️ Mevzuat & Operasyon Analizi")

    if rag_index is None:
        st.warning("⚠️ Veritabanı boş. Lütfen verileri güncelleyin.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Hangi mevzuat konusunu detaylı analiz etmemi istersiniz?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.spinner("Dokümanlar taranıyor ve sentezleniyor..."):
                query_vector = rag_model.encode([prompt])
                # k=10 yapıldı, böylece daha fazla bilgi LLM'e gider
                distances, indices = rag_index.search(np.array(query_vector).astype("float32"), k=10)

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
                with st.spinner("Analiz raporu oluşturuluyor..."):
                    response = ask_llm(full_context, prompt)
                    st.markdown(response)  # Detaylı format için markdown desteği

                    if sources:
                        with st.expander("📚 Analizde Kullanılan Dayanaklar"):
                            for s in sources:
                                st.write(f"- {s}")

            st.session_state.messages.append({"role": "assistant", "content": response})

# --- MOD 2 VE 3 AYNI KALDI ---
elif mode == "Kredi Hesaplayıcı":
    st.header("💳 Detaylı Kredi Simülasyonu")
    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Kredi Tutarı (TL)", min_value=1000, value=100000, step=1000)
    with col2:
        rate = st.number_input("Aylık Faiz Oranı (%)", min_value=0.01, value=3.50, step=0.01)
    with col3:
        term = st.number_input("Vade (Ay)", min_value=1, value=12, step=1)
    tax_option = st.checkbox("KKDF (%15) ve BSMV (%15) Dahil Et", value=True)

    if st.button("Hesaplama Yap"):
        df_plan, summary = banking_tools.calculate_loan_schedule(amount, rate, term, tax_option)
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Aylık Taksit", f"{summary['Aylık Taksit']:,.2f} TL")
        sc2.metric("Toplam Geri Ödeme", f"{summary['Toplam Geri Ödeme']:,.2f} TL")
        sc3.metric("Maliyet Oranı", f"%{summary['Maliyet Oranı (Yıllık Efektif)']}")
        st.dataframe(df_plan, use_container_width=True)

elif mode == "Mevduat & Getiri":
    st.header("💰 Mevduat Getiri Hesaplama")
    c1, c2, c3 = st.columns(3)
    with c1:
        m_amount = st.number_input("Anapara (TL)", value=500000, step=10000)
    with c2:
        m_days = st.number_input("Gün Sayısı", value=32)
    with c3:
        m_rate = st.number_input("Yıllık Faiz (%)", value=45.0)
    stopaj = st.selectbox("Stopaj Oranı", [0.075, 0.05, 0.0, 0.10, 0.15], index=0,
                          format_func=lambda x: f"%{x * 100:.1f}")

    if st.button("Getiri Hesapla"):
        res = banking_tools.calculate_deposit_return(m_amount, m_days, m_rate, stopaj)
        st.success(f"Net Getiri: {res['Net Ele Geçen (Stopaj Düşülmüş)']:,.2f} TL")
        st.json(res)