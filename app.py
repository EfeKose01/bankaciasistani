# app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

# Kendi modüllerimiz
import banking_tools
import scraper_service  # Web Scraping ve PDF indirme modülü

# --- AYARLAR ---
st.set_page_config(page_title="Bankacı Asistanı Pro", page_icon="🏦", layout="wide")
load_dotenv()


# --- RAG MODELİNİ YÜKLE (CACHE) ---
@st.cache_resource
def load_rag_engine():
    """Vektör veritabanını ve embedding modelini yükler."""
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
    """Claude API'ye context ve soruyu gönderir."""
    if not client:
        return "⚠️ API Key eksik. Lütfen .env dosyasını kontrol edin."

    system_prompt = """Sen uzman bir Bankacılık Mevzuat ve Operasyon Asistanısın.
    Sana verilen mevzuat parçalarını (CONTEXT) kullanarak kullanıcının sorusunu yanıtla.

    KURALLAR:
    1. Öncelikle CONTEXT içindeki bilgilere dayan.
    2. Eğer context içinde BDDK, TCMB gibi resmi bir kaynak varsa, cevabında belirt (Örn: "BDDK yönetmeliğine göre...").
    3. Context içinde bilgi yoksa dürüstçe 'Dahili belgelerimde/indirdiğim raporlarda bu bilgi yer almıyor' de.
    4. Asla yatırım tavsiyesi verme. Sadece operasyonel kuralları ve mevzuatı açıkla.
    """

    prompt = f"CONTEXT:\n{context}\n\nSORU: {question}"

    try:
        message = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=1500,
            temperature=0.1,  # Halüsinasyonu önlemek için düşük sıcaklık
            system=system_prompt,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text
    except Exception as e:
        return f"LLM Hatası: {str(e)}"


# --- ARAYÜZ BAŞLANGICI ---

st.title("🏦 Bankacı Operasyon Asistanı (RAG + Live Data)")

# --- SIDEBAR (YAN MENÜ) ---
st.sidebar.title("Menü")
mode = st.sidebar.radio("Çalışma Modu", ["Mevzuat Asistanı (RAG)", "Kredi Hesaplayıcı", "Mevduat & Getiri"])

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Veri Yönetimi")

# Web Scraping Butonu
if st.sidebar.button("🔄 Verileri Güncelle (BDDK & Haberler)"):
    with st.spinner("BDDK ve Haber siteleri taranıyor, yeni PDF'ler indiriliyor..."):
        try:
            # Scraper servisini çalıştır
            status_msg = scraper_service.run_daily_update()

            # Sistemi tazelemek için cache temizle
            st.cache_resource.clear()
            rag_model, rag_index, rag_content = load_rag_engine()

            st.sidebar.success(status_msg)
        except Exception as e:
            st.sidebar.error(f"Güncelleme Hatası: {e}")

st.sidebar.info(f"📚 İndeksli Doküman Parçası: {len(rag_content) if rag_content else 0}")

# --- MOD 1: MEVZUAT ASISTANI (RAG) ---
if mode == "Mevzuat Asistanı (RAG)":
    st.header("⚖️ Mevzuat & Operasyon Soru-Cevap")
    st.caption("Veritabanı: BDDK Yönetmelikleri, TCMB Tebliğleri ve İndirilen Güncel PDF Raporlar")

    if rag_index is None:
        st.warning(
            "⚠️ Vektör veritabanı bulunamadı. Lütfen yan menüden 'Verileri Güncelle' butonuna basın veya data klasörüne dosya atıp rag_indexer.py çalıştırın.")
    else:
        # Chat Geçmişi
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Eski mesajları ekrana yaz
        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        # Yeni Soru Girişi
        if prompt := st.chat_input(
                "Soru sor (Örn: Konut kredisinde LTV oranı nedir? veya BDDK son duyurusunda ne dedi?)"):
            # Kullanıcı mesajını ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            # RAG Arama (En alakalı 5 parçayı getir)
            with st.spinner("Dokümanlar taranıyor..."):
                query_vector = rag_model.encode([prompt])
                distances, indices = rag_index.search(np.array(query_vector).astype("float32"), k=5)

                context_texts = []
                sources = set()

                for i in indices[0]:
                    if i < len(rag_content):
                        # Metni ve kaynağı al
                        text_segment = rag_content[i]['text']
                        source_file = rag_content[i].get('source', 'Bilinmeyen Kaynak')

                        context_texts.append(f"KAYNAK DOSYA: {source_file}\nİÇERİK: {text_segment}")
                        sources.add(source_file)

                full_context = "\n---\n".join(context_texts)

            # LLM Cevap Üretimi
            with st.chat_message("assistant"):
                with st.spinner("Cevap hazırlanıyor..."):
                    response = ask_llm(full_context, prompt)
                    st.write(response)

                    # Kaynakları göster
                    if sources:
                        with st.expander("📚 Kullanılan Kaynaklar"):
                            for s in sources:
                                st.write(f"- {s}")

                    # Debug için tam context (istenirse açılabilir)
                    # with st.expander("AI Context (Debug)"):
                    #     st.text(full_context)

            st.session_state.messages.append({"role": "assistant", "content": response})


# --- MOD 2: KREDİ HESAPLAYICI ---
elif mode == "Kredi Hesaplayıcı":
    st.header("💳 Detaylı Kredi Simülasyonu")

    col1, col2, col3 = st.columns(3)
    with col1:
        amount = st.number_input("Kredi Tutarı (TL)", min_value=1000, value=100000, step=1000)
    with col2:
        rate = st.number_input("Aylık Faiz Oranı (%)", min_value=0.01, value=3.50, step=0.01)
    with col3:
        term = st.number_input("Vade (Ay)", min_value=1, value=12, step=1)

    tax_option = st.checkbox("KKDF (%15) ve BSMV (%15) Dahil Et (Bireysel İhtiyaç)", value=True)

    if st.button("Hesaplama Yap"):
        df_plan, summary = banking_tools.calculate_loan_schedule(amount, rate, term, tax_option)

        # Özet Kartları
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Aylık Taksit", f"{summary['Aylık Taksit']:,.2f} TL")
        sc2.metric("Toplam Geri Ödeme", f"{summary['Toplam Geri Ödeme']:,.2f} TL")
        sc3.metric("Maliyet Oranı", f"%{summary['Maliyet Oranı (Yıllık Efektif)']}")

        tab1, tab2 = st.tabs(["Ödeme Planı Tablosu", "Grafik Analiz"])

        with tab1:
            st.dataframe(df_plan, use_container_width=True)

        with tab2:
            df_melted = df_plan.melt(id_vars=["Taksit No"],
                                     value_vars=["Anapara Ödemesi", "Faiz Ödemesi", "Vergi (KKDF+BSMV)"])
            fig = px.bar(df_melted, x="Taksit No", y="value", color="variable", title="Taksit Dağılımı")
            st.plotly_chart(fig, use_container_width=True)


# --- MOD 3: MEVDUAT ---
elif mode == "Mevduat & Getiri":
    st.header("💰 Mevduat Getiri Hesaplama")

    c1, c2, c3 = st.columns(3)
    with c1:
        m_amount = st.number_input("Anapara (TL)", value=500000, step=10000)
    with c2:
        m_days = st.number_input("Gün Sayısı (Kırık Vade)", value=32)
    with c3:
        m_rate = st.number_input("Yıllık Mevduat Faizi (%)", value=45.0)

    stopaj = st.selectbox("Stopaj Oranı", [0.075, 0.05, 0.0, 0.10, 0.15], index=0,
                          format_func=lambda x: f"%{x * 100:.1f}")

    if st.button("Getiri Hesapla"):
        res = banking_tools.calculate_deposit_return(m_amount, m_days, m_rate, stopaj)

        st.success(f"Net Ele Geçen Tutar: **{res['Net Ele Geçen (Stopaj Düşülmüş)']:,.2f} TL**")

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Brüt Getiri", f"{res['Brüt Getiri']:,.2f} TL")
        with col_b:
            st.metric("Kesilen Stopaj", f"{res['Stopaj Tutarı']:,.2f} TL")

        st.json(res)