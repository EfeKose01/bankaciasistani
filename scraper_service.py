import requests
import os
import datetime
import json
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import rag_indexer
from dotenv import load_dotenv
import trafilatura

load_dotenv()
DATA_DIR = "data"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- 1. ZENGİNLEŞTİRİLMİŞ ARAMA SORGULARI ---
# Sadece haber değil, teknik limitleri de yakalamak için
SEARCH_QUERIES = [
    "TCMB sermaye hareketleri genelgesi güncel değişiklikler 2026",
    "BDDK kredi sınırlandırmaları LTV ve vade oranları güncel",
    "Bankaların döviz kredisi kullanım şartları ve 15 milyon dolar sınırı",
    "Resmi Gazete bankacılık ve finans tebliğleri",
    "KKDF ve BSMV muafiyetleri son dakika"
]

# --- 2. GENİŞLETİLMİŞ RESMİ KAYNAK LİSTESİ ---
# Gemini'nin bildiği yasal dayanakları çekmek için TCMB ve Resmi Gazete odaklı
PDF_SITES = [
    {
        "name": "BDDK_Duyurular",
        "url": "https://www.bddk.org.tr/Duyurular",
        "base_url": "https://www.bddk.org.tr",
        "limit": 5
    },
    {
        "name": "TCMB_Mevzuat_Tebligler",
        "url": "https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/yayinlar/mevzuat/tebligler",
        "base_url": "https://www.tcmb.gov.tr",
        "limit": 5
    },
    {
        "name": "TCMB_Genelgeler",
        "url": "https://www.tcmb.gov.tr/wps/wcm/connect/tr/tcmb+tr/main+menu/yayinlar/mevzuat/genelgeler",
        "base_url": "https://www.tcmb.gov.tr",
        "limit": 3
    },
    {
        "name": "MASAK_Rehberler",
        "url": "https://masak.hmb.gov.tr/sektorel-rehberler",
        "base_url": "https://masak.hmb.gov.tr",
        "limit": 2
    }
]


def clean_filename(title):
    invalid = '<>:"/\|?* '
    for char in invalid:
        title = title.replace(char, '_')
    return title[:100]


def extract_full_article_text(url):
    """Trafilatura kullanarak haberin ana gövdesini reklamdan arındırıp çeker."""
    print(f"      📖 Derin Okuma Yapılıyor: {url[:60]}...")
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded:
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
            if text and len(text) > 250:
                return text

        # Yedek mekanizma
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = "\n\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 60])
        return full_text if len(full_text) > 200 else None
    except Exception as e:
        print(f"      ⚠️ Okuma Hatası: {e}")
        return None


def fetch_and_read_news():
    if not SERPER_API_KEY:
        print("❌ .env dosyasında SERPER_API_KEY eksik!")
        return

    print("📡 Google ve Finans Kaynakları üzerinden teknik veriler toplanıyor...")
    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    daily_content = f"TARİH: {datetime.date.today()}\nBİLGİ TÜRÜ: Güncel Mevzuat ve Haber Analizi\n\n"
    news_count = 0

    for query in SEARCH_QUERIES:
        print(f"   🔎 Teknik Tarama: '{query}'")
        try:
            payload = json.dumps(
                {"q": query, "gl": "tr", "hl": "tr", "num": 4, "tbs": "qdr:m"})  # Son 1 ay (Daha geniş mevzuat için)
            response = requests.post(url, headers=headers, data=payload, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if "organic" in results:
                    for item in results["organic"]:
                        link = item.get("link")
                        title = item.get("title")
                        full_article = extract_full_article_text(link)

                        daily_content += f"BAŞLIK: {title}\nLİNK: {link}\n"
                        if full_article:
                            daily_content += f"İÇERİK ANALİZİ:\n{full_article}\n"
                        else:
                            daily_content += f"ÖZET (Tam metne ulaşılamadı): {item.get('snippet', '')}\n"

                        daily_content += f"{'=' * 50}\n\n"
                        news_count += 1
                        time.sleep(1)
        except Exception as e:
            print(f"   ❌ Arama Hatası: {e}")

    if news_count > 0:
        filename = f"{datetime.date.today()}_Mevzuat_Haber_Analizi.txt"
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(daily_content)
        print(f"✅ {news_count} adet veri kaynağı işlendi.")


def download_pdf(pdf_url, save_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        if response.status_code == 200:
            filepath = os.path.join(DATA_DIR, save_name)
            if os.path.exists(filepath): return False
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
            print(f"✅ Yeni PDF Kaydedildi: {save_name}")
            return True
        return False
    except Exception:
        return False


def fetch_pdfs_from_sites():
    print("📡 Resmi Mevzuat Kanalları Taranıyor (BDDK, TCMB, MASAK)...")
    for site in PDF_SITES:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(site['url'], headers=headers, timeout=20)
            soup = BeautifulSoup(response.content, 'html.parser')
            links = soup.find_all('a', href=True)

            count = 0
            for link in links:
                href = link.get('href')
                if href and href.lower().endswith('.pdf'):
                    full_url = urljoin(site['base_url'], href)
                    text = link.get_text().strip() or "Dokuman"
                    # Dosya ismine tarih ekleyerek versiyonlama sağlıyoruz
                    filename = f"{datetime.date.today()}_{site['name']}_{clean_filename(text)}.pdf"
                    if download_pdf(full_url, filename):
                        count += 1
                    if count >= site['limit']: break
        except Exception as e:
            print(f"⚠️ {site['name']} taranırken hata oluştu.")


def run_daily_update():
    print("\n--- 🚀 BANKACI ASİSTANI VERİ TABANI GENİŞLETME BAŞLADI ---")
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # Haber ve Duyuruları çek
    fetch_and_read_news()
    # Resmi PDF'leri çek (TCMB dahil)
    fetch_pdfs_from_sites()

    print("\n--- 🧠 YENİ BİLGİLER VEKTÖR VERİTABANINA İŞLENİYOR ---")
    rag_indexer.create_index()

    return f"Güncelleme Tamamlandı. Veritabanı en güncel TCMB ve BDDK verileriyle zenginleştirildi."


if __name__ == "__main__":
    run_daily_update()