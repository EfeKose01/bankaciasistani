# scraper_service.py
import requests
import os
import datetime
import json
import time
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import rag_indexer
from dotenv import load_dotenv

# --- YENİ KÜTÜPHANE: Haber Metni Avcısı ---
import trafilatura

load_dotenv()
DATA_DIR = "data"
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# --- ARAMA KONULARI ---
SEARCH_QUERIES = [
    "Türkiye ekonomisi son dakika analiz",
    "BDDK kararları ve bankacılık sektörü",
    "Merkez Bankası faiz kararı yorumları",
    "Borsa İstanbul hisse önerileri ve analizleri"
]

# --- PDF İNDİRİLECEK SİTELER ---
PDF_SITES = [
    {
        "name": "BDDK_Duyurular",
        "url": "https://www.bddk.org.tr/Duyurular",
        "base_url": "https://www.bddk.org.tr",
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


# --- YENİ FONKSİYON: PROFESYONEL METİN KAZIYICI ---
def extract_full_article_text(url):
    """
    Trafilatura kütüphanesi kullanarak haberin ana gövdesini reklamdan arındırıp çeker.
    """
    print(f"      📖 Derin Okuma Yapılıyor: {url[:60]}...")
    try:
        # 1. YÖNTEM: Trafilatura (En temiz metin çekici)
        downloaded = trafilatura.fetch_url(url)

        if downloaded:
            # include_comments=False -> Yorumları alma
            # include_tables=True -> Tabloları (veri varsa) al
            text = trafilatura.extract(downloaded, include_comments=False, include_tables=True,
                                       date_extraction_params={'extensive_search': True})

            if text and len(text) > 250:  # Çok kısa metinleri (hata mesajlarını) ele
                return text

        # 2. YÖNTEM: (Yedek) Klasik BeautifulSoup
        # Trafilatura başarısız olursa burası devreye girer
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')

        paragraphs = soup.find_all('p')
        full_text = ""
        for p in paragraphs:
            pt = p.get_text().strip()
            if len(pt) > 60: full_text += pt + "\n\n"

        return full_text if len(full_text) > 200 else None

    except Exception as e:
        print(f"      ⚠️ Okuma Hatası: {e}")
        return None


# --- MODÜL 1: GOOGLE SERPER İLE HABER BUL VE OKU ---
def fetch_and_read_news():
    if not SERPER_API_KEY:
        print("❌ .env dosyasında SERPER_API_KEY eksik!")
        return

    print("📡 Google üzerinden haberler bulunuyor...")

    url = "https://google.serper.dev/search"
    headers = {'X-API-KEY': SERPER_API_KEY, 'Content-Type': 'application/json'}

    daily_content = f"TARİH: {datetime.date.today()}\nKAYNAK: Google Haberleri (Trafilatura ile Full Metin)\n\n"
    news_count = 0

    for query in SEARCH_QUERIES:
        print(f"   🔎 Konu: '{query}'")
        try:
            payload = json.dumps({"q": query, "gl": "tr", "hl": "tr", "num": 3, "tbs": "qdr:d"})  # Son 24 saat
            response = requests.post(url, headers=headers, data=payload, timeout=10)

            if response.status_code == 200:
                results = response.json()
                if "organic" in results:
                    for item in results["organic"]:
                        link = item.get("link")
                        title = item.get("title")
                        snippet = item.get("snippet", "")

                        # --- KRİTİK NOKTA: İçeriği çek ---
                        full_article = extract_full_article_text(link)

                        daily_content += f"BAŞLIK: {title}\nLİNK: {link}\n"

                        if full_article:
                            # Tam metin bulundu
                            daily_content += f"URUM: TAM METİN OKUNDU\nİÇERİK:\n{full_article}\n"
                        else:
                            # Okunamadı, bari özeti koyalım (HİÇ YOKTAN İYİDİR)
                            daily_content += f"DURUM: ÖZET (Site erişimine izin vermedi)\nÖZET:\n{snippet}\n"

                        daily_content += f"{'=' * 50}\n\n"
                        news_count += 1

                        # Seri istek atıp engellenmemek için bekle
                        time.sleep(2)

        except Exception as e:
            print(f"   ❌ Hata: {e}")

    # Dosyaya Kaydet
    if news_count > 0:
        filename = f"{datetime.date.today()}_Detayli_Haberler.txt"
        filepath = os.path.join(DATA_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(daily_content)
        print(f"✅ {news_count} adet haber işlendi ve '{filename}' dosyasına yazıldı.")
    else:
        print("ℹ️ Haber bulunamadı.")


# --- MODÜL 2: PDF İNDİRME ---
def download_pdf(pdf_url, save_name):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        response = requests.get(pdf_url, stream=True, headers=headers, timeout=30)
        if response.status_code == 200:
            filepath = os.path.join(DATA_DIR, save_name)
            if os.path.exists(filepath): return False
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk: f.write(chunk)
            print(f"✅ PDF İndirildi: {save_name}")
            return True
        return False
    except Exception:
        return False


def fetch_pdfs_from_sites():
    print("📡 Resmi sitelerden PDF raporlar taranıyor...")
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
                    filename = f"{datetime.date.today()}_{site['name']}_{clean_filename(text)}.pdf"
                    if download_pdf(full_url, filename):
                        count += 1
                    if count >= site['limit']: break
        except Exception:
            pass


# --- ANA ÇALIŞTIRICI ---
def run_daily_update():
    print("\n--- 🚀 PROFESYONEL VERİ TARAMASI BAŞLADI ---")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # 1. Haberleri Bul ve İÇİNE GİRİP OKU
    fetch_and_read_news()

    # 2. PDF'leri İndir
    fetch_pdfs_from_sites()

    print("\n--- 🧠 BEYİN GÜNCELLENİYOR (Vektör İndeksleme) ---")
    rag_indexer.create_index()

    return "Güncelleme Başarılı! Haberlerin tamamı (erişilebilenler) okundu."


if __name__ == "__main__":
    run_daily_update()