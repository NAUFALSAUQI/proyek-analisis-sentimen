import streamlit as st
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

# Impor Pustaka Inti (AI/ML)
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer

# ---------------------------------------------------------------------
# KONFIGURASI HALAMAN
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="PoliSense Analytics",
    page_icon="📊",
    layout="wide"
)

# Download resource NLTK jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

# ---------------------------------------------------------------------
# BAGIAN 1: MODEL & PREPROCESSING (CACHE)
# ---------------------------------------------------------------------

@st.cache_resource
def load_bert_sentiment_model():
    """Memuat model IndoBERT."""
    # Ganti device=0 jika menggunakan GPU
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    return pipeline("sentiment-analysis", model=model_name, device=-1)

def preprocess_text_simple(text):
    """Membersihkan teks dengan Regex sederhana."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

@st.cache_data
def run_full_analysis(uploaded_file_data, _classifier):
    """Menjalankan BERTopic dan Sentiment Analysis."""
    # Baca Excel
    try:
        df_all = pd.read_excel(uploaded_file_data)
    except Exception:
        # Fallback jika user upload CSV
        df_all = pd.read_csv(uploaded_file_data)
        
    if 'text' not in df_all.columns:
        return None # Indikasi error kolom

    # Preprocessing
    df_all['text_bersih'] = df_all['text'].apply(preprocess_text_simple)

    # Siapkan Data
    df_posts = pd.DataFrame({'post_id': df_all.index, 'text_untuk_topik': df_all['text_bersih']})
    df_comments = pd.DataFrame({'post_id': df_all.index, 'text_untuk_sikap': df_all['text_bersih'], 'comment_text': df_all['text']})

    # BERTopic
    vectorizer = CountVectorizer(stop_words=["di", "dan", "yang", "ini", "itu", "dengan", "ke", "pada", "dari"])
    topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer, verbose=False, min_topic_size=15)
    topics, _ = topic_model.fit_transform(df_posts['text_untuk_topik'])
    df_posts['topik'] = topics
    df_info_topik = topic_model.get_topic_info()

    # Sentiment Analysis
    list_komentar = df_comments['text_untuk_sikap'].tolist()
    hasil_sikap = _classifier(list_komentar, batch_size=8)
    
    df_comments['sikap_label_raw'] = [result['label'] for result in hasil_sikap]
    label_mapping = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
    df_comments['sikap'] = df_comments['sikap_label_raw'].map(label_mapping)

    # Merge
    df_gabung = pd.merge(df_comments[['post_id', 'sikap', 'comment_text']], df_posts[['post_id', 'topik']], on='post_id', how='left')
    
    # Hitung Statistik
    analisis_final_count = df_gabung.groupby('topik')['sikap'].value_counts().unstack(fill_value=0)
    analisis_final_percent = df_gabung.groupby('topik')['sikap'].value_counts(normalize=True).unstack(fill_value=0)
    
    return df_info_topik, analisis_final_count, analisis_final_percent, df_gabung

# ---------------------------------------------------------------------
# BAGIAN 2: VISUALISASI & EXPORT
# ---------------------------------------------------------------------

def create_visuals(analisis_final_count, analisis_final_percent):
    figs = []
    sns.set(style="whitegrid") # Style plot bersih

    # Chart 1: Bar Chart Absolut
    df_counts_filtered = analisis_final_count[analisis_final_count.index != -1]
    if not df_counts_filtered.empty:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        # Custom warna agar mirip dashboard profesional (Hijau, Merah, Abu)
        colors = {'Positive': '#2ecc71', 'Negative': '#e74c3c', 'Neutral': '#95a5a6'}
        # Pastikan kolom ada sebelum mapping warna
        color_list = [colors.get(col, '#333333') for col in df_counts_filtered.columns]
        
        df_counts_filtered.plot(kind='bar', stacked=True, color=color_list, ax=ax1)
        ax1.set_title('Jumlah Sebaran Sikap per Topik', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Topik', fontsize=10)
        ax1.legend(title='Sikap')
        plt.xticks(rotation=45)
        plt.tight_layout()
        figs.append(fig1)

    # Chart 2: Heatmap
    df_percent = (analisis_final_percent * 100).round(2)
    df_percent_filtered = df_percent[df_percent.index != -1]
    if not df_percent_filtered.empty:
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df_percent_filtered, annot=True, fmt='.1f', cmap='RdYlGn', linewidths=.5, ax=ax2)
        ax2.set_title('Peta Panas (%) Sentimen', fontsize=14, fontweight='bold')
        figs.append(fig2)
    
    return figs

def create_excel_export(df_info, df_count, df_percent, df_raw):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_info.to_excel(writer, sheet_name='Info Topik', index=False)
        df_count.to_excel(writer, sheet_name='Statistik Jumlah')
        (df_percent * 100).round(2).to_excel(writer, sheet_name='Statistik Persen')
        df_raw.to_excel(writer, sheet_name='Data Mentah', index=False)
    return output.getvalue()

# ---------------------------------------------------------------------
# BAGIAN 3: TATA LETAK UTAMA (SESUAI GAMBAR)
# ---------------------------------------------------------------------

def main():
    # --- SIDEBAR: STATUS & LOGO ---
    with st.sidebar:
        # 1. Status Hijau (Model Siap)
        st.success("**Model AI Siap.**")
        
        # 2. Info Biru
        st.info(
            "Aplikasi ini menjalankan proses stemming dan analisis BERT "
            "dengan kecepatan maksimal."
        )
        
        # Spacer agar logo turun ke bawah (opsional, tergantung resolusi layar)
        st.write("") 
        st.write("")
        st.write("")
        
       # 3. Logo Sponsor (IPOL & FISIP)
        st.markdown("---")
        
        # Hapus st.columns agar gambar tersusun vertikal (atas-bawah)
        # Menambahkan caption kosong atau spasi jika perlu
        
        st.image("logo_ipol.png", use_container_width=True)
        # Tambahkan sedikit jarak antar logo jika mau (opsional)
        st.write("") 
        st.image("logo_fisip.png", use_container_width=True)
        # --- MAIN CONTENT ---
    
    # 1. HEADER (Logo PoliSense + Judul)
    col_header_logo, col_header_text = st.columns([1, 4])
    
    with col_header_logo:
        # GANTI URL INI dengan nama file lokal Anda: "logo_polisense.png"
        #st.image("https://via.placeholder.com/200x200/FFA500/000000?text=PS", use_container_width=True)
        st.image("logo_polisense.png", use_container_width=True)

    with col_header_text:
        st.title("Aplikasi Analisis Topik dan Sentimen Politik")
        #st.markdown("### **PoliSense**")

    # 2. DESKRIPSI
    st.markdown("""
    **PoliSense: Wawasan Politik Digital Anda.**
    
    Selami lanskap politik nasional melalui analisis data media sosial X. 
    Dengan teknologi BERT, PoliSense mengidentifikasi topik utama dan menganalisis sentimen publik 
    secara akurat, membuka pemahaman baru tentang opini politik, tren, dan narasi di ruang digital.
    """)
    
    st.markdown("Unggah file Excel Anda (harus punya kolom 'text')")

    # 3. AREA UPLOAD
    uploaded_file = st.file_uploader("", type=["xlsx", "csv"])

    # Load Model (Background)
    classifier = load_bert_sentiment_model()

    # 4. TOMBOL EKSEKUSI
    if uploaded_file is not None:
        
        # Tampilkan nama file yang diupload (opsional, untuk konfirmasi visual)
        st.caption(f"File terdeteksi: {uploaded_file.name}")

        if st.button("Mulai Analisis 🚀", type="primary"):
            
            with st.spinner("Sedang memproses..."):
                hasil = run_full_analysis(uploaded_file, classifier)

            if hasil is None:
                st.error("Gagal memproses. Pastikan file Excel memiliki kolom bernama 'text'.")
            else:
                # 5. HASIL ANALISIS (Box Hijau Sukses)
                st.success("Analisis Selesai!")
                
                df_info, df_count, df_percent, df_raw = hasil
                
                # Header Visualisasi
                st.markdown("## Visualisasi Hasil Analisis")
                
                # Tampilkan Chart
                figs = create_visuals(df_count, df_percent)
                col_chart1, col_chart2 = st.columns(2)
                
                if len(figs) > 0:
                    with col_chart1:
                        st.pyplot(figs[0])
                if len(figs) > 1:
                    with col_chart2:
                        st.pyplot(figs[1])
                
                # Tampilkan Data Tabular (Expandable agar rapi)
                with st.expander("Lihat Detail Data Tabel"):
                    st.dataframe(df_info)
                    st.dataframe(df_raw.head())

                # Tombol Download
                excel_data = create_excel_export(df_info, df_count, df_percent, df_raw)
                st.download_button(
                    label="📥 Download Hasil (Excel)",
                    data=excel_data,
                    file_name="Hasil_Analisis_PoliSense.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

if __name__ == "__main__":
    main()