import streamlit as st
import pandas as pd
import re
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO  # Untuk ekspor Excel

# Impor Pustaka Inti (AI/ML)
from bertopic import BERTopic
from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ---------------------------------------------------------------------
# KONFIGURASI APLIKASI
# ---------------------------------------------------------------------

# Atur layout halaman menjadi "wide" agar chart lebih lega
st.set_page_config(layout="wide")

# Download 'punkt' NLTK sekali saja
nltk.download('punkt')
nltk.download('punkt_tab')

# ---------------------------------------------------------------------
# BAGIAN 1: MEMUAT MODEL (DENGAN CACHE)
# Fungsi-fungsi ini hanya akan dijalankan SEKALI saat aplikasi dimulai.
# ---------------------------------------------------------------------

#@st.cache_resource
#def load_sastrawi_models():
#    """Memuat Stemmer dan Stopwords Sastrawi ke cache."""
#    print("Loading Sastrawi models...")
#    factory_stem = StemmerFactory()
#    stemmer = factory_stem.create_stemmer()
#    factory_stop = StopWordRemoverFactory()
#    stopwords = factory_stop.get_stop_words()
#    print("Sastrawi models loaded.")
#    return stemmer, stopwords

@st.cache_resource
def load_bert_sentiment_model():
    """Memuat model IndoBERT Sentiment dari Hugging Face ke cache."""
    print("Loading BERT sentiment model...")
    # Gunakan 'device=0' jika Anda punya GPU (NVIDIA), atau 'device=-1' untuk CPU
    # Untuk server, CPU (-1) mungkin lebih stabil jika tidak ada GPU
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    classifier = pipeline("sentiment-analysis", model=model_name, device=-1) 
    print("BERT model loaded.")
    return classifier

# ---------------------------------------------------------------------
# BAGIAN 2: FUNGSI PEMROSESAN DATA
# ---------------------------------------------------------------------
# GANTI FUNGSI LAMA DENGAN INI
def preprocess_text_simple(text):
    """Fungsi preprocessing sederhana (hanya regex cleaning)."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Hapus non-huruf
    text = text.strip()
    return text
    
#def preprocess_text_advanced(text, stemmer, stopwords):
#    """Fungsi preprocessing lengkap dari Colab."""
#    if not isinstance(text, str):
#        return ""
    
#    text = text.lower()
#    text = re.sub(r'http\S+', '', text)
#    text = re.sub(r'@[a-zA-Z0-9_]+', '', text)
#    text = re.sub(r'[^a-zA-Z\s]', '', text)
#    text = text.strip()
    
#    tokens = nltk.word_tokenize(text)
#    tokens = [t for t in tokens if t not in stopwords]
    
    # Stemming adalah bagian yang paling lambat
#    tokens = [stemmer.stem(t) for t in tokens] 
#    return " ".join(tokens)

#@st.cache_data
#def run_full_analysis(uploaded_file_data, _stemmer, _stopwords, _classifier):
#    """
#    Fungsi inti yang menjalankan SEMUA proses berat.
#    Streamlit akan meng-cache hasilnya berdasarkan 'uploaded_file_data'.
#    """
    
    # --- 1. Memuat & Preprocessing Data (PROSES SANGAT LAMBAT) ---
#    print("Reading Excel...")
#    df_all = pd.read_excel(uploaded_file_data)
    
#    if 'text' not in df_all.columns:
#        st.error(f"File Error: Kolom 'text' tidak ditemukan. Kolom yang ada: {df_all.columns.tolist()}")
#        return None

#    print("Starting Advanced Preprocessing (Stemming)... THIS WILL TAKE A LONG TIME.")
    # Kita panggil stemmer & stopwords dari cache
#    stemmer, stopwords = load_sastrawi_models() 
#    df_all['text_bersih'] = df_all['text'].apply(lambda x: preprocess_text_advanced(x, stemmer, stopwords))
#    print("Preprocessing finished.")

    # Siapkan DataFrame untuk alur kerja
#    df_posts = pd.DataFrame({'post_id': df_all.index, 'text_untuk_topik': df_all['text_bersih']})
#    df_comments = pd.DataFrame({'post_id': df_all.index, 'text_untuk_sikap': df_all['text_bersih'], 'comment_text': df_all['text']})

    # --- 2. Menjalankan BERTopic ---
#    print("Starting BERTopic...")
#    vectorizer = CountVectorizer()
#    topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer, verbose=False, min_topic_size=15)
#    topics, _ = topic_model.fit_transform(df_posts['text_untuk_topik'])
#    df_posts['topik'] = topics
#    df_info_topik = topic_model.get_topic_info()
#    print("BERTopic finished.")

    # --- 3. Menjalankan Stance Analysis (BERT) ---
#    print("Starting Stance Analysis...")
#    classifier = load_bert_sentiment_model() # Panggil dari cache
#    list_komentar = df_comments['text_untuk_sikap'].tolist()
    
    # batch_size bisa disesuaikan, lebih kecil jika memori CPU terbatas
#    hasil_sikap = classifier(list_komentar, batch_size=8) 
    
#    df_comments['sikap_label_raw'] = [result['label'] for result in hasil_sikap]
#    label_mapping = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
#    df_comments['sikap'] = df_comments['sikap_label_raw'].map(label_mapping)
#    print("Stance Analysis finished.")

    # --- 4. Menggabungkan & Analisis Final ---
#    print("Generating final analysis...")
#    df_gabung = pd.merge(
#        df_comments[['post_id', 'sikap', 'comment_text']],
#        df_posts[['post_id', 'topik']],
#        on='post_id', how='left'
#    )
#    df_gabung = df_gabung[['post_id', 'comment_text', 'topik', 'sikap']]

#    analisis_final_count = df_gabung.groupby('topik')['sikap'].value_counts().unstack(fill_value=0)
#    analisis_final_percent = df_gabung.groupby('topik')['sikap'].value_counts(normalize=True).unstack(fill_value=0)
    
#    print("Analysis complete.")
    # Kembalikan semua hasil yang kita butuhkan
#    return df_info_topik, analisis_final_count, analisis_final_percent, df_gabung

# GANTI FUNGSI LAMA DENGAN INI
@st.cache_data
def run_full_analysis(uploaded_file_data, _classifier): # Hapus _stemmer dan _stopwords
    """
    Fungsi inti yang menjalankan SEMUA proses berat.
    (Versi CEPAT tanpa Sastrawi)
    """
    
    # --- 1. Memuat & Preprocessing Data (PROSES JAUH LEBIH CEPAT) ---
    print("Reading Excel...")
    df_all = pd.read_excel(uploaded_file_data)
    
    if 'text' not in df_all.columns:
        st.error(f"File Error: Kolom 'text' tidak ditemukan. Kolom yang ada: {df_all.columns.tolist()}")
        return None

    print("Starting Simple Preprocessing (Regex only)...")
    # Panggil fungsi BARU yang cepat
    df_all['text_bersih'] = df_all['text'].apply(preprocess_text_simple)
    print("Preprocessing finished.")

    # Siapkan DataFrame untuk alur kerja
    df_posts = pd.DataFrame({'post_id': df_all.index, 'text_untuk_topik': df_all['text_bersih']})
    df_comments = pd.DataFrame({'post_id': df_all.index, 'text_untuk_sikap': df_all['text_bersih'], 'comment_text': df_all['text']})

    # --- 2. Menjalankan BERTopic ---
    print("Starting BERTopic...")
    # Kita tetap butuh CountVectorizer, tapi tanpa stopwords Sastrawi
    vectorizer = CountVectorizer(stop_words=["di", "dan", "yang", "ini", "itu", "dengan", "ke", "pada", "dari"])
    topic_model = BERTopic(language="multilingual", vectorizer_model=vectorizer, verbose=False, min_topic_size=15)
    topics, _ = topic_model.fit_transform(df_posts['text_untuk_topik'])
    df_posts['topik'] = topics
    df_info_topik = topic_model.get_topic_info()
    print("BERTopic finished.")

    # --- 3. Menjalankan Stance Analysis (BERT) ---
    print("Starting Stance Analysis...")
    classifier = load_bert_sentiment_model() # Panggil dari cache
    list_komentar = df_comments['text_untuk_sikap'].tolist()
    
    hasil_sikap = classifier(list_komentar, batch_size=8) 
    
    df_comments['sikap_label_raw'] = [result['label'] for result in hasil_sikap]
    label_mapping = {'LABEL_0': 'Positive', 'LABEL_1': 'Neutral', 'LABEL_2': 'Negative'}
    df_comments['sikap'] = df_comments['sikap_label_raw'].map(label_mapping)
    print("Stance Analysis finished.")

    # --- 4. Menggabungkan & Analisis Final ---
    # (Bagian ini tidak perlu diubah)
    print("Generating final analysis...")
    df_gabung = pd.merge(
        df_comments[['post_id', 'sikap', 'comment_text']],
        df_posts[['post_id', 'topik']],
        on='post_id', how='left'
    )
    df_gabung = df_gabung[['post_id', 'comment_text', 'topik', 'sikap']]

    analisis_final_count = df_gabung.groupby('topik')['sikap'].value_counts().unstack(fill_value=0)
    analisis_final_percent = df_gabung.groupby('topik')['sikap'].value_counts(normalize=True).unstack(fill_value=0)
    
    print("Analysis complete.")
    return df_info_topik, analisis_final_count, analisis_final_percent, df_gabung
    
# ---------------------------------------------------------------------
# BAGIAN 3: FUNGSI UI (VISUALISASI & EKSPOR)
# ---------------------------------------------------------------------

def create_visuals(analisis_final_count, analisis_final_percent):
    """Membuat gambar Bar Chart dan Heatmap."""
    figs = []
    sns.set(style="whitegrid")

    # Visual 1: Bar Chart
    df_counts_filtered = analisis_final_count[analisis_final_count.index != -1]
    if not df_counts_filtered.empty:
        fig1, ax1 = plt.subplots(figsize=(15, 8))
        df_counts_filtered.plot(kind='bar', stacked=True, colormap='RdYlGn', ax=ax1)
        ax1.set_title('Jumlah Absolut Sikap per Topik', fontsize=16, pad=20)
        ax1.set_xlabel('Topik', fontsize=12)
        ax1.set_ylabel('Jumlah Tweet', fontsize=12)
        ax1.legend(title='Sikap', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(rotation=45)
        plt.tight_layout()
        figs.append(fig1)

    # Visual 2: Heatmap
    df_percent = (analisis_final_percent * 100).round(2)
    df_percent_filtered = df_percent[df_percent.index != -1]
    if not df_percent_filtered.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 7))
        sns.heatmap(df_percent_filtered, annot=True, fmt='.2f', cmap='RdYlGn', linewidths=.5, ax=ax2)
        ax2.set_title('Heatmap Distribusi Sikap (%) per Topik', fontsize=16, pad=20)
        ax2.set_xlabel('Sikap', fontsize=12)
        ax2.set_ylabel('Topik', fontsize=12)
        figs.append(fig2)
    
    return figs

def create_excel_export(df_info_topik, analisis_final_count, analisis_final_percent, df_gabung):
    """Membuat file Excel di dalam memori untuk di-download."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df_info_topik.to_excel(writer, sheet_name='Info Topik', index=False)
        analisis_final_count.to_excel(writer, sheet_name='Analisis Hitung Absolut')
        (analisis_final_percent * 100).round(2).to_excel(writer, sheet_name='Analisis Persentase (%)')
        df_gabung.to_excel(writer, sheet_name='Data Lengkap Gabungan', index=False)
    
    processed_data = output.getvalue()
    return processed_data

# ---------------------------------------------------------------------
# BAGIAN 4: ALUR APLIKASI UTAMA (MAIN UI)
# ---------------------------------------------------------------------

#def main():
#    st.title("🚀 Aplikasi Analisis Topik & Sikap")
#    st.markdown("Aplikasi ini menjalankan alur lengkap (Stemming, BERTopic, dan Stance Analysis) pada data teks Anda.")
    
    # 1. Muat model-model berat (ini akan di-cache)
    # Tampilkan ini di sidebar agar tidak mengganggu
#    with st.sidebar:
#        with st.spinner("Mempersiapkan model AI... (Hanya sekali saat start)"):
#            stemmer, stopwords = load_sastrawi_models()
#            classifier = load_bert_sentiment_model()
#        st.sidebar.success("Model AI Siap.")
#        st.sidebar.info("Aplikasi ini dibuat berdasarkan skrip Colab. Proses stemming dan analisis BERT akan memakan waktu lama.")

    # 2. Tombol Upload
#    uploaded_file = st.file_uploader("Unggah file Excel Anda (harus punya kolom 'text')", type=["xlsx"])

#    if uploaded_file is not None:
        
        # 3. Tombol Mulai Analisis
#        if st.button("Mulai Analisis 🚀"):
            
            # 4. Proses Analisis
            # Ini adalah bagian di mana semuanya berjalan, dibungkus spinner
#            with st.spinner("Sedang memproses... Ini bisa memakan waktu 30-60+ menit tergantung data Anda."):
                
                # Kita panggil fungsi analisis utama
                # `_stemmer` dll. hanya sebagai 'kunci' agar cache tahu kapan harus di-refresh
                # (tapi di sini kita anggap modelnya tidak pernah berubah)
#                hasil = run_full_analysis(uploaded_file.getvalue(), stemmer, stopwords, classifier)

#            if hasil is not None:
#                st.success("✅ Analisis Selesai!")
                
                # Unpack hasilnya
#                df_info_topik, analisis_final_count, analisis_final_percent, df_gabung = hasil
                
                # 5. Tampilkan Visualisasi
#                st.header("Visualisasi Hasil Analisis")
#                figs = create_visuals(analisis_final_count, analisis_final_percent)
#                for fig in figs:
#                    st.pyplot(fig)
                
                # 6. Tampilkan Tabel
#                st.header("Tabel Analisis")
#                tab1, tab2, tab3 = st.tabs(["Info Topik", "Analisis Persentase (%)", "Data Gabungan (Contoh)"])
                
#                with tab1:
#                    st.dataframe(df_info_topik)
#                with tab2:
#                    st.dataframe((analisis_final_percent * 100).round(2))
#                with tab3:
#                    st.dataframe(df_gabung.head(100))

                # 7. Tombol Download
#                st.header("Download Laporan Lengkap")
#                excel_data = create_excel_export(df_info_topik, analisis_final_count, analisis_final_percent, df_gabung)
                
#                st.download_button(
#                    label="📥 Download Laporan Lengkap Excel",
#                    data=excel_data,
#                    file_name="Hasil_Analisis_Topik_Sikap.xlsx",
#                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
#                )

# Menjalankan aplikasi
#if __name__ == "__main__":
#    main()
    

# GANTI FUNGSI LAMA DENGAN INI
def main():
    st.title("🚀 Aplikasi Analisis Topik & Sikap")
    st.markdown("Aplikasi ini menjalankan alur lengkap (BERTopic dan Stance Analysis) pada data teks Anda.")
    
    # 1. Muat model-model berat
    with st.sidebar:
        with st.spinner("Mempersiapkan model AI... (Hanya sekali saat start)"):
            # HAPUS SASTRAWI DARI SINI
            classifier = load_bert_sentiment_model()
        st.sidebar.success("Model AI Siap.")
        st.sidebar.info("Aplikasi ini menjalankan proses stemming dan analisis BERT dengan kecepatan maksimal.")

    # 2. Tombol Upload
    uploaded_file = st.file_uploader("Unggah file Excel Anda (harus punya kolom 'text')", type=["xlsx"])

    if uploaded_file is not None:
        
        # 3. Tombol Mulai Analisis
        if st.button("Mulai Analisis 🚀"):
            
            # 4. Proses Analisis
            with st.spinner("Sedang memproses... Ini akan jauh lebih cepat."):
                
                # HAPUS stemmer & stopwords DARI SINI
                hasil = run_full_analysis(uploaded_file.getvalue(), classifier)

            if hasil is not None:
                st.success("✅ Analisis Selesai!")
                
                # (Sisa kode di bawah ini TIDAK PERLU DIUBAH)
                df_info_topik, analisis_final_count, analisis_final_percent, df_gabung = hasil
                
                st.header("Visualisasi Hasil Analisis")
                figs = create_visuals(analisis_final_count, analisis_final_percent)
                for fig in figs:
                    st.pyplot(fig)
                
                st.header("Tabel Analisis")
                tab1, tab2, tab3 = st.tabs(["Info Topik", "Analisis Persentase (%)", "Data Gabungan (Contoh)"])
                
                with tab1:
                    st.dataframe(df_info_topik)
                with tab2:
                    st.dataframe((analisis_final_percent * 100).round(2))
                with tab3:
                    st.dataframe(df_gabung.head(100))

                st.header("Download Laporan Lengkap")
                excel_data = create_excel_export(df_info_topik, analisis_final_count, analisis_final_percent, df_gabung)
                
                st.download_button(
                    label="📥 Download Laporan Lengkap Excel",
                    data=excel_data,
                    file_name="Hasil_Analisis_Topik_Sikap.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

# Menjalankan aplikasi
if __name__ == "__main__":
    main()