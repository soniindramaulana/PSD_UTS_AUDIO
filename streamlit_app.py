# Import library yang dibutuhkan
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import streamlit as st
from streamlit_option_menu import option_menu
import os
import librosa
from scipy.stats import skew, kurtosis, mode, iqr

# ===================================================================
# === 1. PENGATURAN TAMPILAN (CSS) & KONFIGURASI HALAMAN ===
# ===================================================================

# Mengatur konfigurasi halaman (Judul Tab dan Layout)
st.set_page_config(
    page_title="Klasifikasi Audio",
    page_icon="🎵",
    layout="wide"
)

# CSS Kustom untuk mempercantik tampilan
st.markdown("""
<style>
    /* Mengubah font utama */
    html, body, [class*="st-"], .st-emotion-cache-1n4k3ym {
        font-family: 'Helvetica Neue', 'Helvetica', 'Arial', sans-serif;
    }

    /* Mengatur container utama */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }

    /* Kustomisasi Sidebar */
    .st-emotion-cache-16txtl3 {
        padding: 2rem 1rem;
    }
    
    /* Menyembunyikan footer "Made with Streamlit" */
    footer {
        visibility: hidden;
    }
    
    /* Kustomisasi Tombol */
    .stButton>button {
        border: none;
        background-color: #007BFF; /* Biru cerah */
        color: white;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: bold;
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0056b3; /* Biru lebih gelap saat hover */
        color: white;
        transform: scale(1.02);
    }

    /* Kustomisasi File Uploader */
    .stFileUploader {
        border: 2px dashed #007BFF;
        border-radius: 10px;
        padding: 1.5rem;
        background-color: #f8f9fa; /* Latar belakang abu-abu muda */
    }
    
    .stFileUploader label {
        color: #007BFF;
        font-weight: 600;
    }
    
    /* Kustomisasi st.info */
    .st-emotion-cache-q8sbsg {
        background-color: #e6f7ff; /* Biru sangat muda */
        border: 1px solid #007BFF;
        border-radius: 10px;
    }

    /* ===============================================================
    PERBAIKAN CSS: Menghilangkan background biru pada label st.metric
    ===============================================================
    */
    div[data-testid="stMetric"] label {
        background-color: transparent !important;
        padding: 0 !important;
        border-radius: 0 !important;
    }

</style>
""", unsafe_allow_html=True)

# ===================================================================
# === 2. MEMUAT DATA & PENGATURAN SIDEBAR ===
# ===================================================================

# Membaca data dari file csv
@st.cache_data
def load_data():
    data = pd.read_csv('hasil_statistik.csv')
    fitur = data.drop(columns=['Label'], axis =1)
    target = data['Label']
    return data, fitur, target

data, fitur, target = load_data()

# Judul di atas sidebar
st.sidebar.title("Navigasi Aplikasi")
st.sidebar.markdown("---")

with st.sidebar:
    selected = option_menu(
        menu_title=None, # Dihilangkan karena sudah ada judul di atas
        options=["Dataset", "Split Data", "Normalisasi Data", "Hasil Akurasi", "Reduksi Data","Reduksi Data X Grid Search","Prediksi"],
        icons=["database", "layout-split", "sliders", "bar-chart-steps", "graph-down", "grid-3x3-gap", "mic"], # Menambahkan ikon
        default_index=0
    )

# Judul Utama Aplikasi di Halaman Utama
st.title("🎵 Aplikasi Klasifikasi Emosi Audio")
st.markdown("---")

# ===================================================================
# === 3. KONTEN HALAMAN ===
# ===================================================================

if selected == "Dataset":
    st.header("Eksplorasi Dataset")
    st.markdown("Dataset ini berisi **2800 data** audio yang telah diekstraksi menjadi **21 fitur** statistik, ZCR, dan RMSE.")
    
    # Tampilkan dataframe dengan rapi
    st.dataframe(data, use_container_width=True)
    
    # Gunakan 2 kolom untuk deskripsi
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Informasi Data")
        st.metric(label="Jumlah Data", value=f"{data.shape[0]}")
        st.metric(label="Jumlah Fitur", value=f"{data.shape[1] - 1}")
        st.metric(label="Jumlah Label Unik", value=f"{data['Label'].nunique()}")

    with col2:
        st.subheader("Daftar Fitur")
        # Gunakan expander agar tidak memakan tempat
        with st.expander("Klik untuk melihat 21 fitur yang digunakan"):
            st.code("""
1. Mean Audio          12. Mean ZCR
2. Median Audio        13. Median ZCR
3. Modus Audio         14. Std ZCR
4. Maxv Audio          15. Kurtosis ZCR
5. Minv Audio          16. Skew ZCR
6. Std Audio           17. Mean Energy RMSE
7. Skew Audio          18. Median Energy RMSE
8. Kurtosis Audio      19. Kurtosis Energy RMSE
9. Q1 Audio            20. Std Energy RMSE
10. Q3 Audio           21. Skew Energy RMSE
11. IQR Audio
            """)

if selected == "Split Data":
    st.header("Pembagian Data Training & Testing")
    st.markdown("Data dibagi menjadi 80% data training dan 20% data testing (`test_size=0.2`).")

    # Membagi data menjadi data training dan data testing
    fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

    # Tampilkan hasil dalam 3 kolom
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data", f"{fitur.shape[0]}", "100%")
    col2.metric("Data Training", f"{fitur_train.shape[0]}", "80%")
    col3.metric("Data Testing", f"{fitur_test.shape[0]}", "20%")
    
    st.success("Pembagian data berhasil dilakukan dengan `random_state=42` untuk reproduktibilitas.")

if selected == "Normalisasi Data":
    st.header("Normalisasi Data dengan MinMax Scaler")
    st.markdown("Fitur-fitur numerik akan diskalakan ke rentang [0, 1] agar memiliki bobot yang setara.")
    
    #membuat variabel untuk normalisasi menggunakan minmax
    fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

    minmaxscaler = MinMaxScaler()
    minmaxscaler.fit(fitur_train)

    # menyimpan model ke dalam file pickle
    pickle.dump(minmaxscaler, open('minmaxscaler.pkl','wb'))
    
    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))

    minmax_training = minmaxscaler.transform(fitur_train)
    minmax_testing = minmaxscaler.transform(fitur_test)

    st.success("Normalisasi MinMax berhasil di-fit pada data training dan disimpan ke `minmaxscaler.pkl`.")

    # Tampilkan hasil normalisasi di dalam expander
    with st.expander("Lihat Hasil Normalisasi Data Training"):
        st.dataframe(pd.DataFrame(minmax_training, columns=fitur_train.columns), use_container_width=True)
        
    with st.expander("Lihat Hasil Normalisasi Data Testing"):
        st.dataframe(pd.DataFrame(minmax_testing, columns=fitur_test.columns), use_container_width=True)

if selected == "Hasil Akurasi":
    st.header("Pencarian Akurasi KNN Terbaik (Manual)")
    st.markdown("Mencari nilai `k` (dari 1-50) untuk model KNN pasca-normalisasi MinMax.")

    fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size=0.2, random_state=42)
    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
    minmax_training = minmaxscaler.transform(fitur_train)
    minmax_testing = minmaxscaler.transform(fitur_test)

    akurasi_tertinggi = 0
    k_terbaik = []
    log_lines = []
    
    # Placeholder untuk bar kemajuan
    progress_bar = st.progress(0, text="Memulai pencarian...")

    for i, k in enumerate(list(range(1, 51))):
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(minmax_training, target_train)
        y_pred_knn = knn.predict(minmax_testing)

        akurasi_knn = accuracy_score(target_test, y_pred_knn)
        log_lines.append(f"k = {k:02d} | Akurasi: {akurasi_knn:.4f}")

        if akurasi_knn > akurasi_tertinggi:
            akurasi_tertinggi = akurasi_knn
            k_terbaik = [k]
        elif akurasi_knn == akurasi_tertinggi:
            k_terbaik.append(k)
        
        # Update progress bar
        progress_bar.progress((i + 1) / 50, text=f"Menghitung akurasi untuk k={k}...")

    progress_bar.empty() # Hapus progress bar setelah selesai
    
    st.success(f"Pencarian Selesai!")
    
    # Tampilkan hasil terbaik di ATAS dengan st.metric
    st.metric(label="Akurasi KNN Tertinggi", 
              value=f"{akurasi_tertinggi * 100:.2f} %", 
              delta=f"Diperoleh pada k = {k_terbaik}")

    # Tampilkan log di dalam expander yang rapi
    with st.expander("Lihat Log Perhitungan Lengkap"):
        st.code('\n'.join(log_lines))

if selected == "Reduksi Data":
    st.header("Reduksi Data (PCA) dengan K-Manual")
    st.markdown("Mencari kombinasi komponen PCA (dari 20 ke 1) dan `n_neighbors` (dari 1-9) terbaik.")

    fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size=0.2, random_state=42)
    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
    minmax_training = minmaxscaler.transform(fitur_train)
    minmax_testing = minmaxscaler.transform(fitur_test)

    akurasi_list2 = []
    
    # ==========================================================
    # PERBAIKAN ERROR TypeError: Hapus st.container(height=300)
    # ==========================================================
    # log_container = st.container(height=300) # <-- HAPUS BARIS INI
    log_lines = []

    with st.spinner('Melakukan reduksi PCA dan pengujian KNN... Ini mungkin memakan waktu.'):
        for n_components in range(20, 0, -1):
            pca = PCA(n_components = n_components)
            pca.fit(minmax_training)
            pca_train = pca.transform(minmax_training)
            pca_test = pca.transform(minmax_testing)

            for n_neighbors in range(1, 10):
                knn = KNeighborsClassifier(n_neighbors=n_neighbors)
                knn.fit(pca_train, target_train)
                y_pred_knn = knn.predict(pca_test)

                akurasi = accuracy_score(target_test, y_pred_knn)
                akurasi_list2.append((n_components, n_neighbors, akurasi))
                
                log_line = f"PCA={n_components:02d}, k={n_neighbors} | Akurasi: {akurasi:.4f}"
                log_lines.append(log_line)
        
        # ==========================================================
        # PERBAIKAN ERROR TypeError: 
        # Ganti log_container.code(...) menjadi st.code(...)
        # ==========================================================
        # Tampilkan log di container
        # log_container.code('\n'.join(log_lines)) # <-- GANTI BARIS INI
        st.code('\n'.join(log_lines), height=300) # <-- MENJADI SEPERTI INI


    # Cari kombinasi n_components dan n_neighbors dengan akurasi tertinggi
    best_accuracy = max(akurasi_list2, key=lambda x: x[2])
    
    st.success(f"Pencarian Selesai!")
    st.metric(label="Akurasi Tertinggi (PCA + Manual KNN)",
              value=f"{best_accuracy[2] * 100:.2f} %",
              delta=f"Komponen PCA = {best_accuracy[0]}, k = {best_accuracy[1]}")


if selected == "Reduksi Data X Grid Search":
    st.header("Reduksi Data (PCA) & Hyperparameter Tuning (Grid Search)")
    st.markdown("Mencari parameter KNN terbaik (`k`, `weights`, `metric`) terlebih dahulu, kemudian mencari jumlah komponen PCA terbaik.")
    
    df = pd.read_csv('hasil_statistik.csv')
    X = df.drop(columns=['Label'])
    y = df['Label']

    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
    X_scaled = minmaxscaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    knn = KNeighborsClassifier()

    with st.spinner('Menjalankan Grid Search CV untuk KNN... (Langkah 1/2)'):
        grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
    
    st.subheader("Hasil Grid Search KNN (Tanpa PCA)")
    st.json(grid_search.best_params_)
    best_n_neighbors = grid_search.best_params_['n_neighbors']
    best_weights = grid_search.best_params_['weights']
    best_metric = grid_search.best_params_['metric']

    accuracy_dict = {}
    
    progress_bar_pca = st.progress(0, text="Memulai pencarian PCA...")
    log_lines_pca = []
    
    with st.spinner('Mencari komponen PCA terbaik... (Langkah 2/2)'):
        n_components_list = list(range(X_train.shape[1], 0, -1))
        total_steps = len(n_components_list)
        
        for i, n_components in enumerate(n_components_list):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)
            
            best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
            best_knn_model.fit(X_train_pca, y_train)
            y_pred = best_knn_model.predict(X_test_pca)
            grid_knn_pca = accuracy_score(y_test, y_pred)
            accuracy_dict[n_components] = grid_knn_pca
            
            log_lines_pca.append(f"PCA={n_components:02d} | Akurasi: {grid_knn_pca:.4f}")
            progress_bar_pca.progress((i + 1) / total_steps, text=f"Menguji {n_components} komponen PCA...")

    progress_bar_pca.empty()
    
    best_comp = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_comp] * 100
    
    st.subheader("Hasil Akhir (GridSearch + PCA)")
    st.metric(label="Akurasi Terbaik",
              value=f"{best_accuracy:.2f} %",
              delta=f"Dengan {best_comp} Komponen PCA")
    
    with st.expander("Lihat Log Pencarian PCA"):
        st.code('\n'.join(log_lines_pca))

    # Menyimpan model
    hyperparameters = {
        'best_n_neighbors': best_n_neighbors,
        'best_weights': best_weights,
        'best_metric': best_metric,
        'best_comp': best_comp
    }
    model_data = {
        'X_train': X_train, 'y_train': y_train, 'y_test' : y_test,
        'scaler' : X_scaled, 'hyperparameters': hyperparameters
    }
    with open('minmaxgridsearchhmodel.pkl', 'wb') as model_file:
        pickle.dump(model_data, model_file)
    st.success("Model dan hyperparameter terbaik telah disimpan ke `minmaxgridsearchhmodel.pkl`.")


if selected == "Prediksi":
    st.header("Prediksi Emosi Audio 🎤")
    st.markdown("Unggah file audio `.wav` atau `.mp3` untuk mendeteksi emosi.")
    
    # Fungsi kalkulasi (tetap sama)
    @st.cache_data
    def calculate_statistics(audio_buffer):
        # Muat audio dari buffer
        x, sr = librosa.load(audio_buffer, sr=None)
        
        mean = np.mean(x)
        std = np.std(x)
        maxv = np.amax(x)
        minv = np.amin(x)
        median = np.median(x)
        skewness = skew(x)
        kurt = kurtosis(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        mode_v = mode(x, keepdims=True)[0][0] 
        iqr = q3 - q1

        zcr = librosa.feature.zero_crossing_rate(x)[0]
        mean_zcr = np.mean(zcr)
        median_zcr = np.median(zcr)
        std_zcr = np.std(zcr)
        kurtosis_zcr = kurtosis(zcr, axis=None)
        skew_zcr = skew(zcr, axis=None)

        rms_energy = librosa.feature.rms(y=x)[0]
        mean_rms = np.mean(rms_energy)
        median_rms = np.median(rms_energy)
        std_rms = np.std(rms_energy)
        kurtosis_rms = kurtosis(rms_energy, axis=None)
        skew_rms = skew(rms_energy, axis=None)

        return [
            mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr,
            mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr,
            mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms
        ]

    # --- Area Upload ---
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"], label_visibility="collapsed")
    
    # Placeholder untuk hasil
    result_placeholder = st.empty()

    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.type.split("/")[1]}')
        
        # --- Tombol Aksi ---
        col1, col2 = st.columns([1, 1])

        with col1:
            if st.button("Cek Nilai Statistik"):
                with st.spinner("Mengekstraksi fitur..."):
                    statistik = calculate_statistics(uploaded_file)
                    result = {
                        'Mean Audio': statistik[0], 'Median Audio': statistik[1], 'Mode Audio': statistik[2],
                        'Maxv Audio': statistik[3], 'Minv Audio': statistik[4], 'Std Audio': statistik[5],
                        'Skew Audio': statistik[6], 'Kurtosis Audio': statistik[7], 'Q1 Audio': statistik[8],
                        'Q3 Audio': statistik[9], 'IQR Audio': statistik[10], 'Mean ZCR': statistik[11],
                        'Median ZCR': statistik[12], 'Std ZCR': statistik[13], 'Kurtosis ZCR': statistik[14],
                        'Skew ZCR': statistik[15], 'Mean Energy RMSE': statistik[16],
                        'Median Energy RMSE': statistik[17], 'Std Energy RMSE': statistik[18],
                        'Kurtosis Energy RMSE': statistik[19], 'Skew Energy RMSE': statistik[20],
                    }
                    df = pd.DataFrame([result]) # Buat jadi list of dict
                
                result_placeholder.subheader("Hasil Ekstraksi Fitur Audio")
                result_placeholder.dataframe(df, use_container_width=True)

        with col2:
            if st.button("Deteksi Audio"):
                with st.spinner('Menganalisis dan memprediksi emosi...'):
                    # 1. Ekstraksi Fitur
                    audio_features = calculate_statistics(uploaded_file)
                    result = {
                        'Mean Audio': audio_features[0], 'Median Audio': audio_features[1], 'Mode Audio': audio_features[2],
                        'Maxv Audio': audio_features[3], 'Minv Audio': audio_features[4], 'Std Audio': audio_features[5],
                        'Skew Audio': audio_features[6], 'Kurtosis Audio': audio_features[7], 'Q1 Audio': audio_features[8],
                        'Q3 Audio': audio_features[9], 'IQR Audio': audio_features[10], 'Mean ZCR': audio_features[11],
                        'Median ZCR': audio_features[12], 'Std ZCR': audio_features[13], 'Kurtosis ZCR': audio_features[14],
                        'Skew ZCR': audio_features[15], 'Mean Energy RMSE': audio_features[16],
                        'Median Energy RMSE': audio_features[17], 'Std Energy RMSE': audio_features[18],
                        'Kurtosis Energy RMSE': audio_features[19], 'Skew Energy RMSE': audio_features[20],
                    }
                    data_tes = pd.DataFrame([result]) # Buat jadi list of dict

                    # 2. Load Model & Data Training
                    with open('minmaxgridsearchhmodel.pkl', 'rb') as model_file:
                        saved_data = pickle.load(model_file)
                    
                    df = pd.read_csv('hasil_statistik.csv')
                    X = df.drop(columns=['Label'])
                    y = df['Label']
                    
                    # Ambil urutan kolom yang benar
                    feature_columns_order = X.columns.tolist()
                    data_tes = data_tes[feature_columns_order] # Paksa urutan kolom

                    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
                    X_scaled = minmaxscaler.fit_transform(X) # Fit scaler pada SEMUA data X
                    
                    X_train = saved_data['X_train'] # Gunakan X_train dari model
                    y_train = saved_data['y_train'] # Gunakan y_train dari model

                    # 3. Access Hyperparameters
                    hyperparameters = saved_data['hyperparameters']
                    best_n_neighbors = hyperparameters['best_n_neighbors']
                    best_weights = hyperparameters['best_weights']
                    best_metric = hyperparameters['best_metric']
                    best_comp = hyperparameters['best_comp']

                    # 4. Transformasi Data Uji
                    pca = PCA(n_components=best_comp)
                    
                    # Gunakan scaler yang sudah di-fit pada X
                    X_test_minmax = minmaxscaler.transform(data_tes) 

                    X_train_pca = pca.fit_transform(X_train)
                    X_test_pca = pca.transform(X_test_minmax)

                    # 5. Prediksi
                    best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
                    best_knn_model.fit(X_train_pca, y_train)
                    predicted_label = best_knn_model.predict(X_test_pca)
                    
                    # 6. Tampilkan Hasil
                    # Gunakan st.metric untuk tampilan hasil yang keren
                    result_placeholder.metric("Hasil Deteksi Emosi", f"🔊 {predicted_label[0]} 🔊")
