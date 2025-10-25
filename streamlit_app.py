# Import library yang dibutuhkan untuk melakukan data preprocessing
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



# Membaca data dari file csv
data = pd.read_csv('hasil_statistik.csv')
# Memisahkan kolom target (label) dari kolom fitur
fitur = data.drop(columns=['Label'], axis =1)  # Kolom fitur
target = data['Label']  # Kolom target


st.write("# Ekstraksi Ciri Audio Untuk Klasifikasi Audio")

with st.sidebar:
  selected = option_menu(
      menu_title="Main Menu",
      options=["Dataset", "Split Data", "Normalisasi Data", "Hasil Akurasi", "Reduksi Data","Reduksi Data X Grid Search","Prediksi"],
      default_index=0
  )


if selected == "Dataset":
    st.write('''## Dataset''')
    st.write(data)
    st.write('''Dataset ini merupakan hasil Ekstraksi Ciri Audio yang mana audio yang digunakan berasal dari website Kaggle.''')
    st.write('''Dataset ini memiliki jumlah data sebanyak 2800 dengan 21 fitur.''')
    st.write('''#### Fitur-Fitur Pada Dataset''')
    st.info('''
    Fitur yang akan digunakan adalah sebagai berikut :
    1. Mean Audio
    2. Median Audio
    3. Modus Audio
    4. Nilai maksimum Audio
    5. Nilai minimum Audio
    6. Standar deviasi Audio
    7. Nilai kemiringan (skewness) Audio
    8. Nilai Keruncingan (kurtosis) Audio
    9. Nilai kuartil bawah (Q1) Audio
    10. Nilai kuartil atas (Q3) Audio
    8. Nilai IQR Audio
    12. Mean ZCR
    13. Median ZCR
    14. Std ZCR
    15. Kurtosis ZCR
    16. Skew ZCR
    17. Mean Energy RMSE
    18. Median Energy RMSE
    19. Kurtosis Energy RMSE
    20. Std Energy RMSE
    21. Skew Energy RMSE
     ''')

if selected == "Split Data":

  # Membagi data menjadi data training dan data testing
  fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

  st.write('''## Membagi Data Menjadi Data Training Dan Data Testing''')
  st.write('Data dibagi menjadi 20% sebagai data testing dan 80% data training')
  st.success(f'''
  ##### Diperoleh:
  - Banyaknya Data : {fitur.shape[0]}
  - Banyak Data Training : {fitur_train.shape[0]}
  - Banyak Data Testing : {fitur_test.shape[0]}
  ''')

if selected == "Normalisasi Data":
  st.write('''## Normalisasi Menggunakan Minmax''')
  st.write('Untuk melakukan normalisasi MInmax pada python bisa menggunakan Library MinMaxScaler().')
  #membuat variabel untuk normalisasi menggunakan minmax
  fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size = 0.2, random_state=42)

  minmaxscaler = MinMaxScaler()
  minmaxscaler.fit(fitur_train)

  # menyimpan model ke dalam file pickle
  pickle.dump(minmaxscaler, open('minmaxscaler.pkl','wb'))


  # memanggil kembali model normalisasi minmaxscaler dari file pickle
  minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))

  # menerapkan normalisasi minmax pada data training dan testing
  minmax_training = minmaxscaler.transform(fitur_train)
  minmax_testing = minmaxscaler.transform(fitur_test)

  st.write("#### Hasil Normalisasi MinMaxScaler pada X-train ")
  st.write(minmax_training)
  st.write("#### Hasil Normalisasi MinMaxScaler pada X-test ")
  st.write(minmax_testing)

if selected == "Hasil Akurasi":
  st.write("## Hasil Akurasi Dari Normalisasi Minmax Dengan Model KNN")

  # Membagi data menjadi data training dan data testing
  fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size=0.2, random_state=42)

  # memanggil kembali model normalisasi minmaxscaler dari file pickle
  minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))

  # menerapkan normalisasi minmax pada data training dan testing
  minmax_training = minmaxscaler.transform(fitur_train)
  minmax_testing = minmaxscaler.transform(fitur_test)

  akurasi_tertinggi = 0
  k_terbaik = []

  for k in list(range(1, 51)):

      # membangun model KNN
      knn = KNeighborsClassifier(n_neighbors = k)
      knn.fit(minmax_training, target_train)
      y_pred_knn = knn.predict(minmax_testing)

      # akurasi
      akurasi_knn = accuracy_score(target_test, y_pred_knn)
      st.write(f"Hasil akurasi dengan k = {k} : {akurasi_knn}")

      if akurasi_knn > akurasi_tertinggi:
          akurasi_tertinggi = akurasi_knn
          k_terbaik = [k]
      elif akurasi_knn == akurasi_tertinggi:
          k_terbaik.append(k)

  st.success(f"Hasil akurasi tertinggi adalah {akurasi_tertinggi} pada k = {k_terbaik}")

if selected == "Reduksi Data":
  st.write("## Mereduksi Data Berdasarkan Pencarian Parameter K- Manual ")

  # Membagi data menjadi data training dan data testing
  fitur_train, fitur_test, target_train, target_test = train_test_split(fitur, target, test_size=0.2, random_state=42)

  #Membaca dataframe hasil dari normalisasi minmax Scalling sebelumnya
  # memanggil kembali model normalisasi minmaxscaler dari file pickle
  minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))

  # menerapkan normalisasi minmax pada data training dan testing
  minmax_training = minmaxscaler.transform(fitur_train)
  minmax_testing = minmaxscaler.transform(fitur_test)


  st.write("### Reduksi Data Menggunakan Data Hasil Normalisasi MinMaxScaler")
  st.info("Dikarenakan pada hasil akurasi dari normalisasi minmaxscaler dengan model KNN mendapat akurasi terbaik pada k-9, maka pada model PCA ini untuk nilai k diisi dengan 9")

  akurasi_list2 = []

  for n_components in range(20, 0, -1):

      # Membangun model PCA dengan jumlah komponen utama yang sesuai
      pca = PCA(n_components = n_components)
      pca.fit(minmax_training)

      pca_train = pca.transform(minmax_training)
      pca_test = pca.transform(minmax_testing)

      for n_neighbors in range(1, 10):  # Loop untuk n_neighbors dari 1 hingga 9
          knn = KNeighborsClassifier(n_neighbors=n_neighbors)
          knn.fit(pca_train, target_train)
          y_pred_knn = knn.predict(pca_test)

          akurasi = accuracy_score(target_test, y_pred_knn)
          akurasi_list2.append((n_components, n_neighbors, akurasi))  # Simpan akurasi bersama dengan n_components dan n_neighbors

          st.write(f"Jumlah komponen utama:{ n_components} n_neighbors: { n_neighbors} Akurasi:{ akurasi}")

  # Cari kombinasi n_components dan n_neighbors dengan akurasi tertinggi
  best_accuracy = max(akurasi_list2, key=lambda x: x[2])
  st.success(f"Kombinasi terbaik: Jumlah komponen utama = { best_accuracy[0]} n_neighbors = {best_accuracy[1]} Akurasi = { best_accuracy[2]}")


if selected == "Reduksi Data X Grid Search":
    st.write("## Reduksi Data dengan menentukan komponen utama dengan parameter terbaik menggunakan Grid Search ")
    st.write("### Reduksi Data Menggunakan Data Hasil Normalisasi MinMax Scaler")
    # Membaca data dari file CSV
    df = pd.read_csv('hasil_statistik.csv')

    # Memisahkan kolom target (label) dari kolom fitur
    X = df.drop(columns=['Label'])  # Kolom fitur
    y = df['Label']  # Kolom target

    # Normalisasi data menggunakan StandardScaler
    minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
    X_scaled = minmaxscaler.fit_transform(X)

    # Memisahkan data menjadi data latih dan data uji
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Mendefinisikan parameter yang ingin diuji
    param_grid = {
        'n_neighbors': list(range(1, 21)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    # Mendefinisikan model KNN
    knn = KNeighborsClassifier()

    # Mendefinisikan Grid Search dengan model KNN dan parameter yang diuji
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)  # Menggunakan data latih yang belum diproses PCA

    # Menampilkan parameter terbaik
    st.write("Best Parameters:", grid_search.best_params_)

    # Menggunakan PCA dengan komponen utama terbaik
    best_n_neighbors = grid_search.best_params_['n_neighbors']
    best_weights = grid_search.best_params_['weights']
    best_metric = grid_search.best_params_['metric']

    accuracy_dict = {}
    for n_components in range(X_train.shape[1], 0, -1):
        pca = PCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # Membuat model KNN dengan hyperparameter terbaik
        best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
        best_knn_model.fit(X_train_pca, y_train)

        # Membuat prediksi menggunakan model terbaik
        y_pred = best_knn_model.predict(X_test_pca)

        # Mengukur akurasi model terbaik pada data uji
        grid_knn_pca = accuracy_score(y_test, y_pred)

        # Menyimpan akurasi dalam dictionary
        accuracy_dict[n_components] = grid_knn_pca

    # Mencari nilai k terbaik
    best_comp = max(accuracy_dict, key=accuracy_dict.get)
    best_accuracy = accuracy_dict[best_comp] * 100
    st.write(f"\nBest Accuracy pada Grid Search KNN MinMax Scaler : {best_comp} PCA components: {best_accuracy:.2f}%")

    # Store the hyperparameters in a dictionary
    hyperparameters = {
        'best_n_neighbors': best_n_neighbors,
        'best_weights': best_weights,
        'best_metric': best_metric,
        'best_comp': best_comp
    }

    # Store the model and hyperparameters in a dictionary before pickling
    model_data = {
        'X_train': X_train,
        'y_train': y_train,
        'y_test' : y_test,
        'scaler' : X_scaled,
        'hyperparameters': hyperparameters
    }

    # Save the model and hyperparameters using pickle
    with open('minmaxgridsearchhmodel.pkl', 'wb') as model_file:
        pickle.dump(model_data, model_file)

if selected == "Prediksi":

    def calculate_statistics(audio_path):
        x, sr = librosa.load(audio_path)

        mean = np.mean(x)
        std = np.std(x)
        maxv = np.amax(x)
        minv = np.amin(x)
        median = np.median(x)
        skewness = skew(x)
        kurt = kurtosis(x)
        q1 = np.quantile(x, 0.25)
        q3 = np.quantile(x, 0.75)
        
        # PERBAIKAN WARNING: Tambahkan keepdims=True dan ambil [0][0] untuk scalar
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

        # Pastikan urutan return sesuai dengan dictionary 'result'
        return [
            mean, median, mode_v, maxv, minv, std, skewness, kurt, q1, q3, iqr,
            mean_zcr, median_zcr, std_zcr, kurtosis_zcr, skew_zcr,
            mean_rms, median_rms, std_rms, kurtosis_rms, skew_rms
        ]
        
    uploaded_file = st.file_uploader("Pilih file audio...", type=["wav","mp3"])


    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        st.title("Prediksi Class Data Audio Menggunakan MinMax")
        
        if st.button("Cek Nilai Statistik"):
            # Simpan file audio yang diunggah
            audio_path = "audio_diunggah.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Hitung statistik untuk file audio yang diunggah
            # PERBAIKAN BUG: ganti 'audio_features' menjadi 'statistik'
            statistik = calculate_statistics(audio_path)

            results = []
            
            # PERBAIKAN NAMA KOLOM: Samakan persis dengan CSV
            result = {
                'Mean Audio': statistik[0],
                'Median Audio': statistik[1],
                'Mode Audio': statistik[2],
                'Maxv Audio': statistik[3],
                'Minv Audio': statistik[4],
                'Std Audio': statistik[5],
                'Skew Audio': statistik[6],
                'Kurtosis Audio': statistik[7],
                'Q1 Audio': statistik[8],
                'Q3 Audio': statistik[9],
                'IQR Audio': statistik[10],
                'Mean ZCR': statistik[11],
                'Median ZCR': statistik[12],
                'Std ZCR': statistik[13],
                'Kurtosis ZCR': statistik[14],
                'Skew ZCR': statistik[15],
                'Mean Energy RMSE': statistik[16],
                'Median Energy RMSE': statistik[17],
                'Std Energy RMSE': statistik[18],      # Indeks 18
                'Kurtosis Energy RMSE': statistik[19],  # Indeks 19
                'Skew Energy RMSE': statistik[20],
            }
            results.append(result)
            df = pd.DataFrame(results)
            st.write(df)

            # Hapus file audio yang diunggah
            os.remove(audio_path)

        if st.button("Deteksi Audio"):

            # Memuat data audio yang diunggah dan menyimpannya sebagai file audio
            audio_path = "audio_diunggah.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Menghitung statistik untuk file audio yang diunggah
            audio_features = calculate_statistics(audio_path)
            results = []
            
            # PERBAIKAN NAMA KOLOM: Samakan persis dengan CSV
            result = {
                'Mean Audio': audio_features[0],
                'Median Audio': audio_features[1],
                'Mode Audio': audio_features[2],
                'Maxv Audio': audio_features[3],
                'Minv Audio': audio_features[4],
                'Std Audio': audio_features[5],
                'Skew Audio': audio_features[6],
                'Kurtosis Audio': audio_features[7],
                'Q1 Audio': audio_features[8],
                'Q3 Audio': audio_features[9],
                'IQR Audio': audio_features[10],
                'Mean ZCR': audio_features[11],
                'Median ZCR': audio_features[12],
                'Std ZCR': audio_features[13],
                'Kurtosis ZCR': audio_features[14],
                'Skew ZCR': audio_features[15],
                'Mean Energy RMSE': audio_features[16],
                'Median Energy RMSE': audio_features[17],
                'Std Energy RMSE': audio_features[18],      # Indeks 18
                'Kurtosis Energy RMSE': audio_features[19],  # Indeks 19
                'Skew Energy RMSE': audio_features[20],
            }
            results.append(result)
            data_tes = pd.DataFrame(results) # data_tes sekarang punya nama kolom yang BENAR


            # Load the model and hyperparameters
            with open('minmaxgridsearchhmodel.pkl', 'rb') as model_file:
                saved_data = pickle.load(model_file)

            df = pd.read_csv('hasil_statistik.csv')

            # Memisahkan kolom target (label) dari kolom fitur
            X = df.drop(columns=['Label'])  # Kolom fitur
            y = df['Label']  # Kolom target

            # Normalisasi data menggunakan StandardScaler
            minmaxscaler = pickle.load(open('minmaxscaler.pkl','rb'))
            
            # Scaler di-fit ulang ke X (yang punya nama kolom BENAR)
            X_scaled = minmaxscaler.fit_transform(X) 

            # Memisahkan data menjadi data latih dan data uji
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Access hyperparameters
            best_n_neighbors = saved_data['hyperparameters']['best_n_neighbors']
            best_weights = saved_data['hyperparameters']['best_weights']
            best_metric = saved_data['hyperparameters']['best_metric']
            best_comp = saved_data['hyperparameters']['best_comp']

            # Melakukan PCA pada data audio yang diunggah
            pca = PCA(n_components=best_comp)

            # Baris ini sekarang akan SUKSES karena:
            # 1. 'minmaxscaler' di-fit pada X (nama kolom: 'Mean Audio', 'IQR Audio')
            # 2. 'data_tes' juga punya nama kolom: 'Mean Audio', 'IQR Audio'
            X_test_minmax = minmaxscaler.transform(data_tes)

            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test_minmax)

            # Membuat model KNN dengan hyperparameter terbaik
            best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, metric=best_metric)
            best_knn_model.fit(X_train_pca, y_train)

            predicted_label = best_knn_model.predict(X_test_pca)

            # Menampilkan hasil prediksi
            st.success(f"Emosi Terdeteksi: {predicted_label[0]}")


            # Menghapus file audio yang diunggah
            os.remove(audio_path)


