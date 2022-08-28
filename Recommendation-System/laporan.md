# Domain Proyek
Proyek yang dikembangkan adalah sistem rekomendasi terhadap judul film berdasarkan rating yang telah diberikan pengguna yang diperoleh dari website penyedia dataset, Kaggle. Permasalahan yang diselesaikan dalam proyek ini adalah supaya pengguna dapat memdapatkan rekomendasi film yang memilikii kemiripan dengan preferensi pengguna di masa lalu [[1]](https://www.dicoding.com/academies/319/tutorials/19662). Teknik ini disebut dengan *collaborative filtering*.

# Business Understanding
## Problem statements
- Bagaimana membuat model yang memungkinkan untuk memberikan rekomendasi kepada pengguna?
- Bagaimana membuat sistem rekomendasi yang dapat memberikan rekomendasi berdasarkan kemiripan preferensi pengguna?

## Goals
- Memberikan rekomendasi sejumlah judul film yanf sesuai dengan preferensi pengguna berdasarkan rating yang telah diberikan sebelumya.
- Mengidentifikasi judul-judul film yang mirip dan belum pernah dikunjungi oleh pengguna untuk direkomendasikan.

## Solution statements
Untuk mencapai tujuan, masalah ini dapat diselesaikan dengan menerapkan teknsik yang disebut dengan *collaborative filtering*. metode ini menghasilkan rekomendasi berdasarkan pola penggunaan tanpa memerlukan informasi eksogen tentang item atau pengguna [[2]](https://link.springer.com/chapter/10.1007/978-1-0716-2197-4_3). Algoritma *collaborative filtering* telah menunjukkan kualitas prediksi yang bagus baik dalam penelitian akademis maupun dalam aplikasi industri. Kualitas rekomendasi yang diberikan dengan menggunakan metode ini sangat bergantung dari opini pengguna lain (neighbor) terhadap suatu item. Belakangan diketahui bahwa melakukan reduksi neighbor (yaitu dengan memotong neighbor sehingga hanya beberapa pengguna yang memiliki kesamaan / similiarity tertinggi sajalah yang akan digunakan dalam perhitungan) mampu meningkatkan kualitas rekomendasi yang diberikan [[3]](http://www.jurnal.stmik-mi.ac.id/index.php/jcb/article/view/167/189).

# Data Understanding
![image](https://user-images.githubusercontent.com/36911342/187083551-b84e8de5-5be4-413d-bf73-4ca74e5ac4e3.png)

Dataset yang digunakan adalah data [Movie Recommendation System](https://www.kaggle.com/datasets/dev0914sharma/dataset) yang memiliki 2 buah dataset, yaitu **Dataset.csv** dan **Movie_Id_Titles.csv**.

- **Dataset.csv**
    
    Dataset ini memiliki 4 kolom dan 100003 baris data. Diantaranya adalah sebagai berikut.
    - **user_id** merupakan id dari pengguna yang memberikan rating kepada film.
    - **item_id** meruoakan id dari film yang diberikan rating oleh pengguna.
    - **rating** merupakan rating atau penilaian yang diberikan oleh pengguna kepada film yang dipilih.
    - **timestamp** merupakan waktu kapan pengguna memberikan penilaiian kepada film.

- **Movie_Id_Titles.csv**.

    Dataset ini memiliki 2 kolom dan 1682 baris data. Diantaranya adalah sebagai berikut.
    - **item_id** merupakan id dari film yang tersedia.
    - **title** meruakan judul film yang tersedia.
 
 # Data Preparation
 ## Univariate Explanatory Data Analysis
Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- **rating** : merupakan penilaian movie yang diberikan oleh pengguna.
- **movie** : merupakanjudul movie yang tersedia.

Pada tahap ini, akan dilakukan explorasi terhadap variabel-variabel tersebut.
Dari hasil *Univariate Explanatory Data Analysis*, dapat diketahui bahwa:
- Dataset movie ini memiliki 2 kolom dengan 1682 baris data.
- Tipe data **item_id** adalah numerik dan **title** adalah kategori.
- Dataset rating ini memiliki 4 kolom dengan 100003 baris data.
- Tipe data pada semua fitur adalah numerik.
- Skala penilaian adalah 1 sampai 5.

## Penggabungan Dataset
Pada tahap ini, yang harus dilakukan adalah menggabungkan 2 dataset yang terpisah berdasarkan item_id.
```python
movies = pd.merge(movie, rating, on='item_id')
movies
```
![image](https://user-images.githubusercontent.com/36911342/187083917-577555c4-d678-4aae-bdd2-59aaf6b4980b.png)

## Split Dataset
Komposisi yang digunakan untuk membagi dataset ini adalah 80:20. Sebelum itu, diperlukkan mapping data user dan resto menjadi satu nilai terlebih dahulu. Lalu, membuat rating dalam sekala 0 - 1 agar mempermudah dalam proses training.
```python
rating = rating.sample(frac=1, random_state=43)
rating
```
Split dataset:

```python
# Membuat variabel x untuk mencocokkan data user dan movie menjadi satu nilai
x = rating[['user', 'movie']].values

# Membuat variabel y untuk membuat rating dari hasil
y = rating['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values

# Membagi menjadi 80% data training dan 20% data validasi
train_indices = int(0.8 * rating.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:]
)
```

# Modelling
Pada tahap ini, model menghitung skor kecocokan antara pengguna dengan teknik embedding.

1. Melakukan proses embedding terhadap data user dan film.
2. Melakukan operasi perkalian dot product antara embedding user dan film.
3. Dapat juga dengan menambahkan bias untuk setiap user dan film.
4. Skor kecocokan ditetapkan dalam skala [0, 1] dengan fungsi aktivasi sigmoid.

# Evaluation
![image](https://user-images.githubusercontent.com/36911342/187084151-ff152298-e467-470d-ba90-36efec8f5419.png)

Dari hasil visualiasi pada gambar di atas, diperoleh informasi sebagai berikut.

- RMSE pada data training dan validasi berada di angka tidak lebih dari 0.25
- Semakin banyak epoch, semakin kecil RMSE yang diperoleh

Hasil pengujian mendapatkan rekomendasi film.

![image](https://user-images.githubusercontent.com/36911342/187084308-ecb85587-f9ab-4cba-a234-1114d402c5b7.png)

# Kesimpulan
Berdasarkan serangkaian proses yang telah dilakukan, hasil dari rekomendasi yang diberikan oleh sistem ini dengan menerapkan teknik *collaborative filtering* cukup baik dan akurat dengan hasil RMSE yang kecil.

# Referensi
# Referensi
[1] Machine Learning Terapan, Dicoding

[2] Koren, Yehuda, Steffen Rendle, and Robert Bell. "Advances in collaborative filtering." Recommender systems handbook (2022): 91-142.

[3] Wijaya, Anderias, and Deni Alfian. "Sistem Rekomendasi Laptop Menggunakan Collaborative Filtering Dan Content-Based Filtering." Jurnal Computech & Bisnis 12.1 (2018): 11-27.
