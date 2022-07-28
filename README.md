# Klasifikasi Diabetes pada Pasien - Diki Hamdani

# 1. Domain Proyek
Dilansir dari [Halodoc](https://www.halodoc.com/kesehatan/diabetes), diabetes adalah penyakit kronis atau yang berlangsung jangka panjang. Penyakit ini ditandai dengan meningkatnya kadar gula darah (glukosa) hingga di atas nilai normal. Diabetes terjadi ketika tubuh pengidapnya tidak lagi mampu mengambil gula (glukosa) ke dalam sel dan menggunakannya sebagai energi. Kondisi ini pada akhirnya menghasilkan penumpukan gula ekstra dalam aliran darah tubuh.

Salah satu faktor penyebab diabetes adalah karena adanya gangguan dalam tubuh, sehingga tubuh tidak mampu menggunakan glukosa dara ke dalam hati. Sehingga, glukosa menumpuk dalam darah. Oleh karena itu, salah satu cara yang dapat dilakukan untuk mencegah diabetes dengan mengembangkan model machine learning yang dapat memprediksi apakah seseorang terindikasi diabetes atau tidak berdsarkan parameter-paremeter tertentu [[1]](https://www.halodoc.com/kesehatan/diabetes).

# 2. Business Understanding
## 2.1 Problem statements
- Bagaimana membuat model yang memungkinkan untuk melakukan prediksi diabetes pada seseorang?
- Model machine learning manakah yang dapat menyelasikan permasalahan dengan baik?

## 2.2 Goals
- Mengetahui karakteristik yang berpengaruh terhadap diabetes.
- Mengetahui model yang terbaik untuk memprediksi diabetes pada seseorang.

## 2.3 Solution statements
Untuk mencapai tujuan, masalah ini dapat menggunakan perbangingan dari beberapa model, diantaranya adalah sebagai berikut.
- K-Nearest Neighbor
K-Nearest Neighbor (KNN) adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan. KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). KNN bisa digunakan untuk kasus klasifikasi dan regresi sehingga cocok digunakan untuk permasalah klasifikasi diabetes [[2]](https://www.dicoding.com/academies/319/tutorials/18580).

- Random Forest
Random Forest adalah algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni. Algoritma ini juga merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Dibalik model ensemble adalah sekelompok model yang bekerja bersama menyelesaikan masalah. Sehingga, tingkat keberhasilan akan lebih tinggi dibanding model yang bekerja sendirian. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir [[3]](https://www.dicoding.com/academies/319/tutorials/18585?from=18580). 

- Boosting
Sama halnya seperti algoritma random forest, algortima Boosting juga merupakan salah satu algoritma machine learning yang termasuk ke dalam kategori ensembel. Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan [[4]](https://www.dicoding.com/academies/319/tutorials/18590?from=18585).

    Algoritma boosting bertujuan untuk meningkatkan performa akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa.

# 3. Data Understanding
Dataset yang digunakan adalah data [diabetes](https://www.kaggle.com/code/swetarajsinha/prediction-diabetes-logistic-regression/data) yang memiliki 2000 baris dan 9 kolom. Berikut adalah 9 kolom yang akan digunakan.
<img width="755" alt="image" src="https://user-images.githubusercontent.com/36911342/181467294-b1548b6a-7833-4d0c-92fd-3e46ff9d2218.png">


- **Pregnancies**: Kategori kehamilan
- **Glucose**: Kadar gula pada tubuh
- **BloodPressure**: Tekanan darah
- **SkinThickness**: Tingkat ketebalan kulit
- **Insulin**: Insulin
- **BMI**: Berat badan
- **DiabetesPedigreeFunction**: Fungsi silsilah diabetes
- **Age**: Usia
- **Outcome**: Indikasi apakah pasien terindikasi diabetes (1) atau tidak (0).

# 4. Data Loading
<img width="654" alt="image" src="https://user-images.githubusercontent.com/36911342/181466898-4930f705-80fc-4be8-abcc-ed53e3c082d7.png">

# 5. Data Preparation
## 5.1 Deskripsi Variabel
<img width="832" alt="image" src="https://user-images.githubusercontent.com/36911342/181467816-d6996e7f-419d-41bd-b0f2-987ca0d138c7.png">

## 5.2 Menangani Missing Value
<img width="814" alt="image" src="https://user-images.githubusercontent.com/36911342/181468053-99a7ecaa-ddb9-48d3-a3da-5245cfe0e74f.png">

## 5.3 Menangani Outliers
```Python
Q1 = diabetes.quantile(0.25)
Q3 = diabetes.quantile(0.75)
IQR = Q3 - Q1

diabetes = diabetes[~((diabetes < (Q1 - 1.5 * IQR)) | (diabetes > (Q3 + 1.5 * IQR))).any(axis=1)]
```
## 5.4 Split Datset
<img width="226" alt="image" src="https://user-images.githubusercontent.com/36911342/181468763-633d2ba0-e938-4084-8bf7-c95f6dceafea.png">

# Modelling
<img width="228" alt="image" src="https://user-images.githubusercontent.com/36911342/181468916-d7bcb8d1-b600-4f80-b3c0-34e590ebb6c8.png">

# Evaluation
![image](https://user-images.githubusercontent.com/36911342/181469005-bbded2c6-c0c5-4da3-808a-b44a8d4164e3.png)

Score:

<img width="906" alt="image" src="https://user-images.githubusercontent.com/36911342/181469351-2ac91df0-60ac-487b-ab89-cb09732302ed.png">

