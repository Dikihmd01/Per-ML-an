# Klasifikasi Diabetes pada Pasien - Diki Hamdani

## Domain Proyek
Dilansir dari [Halodoc](https://www.halodoc.com/kesehatan/diabetes), diabetes adalah penyakit kronis atau yang berlangsung jangka panjang. Penyakit ini ditandai dengan meningkatnya kadar gula darah (glukosa) hingga di atas nilai normal. Diabetes terjadi ketika tubuh pengidapnya tidak lagi mampu mengambil gula (glukosa) ke dalam sel dan menggunakannya sebagai energi. Kondisi ini pada akhirnya menghasilkan penumpukan gula ekstra dalam aliran darah tubuh.

Salah satu faktor penyebab diabetes adalah karena adanya gangguan dalam tubuh, sehingga tubuh tidak mampu menggunakan glukosa dara ke dalam hati. Sehingga, glukosa menumpuk dalam darah. Oleh karena itu, salah satu cara yang dapat dilakukan untuk mencegah diabetes dengan mengembangkan model machine learning yang dapat memprediksi apakah seseorang terindikasi diabetes atau tidak berdsarkan parameter-paremeter tertentu [[1]](https://www.halodoc.com/kesehatan/diabetes).

## Business Understanding
### Problem statements
- Bagaimana membuat model yang memungkinkan untuk melakukan prediksi diabetes pada seseorang?
- Model machine learning manakah yang dapat menyelasikan permasalahan dengan baik?

### Goals
- Mengetahui karakteristik yang berpengaruh terhadap diabetes.
- Mengetahui model yang terbaik untuk memprediksi diabetes pada seseorang.

### olution statements
Untuk mencapai tujuan, masalah ini dapat menggunakan perbangingan dari beberapa model, diantaranya adalah sebagai berikut.
- K-Nearest Neighbor
K-Nearest Neighbor (KNN) adalah algoritma yang digunakan untuk melakukan klasifikasi terhadap suatu objek, berdasarkan *k* buah data latih yang jaraknya saling berdekatan dengan objek tersebut. Syarat nilai *k* dan lebih dari satu. Dekat atau jauhnya jarak data latih yang paling dekat denan objek yang akan diklasifikasi dapat dihitung dengan metode cosine[[2]](https://jsi.cs.ui.ac.id/index.php/jsi/article/view/500).

- Random Forest
Algoritma Random forest merupakan salah satu metode yang digunakan untuk klasifikasi dan regresi. Metode ini merupakan sebuah ensemble (kumpulan) metode pembelajaran menggunakan pohon keputusan sebagai *base classifier* yang dibangun dan dikombinaskan. Ada tiga aspek penting dalam metode random forest, yaitu: melakukan bootstrap sampling untuk membangun pohon prediksi, masing-masing pohon keputusan memprediksi dengan prediktor acak, lalu random forest melakukan prediksi dengan mengombinasikan hasil dari setiap pohon keputusan dengan cara majority vote untuk klasifikasi atau rata-rata untuk regresi [[3]](hhttp://ejournal.uin-suska.ac.id/index.php/IJAIDM/article/view/4903/3023). 

- Boosting
Sama halnya seperti algoritma random forest, algortima Boosting juga merupakan salah satu algoritma machine learning yang termasuk ke dalam kategori ensembel. Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan [[4]](https://www.dicoding.com/academies/319/tutorials/18590?from=18585).

    Algoritma boosting bertujuan untuk meningkatkan performa akurasi prediksi dengan menggabungkan beberapa model sederhana dan dianggap lemah (weak learners) sehingga membentuk suatu model yang kuat (strong ensemble learner). Algoritma boosting muncul dari gagasan mengenai apakah algoritma yang sederhana seperti linear regression dan decision tree dapat dimodifikasi untuk dapat meningkatkan performa.

## Data Understanding
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

## Data Preparation
### Menangani Missing Value
Pada dataset diabetes ini, terdapat missing value pada beberapa kolom. Untuk menanganinya, pada proyek ini akan mengganti missing value dengan rata-rata pada setiap kolom.
<img width="814" alt="image" src="https://user-images.githubusercontent.com/36911342/181468053-99a7ecaa-ddb9-48d3-a3da-5245cfe0e74f.png">

### Menangani Outliers
Pada dataset diabetes ini, terdapat outliers pada beberapa kolom. Untuk menanganinya, pada proyek ini akan menggunakan metode IQR (InterQuartile Range). Berikut adalah code-nya.
```Python
Q1 = diabetes.quantile(0.25)
Q3 = diabetes.quantile(0.75)
IQR = Q3 - Q1

diabetes = diabetes[~((diabetes < (Q1 - 1.5 * IQR)) | (diabetes > (Q3 + 1.5 * IQR))).any(axis=1)]
```
### Split Datset
Split data adalah pembagian dataset menjadi 2 bagian, yaitu data train dan data test dengan menggunakan librari dari keras, yaitu **train_test_split()**.
<img width="226" alt="image" src="https://user-images.githubusercontent.com/36911342/181468763-633d2ba0-e938-4084-8bf7-c95f6dceafea.png">

## Modelling
Seperti yang telah dijelaskan pada solution statements, pada proyek ini akan membangun model dengan 3 algoritma, yitu KNN, Random Forest, dan Boosting. Dengan nilai akurasi dari algoritma adalah sebagai berikut.
|  Algorithm |  Accuracy |
|---|---|
|  KNeighborsClassifier | 0.780645  |
|  RandomForestClassifier |  0.787097 |
|  AdaBoostClassifier |  0.780646 |

Berdasarkan hasil uji model di atas, model denan akurasi terbaik adalah algortima *Rando Forest*.

## Evaluation
![image](https://user-images.githubusercontent.com/36911342/181469005-bbded2c6-c0c5-4da3-808a-b44a8d4164e3.png)

Berdasarjan hasil pengembangan model dengan menggunakan algoritma KNN, Random Forest, dan AdaBoosting di atas. Dapat disimpulkan bahwa model yang menggunakan algoritma Random Forest memiliki akurasi tertinggi, yaitu 78.7%. Sehingga model ini solusi terbaik untuk melakukan klasifikasi diabetes.
Score:

|   |  Score |
|---|---|
|  Accuracy | 0.812903  |
|  Precision |  0.742857 |
|  Recall |  0.565217 |
|  Kappa |  0.518479 |

Berdasarkan hasil evaluasi model, diperoleh akurasiprediksi 81% dengan menggunakan algoritma Random Forest. Dengan demikian, dapat disimpulkan bahwa algoritma Random Forest adalah model yang baik dan cocok untuk melakukan klasifikasi diabetes terhadap pasien.

## Referensi
[1] [dr. Fadhli Rizal Muslim, Diabetes](https://www.halodoc.com/kesehatan/diabetes)

[2] [Rivki, Muhammad, and Adam Mukharil Bachtiar. "Implementasi algoritma K-Nearest Neighbor dalam pengklasifikasian follower twitter yang menggunakan Bahasa Indonesia." Jurnal Sistem Informasi 13, no. 1 (2017): 31-37.](https://jsi.cs.ui.ac.id/index.php/jsi/article/view/500)

[3] [Primajaya, Aji, and Betha Nurina Sari. "Random Forest Algorithm for Prediction of Precipitation." Indonesian Journal of Artificial Intelligence and Data Mining 1, no. 1 (2018): 27-31](http://ejournal.uin-suska.ac.id/index.php/IJAIDM/article/view/4903/3023)

[4] [Macine Learning Terapan, Dicoding](https://www.dicoding.com/academies/319/tutorials/18590?from=18585)
