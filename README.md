# Segmentasi Area Sawah dan Area Non Sawah

Segmentasi Area Sawah Web App adalah aplikasi web interaktif berbasis Streamlit yang dirancang untuk melakukan segmentasi area sawah dan non-sawah dari citra satelit. Proyek ini bertujuan mempermudah analisis spasial pertanian dengan mengekstraksi informasi lokasi sawah secara otomatis dari data citra satelit, tanpa perlu pemrosesan manual yang memakan waktu.

Sistem ini memanfaatkan algoritma Split and Merge, yang membagi citra menjadi segmen-segmen kecil dan kemudian menggabungkan area serupa untuk menghasilkan segmentasi yang lebih akurat. Untuk membedakan area sawah dan non-sawah, aplikasi memanfaatkan beberapa fitur citra Standard deviation map, GLCM (Gray Level Co-occurrence Matrix), dan Warna LAB.

Setelah proses segmentasi, aplikasi menampilkan output berupa citra satelit yang sudah diberi warna sesuai area: area sawah dan non-sawah ditandai secara visual sehingga memudahkan identifikasi distribusi sawah di wilayah tersebut.

## 🚀 Try the App
Coba aplikasi web-nya disini:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://segmentasi-sawah-nonsawah.streamlit.app/)

## Algoritma yang digunakan
Apliaksi ini menggunakan algoritma *Split and Merge* sebagai inti dari segmentasinya. Algoritma Split and Merge merupakan teknik segmentasi pada pemrosesan citra yang digunakan untuk mengelompokan pixel-pixel berdasarkan homogenitas atau kemiripan tertentu dalam suatu citra. Proses split dilakukan secara berturut-turut dengan membagi citra menjadi beberapa blok berdasarkan kriteria homogenitasnya yang diukur menggunakan standard deviation map, blok dengan variasi piksel yang tidak mempunyai kemiripan akan dipecah hingga memenuhi batas yang telah ditentukan. 

<p align="center">
  <img src="images/ss split.png" alt="Hasil split" width="250"/>
</p>

Sedangkan pada proses merge, penggabungan blok akan dilakukan dengan mempertimbangkan beberapa aspek yang digunakan yaitu standard deviation map untuk melihat homogenitas intensitas, fitur warna lab untuk menangkap perbedaan warna agar hasil lebih akurat, dan Gray-Level Co-occurrence matrix (GLCM) untuk mendapatkan ciri tekstur pada suatu citra seperti perulangan pola, distribusi spasial, dan susunan warna dan intensitas.

<p align="center">
  <img src="images/ss merge.png" alt="Hasil merge" width="250"/>
</p>

## Fitur yang digunakan
Aplikasi ini menggunakan beberapa fitur citra untuk melakukan segmentasi area sawah dan non-sawah dari citra satelit menggunakan algoritma Split and Merge:
- Standard deviation map untuk Membantu mendeteksi variasi intensitas pixel, sehingga area dengan perbedaan tekstur atau warna yang signifikan dapat dipisahkan.
- GLCM (Gray Level Co-occurrence Matrix) untuk menangkap pola tekstur sehingga segmen dengan pola serupa bisa digabungkan.
- Warna LAB untuk membedakan area berdasarkan komponen warna, sehingga membantu memisahkan sawah dari non-sawah yang memiliki spektrum warna berbeda.

## 🎥 Demo Video
Tonton video demo-nya untuk melihat bagaimana app-nya berjalan:

<a href="https://youtu.be/QuMVLwrdL3s">
  <img src="https://img.youtube.com/vi/QuMVLwrdL3s/0.jpg" width="500">
</a>

## 🛠️ Tech Stack
- **Web App / UI:** Streamlit
- **Image Processing:** OpenCV, Pillow
- **Numerical Computing:** NumPy
- **Visualization:** Matplotlib

## 👤 Authors
Project ini dikembangkan oleh:
- Muhammad Ariq Hibatullah
- Firdaini Azmi
- Reva Deshinta Isyana
