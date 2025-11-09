# Segmentasi Area Sawah dan Area Non Sawah

Segmentasi Area Sawah Web App adalah aplikasi web interaktif berbasis Streamlit yang dirancang untuk melakukan segmentasi area sawah dan non-sawah dari citra satelit. Proyek ini bertujuan mempermudah analisis spasial pertanian dengan mengekstraksi informasi lokasi sawah secara otomatis dari data citra satelit, tanpa perlu pemrosesan manual yang memakan waktu.

Sistem ini memanfaatkan algoritma Split and Merge, yang membagi citra menjadi segmen-segmen kecil dan kemudian menggabungkan area serupa untuk menghasilkan segmentasi yang lebih akurat. Untuk membedakan area sawah dan non-sawah, aplikasi memanfaatkan beberapa fitur citra Standard deviation map, GLCM (Gray Level Co-occurrence Matrix), dan Warna LAB.

Setelah proses segmentasi, aplikasi menampilkan output berupa citra satelit yang sudah diberi warna sesuai area: area sawah dan non-sawah ditandai secara visual sehingga memudahkan identifikasi distribusi sawah di wilayah tersebut.
