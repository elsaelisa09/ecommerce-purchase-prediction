# E-Commerce Customer Behavior & Purchase Prediction

Proyek ini membangun model klasifikasi untuk memprediksi pembelian berdasarkan karakteristik sesi pengunjung. Dataset berisi 12.330 sesi dengan 17 fitur dan target `Revenue`.

## Metode

Data dibagi menjadi 7.406 sesi training, 2.454 validation, dan 2.470 test. Kelompok pola fitur yang ditetapkan saat audit ditempatkan pada bagian yang sama, sehingga tidak ada kelompok yang tersebar antarbagian.

Eksperimen mencakup Logistic Regression, Random Forest, XGBoost, dan CatBoost, dengan dan tanpa `PageValues`. Pengujian tambahan mencakup DummyClassifier sebagai pembanding, class weighting pada Random Forest dan XGBoost, serta SMOTENC pada Random Forest.

Pengembangan XGBoost mencakup pencarian hyperparameter dengan lima fold `StratifiedGroupKFold` pada bagian training. Preprocessing dan model dilatih ulang hanya menggunakan bagian training setiap fold. Bobot kelas otomatis juga dihitung hanya dari target training yang sedang digunakan. Konfigurasi, metrik, pipeline, dan laporan eksperimen didokumentasikan melalui W&B.

## Model yang digunakan aplikasi

Aplikasi Streamlit menggunakan XGBoost berbobot dengan seluruh 17 fitur dari run `d9n08p0g`. Konfigurasi utamanya adalah 300 pohon, `max_depth=3`, `learning_rate=0.03`, dan `scale_pos_weight=6223/1183`, yaitu sekitar 5,26.

Threshold 0,7 dipilih karena memberikan F1 tertinggi di antara threshold yang diuji pada validation untuk model tersebut. Model ini juga memberikan F1 validation tertinggi pada tabel perbandingan eksperimen yang telah dilakukan.

| Metrik validation | Hasil |
|---|---:|
| Precision | 0,6040 |
| Recall | 0,7378 |
| F1 | 0,6642 |
| Average Precision | 0,7390 |
| ROC-AUC | 0,9324 |

Pada 2.454 sesi validation, model mendeteksi 273 dari 370 sesi pembelian, melewatkan 97 pembelian, dan menghasilkan 179 kesalahan positif. Hasil ini merupakan evaluasi validation setelah pemilihan threshold; belum ada hasil test independen yang dilaporkan untuk konfigurasi berbobot ini.

## Evaluasi test model awal

Sebelum pengembangan lanjutan, XGBoost awal dari run `u1b7ipvz`, tanpa pembobotan tambahan (`scale_pos_weight=1`), telah dievaluasi pada test menggunakan threshold 0,3 yang dipilih melalui validation. Hasil berikut tetap dicatat sebagai riwayat model awal.

| Metrik test | Hasil |
|---|---:|
| Precision | 0,6107 |
| Recall | 0,7690 |
| F1 | 0,6808 |
| Average Precision | 0,7325 |
| ROC-AUC | 0,9347 |

Model awal mendeteksi 273 dari 355 sesi pembelian, melewatkan 82 pembelian, dan menghasilkan 174 kesalahan positif. Angka test ini tidak mewakili model berbobot yang digunakan aplikasi sekarang. F1 test 0,6808 dan F1 validation 0,6642 berasal dari model dan subset berbeda, sehingga tidak digunakan sebagai perbandingan langsung untuk menentukan model terbaik.

## Insight bisnis dan interpretasi

Pada validation, model berbobot menandai 452 sesi atau sekitar 18,4% dari seluruh sesi, dan mencakup 73,8% sesi yang benar-benar membeli. Proporsi pembelian pada kelompok yang ditandai adalah 60,4%, dibandingkan 15,1% pada seluruh validation. Hasil ini menunjukkan potensi untuk memprioritaskan sesi, tetapi belum membuktikan peningkatan penjualan akibat promosi.

Eksperimen tanpa `PageValues` menunjukkan penurunan performa yang besar. Analisis SHAP yang tersimpan pada notebook 03 masih menjelaskan model awal dan menunjukkan kontribusi dominan `PageValues`. Analisis tersebut perlu dilengkapi untuk model berbobot agar penjelasannya sesuai dengan model aplikasi.

Penyesuaian threshold mengubah keseimbangan precision dan recall. Penggunaan bisnis perlu mempertimbangkan biaya intervensi, nilai transaksi, serta kapasitas tim dalam menindaklanjuti sesi yang ditandai.

## Demo dan batasan

Demo Streamlit memuat pipeline tersimpan untuk simulasi input sesi dan menampilkan skor kecenderungan membeli serta keputusan berdasarkan threshold 0,7.

Hasil proyek merupakan evaluasi klasifikasi berdasarkan data sesi yang tersedia. Penerapan real-time memerlukan pemeriksaan ketersediaan setiap fitur pada waktu prediksi, terutama `PageValues` dan metrik agregat sesi. Skor model berbobot belum divalidasi sebagai probabilitas yang terkalibrasi.

Test model awal sudah dilihat sebelum pengembangan lanjutan. Penilaian generalisasi model terbaru perlu memperhitungkan riwayat tersebut dan sebaiknya dilengkapi dengan data evaluasi baru yang independen.

SHAP menjelaskan perilaku model, bukan hubungan sebab-akibat. Dampak promosi terhadap transaksi tambahan dan margin belum diuji melalui eksperimen bisnis.

## Dokumentasi eksperimen

- [Run training XGBoost berbobot — metrik training run dicatat pada threshold awal 0,3](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/d9n08p0g)
- [Evaluasi threshold model berbobot — pilihan threshold 0,7](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/83s5lg90)
- [Evaluasi test model awal — threshold 0,3](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/y333ouof)
