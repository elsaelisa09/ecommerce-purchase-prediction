# E-Commerce Customer Behavior & Purchase Prediction

Proyek ini membangun model klasifikasi untuk memprediksi pembelian berdasarkan karakteristik sesi pengunjung. Dataset berisi 12.330 sesi dengan 17 fitur dan target `Revenue`.

**Metode**

Data dibagi menjadi 7.406 sesi training, 2.454 validation, dan 2.470 test. Kelompok dengan pola fitur identik ditempatkan pada bagian yang sama untuk mencegah pola tersebut tersebar antarbagian.

Eksperimen mencakup Logistic Regression, Random Forest, dan XGBoost, masing-masing dengan dan tanpa `PageValues`. Pengujian tambahan meliputi DummyClassifier, class weighting, dan SMOTENC pada Random Forest. Konfigurasi, metrik, pipeline, dan grafik dicatat melalui W&B.

**Model terpilih dan hasil akhir**

XGBoost dengan seluruh fitur dipilih berdasarkan F1 validation tertinggi dari percobaan yang dilakukan. Threshold ditetapkan sebesar 0,3 menggunakan validation sebelum evaluasi test.

| Metrik test | Hasil |
|---|---:|
| Precision | 0,6107 |
| Recall | 0,7690 |
| F1 | 0,6808 |
| Average Precision | 0,7325 |
| ROC-AUC | 0,9347 |

Model mendeteksi 273 dari 355 sesi pembelian, melewatkan 82 pembelian, dan menghasilkan 174 kesalahan positif.

**Insight bisnis**

SHAP menunjukkan bahwa `PageValues` memberikan kontribusi paling dominan, diikuti indikator bulan dan `ExitRates`. Penghapusan `PageValues` menurunkan performa ketiga algoritma secara besar pada validation.

Penyesuaian threshold meningkatkan deteksi pembelian dengan konsekuensi bertambahnya kesalahan positif. Pemilihan threshold untuk penggunaan bisnis perlu mempertimbangkan biaya intervensi dan nilai transaksi.

**Batasan**

Hasil ini merupakan evaluasi klasifikasi berdasarkan data sesi yang tersedia. Penerapan real-time memerlukan pemeriksaan ketersediaan setiap fitur pada waktu prediksi, terutama `PageValues` dan metrik agregat sesi.

SHAP menjelaskan perilaku model, bukan hubungan sebab-akibat. Dampak promosi terhadap transaksi tambahan dan margin belum diuji.

[Dokumentasi evaluasi test di W&B](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/y333ouof)