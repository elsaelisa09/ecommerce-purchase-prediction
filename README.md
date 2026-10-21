# E-Commerce Customer Behavior & Purchase Prediction

Proyek machine learning untuk memprediksi pembelian berdasarkan karakteristik sesi pengunjung menggunakan Online Shoppers Purchasing Intention Dataset.

Proyek mencakup audit data, EDA, perbandingan model, penanganan ketidakseimbangan kelas, pemilihan threshold, interpretasi SHAP, dan aplikasi simulasi Streamlit.

## Dataset dan Pembagian Data

Dataset terdiri dari 12.330 sesi dengan 17 fitur dan target `Revenue`.

| Bagian | Jumlah sesi |
|---|---:|
| Training | 7.406 |
| Validation | 2.454 |
| Test | 2.470 |

Pembagian mempertahankan kelompok pola fitur pada bagian yang sama. Kelompok dibentuk tanpa menggunakan target dan `PageValues`.

[Sumber dataset: UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)

## Eksperimen

- Logistic Regression, Random Forest, dan XGBoost, masing-masing dengan dan tanpa `PageValues`.
- DummyClassifier sebagai referensi.
- Class weighting dan SMOTENC pada Random Forest.
- Perbandingan threshold pada XGBoost dan Random Forest.
- Pencatatan konfigurasi, metrik, pipeline, dan grafik melalui W&B.

## Hasil Akhir

Model terpilih adalah **XGBoost dengan seluruh fitur**, menggunakan **threshold 0,3**. Pemilihan dilakukan berdasarkan F1 validation dari percobaan yang dijalankan sebelum evaluasi test.

| Metrik test | Hasil |
|---|---:|
| Precision | 0,6107 |
| Recall | 0,7690 |
| F1 | 0,6808 |
| Average Precision | 0,7325 |
| ROC-AUC | 0,9347 |

[Hasil evaluasi test di W&B](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/y333ouof)

## Menjalankan Proyek

Gunakan Python 3.10 dalam environment Conda. Jalankan perintah dari folder utama proyek.

Pasang paket:

```powershell
python -m pip install -r requirements.txt
```

Untuk training, sediakan:

- `data/raw/online_shoppers_intention.csv`
- `data/splits/split_manifest.csv`

Sesuaikan eksperimen pada `configs/experiment.yaml`, kemudian jalankan:

```powershell
wandb login
python -m src.main
```

## Menjalankan Demo

Demo membaca pilihan model dari `configs/final_selection.yaml`.

Untuk konfigurasi yang disertakan, sediakan pipeline di:

```text
outputs/u1b7ipvz/pipeline.joblib
```

Pipeline tersedia pada artifact model dari [run training terpilih](https://wandb.ai/elsaelisa/ecommerce-purchase-prediction/runs/u1b7ipvz).

Jalankan aplikasi:

```powershell
python -m streamlit run app.py
```

Demo menggunakan file lokal dan tidak memerlukan login W&B saat digunakan. Jika memakai hasil training baru, sesuaikan `source_run_path` pada konfigurasi pilihan model.

## Interpretasi dan Batasan

SHAP menunjukkan `PageValues` sebagai fitur dengan kontribusi paling dominan pada model terpilih. Grafik global, beeswarm, dan waterfall tersedia melalui notebook interpretasi.

Hasil proyek merupakan klasifikasi berdasarkan data sesi yang tersedia. Penggunaan real-time memerlukan validasi ketersediaan fitur saat prediksi. SHAP menjelaskan perilaku model, bukan hubungan sebab-akibat atau dampak promosi terhadap transaksi tambahan.

Ringkasan lebih lanjut tersedia di `docs/project_summary.md`.