# Hasil training YOLO untuk ARDD26 dan CH-RDD2022

Koleksi ini menyeleksi hasil dari `E:\Hasil` pada 21 Oktober 2026. Terdapat 23 run unik: 20 run dengan
artefak evaluasi mandiri dan 3 run seed 123 yang metriknya diekstrak persis dari keluaran notebook.
Ini adalah arsip hasil eksperimen, bukan implementasi model baru pada branch `main`.

## Isi yang dipertahankan

- `summary.csv`: metrik validation/test dengan presisi asli, satu baris per run dan split.
- `<dataset>/<model>/seed-<n>/`: JSON evaluasi, konfigurasi asli, CSV per epoch, diagram arsitektur YAML
  jika tersedia, catatan revisi/transfer/pruning, serta grafik hasil, PR dan confusion matrix terpilih.
- `<dataset>/notebook-evidence/`: lima notebook hasil multi-seed; kode dan keluaran teks dipertahankan.
  Gambar tertanam, widget dan metadata eksekusi dibuang. Notebook merupakan bukti historis dan perlu
  menyesuaikan path dataset serta checkout kode sebelum dijalankan ulang.
- `notebook-extraction.json`: lokasi sel/keluaran untuk tiga JSON seed 123 yang dipulihkan.
- `selection.csv`: keputusan untuk seluruh 1.254 file sumber, termasuk ukuran dan SHA-256 file terpilih.
- `manifest.json`: ukuran dan SHA-256 file hasil kurasi (manifest sendiri dikecualikan).

## Kebijakan penyimpanan

| Artefak | Keputusan | Alasan |
| --- | --- | --- |
| Evaluasi JSON, CSV, konfigurasi, provenance | Masuk Git | Bukti hasil dan pelacakan eksperimen |
| Grafik hasil/PR/confusion matrix terpilih | Masuk Git | Diagnostik yang dapat dibaca langsung |
| Notebook hasil multi-seed | Masuk Git, hanya keluaran teks | Bukti seed yang file evaluasinya tertimpa sesi berikutnya |
| `best.pt` | Tetap lokal | Jika perlu distribusi, pilih checkpoint final untuk GitHub Release/model registry |
| `last.pt` dan bobot awal | Tetap lokal | Cadangan melanjutkan training atau dapat diunduh ulang |
| ZIP dan salinan ekspor berulang | Tidak masuk | Menduplikasi hasil yang sudah diekstrak |
| Gambar batch, label/prediksi contoh, grafik berulang | Tidak masuk | Volume besar dan tidak perlu untuk ringkasan evaluasi |
| Salinan library Python, notebook alternatif/belum dilatih | Tidak masuk | Gunakan branch/revisi kode eksperimen yang asli |
| Catatan seminar, SOP akun, laporan analisis kerja | Tidak masuk | Materi kerja lokal; ringkasan ini diturunkan dari bukti metrik |

Semua file asli tetap tersedia di `E:\Hasil`. Tidak ada checkpoint atau dataset yang dihapus.
Tidak ada bobot model yang diunggah ke GitHub Release oleh perubahan ini.

## Catatan identitas dan reproduksibilitas

Nama folder publik memakai `model` pada `args.yaml` dan seed yang tercatat, bukan nama ekspor Kaggle.
Pada ARDD26, ekspor YOLO11 dan YOLOv8 memakai nama berkas `yolo12s_original...`; `args.yaml` menunjukkan
`yolo11s.pt` dan `yolov8s.pt`. Nama file/isi metadata asli dipertahankan untuk audit, sehingga masih
memuat nama lama. Jangan menentukan model hanya dari nama JSON konfigurasi.

Ekspor China `yolo12-original-test-v-1-0 (1)` menduplikasi metrik baseline pada
`fork-of-yolo12-original-test-v-1-0 (2)`; hanya yang terakhir dipakai.
Seed 42 yang dipakai kembali oleh eksperimen multi-seed dihitung sekali dari run asalnya.
Seed 123 ARDD26 baseline serta China baseline/L-Ghost+EMA-32 dipulihkan dari notebook;
CSV training, checkpoint dan grafik seed tersebut tidak tersedia sebagai file mandiri dalam koleksi terpilih.
Tidak ada artefak seed 456 yang diberi label ulang menjadi seed 123.

Path `/kaggle/...` di metadata adalah path historis. Reproduksi membutuhkan dataset asli dan kode
pada revisi eksperimen, bukan otomatis kode `main` saat ini. Revisi/manifest asli disertakan bila ada;
untuk beberapa baseline tidak terdapat commit sumber mandiri. Impor ini tidak memverifikasi ulang
checkpoint, isi dataset, kesamaan split atau menjalankan training/evaluasi GPU.

## Ringkasan multi-seed

Angka berikut dihitung dari JSON terpilih, menggunakan seed 42, 123, 456. Nilai mAP berada pada skala 0–1.
Standar deviasi adalah standar deviasi sampel (`n-1`). Validation dan test ditampilkan terpisah;
hasil test tidak dipakai untuk memilih checkpoint atau menyatakan model terbaik.

| Dataset | Model | Split | n | mAP50 mean ± std | mAP50-95 mean ± std |
| --- | --- | --- | ---: | ---: | ---: |
| ardd26 | yolo12s-baseline | validation | 3 | 0.7376 ± 0.0104 | 0.4714 ± 0.0077 |
| ardd26 | yolo12s-baseline | test | 3 | 0.7566 ± 0.0040 | 0.4952 ± 0.0057 |
| ardd26 | yolo12s-lghost-ema32 | validation | 3 | 0.7321 ± 0.0103 | 0.4689 ± 0.0038 |
| ardd26 | yolo12s-lghost-ema32 | test | 3 | 0.7549 ± 0.0119 | 0.5000 ± 0.0036 |
| ch-rdd2022 | yolo12s-baseline | validation | 3 | 0.8917 ± 0.0041 | 0.6038 ± 0.0039 |
| ch-rdd2022 | yolo12s-baseline | test | 3 | 0.8590 ± 0.0053 | 0.5760 ± 0.0044 |
| ch-rdd2022 | yolo12s-lghost-ema32 | validation | 3 | 0.8860 ± 0.0125 | 0.6025 ± 0.0023 |
| ch-rdd2022 | yolo12s-lghost-ema32 | test | 3 | 0.8648 ± 0.0069 | 0.5814 ± 0.0128 |

Dengan hanya tiga seed, tabel ini bersifat deskriptif dan tidak membuktikan keunggulan statistik.
Kinerja lintas dataset tidak langsung sebanding: ARDD26 memiliki empat kelas, sedangkan konfigurasi
CH-RDD2022 memuat lima kelas termasuk `Repair`.
