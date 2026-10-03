# Katalog riset deteksi kerusakan jalan

Branch kerja: `research/yolo12s-model-catalog`. Branch ini mengumpulkan konfigurasi dan modul model historis dalam satu tempat, disertai notebook, hasil evaluasi ringan, dan jejak asal eksperimen. Branch sumber tetap dipertahankan; `main` tidak diubah. Dokumen ini adalah pintu masuk, bukan pengganti laporan analisis tiap dataset.

## Peta model

| Keluarga | Konfigurasi di branch ini | Status bukti hasil | Asal historis |
|---|---|---|---|
| YOLOv8s | `ultralytics/cfg/models/v8/yolov8.yaml` (`scale: s`) | Aceh | Ultralytics bawaan |
| YOLO11s | `ultralytics/cfg/models/11/yolo11.yaml` (`scale: s`) | Aceh | Ultralytics bawaan |
| YOLO12s asli | `ultralytics/cfg/models/12/yolo12.yaml` (`scale: s`) | China dan Aceh | Ultralytics bawaan |
| EMA-32 | `ultralytics/cfg/models/12/yolo12-ema32.yaml` | China dan Aceh | `yolo12-ema32` (`419ee77`) |
| GhostConv | `ultralytics/cfg/models/12/yolo12-ghost.yaml` | China dan Aceh | `yolo12-ghost-conv` (`57ea1b4`) |
| Direct EMA + GhostConv | `ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml` | China dan Aceh | tag riset `research/yolo12s-ema-ghost-architecture-v1` (`d199dd3`) |
| L-Ghost | `ultralytics/cfg/models/12/yolo12s-lghost.yaml` | China dan Aceh; checkpoint Aceh belum tersedia lokal | `feat/yolo12s-lghost-ema32` (`14cbeb1`) |
| L-Ghost + EMA-32 | `ultralytics/cfg/models/12/yolo12s-lghost-ema32.yaml` | China dan Aceh | `feat/yolo12s-lghost-ema32` (`14cbeb1`) |
| ECA | `ultralytics/cfg/models/12/yolo12-eca.yaml` | Konfigurasi historis; belum ada evaluasi pada dua dataset di katalog | `yolo12-eca-net` (`8633245`) |
| ECA-P3 | `ultralytics/cfg/models/12/yolo12-eca-p3.yaml` | Konfigurasi historis | `yolo12s-eca-p3` (`e1c1226`) |
| Ghost + ECA | `ultralytics/cfg/models/12/yolo12-ghost-eca.yaml` | Konfigurasi historis | `yolo12-ghost-eca` (`4b13197`) |
| Ghost + EMA; Ghost + EMA-32 | `ultralytics/cfg/models/12/yolo12-ghost-ema.yaml`, `yolo12-ghost-ema32.yaml` | Konfigurasi historis | `yolo12-ghost-ema` (`b0061e8`) |
| SPD; SPD + EMA-32 | `ultralytics/cfg/models/12/yolo12-spd.yaml`, `yolo12-spd-ema32.yaml` | Konfigurasi historis | `yolo12-spd-ema32-continuity` (`3d7acd7`) |
| BL-YOLOv8s | `ultralytics/cfg/models/v8/bl-yolov8s.yaml` | Konfigurasi historis | `yolo12-bl-replica` (`7e9ce52`) |

Status **konfigurasi historis** berarti model dapat dibangun dari YAML di branch ini, bukan berarti telah dilatih ulang, tersedia checkpoint, atau terbukti unggul. Nama `EMAResidualAttention` dipakai pada konfigurasi SPD + EMA-32 untuk menghindari benturan dengan implementasi `EMAAttention` asli pada eksperimen EMA-32. Ini adaptasi kompatibilitas, bukan klaim reproduksi checkpoint SPD yang telah diverifikasi; untuk reproduksi persis, lihat commit sumbernya.

## Hasil yang bisa ditelusuri

Semua angka berikut adalah evaluasi **test** atas `best.pt`, bukan nilai validasi selama pelatihan. Nilai berasal dari [analisis China](research/evidence/china/MODEL_COMPARISON_ANALYSIS.md) dan [analisis Aceh](research/evidence/aceh/MODEL_COMPARISON_ANALYSIS.md).

| Dataset | YOLO12s asli mAP50-95 | L-Ghost + EMA-32 mAP50-95 | Selisih absolut | Catatan penting |
|---|---:|---:|---:|---|
| CH-RDD2022 (China) | 0,578334 | 0,592890 | +0,014556 | `best.pt` L-Ghost + EMA-32 pada tahap pruning 3/3; precision baseline lebih tinggi. |
| ARDD26 (Aceh) | 0,497375 | 0,503937 | +0,006562 | `best.pt` pada tahap pruning 2/3; recall turun dari 0,706835 menjadi 0,648893. |

Kenaikan di kedua dataset adalah **peningkatan teramati**, bukan bukti signifikansi statistik: baru satu seed dan satu pembagian data yang terdokumentasi. Metrik China dan Aceh tidak dibandingkan mentah karena himpunan dan kelasnya berbeda. Untuk China, baseline yang sah adalah `fork-of-yolo12-original-test-v-1-0 (2).ipynb`; notebook `(1)` diabaikan. Untuk Aceh, notebook L-Ghost murni secara fisik berada di folder China tetapi isi eksperimennya memakai Aceh.

## Struktur bukti

- `research/notebooks/china/` dan `research/notebooks/aceh/`: 14 notebook sumber dengan kode dan metadata; output eksekusi dihapus agar Git ringan. Berkas ini tidak dengan sendirinya membuktikan eksekusi.
- `research/evidence/{china,aceh}/<model>/`: `evaluation_metrics.json`, `args.yaml`, dan `results.csv` yang tersedia dari 13 folder hasil ekstraksi. Aceh L-Ghost murni hanya memiliki JSON metrik yang ditranskripsi dari output notebook karena ZIP dan checkpoint-nya belum tersedia pada lokasi yang ditelaah.
- [`research/evidence/manifest.json`](research/evidence/manifest.json): pemetaan 14 eksperimen ke notebook dan folder asal `E:/Hasil`, SHA-256 notebook sumber, serta SHA-256/ukuran checkpoint dan metrik sumber bila tersedia.
- Laporan analisis per dataset di atas membahas protokol, metrik per kelas, kompleksitas, trade-off, dan batas klaim.

ZIP hasil dan `best.pt` tidak dimasukkan ke Git karena berukuran besar. Hash dalam manifest memungkinkan pencocokan dengan artefak asli di `E:/Hasil`; keberadaan hash bukan pengganti akses ke checkpoint. Jika artefak dipindah, verifikasi ulang dengan `Get-FileHash -Algorithm SHA256 <berkas>` di PowerShell. Jangan menafsirkan ukuran `best.pt` atau mask pruning sebagai bukti langsung percepatan inferensi/deployment.

## Cara memakai satu branch

1. Ambil branch `research/yolo12s-model-catalog` dan pasang dependensi sesuai `pyproject.toml`.
2. Pilih YAML pada tabel. Untuk varian L-Ghost dengan pruning terjadwal, ikuti [`README_LGHOST_EMA32.md`](README_LGHOST_EMA32.md) dan trainer khususnya; membangun graph saja tidak menjalankan pruning.
3. Cocokkan konfigurasi pelatihan dengan `args.yaml` serta notebook eksperimen yang sesuai sebelum mengulang hasil. Dataset dan checkpoint asli tetap harus tersedia terpisah.
4. Laporkan selalu dataset, split, seed, `best.pt` versus `last.pt`, tahap pruning checkpoint, dan seluruh metrik utama. Jangan menyebut selisih satu run sebagai signifikan secara statistik.

Branch ini memudahkan navigasi serta audit, tetapi tidak menggabungkan riwayat Git semua cabang, tidak menimpa cabang lama, dan tidak menyatakan kesetaraan bit-per-bit setiap eksperimen historis. Commit sumber di tabel tetap menjadi rujukan ketika reproduksi persis diperlukan.
