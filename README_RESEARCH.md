# Katalog riset deteksi kerusakan jalan

Branch kerja: `research/yolo12s-model-catalog`. Folder `ultralytics/cfg/models/12/` pada branch ini sengaja berisi **hanya enam konfigurasi deteksi YOLO12s** yang dipakai dalam penelitian. Notebook, hasil evaluasi ringan, dan jejak asal eksperimen disertakan. Branch sumber tetap dipertahankan; `main` tidak diubah. Dokumen ini adalah pintu masuk, bukan pengganti laporan analisis tiap dataset.

## Peta model

| Keluarga | Konfigurasi di branch ini | Status bukti hasil | Asal historis |
|---|---|---|---|
| YOLO12s asli | `ultralytics/cfg/models/12/yolo12.yaml` (`scale: s`) | China dan Aceh | Ultralytics bawaan |
| EMA-32 | `ultralytics/cfg/models/12/yolo12-ema32.yaml` | China dan Aceh | `yolo12-ema32` (`419ee77`) |
| GhostConv | `ultralytics/cfg/models/12/yolo12-ghost.yaml` | China dan Aceh | `yolo12-ghost-conv` (`57ea1b4`) |
| Direct EMA + GhostConv | `ultralytics/cfg/models/12/yolo12s-ema-ghost.yaml` | China dan Aceh | tag riset `research/yolo12s-ema-ghost-architecture-v1` (`d199dd3`) |
| L-Ghost | `ultralytics/cfg/models/12/yolo12s-lghost.yaml` | China dan Aceh; checkpoint Aceh belum tersedia lokal | `feat/yolo12s-lghost-ema32` (`14cbeb1`) |
| L-Ghost + EMA-32 | `ultralytics/cfg/models/12/yolo12s-lghost-ema32.yaml` | China dan Aceh | `feat/yolo12s-lghost-ema32` (`14cbeb1`) |

Konfigurasi ECA, SPD, Ghost+ECA, Ghost+EMA, dan BL-YOLOv8s yang sempat digabungkan sebelumnya telah dikeluarkan dari branch ini karena **bukan bagian dari enam model penelitian**. Empat konfigurasi bawaan YOLO12 untuk klasifikasi, segmentasi, pose, dan OBB juga tidak disertakan di folder tersebut karena penelitian ini hanya memakai deteksi. Riwayat file-file itu tetap dapat dilihat pada commit terdahulu. YOLOv8s dan YOLO11s masih disebut dalam bukti Aceh semata sebagai **pembanding eksternal**, bukan varian model penelitian atau konfigurasi tambahan di folder YOLO12.

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
