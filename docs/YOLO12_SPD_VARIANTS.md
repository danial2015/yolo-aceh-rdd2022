# Laporan Implementasi YOLO12-Small SPD-Conv dan SPD-Conv + EMA-32

## Ringkasan

Pekerjaan ini menggantikan eksperimen lama `yolo12-spd-eca` dengan tiga varian ablation yang lebih jelas:

| Branch | Arsitektur | Status |
| --- | --- | --- |
| `yolo12-spd-conv` | YOLO12-Small + SPD-Conv | Aktif |
| `yolo12-spd-conv-ema32` | YOLO12-Small + SPD-Conv + EMA-32 | Basis eksperimen awal |
| `yolo12-spd-ema32-continuity` | YOLO12-Small + SPD-Conv + residual EMA-32 | Aktif, inisialisasi pretrained dipertahankan |
| `yolo12-spd-eca` | YOLO12-Small + SPD-Conv + ECA | Dihapus dari lokal dan GitHub |

Ketiga branch ini tidak memakai ECA atau GhostConv. Branch `yolo12-spd-conv-ema32` merupakan turunan dari
`yolo12-spd-conv`, sedangkan branch continuity adalah turunannya; implementasi SPD-Conv inti tetap sama.

## Pembaruan branch continuity

Branch `yolo12-spd-ema32-continuity` diturunkan dari `yolo12-spd-conv-ema32`. Branch ini memperbaiki dua
diskontinuitas terhadap checkpoint `yolo12s.pt`:

1. Tiga convolution downsampling pretrained direparameterisasi secara tepat ke `PixelUnshuffle(2)` diikuti Conv
   stride-1. Pasangannya adalah source-target `3->3`, `5->6`, dan `7->8`. Seluruh bobot tambahan SPD di-zero-kan,
   lalu sembilan irisan kernel per layer ditulis dengan indexed assignment yang diverifikasi dan seluruh state
   BatchNorm dipindahkan. Ini penting: `.copy_()` pada advanced indexing PyTorch hanya mengubah tensor sementara,
   sehingga tidak boleh digunakan untuk transfer tersebut.
2. EMA-32 menjadi residual adapter:

   ```text
   output = input + residual_scale * (attended - input)
   ```

   Nilai awal `residual_scale=0.001` membuat gangguan fitur sangat kecil tetapi tetap memberi gradien kepada bobot
   EMA sejak langkah optimisasi pertama.

Checkpoint `yolo12s.pt` rilis resmi juga memiliki bias pada convolution positional encoding `AAttn`; implementasi
`Conv` dan `AAttn` pada branch ini kompatibel dengan state tersebut sehingga semua tensor YOLO12s non-kustom dapat
ditransfer. Sebelum training, notebook sementara mengatur scale EMA menjadi nol dan membandingkan output model
proposed terhadap `yolo12s.pt` pada input sintetis. Training dihentikan bila keluaran 80 kelas tersebut tidak setara secara numerik.
Setelah pemeriksaan lulus, scale EMA dikembalikan ke `0.001`; ketika trainer menyesuaikan dataset lima kelas, hanya
head deteksi yang memang tidak kompatibel dengan jumlah kelas harus dilatih kembali.

## Perubahan arsitektur

### SPD-Conv

Modul `SPDConv` berada di `ultralytics/nn/modules/conv.py` dan terdaftar pada parser model di
`ultralytics/nn/tasks.py`. Implementasinya adalah:

```text
input B x C x H x W
  -> PixelUnshuffle(2)
  -> B x (4C) x (H/2) x (W/2)
  -> Conv(4C, C_out, kernel=3, stride=1)
```

SPD-Conv menggantikan convolution downsampling backbone bawaan; tidak menambah tahap downsampling lain. Oleh karena
itu ukuran gambar harus habis dibagi 32 agar seluruh pyramid P3/P4/P5 aman untuk `PixelUnshuffle(2)`.

| Varian | Layer SPD-Conv | Posisi |
| --- | --- | --- |
| SPD-Conv | 3, 5, 7 | P3/8, P4/16, P5/32 |
| SPD-Conv + EMA-32 | 3, 6, 8 | P3/8, P4/16, P5/32 |

Konfigurasi model:

- `ultralytics/cfg/models/12/yolo12-spd.yaml`
- `ultralytics/cfg/models/12/yolo12-spd-ema32.yaml`

### EMA-32

Varian kedua menambahkan `EMAAttention` pada layer 5, tepat setelah fitur P3/8. EMA di sini berarti *Efficient
Multi-scale Attention*, bukan *Exponential Moving Average* milik trainer. Modul menggunakan 32 grup kanal dan
mempertahankan bentuk tensor masukan.

Implementasi memvalidasi bahwa jumlah kanal P3 dapat dibagi dengan faktor grup. Pada YOLO12-Small, P3 memiliki 256
kanal sehingga `256 / 32 = 8` kanal per grup dan konfigurasi valid.

## Notebook Kaggle

Setiap branch memiliki notebook mandiri yang melakukan clone branch, instalasi editable repository, pemeriksaan model,
transfer pretrained, training, evaluasi validation/test, dan pembuatan ZIP hasil.

| Branch | Notebook |
| --- | --- |
| `yolo12-spd-conv` | `examples/kaggle_yolo12s_spd_rdd2022.ipynb` |
| `yolo12-spd-conv-ema32` | `examples/kaggle_yolo12s_spd_ema32_rdd2022.ipynb` |
| `yolo12-spd-ema32-continuity` | `examples/kaggle_yolo12s_spd_ema32_continuity_rdd2022.ipynb` dan `examples/kaggle_yolo12s_spd_ema32_clean_logs_rdd2022.ipynb` |

Semua notebook Kaggle di atas menggunakan dataset berikut:

```text
/kaggle/input/datasets/danialalfayyadh/ch-rdd-2022/datasets-china-split-fix
```

Dataset YAML yang dibuat notebook berisi lima kelas: `D00`, `D10`, `D20`, `D40`, dan `Repair`.

### Hyperparameter

| Parameter | Nilai |
| --- | --- |
| Epoch | 160 |
| Image size | 640 |
| Batch fisik | 16 |
| Nominal batch (`nbs`) | 64 |
| Optimizer | SGD |
| Learning rate awal | 0.01 |
| Momentum | 0.937 |
| Weight decay | 0.0005 |
| Seed | 42 |
| Workers | 2 |

`batch=16` dipilih agar lebih aman untuk GPU Kaggle. `nbs=64` mempertahankan nominal batch Ultralytics melalui
gradient accumulation. Notebook memeriksa `IMGSZ % 32 == 0` sebelum model dibangun.

## Pretrained YOLO12-Small

Notebook branch awal memulai dari `yolo12s.pt` resmi dengan mentransfer tensor yang memiliki nama layer dan bentuk
yang kompatibel. Pada branch `yolo12-spd-ema32-continuity`, transfer tersebut diperluas dengan reparameterisasi
exact untuk Conv-to-SPD dan pemeriksaan kontinuitas fungsi.

- Pada varian SPD-Conv, indeks layer tidak berubah sehingga pencocokan dilakukan berdasarkan nama dan bentuk tensor.
- Pada varian SPD-Conv + EMA-32, semua layer original mulai layer 5 dipetakan ke indeks target `source + 1` karena EMA
  disisipkan setelah layer backbone 4.
- Pada branch continuity, tiga Conv stride-2 tersebut diinisialisasi dari tensor pretrained dengan pemetaan
  Conv-to-SPD exact, termasuk BatchNorm. EMA memakai residual scale kecil, bukan langsung mengganti fitur P3.
- Head lima kelas tetap disesuaikan oleh trainer saat fine-tuning karena checkpoint source berisi 80 kelas.

Kebijakan ini mencegah pemuatan tensor dengan bentuk salah, sekaligus tetap memakai representasi pretrained dari
bagian YOLO12-Small yang kompatibel.

## Validasi yang sudah dilakukan

| Pemeriksaan | SPD-Conv | SPD-Conv + EMA-32 |
| --- | --- | --- |
| Validasi struktur notebook dan syntax semua cell | Lulus | Lulus |
| Parser YAML | Lulus | Lulus |
| Layer yang terbentuk | SPD: `[3, 5, 7]` | SPD: `[3, 6, 8]`, EMA: `[(5, 32)]` |
| Forward CPU, input `1 x 3 x 640 x 640` | Output `1 x 9 x 8400` | Output `1 x 9 x 8400` |
| Backward sintetis dan pemeriksaan gradien hingga | Tidak dijalankan terpisah | Lulus |
| Smoke test transfer tensor kompatibel | Lulus | Lulus |
| Pemeriksaan output pretrained source-proposed pada 80 kelas | Tidak ada | Lulus di branch `yolo12-spd-ema32-continuity` |

Jumlah parameter untuk lima kelas adalah sekitar 15.01 juta pada SPD-Conv dan 15.01 juta pada SPD-Conv + EMA-32.
SPD-Conv menambah biaya memori dibanding YOLO12-Small original; karena itu batch lebih besar dari 16 perlu diuji secara
bertahap di GPU tujuan.

## Batasan dan evaluasi berikutnya

Validasi di atas membuktikan bahwa model dapat dibangun, melakukan forward/backward, dan menjalankan transfer
pretrained tanpa kesalahan bentuk tensor. Validasi tersebut bukan bukti peningkatan mAP.

Untuk perbandingan penelitian yang adil, jalankan baseline YOLO12-Small, SPD-Conv, dan SPD-Conv + EMA-32 dengan
split dataset, seed, epoch, image size, batch, dan hyperparameter yang sama. Bandingkan minimal `mAP50`, `mAP50-95`,
precision, recall, jumlah parameter, penggunaan GPU, dan lama training. Hasil final tetap harus dilaporkan dari
`best.pt` pada validation set; test set hanya dievaluasi jika label tersedia.
