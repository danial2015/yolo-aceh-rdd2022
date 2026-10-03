# Analisis Perbandingan Model CH-RDD2022 (China)

Diperbarui 2 Oktober 2026. Laporan ini membandingkan hasil evaluasi split test menggunakan checkpoint terbaik (best.pt). Angka validasi dan test dibahas terpisah.

## 1. Cakupan dan koreksi baseline

Baseline YOLO12s yang digunakan adalah fork-of-yolo12-original-test-v-1-0 (2).ipynb dan folder hasil ekstraksi dengan nama yang sama. Notebook yolo12-original-test-v-1-0 (1).ipynb sengaja **diabaikan** sesuai pembaruan eksperimen. Baseline baru mencatat 160 epoch, citra 640, batch 16, seed 42, serta evaluasi test atas 438 gambar dan 893 objek.

Enam eksperimen yang dibandingkan adalah YOLO12s original, EMA-32, GhostConv batch 16, Direct EMA + GhostConv, L-Ghost, serta L-Ghost + EMA-32. Dua notebook L-Ghost untuk China ialah yolo12-lghost.ipynb dan yolo12-lghostema.ipynb. Notebook bernama Aceh yang berada di folder China tidak termasuk eksperimen CH-RDD2022.

## 2. Protokol dan rancangan varian

Menurut notebook dan args.yaml, eksperimen YOLO12s yang dibandingkan menggunakan 160 epoch, ukuran 640, batch 16, batch nominal 64, SGD, learning rate awal 0,01, momentum 0,937, weight decay 0,0005, patience 0, seed 42, device 0, dan 2 workers. Output evaluasi mencatat 438 gambar/876 objek pada validasi dan 438 gambar/893 objek pada test. Kesamaan jumlah belum membuktikan identitas setiap file citra; audit hash split belum tersedia.

Baseline memakai bobot yolo12s.pt secara langsung. Varian yang mengubah graph memakai transfer bobot sesuai arsitektur, sehingga efek arsitektur dan cara inisialisasi belum dapat dipisahkan sepenuhnya. GhostConv mengganti operasi pada P4/16 dan P5/32. Direct EMA + GhostConv menggabungkan modul EMA pada layer 15 dan 19 dengan GhostConv pada layer 16 dan 20.

L-Ghost menempatkan LGhostBottleneck pada layer 16 dan 20. L-Ghost + EMA-32 menambah modul EMA pada layer 15 dan 19. Kedua notebook L-Ghost memakai source commit 14cbeb1b9a136c17da4ff0acfe01a3f625cd5e3e, transfer semantik 673 tensor secara tepat, serta trainer dengan pruning setelah epoch 32, 64, dan 96. Fraksi grup yang dipertahankan pada tiap tahap adalah 0,75; 0,50; dan 0,25. Ini merupakan adaptasi L-Ghost pada deteksi, bukan replikasi penuh L-GhostNet.

## 3. Kinerja agregat pada test

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLO12s original (baseline baru) | **0,897547** | 0,806689 | 0,863466 | 0,578334 |
| EMA-32 | 0,881140 | 0,822953 | 0,860605 | 0,563886 |
| GhostConv batch 16 | 0,845425 | 0,822901 | 0,851738 | 0,567157 |
| Direct EMA + GhostConv | 0,846730 | **0,838011** | **0,873272** | 0,584336 |
| L-Ghost | 0,867636 | 0,812853 | 0,856272 | 0,573250 |
| L-Ghost + EMA-32 | 0,883231 | 0,830711 | 0,872306 | **0,592890** |

Precision tertinggi terdapat pada baseline, recall dan mAP50 tertinggi pada Direct EMA + GhostConv, serta mAP50-95 tertinggi pada L-Ghost + EMA-32. Tidak ada satu model yang menguasai semua ukuran.

### Selisih absolut L-Ghost + EMA-32

| Pembanding | Δ Precision | Δ Recall | Δ mAP50 | Δ mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline baru | −0,014316 | +0,024022 | +0,008840 | +0,014556 |
| Direct EMA + GhostConv | +0,036501 | −0,007301 | −0,000966 | +0,008554 |
| L-Ghost | +0,015595 | +0,017858 | +0,016034 | +0,019640 |

Delta adalah hasil L-Ghost + EMA-32 dikurangi hasil pembanding pada skala 0–1, bukan persen relatif. Dibanding baseline, kombinasi baru menaikkan recall, mAP50, dan mAP50-95 dengan precision sedikit lebih rendah. Dibanding Direct EMA + GhostConv, mAP50-95 dan precision naik, sedangkan recall dan mAP50 turun sedikit.

## 4. mAP50-95 test per kelas

Nilai pada tabel berikut diambil dari baris kelas dalam output evaluasi test tiap notebook dan dibulatkan tiga desimal.

| Kelas | Baseline baru | EMA-32 | GhostConv 16 | Direct EMA + GhostConv | L-Ghost | L-Ghost + EMA-32 |
|---|---:|---:|---:|---:|---:|---:|
| D00 | 0,526 | 0,529 | 0,516 | 0,523 | **0,539** | 0,526 |
| D10 | 0,538 | 0,546 | 0,552 | 0,536 | 0,539 | **0,564** |
| D20 | 0,532 | 0,541 | 0,500 | 0,535 | 0,527 | **0,566** |
| D40 | 0,546 | 0,498 | 0,534 | **0,570** | 0,533 | 0,568 |
| Repair | 0,750 | 0,705 | 0,735 | **0,756** | 0,728 | 0,740 |

Terhadap baseline baru, L-Ghost + EMA-32 naik sekitar 0,026 pada D10, 0,034 pada D20, dan 0,022 pada D40; D00 sekitar tetap, sedangkan Repair turun sekitar 0,010. Direct EMA + GhostConv tetap sedikit lebih baik pada D40 dan Repair. Jadi, kenaikan agregat mAP50-95 L-Ghost + EMA-32 tidak berarti setiap kelas membaik.

## 5. Kompleksitas dan kecepatan

| Model | Layer graph training | Parameter | GFLOPs graph training | best.pt | Inferensi test |
|---|---:|---:|---:|---:|---:|
| Baseline baru | 272 | 9.255.071 | 23,5 | 18,07 MB | 15,0 ms/gambar |
| EMA-32 | 277 | 9.255.743 | 23,7 | 18,08 MB | 15,2 ms/gambar |
| GhostConv batch 16 | 276 | 8.379.935 | 22,1 | 16,41 MB | 14,7 ms/gambar |
| Direct EMA + GhostConv | 286 | 8.892.079 | 23,2 | 17,39 MB | 16,4 ms/gambar |
| L-Ghost | 306 | 8.737.375 | 23,2 | 17,12 MB | 15,6 ms/gambar |
| L-Ghost + EMA-32 | 314 | 8.738.223 | 23,3 | 17,12 MB | 14,2 ms/gambar |

GFLOPs dan parameter memakai ringkasan graph training, sehingga angka fused dapat berbeda. GhostConv paling ringan menurut parameter dan GFLOPs. L-Ghost + EMA-32 mencatat waktu inferensi test terendah pada output yang tersedia, tetapi pengukuran ini belum merupakan benchmark hardware terkontrol. Jumlah parameter tetap menghitung struktur model; mask pruning dan ukuran checkpoint tidak otomatis sama dengan kompresi fisik atau percepatan deployment.

## 6. Training dan checkpoint pruning

| Model | Epoch validasi terbaik menurut results.csv | mAP50-95 terbaik | mAP50-95 epoch 160 |
|---|---:|---:|---:|
| Baseline baru | 125 | 0,60374 | 0,59807 |
| EMA-32 | 134 | 0,60696 | 0,60469 |
| GhostConv batch 16 | 148 | 0,59988 | 0,59757 |
| Direct EMA + GhostConv | 115 | **0,60881** | 0,60418 |
| L-Ghost | 153 | 0,60296 | 0,59908 |
| L-Ghost + EMA-32 | 159 | 0,60130 | 0,60026 |

Log training dan evaluasi ulang best.pt adalah dua keluaran berbeda; nilai validasi dapat sedikit berbeda dan tidak dicampur dengan test. Kedua best.pt L-Ghost China secara eksplisit dilaporkan pada **tahap pruning 3 dari 3**. Dengan demikian, angka test keduanya berasal dari checkpoint setelah seluruh tahap pruning terjadwal.

## 7. Kesimpulan dan batas klaim

Pada CH-RDD2022, L-Ghost + EMA-32 mencapai mAP50-95 test tertinggi, 0,592890, yaitu +0,014556 absolut dari baseline baru. Kombinasi Direct EMA + GhostConv tetap tertinggi pada recall dan mAP50. Pilihan model bergantung pada metrik prioritas; klaim bahwa satu model selalu unggul tidak didukung.

Hanya satu seed dan satu split yang terlihat. Belum ada uji signifikansi statistik, pengulangan multi-seed, atau benchmark latensi terkontrol. Perubahan trainer dan transfer pretrained pada L-Ghost juga membatasi atribusi kausal terhadap satu modul. Nilai mAP China tidak dibandingkan langsung secara numerik dengan mAP Aceh karena data dan jumlah kelas berbeda.

## 8. Jejak bukti utama

- Baseline baru: fork-of-yolo12-original-test-v-1-0 (2).ipynb dan folder ekstraksinya; evaluation_metrics.json, args.yaml, results.csv, serta best.pt.
- Varian lama yang masih relevan: yolo12-ema32-ch (2).ipynb, yolo12-ghost-conv-v1.ipynb, dan yolo12-ghost-ema-v4 (2).ipynb beserta folder hasilnya.
- L-Ghost: yolo12-lghost.ipynb dan folder yolo12-lghost; metrik JSON, run_manifest.json, results.csv, lghost_pruning.jsonl, serta best.pt.
- L-Ghost + EMA-32: yolo12-lghostema.ipynb dan folder yolo12-lghostema dengan jenis artefak yang sama.
