# Analisis Perbandingan Model ARDD26 (Aceh)

Diperbarui 2 Oktober 2026. Kinerja utama memakai evaluasi split test atas checkpoint terbaik (best.pt). Angka validasi dan test dibahas terpisah.

## 1. Cakupan dan sumber eksperimen

Perbandingan mencakup YOLOv8s, YOLO11s, YOLO12s original, YOLO12s + EMA-32, YOLO12s + GhostConv, Direct EMA + GhostConv, L-Ghost, serta L-Ghost + EMA-32. Dataset Aceh memiliki empat kelas: D00, D10, D20, dan D40. Output test model baru mencatat 527 gambar dan 928 objek.

Sumber L-Ghost + EMA-32 beserta ZIP hasil ekstraksinya berada di folder ARDD26 sebagai yolo12s-l-ghost-ema-32-aceh-road-damage-dete.ipynb. Notebook L-Ghost murni berada di folder China dengan nama yolo12s-l-ghost-aceh-road-damage-detection-202.ipynb; isi dan dataset_root-nya menunjukkan bahwa notebook tersebut adalah eksperimen **Aceh**, sehingga dimasukkan ke laporan ini. Salinan berakhiran (1) memiliki hash SHA-256 yang sama dan tidak dihitung sebagai eksperimen terpisah. Untuk L-Ghost murni, angka presisi berasal dari output notebook karena ZIP hasilnya belum tersedia di folder ekstraksi yang ditelaah.

## 2. Protokol dan rancangan arsitektur

Eksperimen utama YOLO12s memakai 160 epoch, ukuran citra 640, batch 16, batch nominal 64, optimizer SGD, learning rate awal 0,01, momentum 0,937, weight decay 0,0005, patience 0, dan seed 42. Semua hasil test memakai best.pt. Varian lama dan varian L-Ghost mempunyai perbedaan struktur serta cara transfer bobot, sehingga kesetaraan hyperparameter tidak sendirinya membuktikan efek kausal satu modul.

YOLO12s baseline memakai arsitektur original. EMA-32 menambah perhatian pada P3/8. GhostConv mengganti operasi pada P4/16 dan P5/32. Direct EMA + GhostConv menggabungkan EMA pada layer 15/19 dengan GhostConv pada 16/20. L-Ghost memakai LGhostBottleneck pada layer 16/20; kombinasi L-Ghost + EMA-32 menambah EMA pada 15/19. Dua notebook L-Ghost memakai commit 14cbeb1b9a136c17da4ff0acfe01a3f625cd5e3e dan transfer pretrained semantik yang memindahkan 673 tensor secara tepat.

Trainer L-Ghost menjalankan pruning setelah epoch 32, 64, dan 96. Pada kedua blok, fraksi grup yang dipertahankan bertahap menjadi 0,75; 0,50; lalu 0,25. Jadwal ini menambah perbedaan metode terhadap GhostConv biasa. Mask pruning tidak identik dengan pengurangan jumlah parameter tersimpan pada graph.

## 3. Kinerja agregat pada test

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| YOLOv8s | 0,731578 | 0,620511 | 0,712995 | 0,457048 |
| YOLO11s | 0,716616 | 0,667542 | 0,733605 | 0,483569 |
| YOLO12s original | 0,690845 | **0,706835** | 0,753887 | 0,497375 |
| EMA-32 | 0,748889 | 0,663744 | 0,760702 | 0,491517 |
| GhostConv | 0,756210 | 0,657835 | 0,757971 | 0,497173 |
| Direct EMA + GhostConv | 0,727973 | 0,683008 | 0,758103 | 0,473823 |
| L-Ghost | 0,737731 | 0,660858 | **0,764888** | 0,499344 |
| L-Ghost + EMA-32 | **0,756958** | 0,648893 | 0,759719 | **0,503937** |

L-Ghost + EMA-32 mencapai precision dan mAP50-95 test tertinggi di antara hasil yang tersedia. L-Ghost murni memberi mAP50 tertinggi. Baseline YOLO12s masih memiliki recall tertinggi. Perbedaan mAP50-95 L-Ghost + EMA-32 terhadap baseline kecil, sehingga kesimpulan harus ditulis sebagai peningkatan teramati, bukan kemenangan universal.

### Selisih absolut terhadap baseline YOLO12s

| Varian | Δ Precision | Δ Recall | Δ mAP50 | Δ mAP50-95 |
|---|---:|---:|---:|---:|
| Direct EMA + GhostConv | +0,037129 | −0,023827 | +0,004216 | −0,023552 |
| L-Ghost | +0,046886 | −0,045977 | +0,011002 | +0,001969 |
| L-Ghost + EMA-32 | +0,066113 | −0,057942 | +0,005832 | +0,006562 |

Delta pada skala 0–1 dihitung sebagai varian dikurangi baseline. L-Ghost + EMA-32 meningkatkan mAP50-95 dari 0,497375 menjadi 0,503937, dengan penurunan recall dari 0,706835 menjadi 0,648893. Terhadap Direct EMA + GhostConv, mAP50-95 kombinasi baru naik +0,030114; kedua model tersebut memakai modul/trainer berbeda, sehingga selisih ini belum mengisolasi kontribusi L-Ghost saja.

## 4. mAP50-95 test per kelas

Baseline dan Direct EMA + GhostConv tersedia dengan lebih banyak digit pada artefak sebelumnya. Dua varian L-Ghost ditampilkan hingga tiga desimal sesuai output tabel kelas pada notebook.

| Kelas | Baseline YOLO12s | Direct EMA + GhostConv | L-Ghost | L-Ghost + EMA-32 |
|---|---:|---:|---:|---:|
| D00 | 0,444973 | 0,438367 | 0,451 | **0,457** |
| D10 | 0,526961 | 0,463199 | 0,503 | **0,527** |
| D20 | 0,528437 | 0,520339 | **0,546** | 0,536 |
| D40 | 0,489130 | 0,473389 | **0,497** | 0,496 |

L-Ghost + EMA-32 terlihat naik terhadap baseline pada D00, D20, dan D40; D10 sekitar tetap jika dibaca pada ketelitian tiga desimal. L-Ghost murni lebih tinggi daripada kombinasi L-Ghost + EMA-32 pada D20 dan D40. Selisih kecil antarkelas tidak dapat dianggap signifikan tanpa pengulangan.

## 5. Kompleksitas, ukuran checkpoint, dan inferensi

| Model YOLO12s | Layer graph training | Parameter | GFLOPs graph training | best.pt | Inferensi test |
|---|---:|---:|---:|---:|---:|
| Original | 272 | 9.254.684 | 23,5 | 18,05 MB | 8,9 ms/gambar |
| EMA-32 | 277 | 9.255.356 | 23,7 | 18,05 MB | 10,2 ms/gambar |
| GhostConv | 276 | 8.379.548 | 22,1 | 16,38 MB | 9,2 ms/gambar |
| Direct EMA + GhostConv | 286 | 8.891.692 | 23,2 | 17,37 MB | 11,0 ms/gambar |
| L-Ghost | 306 | 8.736.988 | 23,2 | tidak tersedia lokal | 9,9 ms/gambar |
| L-Ghost + EMA-32 | 314 | 8.737.836 | 23,3 | 17,10 MB | 10,1 ms/gambar |

Angka L-Ghost murni berasal dari output notebook; ukuran best.pt tidak dinyatakan karena checkpoint hasil ekstraksinya belum tersedia pada lokasi yang ditelaah. Model gabungan baru lebih kecil daripada baseline menurut jumlah parameter, tetapi output inferensinya lebih lambat 1,2 ms/gambar. Waktu inferensi ini bersifat observasi pada run masing-masing, bukan benchmark hardware terkontrol. GFLOPs graph training juga tidak sama dengan kecepatan deployment.

## 6. Training dan tahap pruning checkpoint terbaik

| Model | Epoch validasi terbaik menurut results.csv | mAP50-95 terbaik | Nilai epoch 160 |
|---|---:|---:|---:|
| YOLO12s original | 81 | 0,46187 | 0,41813 |
| EMA-32 | 81 | 0,46418 | 0,42675 |
| GhostConv | 87 | 0,47034 | 0,43164 |
| Direct EMA + GhostConv | 65 | 0,45600 | 0,42155 |
| L-Ghost + EMA-32 | 92 | 0,46971 | 0,43868 |

Untuk L-Ghost murni, results.csv lokal belum tersedia, sehingga epoch terbaik tidak ditebak dari output training yang dibulatkan. Notebook L-Ghost murni secara eksplisit melaporkan best.pt pada **tahap pruning 1 dari 3**. L-Ghost + EMA-32 melaporkan best.pt pada **tahap 2 dari 3**; epoch terbaiknya 92, sebelum tahap 3 pada epoch 97. Training keduanya mencapai tahap 3, tetapi angka test di atas berasal dari checkpoint terbaik yang belum menyelesaikan semua tahap pruning. Karena itu, angka test Aceh tidak boleh dipresentasikan sebagai hasil model akhir yang sepenuhnya dipangkas. Evaluasi last.pt atau checkpoint tahap 3 diperlukan untuk mengukur trade-off pada kondisi itu.

## 7. Pembacaan lintas dataset dan batas klaim

Pada China, L-Ghost + EMA-32 juga menaikkan mAP50-95 terhadap baseline, dari 0,578334 menjadi 0,592890 (+0,014556). Pada Aceh kenaikannya lebih kecil, dari 0,497375 menjadi 0,503937 (+0,006562), dan recall turun −0,057942. Arah dan besar trade-off berbeda menurut dataset. Angka mAP mentah China dan Aceh tidak boleh dibandingkan langsung karena isi data serta jumlah kelas berbeda; yang dapat dibandingkan secara hati-hati adalah arah perubahan terhadap baseline masing-masing.

Kedua dataset baru menunjukkan kenaikan mAP50-95 teramati untuk L-Ghost + EMA-32 terhadap baseline, tetapi hanya satu seed dan satu split yang tersedia. Belum ada bukti signifikansi statistik atau jaminan generalisasi. Pada Aceh khususnya, pembaca harus mengetahui bahwa checkpoint terbaik berhenti pada tahap pruning 2.

## 8. Jejak bukti utama

- Baseline dan varian lama: notebook YOLOv8s, YOLO11s, YOLO12s baseline, EMA-32, GhostConv, serta aceh-ardd-yolo12-proposed.ipynb di folder ARDD26, bersama JSON evaluasi dan hasil ekstraksinya.
- L-Ghost murni: E:\Hasil\rdd2022-china-result\yolo12s-l-ghost-aceh-road-damage-detection-202.ipynb, terutama tabel test pada output cell evaluasi dan laporan tahap pruning best.pt. Duplikat berakhiran (1) identik.
- L-Ghost + EMA-32: yolo12s-l-ghost-ema-32-aceh-road-damage-dete.ipynb dan folder ekstraksinya di ARDD26; evaluation_metrics.json, args.yaml, results.csv, lghost_pruning.jsonl, dan best.pt.
