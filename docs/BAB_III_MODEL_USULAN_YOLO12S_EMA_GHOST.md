# BAB III — METODOLOGI PENELITIAN

> Dokumen ini adalah draf Bab III tesis. Sesuaikan nomor subbab dengan pedoman kampus dan tambahkan sitasi primer
> untuk YOLO12, Efficient Multi-scale Attention, GhostConv/GhostNet, RDD2022, dan metrik deteksi. Nilai mAP tidak
> ditulis di sini karena merupakan hasil penelitian pada Bab IV.

## 3.1 Desain penelitian

Penelitian ini menggunakan desain eksperimen kuantitatif-komparatif untuk mengevaluasi kemampuan deteksi kerusakan
jalan dari model usulan **YOLO12s–EMA–Ghost**. Model usulan dikembangkan dari YOLO12s standar melalui dua perubahan
yang dibatasi secara terkontrol pada bagian head: penambahan *direct Efficient Multi-scale Attention* (EMA) pada
fitur akhir P3 dan P4, serta penggantian selektif dua konvolusi *bottom-up downsampling* dengan GhostConv.

Backbone YOLO12s, blok A2C2f, fungsi loss, strategi deteksi multiskala, prosedur pembacaan data, dan konfigurasi
pelatihan tidak diubah. Pembatasan tersebut dilakukan agar perbedaan hasil baseline dan model usulan dapat
diatribusikan terutama kepada modul EMA dan GhostConv, bukan kepada perubahan tidak terkontrol pada backbone, loss,
atau data.

Variabel bebas adalah arsitektur model, yaitu YOLO12s baseline dan YOLO12s–EMA–Ghost. Variabel terikat mencakup
precision, recall, mAP@50, mAP@50–95, AP per kelas, jumlah parameter, dan biaya komputasi sebagai informasi
struktural. Hipotesis kerja penelitian adalah bahwa pemurnian fitur P3/P4 dengan EMA dan propagasi fitur *bottom-up*
menggunakan GhostConv berpotensi menghasilkan representasi yang lebih sesuai untuk deteksi kerusakan jalan. Hipotesis
tersebut harus diuji secara empiris; peningkatan kinerja tidak diasumsikan sebelum evaluasi selesai.

> **Catatan:** EMA dalam penelitian ini berarti *Efficient Multi-scale Attention*, bukan *Exponential Moving Average*
> untuk pembaruan parameter model.

## 3.2 Objek penelitian dan dataset

Objek penelitian adalah citra kerusakan jalan pada subset China dari RDD2022 yang telah dipisahkan ke dalam
direktori train, validation, dan test. Dataset menggunakan lima kelas: D00, D10, D20, D40, dan Repair.

| Komponen | Konfigurasi |
| --- | --- |
| Root dataset Kaggle | /kaggle/input/datasets/danialalfayyadh/ch-rdd-2022/datasets-china-split-fix |
| Data latih | train/images |
| Data validasi | val/images |
| Data uji | test/images |
| Jumlah kelas | 5 |
| Kelas | D00, D10, D20, D40, Repair |
| Ukuran masukan | 640 × 640 piksel |

Data validasi digunakan untuk memantau pelatihan dan memilih checkpoint terbaik. Data uji digunakan hanya untuk
evaluasi akhir checkpoint terbaik. Jumlah citra dan instance per split perlu diisi dari keluaran notebook pada saat
penulisan naskah akhir.

## 3.3 Lingkungan implementasi dan reproducibility

Implementasi memakai repository Ultralytics yang telah dimodifikasi dan dibekukan sebelum eksperimen. Pembekuan
memastikan eksperimen tidak bergantung pada branch yang dapat berubah.

| Item | Nilai |
| --- | --- |
| Repository | danial2015/yolo-aceh-rdd2022 |
| Branch implementasi | feat/yolo12-ema-ghost |
| Tag arsitektur beku | research/yolo12s-ema-ghost-architecture-v1 |
| Commit arsitektur beku | d199dd3bac9dd70875370ed3ba16a8a22da1a21b |
| Lingkungan eksekusi | Kaggle Notebook dengan GPU |
| Checkpoint sumber | yolo12s.pt resmi |
| Runner eksperimen | examples/kaggle_yolo12s_ema_ghost_pretrained_rdd2022.ipynb |

Notebook melakukan clone terhadap tag beku, memeriksa SHA commit, lalu memasang repository lokal secara editable.
Setiap eksekusi menghasilkan manifest berisi commit, konfigurasi data, hyperparameter, laporan transfer pretrained,
checkpoint, dan metrik evaluasi. Dengan demikian, hasil dapat ditelusuri dari nilai metrik hingga versi kode yang
digunakan.

## 3.4 Model baseline: YOLO12s

Baseline adalah YOLO12s standar dari Ultralytics tanpa perubahan arsitektur. Model menggunakan backbone untuk
ekstraksi fitur, fusi fitur top-down dan bottom-up pada head, serta Detect head multiskala. Pada input 640 × 640,
tiga tingkat piramida yang digunakan secara konseptual adalah:

| Skala | Resolusi spasial | Peran utama |
| --- | ---: | --- |
| P3 | 80 × 80 | Detail lokal dan objek relatif kecil |
| P4 | 40 × 40 | Objek berskala menengah |
| P5 | 20 × 20 | Konteks semantik dan objek lebih besar |

Baseline dan model usulan menggunakan checkpoint awal yolo12s.pt yang sama. Ketika Detect head disesuaikan dari
80 kelas COCO menjadi lima kelas RDD2022, parameter output klasifikasi yang tidak kompatibel diinisialisasi baru pada
kedua model. Karena itu, model usulan tidak memperoleh keuntungan pretrained yang tidak tersedia bagi baseline.

## 3.5 Arsitektur model usulan YOLO12s–EMA–Ghost

### 3.5.1 Prinsip perancangan

Model usulan mempertahankan backbone YOLO12s secara penuh. Modifikasi ditempatkan setelah fusi fitur akhir pada head,
dekat dengan feature map yang dipakai langsung oleh Detect head. Strategi ini menjaga kontinuitas pengetahuan visual
pretrained pada backbone sekaligus menguji dua mekanisme yang terlokalisasi: attention untuk P3/P4 dan GhostConv
untuk dua transisi bottom-up.

Perubahan arsitektur terdiri atas:

1. EMA langsung setelah fitur P3 hasil fusi akhir;
2. GhostConv pada downsampling P3 menuju P4;
3. EMA langsung setelah fitur P4 hasil fusi akhir;
4. GhostConv pada downsampling P4 menuju P5; dan
5. Detect head dengan masukan EMA-P3, EMA-P4, dan P5 akhir.

Tidak ada EMA pada P5, GhostConv pada stem, atau GhostConv pada backbone awal. Penelitian ini dengan demikian
menguji konfigurasi penempatan tertentu, bukan mengasumsikan semua tingkat piramida memerlukan attention.

### 3.5.2 Alur model

~~~text
Backbone YOLO12s (tidak diubah)
        │
        ├── fusi P3 → A2C2f → EMA-P3 ────────────────────────────┐
        │                         │                                │
        │                    GhostConv s=2                         │
        │                         │                                │
        ├── fitur top-down P4 ─ Concat → A2C2f → EMA-P4 ──────────┤
        │                                            │             │
        │                                       GhostConv s=2      │
        │                                            │             │
        └── backbone P5 ───────────────────────── Concat → C3k2 ──┤
                                                                  │
                                   Detect(EMA-P3, EMA-P4, P5) ◄───┘
~~~

| Peran | Baseline YOLO12s | Model usulan | Keluaran pada 640 |
| --- | --- | --- | --- |
| P3 hasil fusi akhir | A2C2f layer 14 | A2C2f layer 14 | 128 × 80 × 80 |
| Pemurnian P3 | — | EMA layer 15, EMA(128, 32) | 128 × 80 × 80 |
| P3 → P4 | Conv stride 2 layer 15 | GhostConv layer 16 | 128 × 40 × 40 |
| P4 hasil fusi akhir | A2C2f layer 17 | A2C2f layer 18 | 256 × 40 × 40 |
| Pemurnian P4 | — | EMA layer 19, EMA(256, 32) | 256 × 40 × 40 |
| P4 → P5 | Conv stride 2 layer 18 | GhostConv layer 20 | 256 × 20 × 20 |
| P5 akhir | C3k2 layer 20 | C3k2 layer 22 | 512 × 20 × 20 |
| Input Detect | [14, 17, 20] | [15, 19, 22] | P3, P4, P5 |

Secara struktural, baseline memiliki 272 layer dan 9.284.096 parameter, sedangkan model usulan memiliki 286 layer
dan 8.921.104 parameter pada konfigurasi 80 kelas. Jumlah layer meningkat, tetapi parameter dapat berkurang karena
GhostConv menghasilkan sebagian feature map melalui operasi yang lebih murah. Nilai tersebut adalah deskripsi
struktur, bukan klaim kecepatan inferensi.

## 3.6 Direct Efficient Multi-scale Attention

Misalkan masukan EMA adalah feature map:

$$
X \in \mathbb{R}^{B \times C \times H \times W},
$$

dengan $B$ ukuran batch, $C$ jumlah kanal, serta $H$ dan $W$ tinggi dan lebar feature map. Jumlah grup ditetapkan
menjadi $G=32$, lalu feature map dibagi menjadi grup kanal:

$$
X_g = \operatorname{reshape}(X)
\in \mathbb{R}^{(B \cdot G) \times (C/G) \times H \times W}.
$$

Konfigurasi valid pada kedua lokasi penerapan karena:

$$
128 \bmod 32 = 0, \qquad 256 \bmod 32 = 0.
$$

EMA-P3 memiliki empat kanal per grup, sedangkan EMA-P4 memiliki delapan kanal per grup. Representasi koordinat
dibentuk dengan rata-rata sepanjang lebar dan tinggi, kemudian diproses dengan konvolusi $1 \times 1$. Gating
koordinat dan konteks lokal masing-masing dihitung sebagai:

$$
F_1 = \operatorname{GN}
\left(X_g \odot \sigma(H_g) \odot \sigma(W_g^{T})\right),
\qquad
F_2 = \operatorname{Conv}_{3 \times 3}(X_g).
$$

Dua descriptor perhatian diperoleh dari global average pooling dan softmax:

$$
a_1 = \operatorname{Softmax}(\operatorname{GAP}(F_1)),
\qquad
a_2 = \operatorname{Softmax}(\operatorname{GAP}(F_2)).
$$

Peta perhatian gabungan dan keluaran modul adalah:

$$
A = \sigma\left(a_1 \cdot \operatorname{vec}(F_2)
+ a_2 \cdot \operatorname{vec}(F_1)\right),
\qquad
Y = \operatorname{reshape}(X_g \odot A).
$$

EMA diterapkan secara langsung, yaitu $Y=\operatorname{EMA}(X)$, tanpa residual adapter seperti
$X+\alpha\operatorname{EMA}(X)$. Pada mixed precision, softmax dan perkalian matriks affinity dihitung dalam FP32
untuk mengurangi risiko overflow numerik, lalu bobot perhatian dikembalikan ke dtype feature map masukan.

## 3.7 Selective GhostConv

GhostConv digunakan hanya pada dua downsampling bottom-up. Untuk masukan $X$, feature map primer dihitung dengan:

$$
F_p = \operatorname{Conv}_{3 \times 3, s=2}(X).
$$

Feature map tambahan dibuat melalui operasi murah berupa konvolusi depthwise:

$$
F_g = \operatorname{DWConv}_{5 \times 5}(F_p).
$$

Keluaran GhostConv merupakan konkatenasi:

$$
Y = \operatorname{Concat}(F_p, F_g).
$$

GhostConv tidak dipasang pada backbone awal. Backbone awal memuat bagian terbesar representasi visual checkpoint
pretrained, sedangkan dua transisi head merupakan lokasi yang lebih terkontrol untuk menguji efisiensi pembangkitan
fitur pada aliran P3→P4 dan P4→P5.

## 3.8 Inisialisasi pretrained berbasis semantic weight transfer

### 3.8.1 Permasalahan fairness pretrained

Penambahan EMA dan penggantian Conv oleh GhostConv menggeser indeks beberapa layer head. Loader yang hanya
mencocokkan nama tensor dan bentuk tensor dapat kehilangan bobot modul yang sebenarnya tidak berubah secara
semantik. Sebaliknya, pencocokan nama mentah dapat menghasilkan collision pada BatchNorm ketika indeks baru secara
kebetulan sama.

Penelitian ini menggunakan *semantic pretrained transfer*: bobot dipetakan berdasarkan kesetaraan peran arsitektural,
bukan berdasarkan indeks target yang kebetulan sama.

### 3.8.2 Pemetaan semantik

| Layer baseline | Layer proposed | Alasan |
| ---: | ---: | --- |
| 0–14 | 0–14 | Backbone dan jalur top-down/P3 tidak berubah |
| 16 | 17 | Concat P4 yang ekuivalen secara graph |
| 17 | 18 | A2C2f hasil fusi P4 tetap sama |
| 19 | 21 | Concat P5 yang ekuivalen secara graph |
| 20 | 22 | C3k2 P5 akhir tetap sama |
| 21 | 23 | Detect head memiliki peran yang sama |

Conv baseline pada layer 15 dan 18 tidak dipetakan karena benar-benar diganti GhostConv. EMA pada layer 15 dan 19
juga modul baru. Parameter EMA dan GhostConv tetap fresh dan trainable; sistem tidak mengklaim parameter baru identik
dengan Conv baseline.

Pemetaan hanya diaktifkan bila sumber memenuhi fingerprint baseline: graph YOLO12 dengan 22 layer, input Detect
[14, 17, 20], dan tipe modul yang sama pada setiap pasangan yang dipetakan. Checkpoint proposed atau sumber yang
tidak memenuhi fingerprint memakai loader standar berbasis nama dan bentuk tensor.

### 3.8.3 Audit transfer checkpoint resmi

Target lima kelas RDD2022 menggunakan checkpoint resmi yolo12s.pt dengan 699 entri state; model proposed memiliki
715 entri state. Audit menghasilkan:

| Komponen audit | Jumlah |
| --- | ---: |
| Tensor sumber checkpoint | 699 |
| Tensor target proposed | 715 |
| Tensor exact yang ditransfer | 673 |
| Tensor parameter exact yang dipertahankan | 340 |
| Elemen parameter pretrained yang dipertahankan | 8.515.088 |
| Tensor baru dari dua EMA dan dua GhostConv | 36 |
| Tensor class-logit Detect yang baru | 6 |
| Total tensor target fresh | 42 |

State dictionary hasil remap difilter dan tidak digabungkan dengan state dictionary mentah, sehingga collision lama pada
counter BatchNorm tidak dapat terulang. Delapan tensor sumber tanpa pasangan adalah bias positional-convolution legacy
checkpoint resmi, sedangkan dua belas tensor sumber yang tidak dipetakan berasal dari dua Conv stride-2 baseline yang
memang diganti GhostConv. Laporan audit disimpan sebelum epoch pertama sebagai
semantic_pretrained_transfer.json.

## 3.9 Prosedur pelatihan

Pelatihan menggunakan trainer Ultralytics normal dengan pretrained='yolo12s.pt'. Tidak ada transfer manual, custom
trainer, maupun gradient checker tambahan. Trainer membangun target lima kelas, kemudian semantic mapping diterapkan
tepat sekali sebelum pelatihan.

| Hyperparameter | Nilai |
| --- | ---: |
| Epoch | 160 |
| Ukuran citra | 640 |
| Batch size aktual | 16 |
| Nominal batch size (nbs) | 64 |
| Optimizer | SGD |
| Learning rate awal | 0,01 |
| Momentum | 0,937 |
| Weight decay | 0,0005 |
| Cosine learning-rate schedule | Tidak digunakan |
| Patience | 0 |
| Workers | 2 |
| Seed | 42 |
| Checkpoint awal | yolo12s.pt |

Nilai nbs=64 dipakai engine sebagai nominal batch size untuk kebijakan optimisasi/akumulasi gradien, sedangkan batch
aktual 16 disesuaikan dengan memori GPU. Baseline wajib memakai split, ukuran input, epoch, batch, nbs, optimizer,
learning rate, seed, dan checkpoint sumber yang sama.

Tahapan eksperimen:

1. clone tag arsitektur beku dan verifikasi commit;
2. verifikasi struktur dataset dan buat YAML lima kelas;
3. bangun model proposed dengan nc=5;
4. verifikasi posisi EMA, GhostConv, serta input Detect;
5. muat checkpoint melalui semantic loader;
6. verifikasi laporan transfer sebelum epoch pertama;
7. latih model selama 160 epoch;
8. simpan best.pt, last.pt, riwayat loss, plot, dan manifest;
9. evaluasi best.pt pada validation dan test; dan
10. arsipkan seluruh artefak ke ZIP.

Loss mengikuti implementasi native YOLO, yaitu gabungan box loss, classification loss, dan Distribution Focal Loss:

$$
\mathcal{L} =
\lambda_{box}\mathcal{L}_{box} +
\lambda_{cls}\mathcal{L}_{cls} +
\lambda_{dfl}\mathcal{L}_{dfl}.
$$

## 3.10 Metrik evaluasi

Kinerja dievaluasi dengan precision, recall, mAP@50, dan mAP@50–95:

$$
\operatorname{Precision}=\frac{TP}{TP+FP},
\qquad
\operatorname{Recall}=\frac{TP}{TP+FN}.
$$

AP setiap kelas adalah luas area di bawah kurva precision–recall, sedangkan mAP adalah rata-rata AP lima kelas:

$$
\operatorname{mAP}=\frac{1}{5}\sum_{i=1}^{5}AP_i.
$$

mAP@50 memakai ambang IoU 0,50. mAP@50–95 dirata-ratakan pada IoU 0,50 sampai 0,95 dengan interval 0,05. Selain
metrik agregat, hasil per kelas perlu dilaporkan agar perubahan pada suatu kelas tidak tertutup nilai rata-rata.

## 3.11 Rancangan perbandingan, ablation, dan validitas

| Kode | Konfigurasi | Tujuan |
| --- | --- | --- |
| B0 | YOLO12s baseline | Acuan tanpa modifikasi |
| A1 | YOLO12s + EMA pada P3/P4 | Mengukur kontribusi attention |
| A2 | YOLO12s + GhostConv selektif | Mengukur kontribusi GhostConv |
| P | YOLO12s + EMA P3/P4 + GhostConv selektif | Model usulan lengkap |

Semua konfigurasi harus memakai data split, input, checkpoint awal, hyperparameter, seed, dan evaluasi yang sama. Jika
ablation mengubah indeks graph, semantic transfer yang sesuai wajib diaudit agar hasil tidak dipengaruhi ketidakadilan
inisialisasi pretrained.

Selisih terhadap baseline dapat dilaporkan sebagai:

$$
\Delta\operatorname{mAP@50} =
\operatorname{mAP@50}_{proposed} -
\operatorname{mAP@50}_{baseline}.
$$

Validitas internal dijaga melalui baseline tanpa modifikasi, checkpoint sumber yang sama, class-logit lima kelas yang
fresh pada kedua model, transfer pretrained hanya pada layer ekuivalen, test set yang tidak dipakai untuk pemilihan
checkpoint, serta penyimpanan tag, commit, manifest, dan artefak pelatihan.

## 3.12 Batasan dan artefak eksperimen

Penelitian hanya mengevaluasi satu konfigurasi: EMA pada P3/P4 akhir dan GhostConv pada dua transisi bottom-up.
Hasil tidak langsung dapat digeneralisasi ke semua penempatan attention, semua ukuran YOLO12, atau seluruh dataset
kerusakan jalan. Jumlah parameter dan FLOPs adalah deskripsi struktural; kesimpulan latensi perlu benchmark pada
GPU, batch size, dan prosedur pengukuran yang sama.

| Artefak | Fungsi |
| --- | --- |
| best.pt | Checkpoint dengan validation terbaik |
| last.pt | Checkpoint epoch terakhir |
| results.csv | Riwayat loss dan metrik per epoch |
| semantic_pretrained_transfer.json | Bukti audit transfer pretrained |
| Manifest JSON | Tag, commit, data, dan hyperparameter |
| Plot dan confusion matrix | Interpretasi visual evaluasi |
| ZIP Kaggle | Arsip eksperimen yang dapat diunduh |

Gunakan dokumen ini bersama laporan implementasi
[YOLO12S_EMA_GHOST_PROPOSED.md](YOLO12S_EMA_GHOST_PROPOSED.md). Nilai metrik final dan analisis perbandingan
ditempatkan pada Bab IV.
