# YOLO12-Small dengan ECA-Net

## Ringkasan

Branch ini menambahkan _Efficient Channel Attention_ (ECA-Net) pada YOLO12-Small untuk merekalibrasi kanal fitur
penting dengan biaya komputasi yang sangat kecil. Model didefinisikan oleh
[`yolo12-eca.yaml`](../ultralytics/cfg/models/12/yolo12-eca.yaml) dan menggunakan modul
[`ECAAttention`](../ultralytics/nn/modules/conv.py).

Implementasi ini adalah konfigurasi deteksi yang harus dilatih dari awal atau dari checkpoint hasil pelatihan sendiri;
checkpoint ECA-Net terlatih tidak disertakan.

## Perubahan kode

| Komponen | Perubahan |
| --- | --- |
| `ultralytics/nn/modules/conv.py` | Menambahkan modul `ECAAttention`. |
| `ultralytics/nn/modules/__init__.py` | Mengekspor `ECAAttention` agar dapat digunakan oleh konfigurasi model. |
| `ultralytics/nn/tasks.py` | Menambahkan dukungan parser YAML untuk meneruskan jumlah kanal input ke `ECAAttention`. |
| `ultralytics/cfg/models/12/yolo12-eca.yaml` | Konfigurasi YOLO12-Small ECA untuk P3, P4, dan P5. |

## Desain ECA-Net

ECA-Net menghitung perhatian kanal tanpa _fully connected layer_ atau pengurangan dimensi kanal:

1. _Global average pooling_ mengubah fitur `B × C × H × W` menjadi deskriptor kanal `B × C × 1 × 1`.
2. Konvolusi 1D mempelajari hubungan lokal antar-kanal.
3. Sigmoid menghasilkan bobot perhatian, kemudian bobot tersebut dikalikan kembali ke fitur masukan.

Ukuran kernel dihitung secara adaptif dari jumlah kanal:

```text
t = floor(abs((log2(C) + 1) / 2))
k = t jika t ganjil, selain itu t + 1
```

Pada YOLO12-Small, P3, P4, dan P5 masing-masing memiliki 256, 256, dan 512 kanal. Ketiganya menghasilkan kernel
ganjil `k=5`, sehingga konfigurasi memakai tiga modul ECA dengan total 15 parameter tambahan.

```text
P3: C3k2  -> ECAAttention(k=5)
P4: A2C2f -> ECAAttention(k=5)
P5: A2C2f -> ECAAttention(k=5)
```

ECA tidak mengubah bentuk tensor, sehingga koneksi FPN/PAN dan head deteksi YOLO12 tetap kompatibel.

## Cara menggunakan

Jalankan perintah dari akar repository.

### Python

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/12/yolo12-eca.yaml")
model.train(data="path/to/data.yaml", epochs=100, imgsz=640)
```

Untuk inferensi, gunakan checkpoint hasil pelatihan:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("path/to/image.jpg")
```

### CLI

```bash
yolo detect train model=ultralytics/cfg/models/12/yolo12-eca.yaml data=path/to/data.yaml epochs=100 imgsz=640
yolo detect predict model=runs/detect/train/weights/best.pt source=path/to/image.jpg
```

## Validasi implementasi

Validasi lokal yang telah dijalankan:

| Pemeriksaan | Hasil |
| --- | --- |
| Parser konfigurasi YAML | Berhasil membuat model YOLO12-Small ECA. |
| Forward pass CPU | Input `1 × 3 × 64 × 64` menghasilkan prediksi `1 × 7 × 84` untuk 3 kelas. |
| Kernel ECA | Tiga modul, seluruhnya memakai `k=5`. |
| Parameter | YOLO12-Small: 9,284,096; YOLO12-Small ECA: 9,284,111 (`+15`). |

Belum ada benchmark mAP atau latensi. Bandingkan model ini dengan YOLO12-Small standar menggunakan dataset, ukuran
gambar, perangkat, dan hyperparameter yang sama sebelum memutuskan model untuk deployment.
