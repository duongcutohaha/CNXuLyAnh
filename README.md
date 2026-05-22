# Fruit Counter

Du an Python/OpenCV dung de phat hien va dem so luong trai cay trong anh hoac video. Chuong trinh xu ly anh bang grayscale, CLAHE, Gaussian blur, threshold Otsu, morphology va watershed; voi video, chuong trinh dung centroid tracker de gan ID va dem tong so doi tuong xuat hien.

## Cau truc du an

```text
fruit_counter/
├── images/
│   └── sample.jpg          # Anh mau dau vao
├── output/                 # Thu muc luu ket qua
├── utils/
│   ├── preprocess.py       # Tien xu ly anh
│   ├── detect.py           # Tao mask phat hien trai cay
│   └── count.py            # Dem doi tuong bang watershed
├── main.py                 # Dem trai cay tu anh
├── video_main.py           # Dem/tracking trai cay tu video
├── tracker.py              # Centroid tracker cho video
├── video.mp4               # Video mau
└── requirements.txt        # Thu vien can cai
```

## Yeu cau

- Python 3.9 tro len
- pip
- He dieu hanh co ho tro OpenCV GUI neu muon dung `--show` hoac chay video

## Cai dat

Mo terminal tai thu muc chua du an, sau do chay:

```bash
cd fruit_counter
python -m venv .venv
```

Kich hoat moi truong ao tren Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Neu dung Command Prompt:

```cmd
.\.venv\Scripts\activate.bat
```

Neu dung macOS/Linux:

```bash
source .venv/bin/activate
```

Cai dat thu vien:

```bash
pip install -r requirements.txt
```

## Cach su dung

### 1. Dem trai cay tu anh

Chay voi anh mau co san:

```bash
python main.py
```

Mac dinh chuong trinh se doc:

```text
images/sample.jpg
```

Va luu anh ket qua tai:

```text
output/result.jpg
```

Chay voi anh tuy chon:

```bash
python main.py --input images/sample.jpg --output output/result.jpg
```

Hien thi cac cua so debug khi chay:

```bash
python main.py --show
```

Tuy chinh nguong loc dien tich doi tuong:

```bash
python main.py --min-area 700
```

Tuy chinh nguong tach cac trai cay bi dinh nhau:

```bash
python main.py --distance-ratio 0.35
```

Co the ket hop cac tham so:

```bash
python main.py --input images/sample.jpg --output output/result.jpg --min-area 700 --distance-ratio 0.35 --show
```

### 2. Dem trai cay tu video

Chay voi video mau `video.mp4`:

```bash
python video_main.py
```

Khi cua so video hien ra:

- Khung mau xanh la: vung trai cay duoc phat hien
- Cham/ID mau xanh duong: ID cua doi tuong dang duoc tracker theo doi
- `Total`: tong so doi tuong da duoc ghi nhan
- Nhan phim `Esc` de thoat

De dung webcam thay cho file video, sua dong sau trong `video_main.py`:

```python
cap = cv2.VideoCapture("video.mp4")
```

Thanh:

```python
cap = cv2.VideoCapture(0)
```

## Tham so chinh cua `main.py`

| Tham so | Mac dinh | Y nghia |
| --- | --- | --- |
| `--input` | `images/sample.jpg` | Duong dan anh dau vao |
| `--output` | `output/result.jpg` | Duong dan anh ket qua |
| `--min-area` | `700` | Dien tich toi thieu de tinh la mot trai cay |
| `--distance-ratio` | `0.35` | Nguong distance transform de tach trai cay dinh nhau |
| `--show` | Tat | Hien thi cua so xu ly trung gian |

## Luu y

- Neu dem thieu hoac dem du, hay thu dieu chinh `--min-area` va `--distance-ratio`.
- Anh dau vao nen co do tuong phan tot giua trai cay va nen.
- Khi chay `video_main.py`, can co man hinh GUI vi chuong trinh dung `cv2.imshow`.
- Neu OpenCV khong mo duoc cua so hien thi, hay thu chay trong terminal truc tiep thay vi moi truong khong co giao dien.