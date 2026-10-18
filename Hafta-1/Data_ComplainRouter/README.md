# ComplaintRouter AI

**Finansal tüketici şikâyetlerini sınıflandıran uçtan uca NLP projesi**  
**TF-IDF · Logistic Regression · PyTorch MLP · Scikit-learn Pipeline · Terminal Uygulaması**

ComplaintRouter AI, İngilizce tüketici şikâyeti metinlerini dört finansal ürün kategorisinden birine yönlendirmek için geliştirilmiş bir makine öğrenmesi projesidir. Çalışma yalnızca bir notebook deneyinden ibaret değildir: **veri inceleme, model karşılaştırma, hata analizi, modeli diske kaydetme, terminalden tahmin ve düşük olasılıklı sonuçları insan incelemesine yönlendirme** aşamalarını içerir.

> **Proje durumu:** Çalışan bir prototip. Üretim ortamına hazır değildir. Özellikle bağımsız test değerlendirmesi, olasılık kalibrasyonu, alan dışı (OOD) girdi yönetimi ve otomatik testlerin tamamlanması gerekir.

## İçindekiler

- [Problem ve kapsam](#problem-ve-kapsam)
- [Proje akışı](#proje-akışı)
- [Veri seti ve kategoriler](#veri-seti-ve-kategoriler)
- [Modeller ve deney sonuçları](#modeller-ve-deney-sonuçları)
- [Eğitim ve değerlendirme yöntemi](#eğitim-ve-değerlendirme-yöntemi)
- [Terminal uygulaması](#terminal-uygulaması)
- [Human Review mekanizması](#human-review-mekanizması)
- [Yapılan uygulama testleri](#yapılan-uygulama-testleri)
- [Kurulum ve çalıştırma](#kurulum-ve-çalıştırma)
- [Proje yapısı](#proje-yapısı)
- [Sınırlamalar ve sonraki adımlar](#sınırlamalar-ve-sonraki-adımlar)

## Problem ve kapsam

**Girdi:** İngilizce bir `Consumer complaint narrative` (tüketici şikâyet metni).  
**Çıktı:** Tahmin edilen finansal ürün kategorisi, o sınıfa atanan model olasılığı ve yönlendirme durumu.

Örnek terminal kullanımı:

```bash
python notebooks/predict.py --text "My mortgage payment is incorrect"
```

Gerçekleştirilen örnek çalıştırmalardan biri:

```text
ComplaintRouter AI
-------------------------------
Category: Mortgage
Probability: 0.94
Status: Automatic
```

Model yalnızca eğitimdeki **dört kategori** arasından seçim yapabilir. Düşük olasılıklı tahminler, prototipte insan incelemesine yönlendirilir; bu mekanizma alan dışı girdileri güvenilir biçimde tespit eden ayrı bir OOD modeli değildir.

## Proje akışı

### 1. Veri hazırlama ve model geliştirme

```text
complaints_train_5000.csv
          │
          ▼
Veri inceleme ve etiket eşleştirmesi
          │
          ▼
Metin temizleme
          │
          ▼
Stratified eğitim / holdout ayrımı
          │
          ▼
TF-IDF (eğitim setinde fit)
          │
          ├── Logistic Regression (baseline)
          │
          └── PyTorch MLP (128 / 256 gizli nöron)
                      │
                      ▼
         Accuracy · Macro-F1 · Confusion Matrix
```

### 2. Kaydedilmiş model ile tahmin

```text
Terminal: --text "şikâyet metni"
          │
          ▼
Girdi doğrulama
          │
          ▼
Kaydedilmiş Scikit-learn Pipeline
  ├─ FunctionTransformer → clean_texts()
  ├─ TfidfVectorizer (max_features=3000)
  └─ LogisticRegression
          │
          ▼
predict() + predict_proba()
          │
          ▼
Kategori adı + olasılık
          │
          ├─ Olasılık ≥ 0.60 → Automatic
          └─ Olasılık < 0.60 → Human Review Required
```

Tahmin fonksiyonu, temizleme sonrası boş kalan metinleri reddeder. **Pipeline temizlemeyi içerdiğinden, tahmin için ham metin Pipeline'a verilir**; aynı temizleme dışarıda tekrar uygulanarak modele ikinci kez zorunlu tutulmaz.

## Veri seti ve kategoriler

- **Kaynak çalışma dosyası:** `data/complaints_train_5000.csv`
- **Kayıt sayısı:** 5.000
- **Metin alanı:** `Consumer complaint narrative`
- **Hedef alan:** `Product`
- `Complaint ID`, modelin giriş özelliği olarak kullanılmaz.

| Sayısal etiket | Ürün kategorisi |
|---:|---|
| 0 | Credit reporting, credit repair services, or other personal consumer reports |
| 1 | Debt collection |
| 2 | Credit reporting |
| 3 | Mortgage |

**Önemli veri notu:** `0` ve `2` etiketleri, adlarından da anlaşılacağı üzere birbirine yakın kredi raporlama başlıklarıdır. Bu benzerlik modelin sınıfları karıştırmasına katkıda bulunabilir.

Ham şikâyetler kişisel veri içerebilir. **Veri setini paylaşmadan önce lisans koşullarını, kişisel bilgi riskini ve GitHub'a yükleme iznini kontrol edin.**

## Modeller ve deney sonuçları

Aynı proje kapsamında kelime tabanlı TF-IDF özellikleriyle Logistic Regression baseline'ı ve PyTorch çok katmanlı algılayıcılar (MLP) karşılaştırıldı.

| Deney | TF-IDF özellik sayısı | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| **Logistic Regression (baseline)** | **3.000** | **%72,60** | **0,7231** |
| Logistic Regression (baseline) | 1.500 | %72,30 | 0,7197 |
| PyTorch MLP — 128 gizli nöron | 1.500 | %67,30 | 0,6739 |
| PyTorch MLP — 256 gizli nöron | 1.500 | Ölçülmedi / raporlanmadı | 0,6730 |

**Sonuç:** Kaydedilen deneylerde **3.000 özellikli Logistic Regression**, değerlendirilen MLP alternatiflerinden daha iyi Macro-F1 elde etti. Bu nedenle terminal uygulamasında Logistic Regression Pipeline'ı kullanılıyor.

MLP-256 için son eğitim kaybı **0,0406** olmasına rağmen Macro-F1 (**0,6730**), MLP-128'den (**0,6739**) az da olsa düşüktü. Daha düşük *training loss*, otomatik olarak daha iyi genelleme anlamına gelmez. Bu küçük MLP farkı tek başına istatistiksel anlamlılık göstermez.

### Logistic Regression baseline ayarları

Terminal uygulamasında kullanılan kaydedilmiş Pipeline şu bileşenlerden oluşur:

```python
Pipeline(steps=[
    ("clean_text", FunctionTransformer(clean_texts, validate=False)),
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("classifier", LogisticRegression(
        C=1, max_iter=1000, penalty="l2", solver="newton-cg"
    )),
])
```

Metin temizleme adımı; boş/eksik değerleri ele alır, küçük harfe çevirir, `XXXX` benzeri maskeleri ve noktalama işaretlerini kaldırır, fazla boşlukları tekilleştirir. **Aynı fonksiyonun eğitimde ve tahminde kullanılması**, öğrenilen TF-IDF özelliklerinin anlamının korunması açısından önemlidir.

### PyTorch MLP mimarisi

```text
TF-IDF (1.500 boyut)
        ↓
Linear(1500, 128 veya 256) + ReLU
        ↓
Linear(128 veya 256, 32) + ReLU
        ↓
Linear(32, 4) → logits
```

| Özellik | MLP-128 | MLP-256 |
|---|---:|---:|
| Eğitilebilir parametre | 196.388 | 392.612 |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | Adam, lr=0.001 | Adam, lr=0.001 |
| Batch size | 32 | 32 |
| Epoch | 10 | 10 |

`CrossEntropyLoss`, logits üzerinden gerekli olasılık hesabını kendi içinde gerçekleştirdiğinden eğitim modelinin sonuna ayrıca `Softmax` eklenmedi.

### Hata analizi

MLP-128 deneyinde en zor ayrıştırılan iki sınıf:

- **Sınıf 0:** F1 = 0,53
- **Sınıf 2:** F1 = 0,56

Confusion matrix kaydında bu sınıflar arasında **155 karışıklık** görüldü: `0 → 2` için 76, `2 → 0` için 79 örnek. En yüksek F1, **Mortgage** sınıfındaydı (**0,89**). Kredi raporlama etiketlerinin semantik benzerliği olası bir neden olsa da bunun doğrulanması için hatalı sınıflandırılmış metinleri tek tek incelemek gerekir.

## Eğitim ve değerlendirme yöntemi

- `train_test_split(test_size=0.2, random_state=42, stratify=y)` kullanıldı.
- **4.000 eğitim** ve **1.000 holdout** örneği elde edildi.
- Holdout kümesinde her sınıftan **250 örnek** bulunuyordu.
- TF-IDF kelime sözlüğü ve IDF değerleri yalnızca eğitim verisinde öğrenildi (`fit_transform`); holdout üzerinde yalnızca `transform` kullanıldı.
- Model karşılaştırmalarında Accuracy, Macro-F1, sınıf bazlı rapor ve confusion matrix incelendi.

> **Değerlendirme sınırlaması:** Eski deneylerde `val_loader` adını taşıyan veri aslında `X_test`/holdout bölümünden geliyordu. Aynı holdout, hiperparametre/model karşılaştırmasında da kullanıldığı için **tamamen bağımsız, nihai test sonucu** olarak yorumlanmamalıdır. Gelecek iterasyonda ayrı **train / validation / test** ayrımı kullanılacak ve final test kümesine model seçimi tamamlandıktan sonra bakılacaktır.

128 ve 256 nöronlu MLP deneyleri için veri sırası, DataLoader karıştırması ve random seed'in her koşulda birebir eşitlendiği doğrulanmadığından bu sonuçlar sınırlı bir karşılaştırmadır.

## Terminal uygulaması

Uygulama `notebooks/predict.py` dosyasındadır. `argparse` kullanır; notebook açmadan komut satırından çalıştırılabilir.

```bash
python notebooks/predict.py --text "I received a debt collection notice for a debt I already paid."
```

Örnek sonuç:

```text
ComplaintRouter AI
-------------------------------
Category: Debt collection
Probability: 0.99
Status: Automatic
```

Uygulamanın temel işlevleri:

1. **Girdi denetimi:** `None`/string olmayan değerler, boş veya yalnızca boşluk içeren metinler ve temizlik sonrası boş kalan metinler reddedilir.
2. **Ortak preprocessing:** `preprocessing.py` içindeki `clean_text()` ve `clean_texts()` kullanılır.
3. **Model yükleme:** Kaydedilmiş `models/complaintrouter_lr.joblib`, `joblib.load()` ile yüklenir; her tahminde yeniden eğitim yapılmaz.
4. **Tahmin:** `predict()` ile kategori, `predict_proba()` ile sınıf olasılıkları hesaplanır.
5. **Sonuç biçimi:** `category`, `probability` ve `status` alanları bir sözlükten terminale yazdırılır.

Tahmin fonksiyonunun döndürdüğü örnek sözlük:

```python
{
    "category": "Mortgage",
    "probability": 0.94,
    "status": "automatic"
}
```

Terminal arayüzü durumu `Automatic` / `Human Review Required` biçiminde gösterir.

### Model Persistence

Scikit-learn `Pipeline`, temizleme dönüştürücüsünü, **TF-IDF sözlüğünü/IDF değerlerini** ve **öğrenilmiş Logistic Regression parametrelerini** birlikte saklar. Kaydedilen nesne sayesinde her terminal çağrısında model yeniden eğitilmez.

`FunctionTransformer` içindeki kullanıcı tanımlı fonksiyonların yükleme sırasında bulunabilmesi için `preprocessing.py` modülünün aynı Python ortamında erişilebilir olması gerekir.

> **Güvenlik:** `joblib.load()`, pickle tabanlı mekanizma nedeniyle güvenilmeyen dosyalar yüklenirken zararlı kod çalıştırabilir. Rastgele internet kaynaklarından indirilen `.joblib` dosyalarını doğrudan yüklemeyin.

## Human Review mekanizması

Prototipte aşağıdaki iş kuralı uygulanır:

| En yüksek sınıf olasılığı | Durum |
|---|---|
| `>= 0.60` | `Automatic` |
| `< 0.60` | `Human Review Required` |

Örnek olarak **%46** olasılıklı bir Mortgage tahmini insan incelemesine yönlendirilmiştir.

**%60 eşiği doğrulanmış bir güven sınırı değildir.** Gerçek uygulamada eşik; validation sonuçları, olasılık kalibrasyonu, yanlış yönlendirme maliyeti ve iş süreçleriyle belirlenmelidir. `predict_proba()` çıktısındaki %90 değer, otomatik olarak %90 gerçek doğruluk anlamına gelmez.

En yüksek iki sınıf olasılığı arasındaki fark (**prediction margin**) da belirsizlik için ek sinyal olabilir. Örneğin `0.46` ve `0.44` arasındaki fark yalnızca `0.02`'dir. Ancak küçük margin veya düşük maksimum olasılık, tek başına güvenilir bir OOD tespit yöntemi değildir.

## Yapılan uygulama testleri

Aşağıdaki sonuçlar terminal uygulamasında gözlemlenen **örnek çalıştırmalardır**; toplu değerlendirme ölçütlerinin yerine geçmez.

| Senaryo | Girdi / durum | Gözlenen çıktı |
|---|---|---|
| Debt Collection | Önceden ödenmiş borç için tahsilat bildirimi | `Debt collection` · **0.99** · `Automatic` |
| Mortgage | Yanlış aylık konut kredisi ödemesi | `Mortgage` · **0.87** · `Automatic` |
| Credit Reporting | Kredi raporunda hatalı ödeme geçmişi | `Credit reporting` · **0.62** · `Automatic` |
| Boş metin | `--text ""` | Anlaşılır hata mesajı |
| Sadece boşluk | `--text "     "` | Anlaşılır hata mesajı |
| Düşük olasılık | Belirsiz ödeme şikâyeti | `Mortgage` · **0.46** · `Human Review Required` |
| Çok uzun metin | Mortgage cümlesi 1.000 kez tekrarlandı | `Mortgage` · **0.95**; çökmeden tamamlandı |
| Alan dışı (OOD) örneği | `My laptop battery is not working.` | `Credit reporting` · **0.32** · `Human Review Required` |

**OOD yorumu:** Dizüstü bilgisayar arızası dört finans kategorisine ait değildir; buna rağmen kapalı-küme sınıflandırıcı mevcut sınıflardan birini seçmiştir. Düşük olasılık nedeniyle incelemeye yönlendirilmesi yararlı olsa da model **"bilinmeyen kategori"** etiketini doğrudan üretmez ve yüksek olasılıklı OOD hataları hâlâ mümkündür.

**Henüz doğrulanmamış / tamamlanacak testler:** Eksik model dosyası senaryosunun sistematik testi ve `pytest` ile en az üç otomatik test (boş metin, çıktı sözlüğü şeması, kaydet-yükle tahmin eşitliği). Terminalden ayrı çalıştırmalarda model diskteki dosyadan yeniden yüklenerek kullanılabilmiştir.

## Kurulum ve çalıştırma

### Gereksinimler

- Python 3 ve projedeki PyTorch / scikit-learn sürümleriyle uyumlu bir ortam (özellikle sürüm uyumluluğunu kontrol edin)
- Temel paketler: `pandas`, `numpy`, `scikit-learn`, `joblib`
- Notebook/MLP deneyleri için ayrıca: `torch`, `matplotlib`, `jupyter` (kullanılan notebook'a bağlı ek kütüphaneler gerekebilir)

Örnek ortam kurulumu:

```bash
python -m venv .venv
source .venv/bin/activate  # Linux / macOS
python -m pip install pandas numpy scikit-learn joblib torch matplotlib jupyter pytest
```

### Terminalden tahmin

Proje ana klasöründen çalıştırın:

```bash
python notebooks/predict.py --text "My mortgage payment is incorrect"
```

Diğer örnekler:

```bash
python notebooks/predict.py --text "My credit report contains incorrect information."
python notebooks/predict.py --text ""
python notebooks/predict.py --text "My laptop battery is not working."
```

**GitHub'a yüklemeden önce dosya yolu kontrolü:** Geliştirme sırasında `MODEL_PATH` mutlak (`/home/...`) bir yol olarak ayarlandıysa, başka bilgisayarlarda çalışması için `notebooks/predict.py` içinde proje konumundan türetilen yolu kullanın:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "complaintrouter_lr.joblib"
```

`models/complaintrouter_lr.joblib` dosyası dağıtıma dahil edilmemişse önce notebook üzerinden modeli üretip bu konuma kaydetmek gerekir. Model dosyasının güvenilir kaynaktan geldiğinden emin olun.

### Notebook deneylerini çalıştırma

```bash
jupyter notebook
```

Notebook'ları `notebooks/` klasöründen açın. Kodun veri yolu beklentilerine göre `data/` konumunu doğrulayın. TF-IDF/MLP karşılaştırmalarının yeniden üretilebilmesi için `complaints_train_5000.csv` gereklidir.

## Proje yapısı

Aşağıdaki yapı, geliştirme sırasında kullanılan temel dizinleri ve dosyaları gösterir:

```text
Data_ComplainRouter/
├── data/
│   ├── complaints_train_5000.csv
│   ├── complaints_test.csv
│   └── rows.csv
├── models/
│   └── complaintrouter_lr.joblib
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_baseline_model.ipynb
│   ├── 03_pytorch_mlp.ipynb
│   ├── 04-final_model.ipynb
│   ├── preprocessing.py
│   └── predict.py
├── reports/
├── tests/
└── README.md
```

| Dosya / klasör | Amaç |
|---|---|
| `notebooks/01_data_audit.ipynb` | Veri inceleme ve kalite kontrolü |
| `notebooks/02_baseline_model.ipynb` | TF-IDF + Logistic Regression baseline |
| `notebooks/03_pytorch_mlp.ipynb` | PyTorch MLP deneyleri (128/256) |
| `notebooks/04-final_model.ipynb` | Son Logistic Regression Pipeline çalışması ve değerlendirme |
| `notebooks/preprocessing.py` | Ortak metin temizleme fonksiyonları |
| `notebooks/predict.py` | `argparse` tabanlı terminal sınıflandırıcısı |
| `models/complaintrouter_lr.joblib` | Eğitilmiş Pipeline artefaktı |
| `reports/` | Teknik / deney raporları için klasör |
| `tests/` | Planlanan otomatik testler için klasör |

> Önceki deney notlarında geçen `baseline_report.md`, `pytorch_experiment_report.md`, `deployment_report.md` ve grafik dosyaları repoya eklenecekse mevcut konumlarına göre bağlantılar ayrıca düzenlenmelidir. Bu README, mevcut olmayan rapor/görsel bağlantılarına bağımlı değildir.

## Sınırlamalar ve sonraki adımlar

1. **Veri ayrımı:** Train / validation / test kümelerini birbirinden ayırmak; modeli validation ile seçip testte bir kez değerlendirmek.
2. **Olasılık kalibrasyonu:** `predict_proba()` sonuçlarını kalibrasyon metrikleri ve uygun yöntemlerle doğrulamak.
3. **Human Review eşiği:** Deneysel `0.60` sınırını validation verisi ve hata maliyetine göre ayarlamak; gerekirse ilk iki olasılığın farkını kullanmak.
4. **OOD / bilinmeyen konu:** Eğitim kategorileri dışındaki girdileri işaretleyecek ek kontroller geliştirmek; yüksek olasılıklı yanlış yönlendirmeleri incelemek.
5. **Otomatik testler:** `pytest` ile geçersiz giriş, çıktı şeması, model yükleme ve tahmin tutarlılığı testlerini tamamlamak; eksik model dosyası için kontrollü hata üretmek.
6. **Model iyileştirme:** `ngram_range`, `max_features`, düzenlileştirme, karakter n-gramları ve sınıf `0 ↔ 2` hata örneklerini araştırmak.
7. **Tekrarlanabilirlik:** Bağımlılık sürümlerini sabitlemek, deney seed'lerini standartlaştırmak ve model dosyasının sürümünü belgelemek.
8. **Dağıtım güvenliği:** Hassas veriyi korumak, güvenilmeyen `joblib` dosyalarını yüklememek, metin uzunluğu/performans sınırlarını ve gözlemlenebilirliği eklemek.

## Bu projede öğrendiklerim

- TF-IDF ile metinleri sayısallaştırmayı ve **fit/transform ayrımını** doğru uygulamayı.
- Baseline olarak Logistic Regression kullanmayı ve PyTorch MLP ile karşılaştırmayı.
- `nn.Module`, `DataLoader`, `CrossEntropyLoss`, `Adam`, `backward()` ve `optimizer.step()` kavramlarını.
- Accuracy, Macro-F1, sınıf raporu ve confusion matrix ile hata analizi yapmayı.
- `FunctionTransformer` + `TfidfVectorizer` + `LogisticRegression` bileşenlerini tek bir **Pipeline** içinde kullanmayı.
- `joblib.dump()` / `joblib.load()` ile modeli **yeniden eğitmeden** çalıştırmayı.
- `argparse` ile terminal uygulaması oluşturmayı, girdileri doğrulamayı ve sınıf olasılıklarını göstermeyi.
- **Human Review**, olasılık kalibrasyonu ve **OOD** sınırlarının gerçek uygulamalarda neden önemli olduğunu.

---

**Özet:** Bu çalışma, 5.000 şikâyet üzerinde farklı metin sınıflandırıcılarını karşılaştıran ve seçilen Logistic Regression Pipeline'ını komut satırında çalıştıran **uçtan uca bir NLP prototipidir**. En iyi kaydedilmiş baseline sonucu **%72,60 Accuracy / 0,7231 Macro-F1**'dir. Sonraki odak, genelleme ve güvenilir dağıtım için bağımsız değerlendirme, kalibrasyon ve otomatik testlerdir.