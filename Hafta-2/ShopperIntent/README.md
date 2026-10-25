# ShopperIntent 🛒

### PyTorch ile e-ticaret oturumlarında satın alma olasılığı tahmini

ShopperIntent, bir e-ticaret oturumunun **satın almayla (`Revenue=True`) sonuçlanma olasılığını** tahmin eden, eğitimden bağımsız çalıştırılabilen bir makine öğrenmesi projesidir. Python tabanlı komut satırı uygulaması (CLI), kayıtlı bir PyTorch modelini ve eğitimde fit edilmiş preprocessing nesnesini kullanarak tek bir oturuma ait JSON girdisinden tahmin üretir.

**Teknolojiler:** `Python` · `PyTorch` · `scikit-learn` · `pandas` · `NumPy` · `joblib` · `pytest`

**Durum:** Eğitim, validation, bağımsız test değerlendirmesi, model paketleme, CLI ve temel otomatik testler tamamlandı. **Gerçek kampanyalarda kullanım için doğrulanmış bir sistem değildir.**

> [!IMPORTANT]
> Bu model **satın alma olasılığını** tahmin eder; bir müşteriye kampanya mesajı gönderilmesinin **ek satış yaratacağını kanıtlamaz**. Kampanyanın nedensel etkisi için ayrıca kontrollü deney (ör. A/B testi) gerekir.

---

## 1. Problem ve yaklaşım

Bir e-ticaret ekibi, hangi alışveriş oturumlarının satın almayla sonuçlanma olasılığının daha yüksek olduğunu anlamak istiyor. Bu projede amaç, oturum özelliklerinden bir olasılık hesaplayıp bunu **ayrı bir karar politikası** ile kampanya adaylığına dönüştürmek.

İş senaryosunda mesaj başına **2 TL** ve validation grubunda **en fazla 100 kullanıcı** kapasitesi varsayıldı. İncelenen `0.50 / 0.60 / 0.70 / 0.80` eşikleri arasında **`0.60`**, kapasiteyi aşmadan en çok gerçek pozitif kullanıcıyı hedefleyen seçenekti. Bu eşik *validation* üzerinde seçildi; yeni bir grupta hedeflenen kişi sayısını 100 ile sınırlamayı garanti etmez.

```text
JSON oturum verisi (16 ham sütun)
              │
              ▼
      Girdi doğrulama
              │
              ▼
 Kayıtlı Pipeline / ColumnTransformer
  StandardScaler + OneHotEncoder
              │
              ▼
       73 sayısal özellik
              │
              ▼
   ShopperClassifier (PyTorch MLP)
        73 → 64 → 32 → 2
              │
              ▼
   Softmax → Revenue=True olasılığı
              │
              ▼
 Karar politikası (threshold=0.60)
              │
              ▼
     Tahmin sözlüğü / CLI çıktısı
```

## 2. Veri seti

**Kaynak:** [UCI — Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
**Kaynak atfı:** Sakar, C. & Kastro, Y. (2018), UCI Machine Learning Repository, [DOI: 10.24432/C5F88Q](https://doi.org/10.24432/C5F88Q).

| Özellik | Değer |
|---|---|
| Oturum sayısı | 12.330 |
| Hedef | `Revenue` (satın alma: `True` / `False`) |
| Pozitif sınıf oranı | ≈ %15,5 |
| Modelde kullanılan ham özellik | 16 |
| Preprocessing sonrası özellik | 73 |
| Veri ayrımı | %70 train · %15 validation · %15 test |
| Bölme yöntemi | Stratified; `random_state=42` |

**Model girdileri:**

- Sayısal: `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay`.
- Kategorik / kodlanmış: `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

`Revenue` hedef sütunudur ve girdiye **dahil edilmez**. `PageValues`, olası veri sızıntısı riski nedeniyle bilinçli olarak çıkarılmıştır. Preprocessing yalnızca **training verisinde fit edilmiş**, validation/test/serving sırasında yalnızca `transform()` kullanılmıştır.

## 3. Modeller ve deneyler

- **Logistic Regression:** Tabular veri üzerinde referans baseline model.
- **PyTorch MLP:** `Linear(73,64) → ReLU → Linear(64,32) → ReLU → Linear(32,2)`.
- **Kontrollü deneyler:** ReLU/GELU aktivasyonları; Adam/SGD optimizer; gradient normu ve gradient clipping çalışmaları.
- **Paketlenen sürüm:** `ShopperClassifier`, **ReLU + Adam** ile eğitilen checkpoint; **model version `1.0.0`**.

`CrossEntropyLoss` eğitimde ham logits ile kullanılır. Tahmin sırasında `softmax(logits, dim=1)[:, 1]` pozitif sınıf olasılığını üretir. `0.60` threshold modelin ağırlıklarını değiştirmez; yalnızca kampanya adaylığı kararını değiştirir.

## 4. Model performansı

Kaydedilmiş model için validation'da seçilen **`threshold=0.60`**, bağımsız testte değiştirilmeden kullanıldı.

| Metrik | Validation | Bağımsız test |
|:---|---:|---:|
| Accuracy | 0.8399 | **0.8459** |
| Precision | 0.4286 | **0.5072** |
| Recall | 0.1049 | **0.1224** |
| F1 | 0.1685 | **0.1972** |
| Average Precision (AP) | Final checkpoint için burada doğrulanmış değer sunulmuyor | Test AP değeri bu ekran görüntülerinde yer almıyor; `reports/final_test_results.csv` dosyasında hesaplanır |

![Validation ve test metriklerinin karşılaştırması](assets/validation_vs_test.png)

**Nasıl yorumluyorum?** Test Precision değeri ≈ %50,72; pozitif tahminlerin yaklaşık yarısı gerçekten satın almış. Ancak **Recall ≈ %12,24**: gerçek satın alıcıların yaklaşık **%87,76'sı kaçırılmış**. Bu nedenle modelin yalnızca ≈ %84,59 Accuracy sonucuna bakıp başarılı olduğunu söyleyemem. Zaten sınıf dengesizliği nedeniyle sürekli negatif tahmin veren bir yaklaşım yaklaşık %84,5 Accuracy elde edebilir.

Test metriklerinin validation'dan biraz daha iyi olması olumlu olsa da, **gerçek dağıtım ortamında aynı başarıyı garanti etmez**. Final test sonucuna bakarak threshold veya hiperparametreler yeniden seçilmemiştir.

Ayrıntılar: [`reports/week_02_final_report.md`](reports/week_02_final_report.md) · [`reports/threshold_experiments.csv`](reports/threshold_experiments.csv) · [`reports/final_test_results.csv`](reports/final_test_results.csv) (yerel değerlendirme notebook'u çalıştırıldığında üretilir).

## 5. Proje yapısı

```text
ShopperIntent/
├── data/
│   └── online_shoppers_intention.csv    # UCI'den indirilecek veri
├── models/
│   ├── shopper_mlp.pt                  # Eğitilmiş model ağırlıkları
│   ├── preprocessor.joblib             # Eğitimde fit edilmiş preprocessing
│   └── model_config.json               # Mimari, feature schema ve threshold
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_pytorch_experiments.ipynb
│   ├── 04_gradient_stability.ipynb
│   ├── 05_decision_thresholds.ipynb
│   └── 06_final_evaluation.ipynb
├── reports/
│   ├── threshold_experiments.csv
│   ├── final_test_results.csv
│   └── week_02_final_report.md
├── samples/
│   └── session_01.json
├── src/
│   ├── decision_policy.py
│   └── serving.py
├── tests/
│   └── test_prediction.py
├── assets/                          # README ekran görüntüleri
├── predict.py                       # Terminal uygulaması
├── model_card.md                    # Modelin kullanım sınırları
├── pytest.ini
├── requirements.txt
└── README.md
```

> Bazı çıktı dosyaları (özellikle `models/` altındaki ağırlıklar ve `reports/final_test_results.csv`) yerel eğitim/değerlendirme sürecinde oluşturulur. **Bu README paketi model ağırlıklarını veya veri setini içermez.** Dosyalar GitHub'a eklenmemişse aşağıdaki hazırlık adımlarını uygulayın.

## 6. Kurulum ve çalıştırma

**Gerekenler:** Python 3.11+ önerilir; proje geliştirilirken Python 3.14 ortamında CLI ve pytest çalıştırılmıştır. Uyumlu bir PyTorch kurulumu gerekir.

```bash
# 1) Depoyu klonlayın (URL'yi kendi deponuzla değiştirin)
git clone <REPOSITORY_URL>
cd ShopperIntent

# 2) Sanal ortam oluşturun (Linux / macOS)
python -m venv .venv
source .venv/bin/activate

# 3) Bağımlılıkları kurun
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows ortamı:** Etkinleştirme adımında `.venv\Scripts\activate` kullanabilirsiniz.

### 6.1. Veriyi hazırlama

UCI veri setini yukarıdaki bağlantıdan indirin ve şu konuma yerleştirin:

```text
data/online_shoppers_intention.csv
```

CSV'nin aynı içerik ve **aynı satır sırasını** koruması, daha önceki `random_state=42` veri ayrımının yeniden üretilebilmesi için önemlidir.

### 6.2. Model artefaktlarını edinme / üretme

Tahmin için şu **üç dosyanın** mevcut olması gerekir:

```text
models/shopper_mlp.pt
models/preprocessor.joblib
models/model_config.json
```

Eğer bunlar depoda paylaşılmamışsa, **`notebooks/05_decision_thresholds.ipynb` içindeki eğitim ve kaydetme hücrelerini**, veri kümesi ve paket sürümleri aynı tutularak çalıştırın. Fitted preprocessing ve seçilen checkpoint ile uyumlu JSON metadata'yı birlikte kaydedin. Özellikle:

```python
# Eğitim tamamlandıktan sonra, notebook içinde:
import torch
import joblib

torch.save(model.state_dict(), "../models/shopper_mlp.pt")
joblib.dump(pipline, "../models/preprocessor.joblib")
# model_config.json: input_dim, hidden_layers, activation,
# positive_class=1, decision_threshold=0.60 ve feature_columns içermelidir.
```

Bu kod, notebook'un `notebooks/` dizininden çalıştırıldığını ve eğitimde kullanılan pipeline değişkeninin `pipline` olduğunu varsayar. **Kaydettiğiniz checkpoint önceki sonuçları birebir üretmeyebilir**; eğitim rastgeleliğini ve veri sırasını denetleyin. Kaydedilmiş/pretrained model elinizdeyse **yeniden eğitim gerekmez**.

`joblib.load` yalnızca kaynağı güvenilir dosyalarda kullanılmalıdır.

## 7. CLI ile tahmin

Tek bir oturumu JSON dosyasından okuyarak tahmin oluşturun:

```bash
python predict.py --input-file samples/session_01.json
```

Örnek JSON (`samples/session_01.json`):

```json
{
  "Administrative": 2,
  "Administrative_Duration": 25.0,
  "Informational": 0,
  "Informational_Duration": 0.0,
  "ProductRelated": 15,
  "ProductRelated_Duration": 450.0,
  "BounceRates": 0.01,
  "ExitRates": 0.03,
  "SpecialDay": 0.0,
  "Month": "Nov",
  "OperatingSystems": 2,
  "Browser": 2,
  "Region": 1,
  "TrafficType": 2,
  "VisitorType": "Returning_Visitor",
  "Weekend": false
}
```

Projedeki **örnek oturumla alınmış gerçek terminal çıktısı**:

```text
================================
       SHOPPER INTENT AI
================================
Purchase Probability : 0.1372
Predicted Revenue    : False
Campaign Candidate   : No
Decision Threshold   : 0.60
Model Version        : 1.0.0
================================
```

![CLI ile tek oturum tahmini](assets/cli_prediction.png)

`Predicted Revenue`, standart `0.50` sınıflandırma kararını; `Campaign Candidate` ise metadata'daki **`0.60`** kampanya eşiğini kullanır. İki sonuç farklı olabilir. Örnek olasılık, örnek girdi ve bu checkpoint için geçerlidir.

## 8. Otomatik testler

```bash
# Altı testi çalıştır
python -m pytest -v

# Üç hata senaryosunu seçerek çalıştır
python -m pytest -v tests/test_prediction.py -k "missing_column or nan_inf or model_file_not_found"
```

Son yerel çıktıda **6/6 test başarıyla geçti**:

![Pytest: altı başarılı test](assets/pytest_6_passed.png)

Test edilen durumlar: geçerli oturum, eksik sütun, geçersiz sayısal veri, NaN/Inf, eksik model dosyası ve kaydet-yükle tutarlılığı.

Üç kontrollü hata senaryosu ayrıca çalıştırıldı (**3 passed**):

![Pytest: üç hata senaryosu](assets/error_tests_3_passed.png)

Bu testler temel giriş doğrulamasını ve serialization tutarlılığını kapsar; **üretim ortamı için tüm hata ve yük senaryolarının test edildiği anlamına gelmez**.

## 9. Bilinen sınırlamalar ve sorumlu kullanım

1. **Düşük Recall:** `threshold=0.60` için test Recall ≈ `0.1224`; birçok gerçek alıcı kaçırılıyor.
2. **Sınıf dengesizliği:** Oturumların yaklaşık `%84,5`'i satın almayla sonuçlanmıyor; Accuracy tek başına yanıltıcı.
3. **Veri sızıntısı riski:** `PageValues` dışarıda bırakıldı ancak diğer oturum özelliklerinin ne zaman hesaplandığı ayrıca doğrulanmalı.
4. **Tahmin zamanı:** Oturum sonunda oluşan ölçümler, oturum henüz sürerken tahmin vermek için erişilebilir olmayabilir. Üretimde kullanılan feature set'i tahmin anındaki erişilebilirlikle sınırlanmalı.
5. **Kampanya etkisi ölçülmedi:** Yüksek satın alma olasılığı, mesajın ilave satış yaratacağı anlamına gelmez; A/B testi veya uygun nedensel analiz gerekir.
6. **Eşik ve kapasite:** `0.60` validation üzerinde 70 kullanıcıyı hedefledi. Yeni veri akışında kesin 100 kişi sınırı sağlamaz.
7. **Dağılım değişimi:** Farklı mağazalar, sezonlar veya güncel kullanıcı davranışı model performansını değiştirebilir.

Daha fazla bilgi: **[Model Card](model_card.md)** · **[2. Hafta Final Teknik Raporu](reports/week_02_final_report.md)**.

## 10. Çalışmanın kapsamı

Bu repo, bir sınıflandırma modelini yalnızca notebook içinde eğitmekten çıkarıp **kaydedilebilir, yeniden yüklenebilir, CLI üzerinden çağrılabilir ve otomatik test edilebilir** bir Python bileşenine dönüştürmek için geliştirildi. Sonraki aşamada gerçek bir dağıtım kararı verilmeden önce verinin tahmin zamanındaki erişilebilirliği, drift izleme, kalibrasyon ve kampanyanın nedensel etkisi incelenmelidir.

---

**Veri atfı:** Sakar & Kastro (2018), *Online Shoppers Purchasing Intention Dataset*, UCI Machine Learning Repository; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

<details>
<summary>Ek ekran görüntüsü — kısa metrik çıktısı</summary>

![Kısa validation ve test metrik çıktısı](assets/validation_vs_test_compact.png)

</details>
