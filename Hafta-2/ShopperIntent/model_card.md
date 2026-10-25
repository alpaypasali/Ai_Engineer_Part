# ShopperIntent — Model Card

**Model sürümü:** 1.0.0  
**Model türü:** PyTorch MLP (ikili sınıflandırma)  
**Hedef:** `Revenue = True`  
**Seçilen kampanya karar eşiği:** `0.60` (validation üzerinden)

## 1. Modelin amacı ve kullanım alanı

ShopperIntent, bir e-ticaret oturumunun satın almayla sonuçlanma olasılığını tahmin eder. Çıktı, `Revenue=True` sınıfına ait tahmini olasılıktır. Ayrı karar politikası bu olasılığı `0.60` eşiğiyle karşılaştırıp kullanıcıyı kampanya adayı olarak işaretler. Modelin amacı **satın alma olasılığını** tahmin etmektir; bir kampanyanın satış üzerindeki **nedensel etkisini** tahmin etmek değildir.

## 2. Veri kaynağı ve kapsamı

- **Kaynak:** UCI Online Shoppers Purchasing Intention veri seti.
- **Veri boyutu:** 12.330 oturum.
- **Hedef sınıf:** `Revenue` (`True` pozitif sınıf, kodu `1`).
- **Sınıf dağılımı:** Pozitif sınıf yaklaşık %15,5; veri dengesizdir.
- **Değerlendirme yaklaşımı:** Eğitim/validation/test ayrımı yapılmıştır. Bu karttaki sayısal karar eşiği sonuçları **validation** verisine aittir; bağımsız test metrikleri burada raporlanmamıştır.

## 3. Kullanılan özellikler

Model **16 ham giriş sütunu** kullanır. Preprocessing sonrası giriş boyutu **73** özelliktir.

**Sayısal özellikler:** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay`.

**Kategorik veya kategorik kod olarak ele alınan özellikler:** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

**Girdiye dahil edilmeyen sütunlar:** `PageValues` (olası veri sızıntısı riski nedeniyle çıkarıldı) ve `Revenue` (tahmin edilen hedef değişken).

## 4. Preprocessing

Eğitim verisinde fit edilmiş `Pipeline`/`ColumnTransformer` kullanılır. Sayısal özelliklerde **StandardScaler**, kategorik özelliklerde **OneHotEncoder** uygulanır. Kaydedilmiş `models/preprocessor.joblib` nesnesi tahminde yalnızca `transform()` ile kullanılır; yeni veriye `fit()` veya `fit_transform()` uygulanmaz. Girdi sütunlarının isimleri ve sırası `models/model_config.json` içindeki `feature_columns` alanında saklanır.

## 5. Model mimarisi ve paketleme

- **Model sınıfı:** `ShopperClassifier`
- **Mimari:** `Linear(73, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 2)`
- **Eğitim optimizer'ı:** Adam
- **Çıktı:** İki logit; `softmax(logits, dim=1)[:, 1]` ile pozitif sınıf olasılığı
- **Model ağırlıkları:** `models/shopper_mlp.pt`
- **Preprocessing:** `models/preprocessor.joblib`
- **Metadata:** `models/model_config.json`
- **Tahmin modülü:** `src/serving.py`
- **Karar politikası:** `src/decision_policy.py`
- **Komut satırı kullanımı:** `python predict.py --input-file samples/session_01.json`

## 6. Doğrulama sonuçları

Seçilen **threshold = 0.60** için validation sonuçları:

| Metrik | Değer |
|---|---:|
| Accuracy | 0.8399 |
| Precision | 0.4286 |
| Recall | 0.1049 |
| F1 | 0.1685 |
| True Positive (TP) | 30 |
| False Positive (FP) | 40 |
| True Negative (TN) | 1523 |
| False Negative (FN) | 256 |
| Hedeflenen oturum sayısı (TP + FP) | 70 |

**Average Precision (AP):** Önceki model deneylerinde yaklaşık **0.3361** bildirildi. Bu değer, kaydedilmiş nihai model dosyası üzerinde burada yeniden doğrulanmadığı için **nihai test AP'si veya kesin checkpoint metriği olarak sunulmaz**. Pozitif sınıfın validation oranı `286 / 1849 ≈ 0.1547` idi.

**Yorum:** Recall yalnızca yaklaşık **%10,5**; gerçek pozitiflerin büyük çoğunluğu bu karar eşiğinde kaçırılmaktadır. Yüksek Accuracy, negatif sınıf çoğunlukta olduğu için tek başına yeterli başarı göstergesi değildir.

## 7. Karar eşiği ve iş kısıtı

Validation'da incelenen eşikler `0.50`, `0.60`, `0.70`, `0.80` idi. En fazla **100 kullanıcıya** kampanya gönderme varsayımı altında `0.50` **141 kişiyi** hedeflediği için sınıra uymaz. `0.60` **70 kişiyi** hedefler ve kapasiteye uyan bu dört eşik arasında en yüksek TP sayısına (**30**) sahiptir. Bu nedenle `0.60` seçilmiştir.

Mesaj başına **varsayımsal 2 TL** maliyetle `0.60` eşiğinde toplam iletişim maliyeti **140 TL**, yanlış pozitiflere gönderilen mesajların maliyeti **80 TL** olur. Bu maliyetler ders senaryosuna aittir. **0.60 eşiği yeni bir veri grubunda 100 kişilik üst sınırı garanti etmez.**

## 8. Bilinen sınırlamalar ve riskler

1. **Sınıf dengesizliği:** `Revenue=True` oranı yaklaşık %15,5'tir; Accuracy yanıltıcı olabilir.
2. **Düşük Recall:** `0.60` eşiğinde gerçek satın alıcıların yaklaşık %89,5'i pozitif tahmin olarak yakalanmamıştır.
3. **Olası veri sızıntısı ve tahmin zamanı:** `PageValues` dışlanmıştır; buna rağmen oturumun sonuna yakın ölçülebilen diğer özelliklerin (ör. süreler ve çıkış oranları) **tahmin anında gerçekten mevcut olması** ayrıca doğrulanmalıdır. Eğitimdeki bilginin gelecekteki tahmin anından gelmesi engellenmelidir.
4. **Dağılım kayması:** Kullanıcı davranışları, trafik kaynakları ve kategori dağılımları zamanla değişebilir. Validasyondaki performansın üretimde korunacağı garanti değildir.
5. **Karar eşiğinin bağlama bağımlılığı:** `0.60` yalnızca incelenen validation kohortu ve kapasite senaryosuna göre seçilmiştir.
6. **Bağımsız test sonucu eksik:** Bu kartta henüz final test kümesine ait AP, Precision, Recall ve F1 raporlanmamıştır.
7. **Olasılık kalibrasyonu:** Olasılıkların gerçek dünyadaki satın alma sıklıklarıyla ne ölçüde örtüştüğü ayrıca doğrulanmamıştır.

## 9. Kullanım sınırı: Kampanya etkisi kanıtlanmadı

**ShopperIntent modeli, kullanıcıya kampanya gösterilmesinin ek satış oluşturacağını kanıtlamaz.** `TP`, hedeflenen kullanıcının veri setinde satın alma gerçekleştirdiğini belirtir; satın almanın kampanya sayesinde gerçekleştiğini göstermez. Gerçek ek satış, kampanya kârlılığı ve nedensel etki için ayrıca **kontrollü A/B testi veya uygun bir nedensel etki değerlendirmesi** gerekir.

Bu nedenle model, gerçek pazarlama bütçesi üzerinde tek başına otomatik karar verici olarak kullanılmamalıdır; önce bağımsız test değerlendirmesi, tahmin zamanı kontrolleri ve kampanya etki analizi yapılmalıdır.

## 10. Mühendislik doğrulamaları

- `src/serving.py`, kaydedilmiş model ve preprocessing ile CPU üzerinde tahmin oluşturur.
- Orijinal ve yeniden yüklenen modelin aynı oturum için tahmini karşılaştırılmış, `np.isclose(..., atol=1e-6, rtol=0)` kontrolü **True** çıkmıştır.
- `tests/test_prediction.py` altında altı temel senaryo için pytest sonucu **6 passed** olarak doğrulanmıştır: geçerli oturum, eksik sütun, yanlış sayısal değer, NaN/Inf, eksik model dosyası, serialization consistency.

**Not:** Geçen otomatik testler, modelin genel performansını, olasılık kalibrasyonunu veya gerçek kampanya etkisini kanıtlamaz.
