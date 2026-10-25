# ShopperIntent — Hafta 2 Final Teknik Raporu

**Proje:** ShopperIntent | **Veri kaynağı:** UCI Online Shoppers Purchasing Intention  
**Model sürümü:** 1.0.0 | **Seçilen karar eşiği:** 0.60  
**Değerlendirme yöntemi:** %70 train / %15 validation / %15 bağımsız test (stratified, `random_state=42`)  
**Rapor durumu:** Paylaşılan validation ve test sonuçlarına dayanır. Nihai test **Average Precision** değeri bu konuşmada paylaşılmadığı için tamamlanması gereken alan olarak işaretlenmiştir.

## 1. Problem Tanımı

Bu projede, bir e-ticaret sitesindeki ziyaret oturumunun satın almayla (`Revenue=True`) sonuçlanma olasılığını tahmin etmeye çalıştım. Şirketin pazarlama ekibi, sınırlı kampanya kapasitesini hangi oturumlara ayıracağı konusunda bir karar desteğine ihtiyaç duyuyor. Model, oturum özelliklerinden **satın alma olasılığı** üretir; ayrı bir karar politikası bu olasılığı önceden seçilmiş bir eşikle karşılaştırıp kampanya adaylığına dönüştürür.

İş senaryosunda mesaj başına **2 TL** varsayımsal maliyet ve en fazla **100 kişilik** kampanya kapasitesi kullandım. Bu değerler gerçek şirket maliyetleri değildir. Amacım yalnızca Accuracy yükseltmek değil, pozitif sınıfı ne kadar iyi yakaladığımı ve kararın operasyonel sonuçlarını birlikte değerlendirmekti.

**Kritik ayrım:** `Revenue=True` tahmini, kampanya gönderilmesinin **ek satış yaratacağını** göstermez. Model satın alma eğilimini tahmin eder; kampanyanın nedensel etkisini ölçmez.

## 2. Veri Seti

UCI Online Shoppers Purchasing Intention veri setinde **12.330 alışveriş oturumu** bulunur. Hedef sütun `Revenue` ikili sınıftır: `True` satın almayla sonuçlanan oturum, `False` satın almayla sonuçlanmayan oturumdur. Pozitif sınıf oranı yaklaşık **%15,5** olduğu için veri seti dengesizdir.

Modelin ham girdisinde **16 özellik** kullandım; `Revenue` hedef sütununu ve olası veri sızıntısı riski taşıyan `PageValues` sütununu girdiden çıkardım.

- **Sayısal oturum ölçümleri (9):** `Administrative`, `Administrative_Duration`, `Informational`, `Informational_Duration`, `ProductRelated`, `ProductRelated_Duration`, `BounceRates`, `ExitRates`, `SpecialDay`.
- **Kategorik / kodlanmış özellikler (7):** `Month`, `OperatingSystems`, `Browser`, `Region`, `TrafficType`, `VisitorType`, `Weekend`.

Ön işleme sırasında sayısal sütunlara **StandardScaler**, uygun kategorik sütunlara **OneHotEncoder** uyguladım; ikili bazı sütunları doğrudan aktardım. Dönüşümler yalnızca **training** verisinde `fit()` edildi; validation ve testte sadece `transform()` kullanıldı. Böylece 16 ham sütundan, kaydedilen modelin beklediği **73 sayısal özellik** elde edildi.

Aynı satır sırasındaki veri, `random_state=42` ve stratification ile önce %70/%30, sonra kalan %30'un yarı yarıya ayrılmasıyla **8.631 train, 1.849 validation, 1.850 test** kayda bölündü.

## 3. Kullanılan Modeller

**Logistic Regression:** İlk karşılaştırma için daha basit ve yorumlanabilir bir baseline olarak kullandım. Tabular veri için güçlü bir referans olduğundan MLP'nin gerçekten ek katkı sağlayıp sağlamadığını anlamaya yardımcı oldu. Bu raporda Logistic Regression'a ait doğrulanmış sayısal sonuç paylaşılmadığından performans üstünlüğü iddiasında bulunmuyorum.

**PyTorch MLP — ShopperClassifier:** Kaydettiğim modelin mimarisi `73 → 64 → 32 → 2` şeklindedir. İki gizli katmanda **ReLU**, son katmanda iki sınıf için **logits** kullanılır. Eğitimde **CrossEntropyLoss**; olasılık hesaplamasında, yalnızca değerlendirme sırasında **Softmax** kullanıldı.

**Kontrollü deneyler:** ReLU/GELU aktivasyonları ile Adam/SGD optimizer seçeneklerini ayrı deneyler olarak inceledim. Ayrıca Adam ile gradient normu ve gradient clipping (`max_norm=1.0`) deneyleri yaptım. Nihai paketlenen ve karar eşiği analizi yapılan sürüm **ReLU + Adam** modelidir. Deneylerin tamamı için sayısal üstünlük sıralaması vermiyorum; raporun değerlendirme bölümü kaydedilmiş nihai modele aittir.

## 4. Model Değerlendirme

### 4.1 Validation ve bağımsız test karşılaştırması

Perşembe günü validation üzerinde seçilen **0.60 eşiği**, final testte **değiştirilmeden** kullanıldı. Model ve preprocessing dosyaları kaydedilmiş artefaktlardan yüklendi; test kümesinde yeniden eğitim veya `fit()` yapılmadı.

| Metrik | Validation | Bağımsız test |
|---|---:|---:|
| Accuracy | 0.8399 | **0.8459** |
| Precision | 0.4286 | **0.5072** |
| Recall | 0.1049 | **0.1224** |
| F1 | 0.1685 | **0.1972** |
| Average Precision (AP) | Önceki deneylerde ≈0.3361 bildirilmişti; nihai checkpoint için burada doğrulanmadı | **Değer paylaşılmadı — `final_test_results.csv` üzerinden tamamlanmalı** |

**Yorumum:** Testte gösterilen dört metrik validation'dan biraz daha iyi. Bu, modelin test kümesinde belirgin bir düşüş göstermediğine işaret ediyor; ancak küçük farklardan hareketle modelin gerçek hayatta başarılı olacağını söyleyemem. Özellikle testte **Recall = 0.1224**, yani gerçek satın alıcıların yaklaşık **%87,76'sı** pozitif olarak sınıflandırılamamış. Precision'ın yaklaşık %50,72 olması, pozitif tahmin edilen grubun yaklaşık yarısının gerçekten satın aldığını gösteriyor. Bu, yüksek hedefleme güvenilirliği ile geniş kapsam arasında halen ciddi bir ödünleşim olduğunu ortaya koyuyor.

Accuracy de tek başına yanıltıcı olabilir: Test setinde yaklaşık **%84,5** kullanıcı negatif sınıftadır; herkesi `Revenue=False` sayan basit yaklaşım da yaklaşık aynı Accuracy'yi sağlayabilir. Bu nedenle Recall, Precision, F1 ve **AP** değerlerini ayrıca raporlamayı önemsiyorum.

Validation üzerindeki **0.60 eşiği**, denenen dört eşik (`0.50`, `0.60`, `0.70`, `0.80`) arasında **100 kişilik kapasiteyi aşmadan en çok gerçek pozitif kullanıcıyı hedefleyen** seçenekti: **70 hedeflenen oturum, 30 TP, 40 FP, 140 TL varsayımsal mesaj maliyeti**. Bu eşik yeni bir veri grubunda en fazla 100 hedefleme garantisi vermez.

**Metodolojik karar:** Bağımsız test sonuçlarını gördükten sonra threshold'u, optimizer'ı, ağırlıkları veya hiperparametreleri değiştirmiyorum. Aksi hâlde test setini de model seçiminin bir parçasına dönüştürmüş olurum.

### 4.2 Confusion matrix ve AP notu

Ayrıntılı test confusion matrix ve AP değerinin gerçek kaynağı `reports/final_test_results.csv` olmalıdır. Yukarıdaki yuvarlatılmış Precision ve Recall ile 1.850 kayıtlık, stratified test ayrımı birbiriyle uyumludur; fakat bu raporda **doğrudan görülmeyen tam sayı ve AP değerlerini kesin ölçüm diye sunmuyorum**. CSV oluşturulduğunda bu bölüm oradan tamamlanmalıdır.

## 5. Model Packaging

```text
JSON Input (16 ham özellik)
           ↓
Input Validation (eksik/fazla alan, tip, NaN/Inf kontrolü)
           ↓
Preprocessing (kaydedilmiş fitted Pipeline; sadece transform)
           ↓
PyTorch Model (73 → 64 → 32 → 2, state_dict ağırlıkları)
           ↓
Softmax (Revenue=True sınıfının olasılığı)
           ↓
Decision Threshold (kampanya için 0.60)
           ↓
Prediction Result (olasılık, Revenue kararı, kampanya adayı, sürüm)
```

`models/shopper_mlp.pt` dosyası PyTorch ağırlıklarını, `models/preprocessor.joblib` eğitim verisinden öğrenilmiş dönüşümleri, `models/model_config.json` ise özellik sırasını, model giriş boyutunu, aktivasyonu, pozitif sınıfı, sürümü ve karar eşiğini saklar. `src/serving.py` içinde `load_model()` bu dosyaları yükler; `predict_purchase()` ham JSON girdisini doğrular ve olasılık üretir. `src/decision_policy.py` içinde `campaign_decision()` işletmenin karar kuralını ayrı bir fonksiyonda tutar.

`predict.py` dosyasında `argparse` ile çalışan bir CLI geliştirdim. Şirket çalışanı şu komutla örnek oturum için tahmin alabilir:

```bash
python predict.py --input-file samples/session_01.json
```

**Model paketlemenin temel kazanımı:** Tahmin için notebook'un eğitim hücrelerini çalıştırmak gerekmiyor. Tahmin sırasında yeniden `fit()` veya model eğitimi yapılmıyor. CLI'daki `predicted_revenue` alanı **0.50 standart sınıf eşiğini**, `target_campaign` alanı **0.60 iş eşiğini** kullanıyor; bunlar farklı kararlar olabilir.

## 6. Otomatik Testler

`tests/test_prediction.py` ile **6 pytest testi** yazdım ve terminalde **`6 passed`** çıktısını aldım:

| Test | Doğruladığı davranış | Sonuç |
|---|---|---|
| Geçerli oturum | Gerekli alanları taşıyan, 0–1 aralığında olasılık döndürülür | PASSED |
| Eksik `Month` | Eksik özellik anlamlı hatayla reddedilir | PASSED |
| Hatalı sayısal değer | `"hello"` gibi sayısal olmayan değer reddedilir | PASSED |
| `NaN` / `Inf` | Sonlu olmayan sayısal girdiler reddedilir | PASSED |
| Eksik model dosyası | Model yoksa anlamlı `FileNotFoundError` oluşur | PASSED |
| Serialization consistency | Ağırlıklar/preprocessing kaydedilip yüklenince aynı girdiye eşdeğer tahmin üretilir | PASSED |

Daha sonra eksik özellik, NaN/Inf ve eksik model dosyasına odaklanan **3 kontrollü hata testini** tekrar çalıştırdım; üçü de geçti. Gerçek model dosyasını silmeden `monkeypatch`, geçici serialization için `tmp_path` kullanmak güvenli test tasarımına yardımcı oldu.

## 7. Bilinen Sınırlamalar

**Düşük Recall.** Testte Recall ≈ **%12,24**. Model gerçek satın alıcıların çok büyük kısmını kaçırıyor. Bu hâliyle, mümkün olduğunca fazla alıcıya ulaşmanın önemli olduğu senaryolarda zayıf bir aday olabilir.

**Sınıf dengesizliği.** `Revenue=True` oranı yaklaşık **%15,5** olduğundan negatif sınıf baskın. Tek başına yüksek Accuracy, pozitif sınıfın iyi öğrenildiğini kanıtlamaz.

**Olası veri sızıntısı.** `PageValues` girişten çıkarıldı; yine de başka özellikler hedef olay gerçekleştikten sonra hesaplanıyor olabilir. Özellikle tahmin anında elde edilemeyen bilgilerle model eğitmek gerçek kullanım performansını olduğundan iyi gösterebilir.

**Tahmin zamanında özelliklerin erişilebilirliği.** Oturum sonu toplamları/süreleri ve `ExitRates` gibi alanlar, kampanya gönderilmek istenen **erken bir anda** henüz bilinmeyebilir. Modelin kullanılacağı zaman noktası netleştirilmeden gerçek zamanlı hedefleme iddiasında bulunamam.

**Genelleme ve veri dağılımı.** UCI oturumlarından elde edilen performans; başka site, sezon, trafik kaynağı veya kullanıcı kitlesinde aynı kalmayabilir. Zaman temelli veya haricî bir doğrulama henüz raporlanmadı.

**Kampanya etkisinin kanıtlanmamış olması.** Model, kullanıcının zaten satın alma ihtimalini tahmin eder. Mesajın yarattığı **ek satış** ya da **kampanya kârı** bilinmiyor. Gerçek etki için randomize A/B testi veya uygun nedensel etki değerlendirmesi, ayrıca satış marjı ve diğer maliyetler gerekir. Mesaj başına **2 TL** yalnızca öğretici bir varsayımdır.

**Karar kapasitesi.** Validation'da `0.60` ile 70 kişi hedeflenmesi, başka veri grubunda kesinlikle 100 kişinin altında kalınacağını garanti etmez. Sert kapasite kısıtları için ayrıca kampanya gönderim katmanında limit uygulanmalıdır.

## 8. Bu Hafta Ne Öğrendim?

1. **İkili sınıflandırmada tek bir Accuracy değerinin yeterli olmadığını öğrendim.** Dengesiz veri setinde herkese negatif diyen bir model bile yüksek Accuracy sağlayabildiği için Precision, Recall, F1 ve AP'yi ayrı yorumlamam gerektiğini gördüm.
2. **ReLU/GELU ile Adam/SGD deneylerinin aynı şeyi değiştirmediğini öğrendim.** Aktivasyon fonksiyonu ağın dönüşümünü, optimizer ise ağırlıkların gradyanlarla nasıl güncellendiğini etkiliyor. Bu nedenle deneylerde hangi bileşeni değiştirdiğimi açık tutmam gerekiyor.
3. **Gradient normlarını ve clipping'i eğitim kararlılığı için kullanmayı öğrendim.** `loss.backward()` sonrasında norm takibi yapmanın, NaN/Inf kontrolünün ve clipping'in nerede uygulanacağının neden önemli olduğunu uyguladım.
4. **Karar eşiği değiştiğinde modelin yeniden eğitilmediğini anladım.** Aynı olasılık skorlarıyla `0.50`, `0.60`, `0.70`, `0.80` için farklı Precision–Recall ve TP/FP sonuçları aldım; AP'nin ise sabit skorlar üzerinde değişmediğini öğrendim.
5. **İş kısıtını model metriğinden ayırmayı öğrendim.** 100 kişilik kapasite ve 2 TL mesaj maliyetini kullanarak seçtiğim karar kuralının hedeflenen kullanıcı sayısını ve masrafını hesapladım; yüksek Recall'un her zaman uygun iş kararı olmadığını gördüm.
6. **PyTorch modelini notebook dışında çalıştırmayı öğrendim.** `state_dict()` ile ağırlıkları, `joblib` ile fitted preprocessing'i ve JSON ile model metadata'sını saklayıp başka bir Python modülünde yeniden yükledim.
7. **Girdi doğrulamanın tahmin kalitesinin bir parçası olduğunu öğrendim.** Eksik sütun, yanlış sayısal tip, NaN/Inf ve bulunamayan dosya durumlarında yanlış tahmin üretmek yerine anlamlı hata vermeyi uyguladım.
8. **Bir ML modelini CLI ve otomatik testlerle paketlemeyi öğrendim.** `argparse` ile JSON alan bir terminal uygulaması yazdım; pytest ile altı senaryoyu doğruladım ve kaydetme/yükleme tutarlılığını test ettim.
9. **Bağımsız test kümesini geliştirme kararlarından ayrı tutmayı öğrendim.** Threshold'u validation'da seçip dondurdum; final test sonuçlarını yalnızca raporlama için kullandım. Testte daha iyi metrik görmenin modeli tekrar ayarlamak için gerekçe olmadığını anladım.
10. **Tahmin ile nedensel etki arasındaki farkı öğrendim.** Bir kullanıcının satın alma olasılığının yüksek olması, ona kampanya gönderince fazladan satın alma gerçekleşeceğinin kanıtı değildir.

---

### Sonuç

Bu hafta bir baseline ve PyTorch modeliyle başlayan çalışmayı **paketlenebilir, terminalden tahmin üreten, otomatik testlerle kontrol edilen** bir projeye dönüştürdüm. Buna rağmen modelin **Recall'u düşük**, bağımsız test Accuracy'si ise çoğunluk sınıfı referansına çok yakın. Bu nedenle projeyi gerçek kampanya ortamı için hazır olarak nitelendirmiyorum. Sonraki geliştirmeler için veri zamanı/leakage denetimi, pozitif sınıf performansı ve gerçek kampanya etkisinin kontrollü ölçümü kritik olacaktır.

**Tamamlanacak tek rapor alanı:** Bağımsız test **Average Precision** değeri `reports/final_test_results.csv` dosyasından bu rapora aktarılmalıdır. Bu eksik alan için sayı tahmin edilmemiştir.
