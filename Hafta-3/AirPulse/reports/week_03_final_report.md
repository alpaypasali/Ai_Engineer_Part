# AirPulse — 3. Hafta Final Teknik Raporu

**Proje:** AirPulse  
**İstasyon:** Aotizhongxin  
**Görev:** Geçmiş 24 saatlik PM2.5 ölçümünü kullanarak bir sonraki saatin PM2.5 değerini tahmin etmek  
**Kapsam:** Eğitim ve araştırma prototipi; gerçek kullanım için doğrulanmamıştır.

## 1. Problem Definition

AirPulse, **t** saatine kadar mevcut olan geçmiş **24 saatlik PM2.5** ölçümlerinden **t+1** saatindeki PM2.5 yoğunluğunu tahmin eder. Problem, tek değişkenli (univariate) **zaman serisi regresyonudur**. Çıktı, **µg/m³** cinsinden tek bir sayısal tahmindir; doğrudan bir hava kalitesi uyarısı değildir.

## 2. Dataset

- **Kaynak:** UCI Beijing Multi-Site Air Quality.
- **İstasyon:** Yalnızca Aotizhongxin.
- **Veri dosyası:** `data/PRSA_Data_Aotizhongxin_20130301-20170228.csv`.
- **Tarih aralığı:** 2013-03-01 00:00 – 2017-02-28 23:00.
- **Ham veri:** 35.064 saatlik kayıt; PM2.5 sütununda **925 eksik ölçüm**.
- **Modele verilen özellik:** Yalnızca geçmiş PM2.5 değerleri. Meteorolojik değişkenler bu haftaki modellere eklenmedi.

## 3. Data Preparation

`year`, `month`, `day`, `hour` sütunlarından zaman damgası oluşturulup veriler kronolojik sıralandı. **Sliding window** yöntemiyle, eksiksiz ve birbirini izleyen **24 saat** bir giriş örneğini (`X`, `(24, 1)`), sonraki saatin PM2.5 ölçümü de hedefi (`y`) oluşturdu. Eksik PM2.5 veya saat boşluğu bulunan pencereler çıkarıldı. Böylece **30.062 geçerli pencere** elde edildi.

Giriş ve hedef için ayrı `StandardScaler` kullanıldı; her iki scaler yalnızca **training verisinde `fit()`** edildi. Validation ve test verileri bu istatistiklerle dönüştürüldü. Sinir ağlarının tahminleri MAE/RMSE hesabından önce tekrar orijinal **µg/m³** birimine çevrildi.

## 4. Temporal Split

Ayrım, **pencerenin hedef zaman damgasına göre** yapıldı:

| Bölüm | Hedef zamanı | Geçerli pencere |
|---|---|---:|
| Training | 2015-12-18 16:00 öncesi | 21.177 |
| Validation | 2015-12-18 16:00 – 2016-07-24 20:00 öncesi | 4.266 |
| Test | 2016-07-24 20:00 ve sonrası | 4.619 |

Test döneminin ilk pencereleri, önceki validation saatlerinden **geçmiş ölçümleri** içerebilir. Bu, ölçümler hedef saatinden önce mevcut olduğu sürece tek başına **data leakage** değildir. Bölümler arasında hedef zamanları çakışmıyor.

## 5. Naive Baseline

Naive Forecast, bir sonraki saatin değerini son gözlenen ölçüme eşit kabul eder: **ŷ(t+1) = y(t)**. Eğitilen parametresi yoktur. Çok basit ve düşük maliyetli olduğu için güçlü bir **referans noktasıdır**: RNN/LSTM'nin ek karmaşıklığının gerçekten yarar sağlayıp sağlamadığını anlamamıza yardım eder. Daha karmaşık modelin her metrikte Naive'i geçmesi garanti değildir.

## 6. RNN

- **Mimari:** `nn.RNN(input_size=1, hidden_size=32, num_layers=1, batch_first=True)` ve son zaman adımı çıktısına uygulanan `Linear(32, 1)`.
- **Öğrenilebilir parametre:** **1.153**.
- **Tekrar edilen sabit eğitim protokolü:** `torch.manual_seed(42)`, **10 epoch**, **batch size 32**, **Adam (lr=0.001)**, **MSELoss**.
- **Kaydedilmiş ağırlıklar:** `models/rnn_state_dict.pth`.
- **Yeniden eğitilmiş sürüm validation:** MAE **10.5165**, RMSE **21.1358**.
- **Bağımsız test:** MAE **11.2278**, RMSE **20.7872**.

RNN, her saatten sonraki saate güncellenen bir **hidden state** taşır. Uzun süreli bağımlılıkları öğrenirken zorlanabilir.

## 7. LSTM

- **Mimari:** `nn.LSTM(input_size=1, hidden_size=32, num_layers=1, batch_first=True)` ve son hidden state'e uygulanan `Linear(32, 1)`.
- **Öğrenilebilir parametre:** **4.513**.
- **Eğitim:** RNN ile aynı sabit protokol; test sonuçlarına göre tuning veya yeniden eğitim yapılmadı.
- **Kaydedilmiş ağırlıklar:** `models/lstm_state_dict.pth`.
- **Yeniden eğitilmiş sürüm validation:** MAE **10.3180**, RMSE **20.5135**.
- **Bağımsız test:** MAE **11.0416**, RMSE **20.1592**.

Standart RNN'den farklı olarak LSTM hem **hidden state** hem **cell state** taşır. Kapı mekanizmaları, hangi bilginin korunacağını ve güncelleneceğini düzenleyerek uzun süreli ilişkileri öğrenmeyi kolaylaştırır. Bunun bedeli daha fazla parametre ve hesaplamadır.

## 8. Validation Results

**Cuma günündeki ilk karşılaştırma — orijinal model sürümleri** (o gün kaydedilen yuvarlanmış değerler):

| Model | Validation MAE | Validation RMSE |
|---|---:|---:|
| Naive | **10.37** | 21.59 |
| İlk RNN | 10.66 | 21.14 |
| İlk LSTM | 10.47 | **20.78** |

**Tahmin kaynağı hatası ve düzeltmesi:** Başlangıçta `reports/rnn_predictions.npz` dosyasına yanlışlıkla **LSTM tahminleri** yazılmıştı. Bu yüzden RNN/LSTM dizileri birebir eşitti. Orijinal RNN notebook'unda RNN modeliyle tahminler yeniden oluşturulup doğru dosyaya kaydedildi; hedef zamanları ve gerçek değerlerin eşleştiği, RNN ile LSTM tahminlerinin artık **farklı olduğu** doğrulandı. `.npz` dosyaları model ağırlıklarını değil, **validation tahminlerini** saklar.

Orijinal modellerin ağırlıkları `state_dict` olarak kayıtlı değildi. Final test için aynı mimari, seed ve sabit eğitim ayarlarıyla RNN/LSTM **yeniden eğitildi** ve ağırlıkları kaydedildi. Yeni sürümlerin validation metrikleri tekrar hesaplandı. Bu nedenle **ilk sürümün validation sonuçları, sonradan yeniden eğitilen modelin sonuçlarıymış gibi kullanılmadı**.

## 9. Final Test Results

**Final testte kullanılan model sürümlerinin kendi validation ve test sonuçları:**

| Model | Validation n | Test n | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| Naive | 4.266 | 4.619 | 10.3748 | 21.5890 | 11.1121 | 21.2815 |
| RNN (yeniden eğitilen) | 4.266 | 4.619 | 10.5165 | 21.1358 | 11.2278 | 20.7872 |
| **LSTM (yeniden eğitilen)** | 4.266 | 4.619 | **10.3180** | **20.5135** | **11.0416** | **20.1592** |

**Sonuçlar:** Bağımsız testte **en düşük MAE ve RMSE LSTM'de**. RNN, Naive'i **RMSE'de** geçiyor ancak **MAE'de** geçemiyor. Validation'dan teste geçildiğinde üç modelin de **MAE değerleri biraz artarken RMSE değerleri biraz düşüyor**; dolayısıyla her iki metrikte birden iyileşme olduğunu söyleyemeyiz. Önceden belirlenen **RMSE önceliği** test sonuçları görüldükten sonra değiştirilmedi. Özellikle LSTM ve Naive arasındaki test MAE farkı küçüktür.

Metrikler Python ile hesaplanarak `reports/final_test_results.csv` dosyasına kaydedildi. **Test sonuçlarına bakarak yeniden eğitim, hiperparametre değişikliği veya model seçimi yapılmadı.**

## 10. Failure Analysis

Önceki **validation** LSTM hata analizinde büyük hataların özellikle PM2.5'in ani yükseldiği veya düştüğü anlarda görülebildiği belirlendi. Örneğin **2016-02-08 02:00** için gerçek değer **898 µg/m³**, eski LSTM tahmini yaklaşık **408.13 µg/m³** ve mutlak hata yaklaşık **489.87 µg/m³** idi. Yüksek kirlilik seviyelerinin ciddi biçimde düşük tahmin edilmesi gerçek uyarı uygulamalarında önemli bir risk oluşturur.

Bu örnek **önceki model sürümünün validation analizine** aittir; **yeniden eğitilmiş LSTM'nin test hatası olarak sunulmamıştır**. Genel MAE/RMSE'nin makul olması uç olaylarda güvenilirliği kanıtlamaz.

## 11. Input Validation

`src/validation.py` içindeki `validate_pm25_window(timestamps, values, sequence_length=24)` fonksiyonu tahminden önce şu sorunları reddeder:

1. **Uzunluk hatası:** 24 yerine 23 ölçüm veya zaman damgası gibi yanlış sayılar.
2. **Sayısal hata:** PM2.5 değerlerinde `NaN`, `+Inf` veya `-Inf`.
3. **Zaman hatası:** Eksik, tekrarlı, sırasız veya ardışık olmayan saatler.

`tests/test_inference.py` içinde kısa pencere, NaN/Inf ve zaman boşluğu testleri çalıştırıldı: **`3 passed`**. Geçerli ve ardışık 24 saatlik veri kabul ediliyor. Bağımsız demo, bu fonksiyonu **modelden önce** çalıştırıyor.

## 12. Measurable Improvement

**Ölçülebilir güvenilirlik iyileştirmesi: 0/3 → 3/3 geçersiz girdi doğru reddediliyor.** Önceki tahmin akışı 23 ölçümü kabul ediyor, `NaN` girişinden `nan` tahmini üretiyor ve zaman boşluğunu fark etmiyordu. Doğrulama fonksiyonu eklendikten sonra üç senaryonun tamamı tahmin çalıştırılmadan reddedildi. Bu bir **input validation / güvenilirlik** iyileştirmesidir; **MAE veya RMSE iyileştirmesi değildir**.

`demo_predict.py`, eğitim notebook'u çalıştırılmadan kaydedilmiş **LSTM ağırlıklarını**, `input_scaler.joblib` ve `target_scaler.joblib` dosyalarını yükleyerek çalıştı. Gerçek veri setindeki son geçerli 24 saatlik pencerede **giriş bitişi 2017-02-28 23:00**, **tahmin saati 2017-03-01 00:00** ve **tahmin 22.26 µg/m³** görüldü. Bu çıktı yalnızca **tarihsel eğitim demosudur**, canlı uyarı sistemi değildir.

## 13. Limitations

- **Tek istasyon:** Başka istasyonlarda aynı performans garanti edilemez.
- **Sınırlı özellikler:** Yalnızca PM2.5 geçmişi kullanıldı; hava durumu ve diğer değişkenler eklenmedi.
- **Eksik veri:** Kesintisiz ve sonlu 24 ölçüm gerekli; eksik girişlerde mevcut demo tahmin yapmıyor.
- **Ani değişimler:** Büyük PM2.5 yükselişlerinde düşük tahmin riski var; özellikle tehlikeli eşikler için ayrı hata analizi gerekli.
- **Genelleme:** Tek kronolojik bağımsız test dönemi, gelecekteki farklı sezonları, sensör sorunlarını ve dağılım kaymasını temsil etmeyebilir.
- **Operasyonel gereksinimler:** Belirsizlik, veri kalitesi izleme, drift takibi, hata yönetimi ve gerçek dünyada güvenilirlik henüz yeterince değerlendirilmedi.
- **Production Status:** **Gerçek kullanım için doğrulanmadı.** Doğrudan halka yönelik hava kalitesi uyarıları veya sağlık kararları için kullanılmamalıdır.

## 14. What I Learned

Bu hafta zaman serilerinde **kronolojik ayırmanın**, 24 saatlik pencerelerde **eksik ölçüm/ardışıklık kontrolünün** ve scaler'ın yalnızca training verisine `fit()` edilmesinin önemini uyguladım. Naive Forecast'i referans alarak RNN ve LSTM'yi hem **MAE** hem **RMSE** ile değerlendirdim; hidden state, cell state ve LSTM kapılarının farkını öğrendim. RNN/LSTM tahmin dosyalarının kaynağındaki hatayı tespit edip düzelttim; farklı model sürümlerinin sonuçlarını karıştırmadan ağırlıkları ve scaler'ları kaydederek bağımsız test yaptım. Son olarak üç hata senaryosuyla girdi doğrulaması yazdım, **0/3 → 3/3** ölçülebilir iyileştirme elde ettim ve notebook dışında çalışan bir LSTM tahmin demosu oluşturdum.

**Genel sonuç:** Yeniden eğitilen LSTM, tanımladığımız bağımsız testte en düşük MAE ve RMSE'yi sağladı. Ancak Naive'e göre avantajı sınırlı ve sistem **production-ready değil**. Bu haftanın önemli mühendislik kazanımı, geçersiz girdilerin modelden önce denetlenmesi ve **3/3 otomatik test** ile doğrulanmasıdır.
