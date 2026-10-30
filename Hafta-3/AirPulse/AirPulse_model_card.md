# AirPulse — Model Card

**Model Name:** AirPulse  
**Problem Type:** Time Series Regression  
**Status:** Eğitim/araştırma prototipi — **gerçek kullanım (production) için doğrulanmadı**  
**Değerlendirme durumu:** Kronolojik validation tamamlandı; **bağımsız test değerlendirmesi henüz yapılmadı**.  
**Models:** Naive Forecast, Vanilla RNN, LSTM.  
**Missing Values:** Girdi veya hedefte eksik ölçüm olan pencereler çıkarıldı.  
**Data Leakage:** Gelecek ölçümler girdiye dahil edilmedi; scaler yalnızca training verisine fit edildi.

## 1. Model özeti

AirPulse, bir istasyondaki geçmiş **24 ardışık saatlik PM2.5** ölçümlerini kullanarak **gelecek saatin PM2.5 değerini** tahmin etmeye yönelik bir zaman serisi çalışmasıdır. Üç yaklaşım değerlendirilmiştir:

- **Naive Forecast:** Bir sonraki saat için son gözlemlenen PM2.5 değerini kullanır: `ŷ(t+1) = PM2.5(t)`. Öğrenilebilir parametresi yoktur.
- **Vanilla RNN (`AirPulseRNN`):** 24 saatlik diziyi Hidden State üzerinden işler; son gizli temsilden tek sayısal tahmin üretir.
- **LSTM (`AirPulseLSTM`):** Hidden State ve Cell State ile Forget/Input/Output Gate mekanizmalarını kullanır; son Hidden State'ten tek sayısal tahmin üretir.

**Hedef (`Target`):** `PM2.5(t+1)` — gelecek saat, **µg/m³**.  
**Girdi (`Input`):** Geçmiş 24 saatin PM2.5 ölçümleri; RNN/LSTM için `(batch, 24, 1)`.  
**Özellikler:** Yalnızca PM2.5 geçmişi; meteorolojik veya diğer istasyon verileri **girdi değildir**.

## 2. Veri seti ve kapsam

| Alan | Bilgi |
|---|---|
| Dataset | UCI Beijing Multi-Site Air Quality |
| Station | **Aotizhongxin** |
| Kaynak | https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data |
| İstasyon CSV'si | `PRSA_Data_Aotizhongxin_20130301-20170228.csv` |
| Frekans | Saatlik |
| Ham kayıt sayısı | **35.064** |
| Eksik PM2.5 kaydı | **925** |
| Geçerli 24-saat giriş + sonraki-saat hedef penceresi | **30.062** |
| Training sequence sayısı | **21.177** |
| Validation sequence sayısı | **4.266** |

Veri tek bir istasyonu temsil eder. Başka istasyon, şehir veya farklı veri dönemlerinde benzer sonuçlar **garanti değildir**.

## 3. Veri hazırlama, eksik ölçümler ve sızıntı önlemleri

- Kayıtlar `timestamp` sırasına konuldu. Girdi **24 ardışık saat** ve hedef **hemen sonraki saat** olacak şekilde pencereler oluşturuldu.
- 24 giriş saatinden herhangi birinde veya hedefte **eksik PM2.5** varsa ilgili pencere **çıkarıldı**. Eksik değerler doldurulmadı.
- Giriş ile hedef arasındaki 24 zaman farkının her birinin **tam 1 saat** olması kontrol edildi.
- Örnekler **hedef zamanına göre kronolojik** ayrıldı. Validation sınırı **2015-12-18 16:00**, test sınırı **2016-07-24 20:00**. Eksik pencereler sebebiyle ilk geçerli validation hedefi **2015-12-18 20:00** oldu.
- `input_scaler` ve `target_scaler` (`StandardScaler`) **yalnızca training verisi üzerinde fit edildi**; validation verisinde aynı scaler'lara yalnızca `transform()` uygulandı. Sonuç metrikleri için tahminlere `target_scaler.inverse_transform()` uygulandı.
- Gelecekteki PM2.5 ölçümleri giriş pencerelerine dahil edilmedi. Training hedefleri validation hedeflerine taşmadı; ortak hedef timestamp sayısı **0**.
- İlk validation penceresindeki geçmiş ölçümlerin bir bölümü kronolojik training dönemindendi. Bu ölçümler hedef saatinden önce biliniyor olduğundan **tek başına veri sızıntısı değildir**.
- Veri kalitesi denetiminde **0 tekrar timestamp**, **0 tamamen yinelenen satır**, **0 çelişkili timestamp** bulundu.

Bu kontroller, çalışma kapsamındaki kronolojik sızıntı risklerini ele alır; her tür olası sızıntının bulunmadığını garanti etmez.

## 4. Model mimarileri ve eğitim

| Özellik | Naive Forecast | RNN | LSTM |
|---|---|---|---|
| Giriş | Son gözlem | `(batch,24,1)` | `(batch,24,1)` |
| `input_size` | — | 1 | 1 |
| `hidden_size` | — | 32 | 32 |
| `num_layers` | — | 1 | 1 |
| `batch_first` | — | `True` | `True` |
| Son temsil | — | `output[:, -1, :]` | `hidden[-1]` |
| Çıkış | Son gözlem | `Linear(32,1)` | `Linear(32,1)` |
| Eğitim | Yok | Adam, LR `0.001`, 10 epoch | Adam, LR `0.001`, 10 epoch |

RNN/LSTM eğitimi: batch size **32**, seed **42**, `MSELoss` (normalize edilmiş hedefte). Bu model card'da ek özellik, yeni mimari denemesi veya hiperparametre ayarı yapılmamıştır.

## 5. Değerlendirme yöntemi

**Evaluation:** Kronolojik **validation**. Üç yaklaşım aynı **4.266 hedef saati**, aynı sıra ve aynı gerçek PM2.5 değerleri üzerinde karşılaştırıldı. Ayrı kaydedilmiş gerçek RNN ve LSTM tahminlerinin farklı olduğu kontrol edildi; önceki yanlış RNN tahmin dışa aktarımı düzeltildi.

**Metrics:**

- **MAE (Mean Absolute Error):** Ortalama mutlak hata; **µg/m³** cinsinden.
- **RMSE (Root Mean Squared Error):** Büyük hatalara daha duyarlı hata ölçüsü; **µg/m³** cinsinden.

| Model | Validation örneği | MAE ↓ (µg/m³) | RMSE ↓ (µg/m³) |
|---|---:|---:|---:|
| **Naive Forecast** | 4.266 | **10.37** | 21.59 |
| **Vanilla RNN** | 4.266 | 10.66 | 21.14 |
| **LSTM** | 4.266 | 10.4699 | **20.7776** |

**Yorum:** Naive Forecast en düşük **MAE**, LSTM en düşük **RMSE** değerine sahiptir. LSTM'nin RNN'e göre iyileşmesi küçük kalmıştır; ek karmaşıklık her ölçütte üstünlük sağlamamıştır. RNN/LSTM eğitim kayıpları normalize edilmiş **MSE** olup tablodaki orijinal birim MAE/RMSE ile doğrudan aynı ölçek değildir.

**Not:** Salı günkü farklı **4.789 örnekli** Naive Forecast sonucu bu tabloda kullanılmadı; karşılaştırma yalnızca eşleşen 4.266 örnek üzerindedir.

## 6. Hata analizi: ani PM2.5 değişimleri

LSTM'nin validation dönemindeki en büyük beş **mutlak** hatası:

| Hedef zamanı | Gerçek PM2.5 | LSTM tahmini | Mutlak hata |
|---|---:|---:|---:|
| 2016-02-08 02:00 | 898.00 | 408.13 | **489.87** |
| 2016-02-08 01:00 | 494.00 | 171.49 | **322.51** |
| 2015-12-24 22:00 | 312.00 | 47.91 | **264.09** |
| 2016-03-09 14:00 | 270.00 | 15.79 | **254.21** |
| 2016-02-08 09:00 | 37.00 | 275.39 | **238.39** |

Bu **beş örneğin dördü ani yükseliş**, biri **ani düşüş** sırasında oluştu. Örneğin 2016-02-08 02:00 hedefinde son ölçüm **494**, gerçek sonraki saat **898** iken LSTM **408.13** tahmin etti. 2016-02-08 09:00'da son ölçüm **374**, gerçek değer **37** iken tahmin **275.39** kaldı. Bu gözlemler, sert değişimlerde tahminlerin gecikebileceğini düşündürür; tek başına bütün zamanlar için genel başarı/başarısızlık hükmü değildir.

## 7. Intended Use — amaçlanan kullanım

- **Eğitim, öğrenme, yöntem karşılaştırması ve araştırma prototipi**.
- Saatlik PM2.5 zaman serisinde basit baseline ile tekrarlayan sinir ağlarını karşılaştırma ve hata örneklerini inceleme.
- Yalnızca doğrulama çalışmasındaki veri kapsamı içinde **deneysel tahmin analizi**.

**Amaçlanmayan kullanım:** Canlı hava kalitesi uyarısı, sağlık veya kamu güvenliği kararı, yasal/operasyonel eşik yönetimi ya da doğrulanmamış istasyonlara hazır çözüm olarak aktarım. Bu amaçlarla kullanılabilecek doğruluk, sağlamlık ve güvenilirlik **kanıtlanmadı**.

## 8. Limitations — sınırlamalar ve güvenilirlik riskleri

1. **Tek istasyon:** Yalnızca Aotizhongxin verileri kullanıldı. Diğer istasyon veya coğrafyalara genellenebilirlik değerlendirilmedi.
2. **Eksik ölçümler:** Girişteki geçmiş **24 saatin herhangi biri** eksikse mevcut yöntem tahmin için geçerli pencere oluşturmuyor. Eksik verili durumlar için alternatif çıkarım yöntemi geliştirilmedi.
3. **Ani yükseliş/düşüşler:** Hata analizinde ani PM2.5 değişimlerinde **çok büyük hatalar** görüldü. Özellikle tehlikeli yükselişleri olduğundan düşük tahmin etme riski önemlidir.
4. **Sınırlı özellikler:** Modeller yalnızca PM2.5 geçmişine bakar; **sıcaklık, rüzgâr ve diğer meteorolojik değişkenler kullanılmadı**. Bunlar bu görev kapsamında eklenmedi.
5. **Sabit 24 saatlik geçmiş:** Daha uzun zaman ölçeklerindeki etkiler gözden kaçabilir. 24 saatin en uygun pencere olduğu ayrıca test edilmedi.
6. **Bağımsız test eksikliği:** **Test dönemi henüz değerlendirilmedi**. Validation metrikleri, gelecekteki veya gerçek ortam performansı olarak sunulmamalıdır.
7. **Üretim hazırlığı:** Gerçek zamanlı veri akışı, eksik veri toleransı, güvenilirlik/izleme, model güncellemeleri ve uygulama koşulları için üretim doğrulaması yapılmadı.

## 9. Production Status ve değerlendirme kararı

**Production Status: NOT VALIDATED FOR PRODUCTION.**

AirPulse mevcut hâliyle **eğitim ve araştırma amaçlıdır**. Bağımsız test tamamlanmadan ve kullanım koşulları ayrıca doğrulanmadan gerçek dünyadaki hava kalitesi veya sağlıkla ilişkili kararları yönlendirmek için kullanılmamalıdır.

---

**İlgili çalışma dosyaları:** `notebooks/04_rnn_baseline.ipynb`, `notebooks/05_lstm_comparison.ipynb`, `notebooks/06_model_comparison_audit.ipynb`, `reports/day_03_rnn_baseline.md`, `reports/day_04_lstm_comparison.md`.  
**Model Card sürümü:** 1.0 — mevcut validation bulgularına dayalı.
