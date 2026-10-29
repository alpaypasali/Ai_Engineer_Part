# AirPulse — Gün 4: LSTM ve RNN karşılaştırması

**Notebook:** `notebooks/05_lstm_comparison.ipynb`  
**Veri:** UCI Beijing Multi-Site Air Quality — Aotizhongxin istasyonu  
**Problem:** Geçmiş 24 saatlik PM2.5 değerleriyle bir sonraki saati tahmin eden tek adımlı zaman serisi regresyonu.  
**Kapsam:** Train ve validation; **test kümesi kullanılmadı**.

## 1. LSTM ile RNN arasındaki temel fark

Vanilla RNN, mevcut girdi ve önceki **Hidden State** üzerinden yeni bir Hidden State oluşturur. LSTM ise Hidden State (`h_t`) yanında bir **Cell State** (`C_t`) taşır. Forget, Input ve Output Gate mekanizmalarıyla geçmiş bilginin korunmasını, yeni bilginin eklenmesini ve gizli duruma yansıtılmasını düzenler. Bu yapı uzun süreli bağımlılıkları öğrenmeyi kolaylaştırabilir, ancak her zaman daha iyi performans garantisi vermez.

## 2. Forget Gate ne işe yarar?

Önceki Cell State'te hangi bilginin ne ölçüde korunacağını belirler. Sigmoid çıktısı `0`'a yakınsa bilgi büyük ölçüde unutulur, `1`'e yakınsa korunur. `f_t ⊙ C_(t−1)` işlemiyle eski bilginin korunacak kısmı belirlenir.

## 3. Input Gate ne işe yarar?

Yeni bilgilerin Cell State'e eklenme oranını kontrol eder. Sigmoid ile hesaplanan `i_t` (0–1), tanh ile oluşturulan aday bellek `C̃_t` (−1–1) ile eleman bazında çarpılır. Cell State güncellemesi: `C_t = f_t ⊙ C_(t−1) + i_t ⊙ C̃_t`.

## 4. Output Gate ne işe yarar?

Güncellenmiş Cell State'ten Hidden State'e hangi bilginin aktarılacağını kontrol eder: `h_t = o_t ⊙ tanh(C_t)`. Burada tanh **aday belleğe değil güncellenmiş Cell State'e** uygulanır.

## 5. LSTM mimarisi ve veri hazırlama

| Parametre | Değer |
|---|---|
| Model | `AirPulseLSTM`: `nn.LSTM` + `nn.Linear` |
| `input_size` | 1 — PM2.5 |
| `hidden_size` | 32 |
| `num_layers` | 1 |
| `batch_first` | `True` |
| Sequence Length | 24 saat |
| Son Hidden State | `hidden[-1]` |
| Çıkış | `Linear(32, 1)` — 1 sonraki-saat tahmini |
| Giriş / çıkış Tensor'ları | `(batch, 24, 1)` / `(batch, 1)` |
| Training / validation örnekleri | 21.177 / 4.266 |
| Batch size | 32 |

**Veri tutarlılığı:** Çarşamba günkü RNN çalışmasının 24 saatlik pencere kuralları kullanıldı. Giriş veya hedefte eksik PM2.5 bulunan ya da saatleri kesintisiz olmayan pencereler çıkarıldı. Kronolojik ayırmada hedef zamanı kullanıldı: validation başlangıcı `2015-12-18 16:00`, test başlangıcı `2016-07-24 20:00`. Input ve Target `StandardScaler` yalnızca **training** üzerinde `fit` edildi, validation'da sadece `transform` kullanıldı. Tensor'lar `float32`.

Tek katmanlı, tek yönlü mimarimizde `output[:, -1, :]` ve `hidden[-1]` son zaman adımı için aynı Hidden State'i verir; `cell` ise farklı bir bellek durumudur.

## 6. Training Loss nasıl değişti? — 10 epoch

**Eğitim ayarları:** Adam, learning rate `0.001`, `MSELoss`, seed `42`, batch `32`, `10` epoch.

| Epoch | Training MSE Loss | Validation MSE Loss |
|---:|---:|---:|
| 1 | 0.2456 | 0.1635 |
| 2 | 0.0759 | 0.1006 |
| 3 | 0.0621 | 0.0855 |
| 4 | 0.0603 | 0.0819 |
| 5 | 0.0600 | 0.0798 |
| 6 | 0.0597 | 0.0781 |
| 7 | 0.0594 | 0.0767 |
| 8 | 0.0592 | 0.0755 |
| 9 | 0.0590 | 0.0746 |
| 10 | **0.0589** | **0.0739** |

Training Loss **0.2456 → 0.0589** seviyesine geriledi. En büyük iyileşme ilk epoch'larda; sonrasında düşüş devam etti ama yavaşladı.

## 7. Validation Loss nasıl değişti?

Validation Loss **0.1635 → 0.0739** düzeyine indi. Eğitim ve validation loss birlikte azaldı; bu 10 epoch'luk süreçte belirgin overfitting işareti gözlenmedi. Bu sonuç bağımsız test performansını garanti etmez.

**Ölçüm ayrımı:** Training/Validation MSE Loss değerleri **normalize edilmiş hedef** üzerinden, MAE/RMSE ise `target_scaler.inverse_transform()` sonrası **µg/m³** cinsinden hesaplanmıştır.

## 8. LSTM Validation MAE ve RMSE

- **Validation MAE:** **10.4699 µg/m³**
- **Validation RMSE:** **20.7776 µg/m³**
- **Validation örnek sayısı:** **4.266**

Yüksek RMSE, bazı örneklerde büyük tahmin hatalarının bulunduğunu gösterir.

## 9. RNN ile LSTM arasındaki fark — Naive karşılaştırması

| Model | Validation MAE (µg/m³) ↓ | Validation RMSE (µg/m³) ↓ |
|---|---:|---:|
| Naive Forecast — 24 saatlik sequence örnekleri | **10.37** | 21.59 |
| RNN | 10.66 | 21.14 |
| LSTM | 10.4699 | **20.7776** |

RNN ile karşılaştırıldığında LSTM, MAE'yi yaklaşık **0.19**, RMSE'yi yaklaşık **0.36 µg/m³** düşürdü. **MAE'de Naive Forecast, RMSE'de LSTM** en iyi. LSTM'nin ek karmaşıklığının getirdiği fayda **küçük**, hesaplama maliyeti ayrıca ölçülmedi. Salı günkü **4.789 örnekli** Naive (MAE 10.19, RMSE 20.82) sonucu adil karşılaştırma tablosuna dahil edilmedi.

**Karşılaştırmanın doğrulama durumu:** Bu metrikler aynı hazırlama kurallarına ve 4.266 validation örneğine göre raporlandı; ancak ayrı RNN/LSTM notebook'larındaki **hedef zamanlarının satır satır aynı olduğunu gösteren test çıktısı henüz paylaşılmadı**. Sonucun kesin adil karşılaştırma sayılması için hedef zamanları **aynı sırayla**, gerçek hedefleri de aynı olacak şekilde `np.array_equal` ve `np.allclose` ile doğrulanmalıdır. CSV'nin yenilenmesi için verilen kod bu doğrulamayı yapar.

## 10. En büyük beş hata (gerçek failure samples)

| Target Timestamp | Gerçek PM2.5 | LSTM Tahmini | Mutlak Hata |
|---|---:|---:|---:|
| 2016-02-08 02:00 | 898.00 | 408.13 | **489.87** |
| 2016-02-08 01:00 | 494.00 | 171.49 | **322.51** |
| 2015-12-24 22:00 | 312.00 | 47.91 | **264.09** |
| 2016-03-09 14:00 | 270.00 | 15.79 | **254.21** |
| 2016-02-08 09:00 | 37.00 | 275.39 | **238.39** |

**24 saatlik geçmişe bakıldığında:**

- **8 Şubat 02:00:** Son ölçüm `494`, gerçek hedef `898` (**+404**); penceredeki son üç değer `61 → 152 → 494`. Model yükselişi eksik tahmin etti.
- **8 Şubat 01:00:** Son ölçüm `152`, gerçek `494` (**+342**); model `171.49` tahmin etti.
- **24 Aralık 22:00:** Son ölçüm `44`, gerçek `312` (**+268**); model `47.91` ile sıçramayı yakalayamadı.
- **9 Mart 14:00:** Son ölçüm `23`, gerçek `270` (**+247**); model `15.79` tahmin etti.
- **8 Şubat 09:00:** Son ölçüm `374`, gerçek `37` (**−337**); geçmişte `898`, `713`, `615` gibi yüksek değerler vardı. Model `275.39` ile düşüşü yeterince izleyemedi.

**Gözlem:** Beş büyük hatanın **dördü ani yükselişte, biri ani düşüşte** gerçekleşti. Bu örnekler LSTM'nin keskin değişimlere gecikmeli tepki verebileceğini düşündürür; tüm validation dönemi için genel hüküm değildir.

## 11. Bugün ne öğrendim? — Birinci şahıs taslak

> LSTM'de Hidden State'e ek olarak Cell State bulunduğunu, Forget Gate'in hangi eski bilgilerin korunacağını, Input Gate'in hangi yeni bilgilerin ekleneceğini, Output Gate'in ise Hidden State'e ne aktarılacağını kontrol ettiğini öğrendim. LSTM'nin Vanishing Gradient riskini azaltabildiğini ancak problemi tamamen ortadan kaldırmadığını anladım. Aynı validation saatleri ve sadece training üzerinde fit edilen scaler kullanılmadan model karşılaştırmasının yanıltıcı olacağını gördüm. Hata analizinde en büyük hataların ani PM2.5 artış ve düşüşlerinde olduğunu gözlemledim.

*Bu birinci şahıs paragrafı sohbet içindeki öğrenim cevapların doğrultusunda düzenlenmiş bir taslaktır; kendi ifadenle son halini verebilirsin.*

## 12. Sınırlamalar

Yalnızca PM2.5 geçmişi kullanıldı; rüzgâr ve meteorolojik değişkenler yok. 24 saatlik pencere tüm etkileri temsil etmeyebilir. Eksik ölçüm pencerelerini çıkarma veri kapsamını azaltır. Ani değişimlerde hatalar yüksektir. LSTM'nin eğitim/hız maliyeti ölçülmedi. **Test kümesi değerlendirilmedi.** Teknik testler ve aynı hedef zamanlarının eşleşmesi notebook'ta çalıştırılarak teyit edilmelidir.

**Çıktı:** `reports/lstm_comparison.csv` dosyası Naive, RNN ve LSTM için validation metriklerini içerir. Eğitimi olmayan Naive modelinin `hidden_size`, `epochs`, `learning_rate` ve loss sütunları boş bırakılır.
