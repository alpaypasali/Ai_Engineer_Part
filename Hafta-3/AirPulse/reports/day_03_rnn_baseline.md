# AirPulse — Gün 3: RNN Baseline Teknik Raporu

**Proje:** AirPulse · **Notebook:** `notebooks/04_rnn_baseline.ipynb`  
**Veri:** UCI Beijing Multi-Site Air Quality, Aotizhongxin istasyonu  
**Değerlendirme:** Kronolojik validation dönemi; test kümesi kullanılmadı.

## 1. Problem

Amaç, Aotizhongxin istasyonunda geçmiş **24 saatlik PM2.5** ölçümlerinden **bir sonraki saatin PM2.5 değerini** tahmin etmektir. Bu bir **tek adımlı zaman serisi regresyonu** problemidir. Çıktı birimi **µg/m³**'tür.

## 2. Sequence Length — Neden 24 saat?

Giriş olarak son 24 ardışık saat kullanıldı; her pencerenin hedefi, bu 24 saatin hemen ardından gelen saatin PM2.5 değeridir. 24 saat seçimi, bir tam günlük değişim örüntüsünü modele sunmak için basit ve yorumlanabilir bir başlangıçtır; en iyi pencere uzunluğu olduğu ayrıca test edilmedi.

- Girdi: `(batch, 24, 1)`; saat başına tek özellik `PM2.5`.
- Hedef: `PM2.5(t+1)`; örnek başına bir sürekli değer.
- Oluşturulan geçerli pencereler: **30.062**.
- RNN train pencereleri: **21.177**; validation pencereleri: **4.266**.

## 3. Missing Values — Geçersiz pencereler

Ham veri zaman sırasına göre düzenlendi. Her 24 saatlik giriş penceresi ve onu izleyen hedef saati için:

1. Girişte veya hedefte `NaN` PM2.5 varsa örnek **atlandı**.
2. 24 giriş saati ve hedef saati arasındaki **24 zaman farkının tamamının 1 saat** olması şartı arandı.
3. Eksik değerler doldurulmadı; eksik satırlar silinerek saatlerin sanki bitişikmiş gibi birleştirilmesine izin verilmedi.

Bu seçim, daha az örnek kullanılmasına yol açsa da her girdinin gerçekten ardışık 24 saati temsil etmesini sağlar.

## 4. Normalizasyon ve kronolojik bölme

Örnekler hedef zaman damgasına göre kronolojik olarak bölündü:

- **Validation başlangıcı:** `2015-12-18 16:00`
- **Test başlangıcı:** `2016-07-24 20:00`
- **Train:** hedef zamanı validation başlangıcından önce olan örnekler.
- **Validation:** hedef zamanı validation başlangıcından test başlangıcına kadar olan örnekler.

`input_scaler = StandardScaler()` yalnızca **training giriş verisine** `fit` edildi. `target_scaler = StandardScaler()` yalnızca **training hedef verisine** `fit` edildi. Validation verilerine aynı scaler'lar ile sadece `transform` uygulandı. Değerlendirme tahminlerine `target_scaler.inverse_transform()` uygulandı ve metrikler **orijinal PM2.5 biriminde** hesaplandı.

## 5. RNN mimarisi

| Bileşen | Ayar |
|---|---|
| Model | `AirPulseRNN`, `nn.RNN` |
| `input_size` | `1` |
| `hidden_size` | `32` |
| `num_layers` | `1` |
| `batch_first` | `True` |
| Son zaman adımı | `output[:, -1, :]` |
| Çıkış katmanı | `nn.Linear(32, 1)` |
| Problem | Regresyon |

Model 24 saatin dizisini RNN'e verir, son zaman adımının 32 boyutlu temsilini alır ve Linear katmanıyla tek bir sonraki-saat tahmini üretir.

## 6. Hidden State — Geçmiş bilgi nasıl taşınıyor?

RNN her zaman adımında **mevcut PM2.5 girdisiyle bir önceki hidden state'i** birleştirerek yeni bir hidden state hesaplar. Böylece saatler ilerledikçe önceki saatlerden gelen bilgi duruma taşınır. Bu, 24 saatin tümünü kusursuz hatırladığı anlamına gelmez; özellikle uzun bağımlılıkları öğrenmek zor olabilir.

`output`, bu tek katmanlı tek yönlü RNN'de her saat için hidden state'leri içerir: `(batch, 24, 32)`. `hidden`, son gizli durumları katman bazında döndürür: `(1, batch, 32)`. Mevcut mimaride `output[:, -1, :]` ve `hidden[-1]` aynı son durum temsilini verir.

## 7. Training — 10 epoch sonuçları

Eğitim ayarları: **Adam**, learning rate **0.001**, **MSELoss**, batch size **32**, **10 epoch**, random seed **42**. Validation sırasında `model.eval()` ve `torch.no_grad()` kullanıldı; validation aşamasında ağırlık güncellenmedi.

| Epoch | Train MSE Loss | Validation MSE Loss |
|---:|---:|---:|
| 1 | 0.2154 | 0.1558 |
| 2 | 0.0796 | 0.1032 |
| 3 | 0.0639 | 0.0846 |
| 4 | 0.0613 | 0.0805 |
| 5 | 0.0614 | 0.0793 |
| 6 | 0.0614 | 0.0785 |
| 7 | 0.0614 | 0.0779 |
| 8 | 0.0614 | 0.0774 |
| 9 | 0.0613 | 0.0770 |
| 10 | **0.0612** | **0.0765** |

**Yorum:** Training Loss başlangıçta belirgin biçimde azalıp yaklaşık 0.061 düzeyinde yataylaştı. Validation Loss da 0.1558'den 0.0765'e düştü; bu 10 epoch'ta belirgin bir overfitting işareti gözlenmedi. Bu loss değerleri normalize edilmiş hedeflerin MSE değerleridir, doğrudan µg/m³ cinsinden değildir.

## 8. Validation — MAE ve RMSE

Eğitilmiş RNN, **4.266 validation örneği** üzerinde değerlendirildi. Tahminler `target_scaler.inverse_transform()` ile orijinal birime getirildi.

- **RNN MAE:** **10.66 µg/m³**
- **RNN RMSE:** **21.14 µg/m³**

## 9. Naive Forecast ile adil karşılaştırma

Naive tahmin, her örneğin **son gözlemlenen PM2.5** değerini bir sonraki saat için kullanır: `X_val_seq[:, -1, 0]`. RNN ile aynı hedef saatlerini, aynı `y_val_seq` gerçek değerlerini ve aynı ölçüm birimini kullanmıştır.

| Model | Validation örneği | MAE (µg/m³) | RMSE (µg/m³) |
|---|---:|---:|---:|
| Naive — Salı referansı | 4.789 | 10.19 | 20.82 |
| **Naive — RNN ile aynı 24 saatlik pencereler** | **4.266** | **10.37** | 21.59 |
| **RNN (24 saat)** | **4.266** | 10.66 | **21.14** |

**Sonuç:** Adil eşleştirilmiş karşılaştırmada **Naive Forecast MAE'de 0.29 µg/m³ daha iyi**, **RNN ise RMSE'de 0.45 µg/m³ daha iyi**. Tek bir metrikte mutlak üstünlük yoktur. Salı günkü 4.789 örnekli Naive sonucu RNN ile doğrudan karşılaştırılmaz.

## 10. Hata analizi — En yüksek beş mutlak hata

**Tamamlanacak:** Bu beş satırın gerçek değerleri mevcut sohbette paylaşılmadı. Tahmini veya uydurma sonuç yazılmadı. `04_rnn_baseline.ipynb` dosyasındaki `top5_errors` tablosunun çıktısı buraya aktarılmalıdır.

| Hedef zamanı | Gerçek PM2.5 (µg/m³) | RNN tahmini (µg/m³) | Mutlak hata (µg/m³) |
|---|---:|---:|---:|
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |
| — | — | — | — |

Hata sıralama yöntemi: `abs(y_val_seq - predictions_original.ravel())` hesaplanır, hedef zamanlarıyla eşleştirilir ve en büyük beş mutlak hata seçilir. Bu bölüm doldurulduktan sonra hataların ani yükseliş/düşüşlerle ilişkisi incelenebilir.

## 11. Sınırlamalar

- Model **yalnızca PM2.5 geçmişini** kullanıyor; meteorolojik değişkenler veya diğer istasyonlar girdi değil.
- 24 saatlik pencere, daha uzun süreli etkileri her zaman yakalayamayabilir.
- Eksik ölçüm içeren pencereler çıkarıldığı için veri kapsamı daralır; eksikliklerin rastgele olmaması sonuçları etkileyebilir.
- Tek katmanlı basit RNN, uzun süreli bağımlılıkları öğrenmekte zorlanabilir.
- Ani kirlenme sıçramalarında veya sert düşüşlerde gecikmeli tahminler üretebilir.
- Bu sonuçlar **validation dönemine** aittir; bağımsız test dönemi performansı bu raporda değerlendirilmedi.

## 12. Teknik kontroller

- Giriş boyutu kontrolü: `X_train_tensor.ndim == 3` ve sequence uzunluğu `24`.
- Saat bütünlüğü kontrolü: her geçerli pencerenin 24 giriş saati ve bir hedef saati arasında 1'er saat bulunması.
- Kronolojik sızıntı kontrolü: en son train hedefi ilk validation hedefinden önce; her validation girdisinin son saati kendi hedef saatinden önce.

Bu kontroller notebook'ta başarılı olarak bildirildi.

---

**CSV özeti:** `reports/rnn_baseline.csv` 10. epoch'un Training/Validation MSE Loss değerlerini ve validation dönemine ait MAE/RMSE sonuçlarını içerir. Ekrana yazdırılmış yuvarlanmış değerler kullanılmıştır.
