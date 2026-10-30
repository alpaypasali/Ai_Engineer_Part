# AirPulse — Gün 5: Model Karşılaştırması ve Veri Kalitesi Denetimi

**Proje:** AirPulse  
**Veri:** UCI Beijing Multi-Site Air Quality — Aotizhongxin istasyonu  
**Görev:** RNN/LSTM tahmin kaynağı denetimi, Duplicate ve Near-Duplicate analizi, karşılaştırma  
**Değerlendirme:** Kronolojik validation, aynı **4.266 hedef zamanı**; bağımsız test dönemi değerlendirilmedi.

## 1. RNN ve LSTM sonuçlarında hangi tutarsızlık düzeltildi?

Önceki `rnn_predictions.npz` dosyasındaki tahminlerin **gerçek RNN tahminleri olmadığı** fark edildi: RNN ve LSTM dosyalarındaki tahmin dizileri birebir aynıydı (`np.allclose=True`, en büyük fark `0.0`). İki dosyadan da **MAE 10.4699 ve RMSE 20.7776** hesaplanması, dosyaya LSTM tahminlerinin yanlışlıkla RNN etiketiyle kaydedilmiş olduğunu düşündürdü.

Daha sonra `04_rnn_baseline.ipynb` içinde modelin sınıfının **`AirPulseRNN`** olduğu doğrulandı. Model **yeniden eğitilmeden**, eğitilmiş gerçek RNN ile validation tahminleri yeniden üretildi ve `rnn_predictions.npz` güncellendi. Yeni kayıtlarda RNN ile LSTM tahminleri artık farklıydı (`np.allclose=False`); en büyük mutlak tahmin farkı yaklaşık **75.00305 µg/m³** bulundu. Her iki dosyadaki **4.266 hedef zamanının aynı sırada** ve gerçek değerlerin aynı olduğu `assert` kontrolleriyle doğrulandı. Böylece önceki yanıltıcı, üst üste binen RNN/LSTM grafiği düzeltildi.

## 2. Üç modelin doğrulanmış MAE ve RMSE sonuçları

Aşağıdaki değerler aynı **4.266 validation örneğine** ait önceki çalışmaların ve model kaynağı düzeltmesinin sonuçlarıdır. Notebook'ta üretilen `model_comparison_verified.csv`, değerleri elle girmeden **kaydedilmiş gerçek model tahminlerinden** yeniden hesaplar.

| Model | Validation örneği | MAE (µg/m³) ↓ | RMSE (µg/m³) ↓ |
|---|---:|---:|---:|
| **Naive Forecast** | 4.266 | **10.37** | 21.59 |
| **RNN** | 4.266 | 10.66 | 21.14 |
| **LSTM** | 4.266 | 10.4699 | **20.7776** |

**Uyarı:** Salı günkü **4.789 örnekli** farklı Naive değerlendirmesi bu tabloya alınmadı. CSV oluşturma kodu, sayıları sabit kullanmaz; hedef saatlerinin ve gerçek değerlerin eşleşmesini tekrar kontrol eder.

## 3. Duplicate ile Near-Duplicate arasındaki fark nedir?

- **Duplicate:** Aynı zaman damgasının tekrar etmesi veya bir satırın bütün sütunlarıyla birebir aynı olmasıdır. Ayrıca aynı timestamp'e ait *farklı* PM2.5 değerleri varsa veri tutarsızlığı söz konusu olabilir.
- **Near-Duplicate:** Kayıtlar birebir aynı değilken büyük ölçüde örtüşür. Örneğin birbirini izleyen iki **24 saatlik pencere 23 zaman damgasını** paylaşır.

Veri kalitesi denetiminin sonuçları: **35.064 kayıt**, **0 tekrarlanan timestamp**, **0 tekrarlanan tam satır**, **0 çelişkili timestamp**. Dolayısıyla bu nedenle kayıt silinmedi. Sonuçlar `duplicate_audit.csv` dosyasında veri üzerinden tekrar hesaplanacaktır.

## 4. Ardışık 24 saatlik pencerelerin örtüşmesi normal mi?

**Evet.** Bir saat kaydırılan pencerelerden biri `2013-03-01 00:00 → 23:00`, diğeri `2013-03-01 01:00 → 2013-03-02 00:00` aralığındaydı. **23 ortak timestamp / 24 = %95,83 örtüşme** hesaplandı. Bu yapı kayan pencere (sliding window) yaklaşımının doğal sonucudur ve tek başına veri sızıntısı kanıtı değildir.

## 5. Train/validation ortak geçmişi neden her zaman leakage değildir?

Validation'ın **ilk geçerli** örneğinin giriş saatleri `2015-12-17 20:00 → 2015-12-18 19:00`, hedefi `2015-12-18 20:00` idi. Bu 24 giriş saatinin **20'si training takvim döneminden**, **4'ü validation döneminden** geliyordu. Ancak tamamı hedef saatinden önce ölçülmüştü; tahmin anında bilinebilen geçmiş verinin kullanımı tek başına sızıntı değildir.

Training ve validation ayrımı **hedef zamanına göre** yapıldı. Son geçerli training hedefi `2015-12-17 16:00`, ilk geçerli validation hedefi `2015-12-18 20:00` idi. Ortak hedef timestamp sayısı **0** bulundu. Normalizasyon scaler'ları yalnızca training verisine fit edildi. Bu bulgular, ele alınan kronolojik sızıntı risklerini kontrol eder; her tür sızıntının imkânsız olduğunu kanıtlamaz.

## 6. MAE'de hangi model daha iyi?

**Naive Forecast (10.37 µg/m³).** Genel mutlak tahmin hatasını ortalama olarak en aza indirmede en düşük validation MAE değerini verdi. RNN 10.66, LSTM 10.4699 elde etti.

## 7. RMSE'de hangi model daha iyi?

**LSTM (20.7776 µg/m³).** RNN'in RMSE'si 21.14, Naive Forecast'inki 21.59 oldu. RMSE büyük hataların etkisini daha fazla artırdığı için farklı bir modelin bu ölçütte öne çıkması mümkündür. LSTM'nin RNN'e göre kazanımı vardır ancak **sınırlıdır**; daha karmaşık olması her ölçütte üstünlük sağlamadı.

## 8. AirPulse hangi sınırlamalarla kullanılabilir?

1. **Tek istasyon:** Yalnızca Aotizhongxin verisinde geliştirildi. Başka istasyon veya dönemlerde aynı performans garanti edilmez.
2. **Eksik ölçümler:** Geçmiş 24 saat veya hedef eksikse ilgili pencere dışarıda bırakılıyor; eksik değerlerle canlı tahmin mekanizması yok.
3. **Ani değişimler:** 2016-02-08 gibi sert PM2.5 sıçramalarında veya düşüşlerde çok büyük hatalar görüldü; özellikle yüksek kirliliği olduğundan düşük tahmin etme riski var.
4. **Sınırlı özellikler:** Sadece geçmiş PM2.5 kullanılıyor; rüzgâr, sıcaklık ve diğer meteorolojik özellikler kullanılmıyor.
5. **Sabit 24 saat:** Daha uzun geçmiş ilişkilerini kaçırabilir; en uygun pencere uzunluğu ayrıca araştırılmadı.
6. **Bağımsız test yok:** Test dönemi üzerinde nihai performans ölçülmedi. Validation metrikleri gerçek dünya performansı gibi sunulmamalı.
7. **Üretim onayı yok:** Canlı hava kalitesi uyarısı, sağlık ya da kamu güvenliği kararı için doğrulanmış değil.

**Intended Use:** Eğitim, yöntem karşılaştırması ve araştırma prototipi.  
**Production Status:** **NOT VALIDATED FOR PRODUCTION**.

---

### Üretilen dosyalar ve tekrar üretim

- `reports/model_comparison_verified.csv` — **gerçek RNN/LSTM `.npz` tahminleri**, ham PM2.5 verisinden üretilen Naive Forecast ve aynı hedef zamanlarının doğrulanması ile oluşturulur.
- `reports/duplicate_audit.csv` — ham CSV'nin **bütün satırları** üzerinden duplicate ve çelişki sayıları hesaplanır.
- Kod: `reports/generate_day05_csv.py`. Kodun çalışması için proje içindeki `reports/rnn_predictions.npz`, `reports/lstm_predictions.npz` ve `data/PRSA_Data_Aotizhongxin_20130301-20170228.csv` gereklidir.

**Denetim ilkesi:** İki modelin hedef zamanları, gerçek değerleri veya tahmin kaynağı uyuşmuyorsa `model_comparison_verified.csv` oluşturulmamalıdır.
