# AirPulse — Hafta 3 Kavram Tekrarı

**Proje:** AirPulse  
**Veri seti:** UCI Beijing Multi-Site Air Quality — Aotizhongxin  
**Problem:** Geçmiş 24 saatteki PM2.5 ölçümlerinden sonraki saatin PM2.5 değerini tahmin etme  
**Not:** Bu rapor, sekiz soruya verdiğim cevapların değerlendirme sonrasında teknik olarak düzeltilmiş ve tamamlanmış hâlidir. Yeni bir modelleme deneyi veya test değerlendirmesi yapılmamıştır.

## 1. Time Series

Zaman serilerinde geçmişteki verileri kullanarak geleceği tahmin etmeye çalıştığımız için zaman sırası önemlidir. Rastgele train/test ayrımı yaparsak eğitim kümesine daha ileri tarihlerden örnekler girerken değerlendirme kümesinde daha eski örnekler kalabilir. Bu, gerçek hayatta henüz bilinmeyen bir geleceği tahmin etme senaryosunu doğru temsil etmez ve sonuçları olduğundan iyi gösterebilir. Bu nedenle AirPulse'da örnekleri hedef zamanlarına göre kronolojik olarak ayırdık.

## 2. Data Leakage

Saat 15.00'in PM2.5 değerini tahmin ederken 14.00 ve önceki saatlerin ölçümlerini kullanabiliriz, çünkü bu bilgiler tahmin anında zaten mevcuttur. Saat 16.00'nın ölçümünü kullanmak yanlıştır; 15.00 için tahmin yaparken bu gelecekteki bilgiye erişemeyiz. Ayrıca 15.00'in gerçek PM2.5 değeri eğitim sırasında hedef (`y`) olabilir ama aynı saati tahmin etmek için giriş (`X`) olarak verilirse **target leakage** oluşur. Data leakage ile overfitting aynı kavram değildir.

## 3. Sequence Window

`(32, 24, 1)` tensöründeki **32**, modelin bir batch içinde işlediği örnek sayısıdır. **24**, her örnekte kullanılan geçmiş 24 saatlik zaman adımlarını gösterir. **1**, her saat için tek giriş özelliği kullandığımızı, yani yalnızca PM2.5 ölçümü olduğunu ifade eder; tahmin edilen çıktıyı belirtmez. Geçmiş 24 saati kullanmamızın amacı modelin zamansal yükseliş, düşüş ve diğer örüntüleri öğrenmesine imkân tanımaktır.

## 4. RNN

RNN'deki **Hidden State**, geçmiş zaman adımlarından elde edilen bilgilerin sayısal bir özetidir ve kısa süreli hafıza gibi düşünülebilir. RNN her yeni saatte o saatin PM2.5 ölçümünü ve önceki Hidden State'i birlikte işleyerek yeni Hidden State'i oluşturur. Böylece önceki saatlerden gelen bilgiler sonraki adımlara aktarılır. AirPulse'da 24. saatten sonraki son Hidden State'i `Linear` katmanına vererek 25. saatin PM2.5 tahminini ürettik; son Hidden State doğrudan tahminin kendisi değildir.

## 5. LSTM

LSTM'deki **Hidden State** mevcut zaman adımının öğrenilmiş temsilini, **Cell State** ise daha uzun süre boyunca aktarılabilen bellek bilgisini taşır; ikisi de sayısal vektörlerdir. **Forget Gate**, Cell State'teki bilgilerin ne kadarının korunacağına veya unutulacağına karar verir. Burada amaç yalnızca eski bilgileri silmek değildir; eski bir ölçüm hâlâ yararlıysa bilgi korunabilir. Bu kapı mekanizmaları, LSTM'nin standart RNN'e kıyasla uzun süreli bağımlılıkları daha kolay öğrenmesini sağlar.

## 6. Normalization

`StandardScaler`, ölçümleri eğitim verisinden hesaplanan ortalama ve standart sapmayla standartlaştırır: `(x - ortalama) / standart_sapma`. `fit()` işlemini yalnızca training verisinde yapmalıyız, çünkü validation ve test dönemlerinin istatistiklerini önceden öğrenmek istemeyiz. Validation ve test için training'den öğrenilmiş aynı ölçekle yalnızca `transform()` uygulanır. Bu dönemlerde yeniden `fit()` yapmak veri sızıntısı riskine ve tutarsız ölçeklendirmeye yol açabilir.

## 7. MAE ve RMSE

**MAE**, mutlak tahmin hatalarının ortalamasını ölçerken **RMSE**, hataların karelerini kullandığı için büyük sapmalara daha duyarlıdır. Bu yüzden Naive Forecast, ortalama mutlak hata bakımından daha iyi olsa da LSTM'nin büyük hatalarının toplam etkisi daha düşük olabilir ve RMSE'de öne geçebilir. AirPulse validation sonuçlarında Naive Forecast en düşük **MAE (10.37)**, LSTM ise en düşük **RMSE (20.78)** değerini elde etti. Hava kirliliğindeki büyük hataları önemserken RMSE'yi izlemek yararlıdır; özellikle tehlikeli yükselişlerin olduğundan düşük tahmin edilmesini de ayrıca değerlendirmek gerekir.

## 8. Duplicate ve Near-Duplicate

**Duplicate**, aynı kaydın tekrar bulunması; **Near-Duplicate** ise birebir aynı olmayan fakat birbirine çok benzeyen örneklerdir ve bunun için zorunlu bir %95 eşiği yoktur. Sliding Window yönteminde 24 saatlik pencereyi bir saat ilerlettiğimiz için iki ardışık pencerenin 23 ölçümü paylaşması normaldir. Training ve validation pencerelerinin ortak geçmiş ölçümler içermesi tek başına veri sızıntısı değildir; önemli olan validation hedefini tahmin ederken kullanılan bütün girişlerin o anda biliniyor olmasıdır. Ancak gelecekteki bir ölçüm girişe eklenirse veya zaman sırasını dikkate almayan bölme işlemi gelecekteki bilgilerin eğitime taşınmasına neden olursa veri sızıntısı ya da yanıltıcı değerlendirme riski doğabilir.

---

## Hafta 3 — Doğrulanmış validation sonuçları

| Model | Validation örnek sayısı | MAE | RMSE |
|---|---:|---:|---:|
| Naive Forecast | 4.266 | **10.37** | 21.59 |
| RNN | 4.266 | 10.66 | 21.14 |
| LSTM | 4.266 | 10.47 | **20.78** |

**Değerlendirme durumu:** Bu sonuçlar kronolojik validation dönemine aittir; bağımsız test dönemi için nihai değerlendirme henüz yapılmamıştır.

## Öğrendiğim kritik ayrımlar

- **Data leakage ≠ Overfitting:** Tahmin anında bilinmeyen bilgilerin kullanılması ile eğitim verisine aşırı uyum farklı problemlerdir.
- **Input size ≠ Output size:** `(32, 24, 1)` içindeki son `1`, saat başına giriş özelliği sayısıdır.
- **Sliding Window ≠ Otomatik güvenli ayrım:** Pencereler örtüşebilir; sızıntı olup olmadığı girişlerin tahmin anında erişilebilirliğine ve değerlendirme tasarımına bağlıdır.
- **Validation ≠ Nihai test:** Validation performansı üretim ortamı performansı olarak sunulmamalıdır.
