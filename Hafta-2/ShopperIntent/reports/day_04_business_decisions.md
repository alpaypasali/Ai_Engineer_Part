# ShopperIntent — Gün 4: Threshold ve İş Kararları Teknik Raporu

**Proje:** Hafta 2 · ShopperIntent  
**Tarih:** 23 Ekim 2026  
**Deney dosyası:** `reports/threshold_experiments.csv`  
**Veri bölümü:** Validation; 1.849 oturum (286 `Revenue=True`, 1.563 `Revenue=False`)  
**Karar kuralı:** `Revenue=True` tahmini, `positive_probability >= threshold` olduğunda verilir.  
**İş varsayımları:** Her hedeflenen kullanıcıya bir mesaj; mesaj başına **2 TL**; kapasite **en fazla 100 kullanıcı**.

## Deney sonuçlarının özeti

Aşağıdaki metrikler, aynı eğitilmiş modelin pozitif sınıf olasılıkları üzerinde yalnızca karar eşiği değiştirilerek elde edildi. Modelin ağırlıkları bu deneyde değişmedi.

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN | Hedeflenen | Toplam maliyet (TL) | Yanlış alarm maliyeti (TL) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 0.828556 | 0.390071 | 0.192308 | 0.257611 | 55 | 86 | 1477 | 231 | 141 | 282 | 172 |
| 0.60 | 0.839913 | 0.428571 | 0.104895 | 0.168539 | 30 | 40 | 1523 | 256 | 70 | 140 | 80 |
| 0.70 | 0.849108 | 0.666667 | 0.048951 | 0.091205 | 14 | 7 | 1556 | 272 | 21 | 42 | 14 |
| 0.80 | 0.846944 | 1.000000 | 0.010490 | 0.020761 | 3 | 0 | 1563 | 283 | 3 | 6 | 0 |

**Hesaplama:** `targeted_users = TP + FP`; `contact_cost = (TP + FP) × 2 TL`; `false_positive_cost = FP × 2 TL`. Yanlış alarm maliyeti, toplam mesaj maliyetinin **bir parçasıdır**; iki maliyet toplanmaz.

## Teknik sorular

### 1. Adam optimizer'da birinci ve ikinci moment ne işe yarar?

Adam'da **birinci moment**, gradyanların üstel hareketli ortalamasıdır. Momentum gibi çalışarak güncelleme yönünü yumuşatır. **İkinci moment**, gradyan **karelerinin** üstel hareketli ortalamasıdır; her parametrede gradyanın büyüklüğüne göre güncellemeyi ölçeklemeyi sağlar. Adam bu iki bilgiyi birleştirerek parametreye özgü, uyarlanabilir güncellemeler yapar. Başlangıçtaki yanlılığı azaltmak için moment tahminlerine *bias correction* uygular. Bu deneyde Adam kullanılmış olması, tek başına belirli bir threshold'un başarısını açıklamaz: burada optimizer sabit, karar eşiği değişkendir.

### 2. Weight Initialization modelin öğrenmesini nasıl etkileyebilir?

Ağırlıklar başlangıçta çok büyükse aktivasyonlar ve gradyanlar büyüyerek eğitimi kararsızlaştırabilir; çok küçükse özellikle derin ağlarda sinyaller ve gradyanlar zayıflayabilir. Bir katmandaki bütün nöronları aynı ağırlıklarla başlatmak simetri sorunu yaratır: nöronlar aynı davranışı öğrenebilir. **ReLU** tabanlı katmanlarda **Kaiming (He) initialization**, sinyalin katmanlar boyunca uygun ölçekte kalmasına yardımcı olabilecek bir başlangıç yöntemidir. Kontrollü model karşılaştırmalarında başlangıç ağırlıklarının eşitlenmesi ve seed kullanılması, rastgelelikten kaynaklanan farkları azaltır. Bu günkü eşik deneyinde ağırlıkları yeniden başlatmadım.

### 3. Exploding Gradient problemini nasıl tespit edebilirsin?

Eğitim sırasında `loss.backward()` sonrasında parametre gradyanlarının **global L2 normunu** (ve gerekirse katman bazlı normlarını) batch/epoch boyunca izlerim. Normların sürekli ve olağan dışı büyümesi, loss'un sert dalgalanması veya büyümesi ve gradyan/loss değerlerinde **NaN/Inf** oluşması uyarı işaretleridir. `torch.isfinite()` ile sonlu değer kontrolü yapar; gradyan clipping uygulanıyorsa normu clipping **öncesinde** de kaydederim. Tek bir yüksek norm değeri tek başına exploding gradient kanıtı değildir. **Bu günün threshold tablosunda gradient normu bulunmadığı için modelimde exploding gradient yaşandığını iddia edemem.**

### 4. Threshold artırıldığında Precision ve Recall nasıl değişti?

Threshold **0.50'den 0.80'e** çıkarıldığında **Precision 0.3901'den 1.0000'e yükseldi**, fakat **Recall 0.1923'ten 0.0105'e düştü**. Pozitif tahminler daha seçici hâle geldi: **TP 55'ten 3'e**, **FP 86'dan 0'a** geriledi; satın alan fakat hedeflenmeyen kullanıcı sayısı (**FN**) 231'den 283'e çıktı. 0.80'deki %100 Precision yalnızca **üç pozitif tahmine** dayanıyor; bütün satın alıcıları iyi yakaladığımız anlamına gelmiyor. **F1 de 0.2576'dan 0.0208'e düştü.** Bu gözlemler bu deneyin dört threshold'u için geçerli; Precision'ın threshold yükseldikçe her zaman artacağı garanti değildir.

### 5. Hangi threshold en fazla kullanıcıyı hedefledi?

**0.50** en çok kullanıcıyı hedefledi: **TP + FP = 55 + 86 = 141 kullanıcı**. Bunun mesaj maliyeti **282 TL**. Ancak 141 hedefleme, şirketin **100 kişilik kapasite sınırını** aşıyor.

### 6. Hangi threshold'un yanlış alarm maliyeti en düşüktü?

**0.80**: **FP = 0**, dolayısıyla **yanlış alarm maliyeti 0 × 2 = 0 TL**. Bununla birlikte toplam üç kullanıcı hedeflendi (**6 TL toplam mesaj maliyeti**) ve **283 gerçek satın alıcı hedeflenmedi**. Dolayısıyla en düşük yanlış alarm maliyeti tek başına en iyi iş sonucu değildir.

### 7. 100 kişilik kampanya kapasitesi için hangi threshold daha uygun?

Kapasiteyi aşmayan seçenekler **0.60 (70 kişi)**, **0.70 (21 kişi)** ve **0.80 (3 kişi)**. Dört denenen eşik içinde, **mümkün olduğunca çok gerçek satın alıcıyı hedefleme** hedefiyle **0.60'ı** seçerim: kapasiteyi aşmaz ve uygun seçenekler arasında en yüksek **TP = 30** değerine sahiptir. Toplam mesaj maliyeti **140 TL**, yanlış pozitif mesaj maliyeti **80 TL**. Daha yüksek Precision veya daha düşük maliyet öncelikliyse 0.70 düşünülebilir. Ayrıca **0.50–0.60 arasında** yeni eşikler denenebilir veya olasılığı en yüksek **ilk 100 kullanıcı** seçilebilir; bunlar mevcut dört eşik dışındaki alternatif karar politikalarıdır. **TP sayıları kampanyanın neden olduğu ek satış sayıları değildir.**

### 8. Şirket yalnızca Accuracy üzerinden karar verirse ne tür sorunlar yaşayabilir?

Validation setinde **286/1849 ≈ %15.47** satın alan, **1563/1849 ≈ %84.53** satın almayan kullanıcı vardır. Herkese `Revenue=False` diyen basit bir kural bile **%84.53 Accuracy** sağlayabilir. Denenen eşiklerde en yüksek Accuracy **0.70'te %84.91**, yani bu referansın yalnızca yaklaşık **0.38 yüzde puan** üzerindedir; aynı eşikte **Recall %4.90** olduğundan gerçek satın alıcıların **272/286'sı kaçırılır**. Accuracy'ye tek başına bakmak azınlık sınıfının kaçırılmasını gizler. Bu yüzden **Precision, Recall, F1, Average Precision** ve kapasite/maliyet ölçütlerini birlikte incelerim.

### 9. Bu sonuçlardan gerçek kampanya kârlılığını neden doğrudan hesaplayamayız?

Bu deney yalnızca **kime mesaj gönderileceğini** ve mesaj maliyetini simüle ediyor. `Revenue=True` etiketinin var olması, satışın kampanya **sayesinde** gerçekleştiğini göstermez; TP'lerin bazıları mesaj gönderilmese de satın alabilirdi. Ayrıca satış tutarları, ürünün katkı kârı, kampanyanın dönüşümde yarattığı **ek artış (uplift)** ve diğer işletme maliyetleri elimizde yok. Gerçek kârlılık için mesaj alan ve almayan benzer kullanıcıları karşılaştıran **randomize kontrol/A-B testi** gibi nedensel kanıtlarla ek satış ve katkı kârı ölçmek gerekir. Bu nedenle burada güvenilir biçimde hesapladığım değer **mesajlaşma maliyetidir**, net kampanya kârı değil.

## Son karar ve sınırlamalar

- **Deney sonuçlarına dayalı mevcut tercih:** 100 kişilik sınır ve gerçek satın alıcıları daha fazla hedefleme amacı için, denenen dört değer arasında **threshold = 0.60**.
- **Model seçimiyle karar politikası farklıdır:** Threshold'u değiştirmek eğitilmiş modelin ağırlıklarını veya aynı skorlar üzerinde hesaplanan **Average Precision** değerini değiştirmez.
- **Değerlendirme disiplini:** Bu seçimler **validation** sonuçlarıyla yapılmıştır. Model ve eşik seçimi tamamlandıktan sonra bağımsız **test setinde bir kez** nihai değerlendirme yapılmalıdır; test seti eşik ayarlamak için kullanılmamalıdır.
- **İş etkisi sınırlaması:** Mesaj maliyetleri varsayımsal **2 TL** üzerinden hesaplandı; satış artışı ve kârlılık kanıtlanmış değildir.

## GitHub'a ekleme

Proje kök dizininde:

```bash
git add reports/threshold_experiments.csv reports/day_04_business_decisions.md
git commit -m "Add threshold experiments and business decision report"
git push
```
