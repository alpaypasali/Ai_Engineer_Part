# AirPulse — Day 02: Temporal Split and Lag Features

## 1. Kronolojik Veri Ayrımı

Zaman serilerinde verilerin kronolojik sırayla ayrılması gerekir. Çünkü rastgele ayrım yaparsak gelecekteki gözlemler eğitim verisine dahil olabilir.

Bu nedenle verileri %70 Training, %15 Validation ve %15 Test olarak kronolojik şekilde ayırdım.

Ayrıca yalnızca mevcut zamanı değil, tahmin edilen hedefin zamanını da kontrol ettim.

## 2. Veri Ayrım Tarihleri

- Training: [Başlangıç] – [Bitiş]
- Validation: [Başlangıç] – [Bitiş]
- Test başlangıç sınırı: [Tarih]
- Veri bitiş tarihi: [Tarih]

## 3. Lag Features

Lag Features, geçmiş zaman adımlarındaki ölçümleri modelin giriş özellikleri olarak kullanmamızı sağlar.

AirPulse projesinde mevcut PM2.5 değerini ve 1, 2, 3, 6, 12, 24 saat önceki PM2.5 ölçümlerini kullandım.

Böylece toplam 7 giriş özelliği oluşturdum.

## 4. Eksik Değerlerin Yönetimi

Lag özelliklerini oluşturmadan önce eksik PM2.5 kayıtlarını silmedim.

Çünkü eksik satırları önceden silmek, farklı saatlere ait ölçümlerin yanlış eşleşmesine neden olabilir.

Önce zaman kontrollerini yaptım. Ardından gerekli özelliklerden veya hedef değerden herhangi biri eksik olan kayıtları ayırdım.

Sonuç olarak:

- Toplam kayıt: 35.064
- Geçerli kayıt: 32.344
- Kullanılmayan kayıt: 2.720

Eksik değerleri doldurmadım.

## 5. Naive Forecast Validation Sonuçları

Naive Forecast, bir sonraki saatin PM2.5 değerinin mevcut saatin değerine eşit olacağını varsayar.

Validation döneminde elde ettiğim sonuçlar:

| Metrik | Sonuç |
|---|---:|
| MAE | 10.19 |
| RMSE | 20.82 |

MAE, tahminlerimin gerçek değerlerden ortalama mutlak uzaklığını gösteriyor.

RMSE'nin daha yüksek olması, bazı büyük tahmin hatalarının etkili olduğunu düşündürüyor.

Bu sonuçları daha sonra geliştireceğim RNN ve LSTM modelleriyle karşılaştırmak için referans olarak kullanacağım.

## 6. PyTorch Tensor Hazırlığı

Train ve validation verilerini `torch.float32` tipine dönüştürdüm.

Tensor boyutları:

| Tensor | Shape |
|---|---|
| X_train | (22634, 7) |
| y_train | (22634,) |
| X_val | (4789, 7) |
| y_val | (4789,) |

Şu anda giriş verilerim iki boyutlu.

RNN modellerinde ise giriş genellikle üç boyutlu olur:

`(batch_size, sequence_length, input_size)`

Bu boyutlar sırasıyla batch içindeki örnek sayısını, zaman adımı sayısını ve her zaman adımındaki özellik sayısını ifade eder.

## 7. Bugün Ne Öğrendim?

Bugün zaman serilerinde kronolojik veri ayrımının neden önemli olduğunu öğrendim.

Geçmiş saatlerin ölçümlerini `shift()` kullanarak lag özelliklerine dönüştürdüm. Ancak `shift()` fonksiyonunun saatlere değil, satırlara göre çalıştığını ve zaman damgalarını ayrıca kontrol etmem gerektiğini öğrendim.

Eksik ölçümleri lag oluşturmadan önce silmenin yanlış zaman eşleşmelerine neden olabileceğini gördüm.

Naive Forecast modelini validation döneminde ayrı değerlendirdim ve MAE ile RMSE sonuçlarını hesapladım.

PyTorch Tensor oluşturmayı ve RNN için gerekli üç boyutlu giriş yapısının mevcut tablosal verilerden nasıl farklı olduğunu öğrendim.

Son olarak küçük SGD deneyinde Loss ve Gradient Norm değerlerinin hızla yükseldiğini gözlemleyerek kararsız eğitim davranışını daha iyi anladım.

## 8. Sonuç

Kronolojik veri ayrımını, lag özelliklerini ve validation değerlendirmesini tamamladım.

Test performansını hesaplamadım. Test dönemini nihai değerlendirme için koruyorum.