# ShopperIntent — 2. Gün PyTorch Deney Raporu

## Ne yaptım?

Bugün ShopperIntent verisiyle PyTorch kullanarak üç MLP deneyi yaptım. Amacım en yüksek skoru yakalamaktan çok, **aktivasyon fonksiyonu ve optimizer değiştiğinde modelin öğrenmesinin nasıl etkilendiğini görmekti**.

Dünkü Logistic Regression deneyindeki veri ayrımını korudum: **8.631 train, 1.849 validation, 1.850 test**. Veri sızıntısı ihtimaline karşı `PageValues` sütununu çıkardım. `StandardScaler` ve `OneHotEncoder` dönüşümlerini yalnızca train üzerinde `fit` ederek 16 özelliği 73 sayısal özelliğe dönüştürdüm.

Üç deneyde de **73 → 64 → 32 → 2** mimarisini, `CrossEntropyLoss`, batch size **32**, **15 epoch**, learning rate **0.001** ve seed **42** kullandım. Son katmandan ham logits aldım; loss hesaplamadan önce Softmax uygulamadım.

| Deney | Aktivasyon | Optimizer | Son train loss | Son val loss | Recall | F1 | AP |
|---|---|---|---:|---:|---:|---:|---:|
| MLP-A | ReLU | Adam | 0.3029 | 0.4115 | 0.1783 | 0.2446 | 0.3024 |
| MLP-B | GELU | Adam | 0.7437 | 0.7439 | 0.1783 | 0.2446 | 0.3024 |
| MLP-C | ReLU | SGD | 0.2812 | 0.4116 | 0.1783 | 0.2446 | 0.3024 |

> **Sonuçların durumu:** Bunlar *ilk notebook çıktım*, doğrulanmış nihai sonuçlar değil. Üç modelin sınıflandırma metriklerinin birebir aynı çıkması şüpheliydi. Değerlendirme kodunu her deneyin kendi modelini kullanacak şekilde düzelttim. Gerçek veriyle yeniden çalıştırılmış çıktılar henüz olmadığı için bu tabloya dayanarak kesin model sıralaması yapmıyorum.

## Teknik sorulara cevaplarım

### 1. ReLU ve GELU arasında nasıl bir fark gördüm?

İlk tabloda **ReLU + Adam** deneyinin loss değerleri **GELU + Adam** deneyinden düşük görünüyor. Ancak GELU'nun loss'u çok daha yüksek olmasına rağmen bütün sınıflandırma metriklerinin aynı çıkması güven vermedi. Bu yüzden ReLU'nun kesinlikle daha iyi olduğunu söyleyemem. ReLU negatif değerleri sıfırlarken GELU daha yumuşak bir aktivasyon sağlıyor; bu farkın etkisini görmek için sonuçları doğru şekilde yeniden ölçmem gerekiyor.

### 2. Adam ve SGD'nin training loss eğrileri nasıl değişti?

İlk çıktıdaki son training loss değerleri Adam için **0.3029**, SGD için **0.2812**. Fakat yalnızca son epoch değerlerine bakarak hangi optimizer'ın daha hızlı öğrendiğini söyleyemem. Bunun için 15 epoch boyunca kaydedilen loss eğrilerini karşılaştırmalıyım. Bu deneyden öğrendiğim şey, **aynı learning rate'in Adam ve SGD için aynı güncelleme davranışını sağlamadığı**.

### 3. En düşük training loss, en iyi validation F1 anlamına geldi mi?

İlk tabloda son training loss'u en düşük model **ReLU + SGD**. Buna rağmen üç modelin validation F1 değeri **0.2446** olarak kaydedilmiş. Bu değerler şüpheli olduğu için kesin sonuç çıkarmıyorum. Yine de training loss'un düşmesinin, validation verisinde daha yüksek F1 elde edeceğimizi garanti etmediğini öğrendim.

### 4. Recall, Logistic Regression baseline'ına göre arttı mı?

Logistic Regression baseline'ımın Recall değeri **0.0350**, ilk MLP çıktısında ise **0.1783** görünüyor. Bu doğrulansaydı daha fazla satın alma örneğini yakaladığımız anlamına gelirdi. Ama MLP değerlendirmesi henüz doğrulanmadığından bunu kesin bir iyileşme olarak yazmıyorum. Ayrıca baseline'ın **AP = 0.3203** değeri, ilk MLP tablosundaki **0.3024** değerinden yüksek. Tek metriğe bakarak karar vermemeliyim.

### 5. Neden yalnızca Accuracy'ye bakmıyorum?

Verinin yaklaşık **%84,5'i `Revenue=False`**. Herkese “satın almayacak” diyen DummyClassifier bile **0.8453 Accuracy** elde etti; buna rağmen satın alan hiçbir müşteriyi bulamadı. Bu nedenle benim için **Recall, Precision, F1 ve Average Precision** daha açıklayıcı. Özellikle kampanya senaryosunda hem kaçırdığımız olası müşterileri hem de gereksiz kampanya maliyetine yol açabilecek yanlış pozitifleri düşünmeliyim.

### 6. Deneylerin tekrarlanabilirliğini nasıl kontrol ettim?

Aynı train/validation kayıtlarını, mimariyi, batch size'ı, epoch sayısını, learning rate'i ve **seed 42** ayarını korudum. Her deney için **yeni model ve optimizer** oluşturdum; batch karıştırma sırasını da sabit seed ile kontrol edecek şekilde notebook'u düzenledim. Ancak aynı deneyi iki kez çalıştırıp aynı sonuçları aldığımı henüz doğrulamadım. Bu yüzden tam tekrarlanabilirlik iddiasında bulunmuyorum.

## Sonuç

Bu çalışmada `DataLoader`, forward pass, `loss.backward()`, `optimizer.step()` ve validation değerlendirmesini bir arada kullanmayı pekiştirdim. En önemli dersim şuydu: **Kodun çalışması, deneyin doğru ölçüldüğü anlamına gelmiyor.** Sonuçlar şüpheli olduğunda model ve değerlendirme akışını kontrol etmek gerekiyor.

Düzeltilmiş notebook'u gerçek veriyle baştan çalıştırdıktan sonra `reports/optimizer_ablation.csv` dosyasını doğrulanmış metriklerle kaydedeceğim. **Test kümesini henüz değerlendirmeye dahil etmedim.**
