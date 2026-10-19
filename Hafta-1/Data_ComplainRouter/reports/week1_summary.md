# 1. Hafta Özeti — ComplaintRouter AI

**Proje:** ComplaintRouter AI  
**Konu:** NLP, makine öğrenmesi, modelin kaydedilmesi, CLI, FastAPI ve testler

## 1. Bir haftada neler geliştirdim?

Bu hafta İngilizce finansal tüketici şikâyetlerini dört kategoriye ayıran bir NLP projesi geliştirdim. Yaklaşık 5.000 şikâyet metniyle çalıştım. İlk olarak veriyi inceledim, kategori etiketlerini sayısal değerlere dönüştürdüm ve metinleri temizledim. Ardından TF-IDF kullanarak metinleri makine öğrenmesi modellerinin anlayabileceği sayısal özelliklere çevirdim.

İlk modelim Logistic Regression oldu. Daha sonra PyTorch ile farklı boyutlarda MLP modelleri deneyerek sonuçları karşılaştırdım. Modelleri sadece accuracy ile değil, Macro-F1 ve confusion matrix gibi ölçütlerle de değerlendirdim. En iyi kaydedilmiş Logistic Regression sonucum **%72,60 accuracy** ve **0,7231 Macro-F1** oldu.

Son aşamada metin temizleme, TF-IDF ve sınıflandırma adımlarını tek bir scikit-learn Pipeline içinde birleştirdim. Bu Pipeline'ı `joblib` ile kaydettim ve tekrar yüklediğimde aynı örnek için aynı tahmini verdiğini kontrol ettim. Ardından `predict.py` ile terminalden çalışan bir uygulama hazırladım. FastAPI kullanarak `/health` ve `/predict` endpoint'lerini oluşturdum; böylece modele HTTP üzerinden istek gönderilebilecek bir arayüz de geliştirdim. `pytest` ile hatalı girdiler, tahmin çıktıları ve API doğrulaması için testler yazmaya başladım. Ekran görüntülerimde **6 tahmin testi ve 2 API testinin geçtiğini** doğruladım; daha kapsamlı API testlerinin tamamlandığını henüz kesinleştirmedim.

## 2. Logistic Regression neden MLP'den daha başarılı oldu?

Benim deneylerimde Logistic Regression, MLP modellerinden daha iyi sonuç verdi. **3.000 TF-IDF özelliği** kullanan Logistic Regression modelinin Macro-F1 değeri **0,7231** iken, **1.500 özellikli** MLP-128 modeli **0,6739**, MLP-256 modeli ise **0,6730** değerine ulaştı.

Bunun muhtemel nedenlerinden biri TF-IDF verisinin seyrek ve yüksek boyutlu olması. Logistic Regression gibi doğrusal modeller bu tür metin özellikleri üzerinde güçlü bir başlangıç yöntemi olabiliyor. MLP ise daha fazla parametre öğreniyor ve elimdeki veri miktarı bu ek karmaşıklığı verimli kullanmak için yeterli olmamış olabilir. MLP'nin düşük eğitim kaybı üretmesi de yeni verilerde daha iyi sonuç vereceği anlamına gelmiyor.

Yine de **“Logistic Regression her zaman MLP'den iyidir”** sonucuna varamam. Çünkü modeller tamamen aynı koşullarda karşılaştırılmadı: Logistic Regression için 3.000, MLP'ler için 1.500 TF-IDF özelliği kullanıldı. Daha adil bir karşılaştırma için aynı veri ayrımı, özellik sayısı ve kontrollü deney ayarlarıyla yeniden denemem gerekir.

## 3. Pipeline hangi problemi çözüyor?

Pipeline, modelin ihtiyaç duyduğu işlem adımlarını **tek bir sıra** hâline getiriyor. Benim projemde bu sıra şöyle:

```text
Ham şikâyet metni
       ↓
Metin temizleme
       ↓
TF-IDF
       ↓
Logistic Regression
       ↓
Kategori tahmini
```

Bu yapı olmasaydı eğitimde metni bir şekilde temizleyip tahmin sırasında başka şekilde temizleme riskim olurdu. Ayrıca TF-IDF'nin eğitim sırasında öğrendiği kelime sözlüğünü ve IDF değerlerini tahminde aynı şekilde kullanmak zorundayım. Pipeline bu adımları birlikte saklamayı ve tutarlı uygulamayı kolaylaştırıyor.

Bir başka faydası, **cross-validation sırasında doğru kurulduğunda veri sızıntısını önlemeye yardımcı olması**. TF-IDF'yi Pipeline'ın içine koyup bütün Pipeline'ı cross-validation'a verirsem her katta kelime sözlüğü sadece o katın eğitim bölümünden öğrenilir.

## 4. Model kaydetme ile model eğitme arasındaki fark nedir?

**Model eğitme**, veriyi kullanarak modelin parametrelerini öğrenmesidir. Örneğin `pipeline.fit(X_train, y_train)` çalıştırdığımda TF-IDF eğitim verisinden özelliklerini öğrenir ve Logistic Regression kendi katsayılarını hesaplar. Bu işlem zaman ve işlem gücü gerektirir.

**Model kaydetme** ise eğitim tamamlandıktan sonra öğrenilmiş yapıyı dosyaya yazmaktır. Ben bunun için `joblib.dump()` kullandım. Daha sonra `joblib.load()` ile aynı Pipeline'ı tekrar açıp **yeniden eğitmeden** tahmin yapabiliyorum.

Kısacası `fit()` öğrenme işlemini, `dump()` saklamayı, `load()` ise saklanan modeli geri getirmeyi sağlar. API'de her istekte eğitim yapmak yerine önceden eğittiğim modeli kullanmamın nedeni de budur. Güvenmediğim `.joblib` dosyalarını açmamam gerekir; bu biçim yüklenirken kod çalıştırabilen pickle mekanizmasına dayanır.

## 5. CLI ve API arasındaki fark nedir?

CLI (Command-Line Interface), programı terminal komutlarıyla kullanmamı sağlıyor:

```bash
python predict.py --text "My mortgage payment is incorrect"
```

Bu yöntem geliştirme, hızlı deneme ve komut satırı üzerinden kullanım için pratik. API ise başka uygulamaların modelime HTTP isteği gönderebilmesini sağlıyor. Örneğin FastAPI'deki `POST /predict` endpoint'ine şöyle bir JSON gönderilebiliyor:

```json
{
  "text": "My mortgage payment is incorrect"
}
```

API'nin cevabı da JSON biçiminde geliyor. CLI çıktısı ise terminalde okunabilir metin olarak gösteriliyor. **İki arayüzün de aynı `predict_complaint()` fonksiyonunu kullanması** önemli: Tahmin mantığını iki ayrı yere kopyalarsam birini güncelleyip diğerini unutabilirim. Ortak fonksiyon kullanmak hem bakımı hem de test etmeyi kolaylaştırıyor.

## 6. Cross-validation sırasında veri sızıntısı nasıl oluşabilir?

Veri sızıntısı (*data leakage*), doğrulama veya test verisinden gelen bilginin model eğitimi sırasında yanlışlıkla kullanılmasıdır. Böyle bir durumda değerlendirme sonucu olduğundan iyi görünebilir; gerçek dünyadaki performans aynı olmayabilir.

Örneğin önce bütün veriye TF-IDF `fit_transform()` uygular, sonra cross-validation yaparsam doğrulama katındaki kelimeler ve IDF istatistikleri TF-IDF'nin öğrenme aşamasına karışmış olur. Modelin sınıf etiketlerini doğrudan görmesi şart değildir; doğrulama verisinden öğrenilen bu özellik bilgisi bile sızıntıdır.

**Benim notebook'umda da dikkat etmem gereken nokta bu:** Önce `X_train_tfidf = vectorizer.fit_transform(X_train)` yaptım, ardından `GridSearchCV` işlemini `X_train_tfidf` üzerinde uyguladım. Böylece dışarıda tuttuğum `X_test` doğrudan TF-IDF eğitimine girmese de **5 katlı cross-validation'ın kendi doğrulama katları** TF-IDF oluşturulurken kullanılmış oldu.

Daha doğru yaklaşım, TF-IDF ve modeli aynı Pipeline içine yerleştirip `GridSearchCV`'ye ham metinleri vermektir:

```python
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=3000)),
    ("classifier", LogisticRegression(max_iter=1000))
])

search = GridSearchCV(pipeline, param_grid, cv=5)
search.fit(X_train, y_train)
```

Buradaki `param_grid` anahtarlarının `classifier__C` gibi Pipeline adımlarıyla uyumlu olması gerekir. Kendi projemde metin temizleme adımını da aynı Pipeline'a eklemeliyim. Ayrıca model ve hiperparametre seçimi bittikten sonra, daha önce kullanılmamış **bağımsız bir test kümesiyle** son değerlendirmeyi yapmak gerekir. Mevcut deneylerimde holdout kümesini model karşılaştırması için de kullandığımdan sonucu tamamen bağımsız nihai test başarısı olarak görmüyorum.

## 7. Bir ML modelinin production ortamına hazır olduğunu söylemek için başka nelere ihtiyacımız var?

Şu anki projemi **çalışan bir prototip** olarak görüyorum; henüz production-ready değil. Gerçek kullanıcıların güvenle kullanması için şu konuları tamamlamam gerekir:

1. **Güvenilir değerlendirme:** Train / validation / test ayrımını netleştirmek; son test kümesini model seçiminde kullanmamak ve farklı gerçekçi şikâyet örnekleriyle performansı ölçmek.
2. **Olasılık kalibrasyonu:** Modelin verdiği `0.90` gibi değerlerin gerçek doğrulukla ne kadar örtüştüğünü incelemek. Özellikle `%60` Human Review eşiğini sonuçlara ve yanlış yönlendirme maliyetine göre seçmek.
3. **Bilinmeyen girdiler (OOD):** Finans dışı metinleri yanlışlıkla dört kategoriden birine güvenle atamamak için ek kontroller ve gerektiğinde insan incelemesi eklemek.
4. **Test kapsamı:** Unit ve API testlerinin yanında gerçek model dosyasını yükleyen entegrasyon testleri, başarısız yükleme ve yüksek trafik senaryoları hazırlamak.
5. **Performans ve dayanıklılık:** Yanıt süresini, bellek kullanımını, eşzamanlı istekleri ve uzun metinlerin etkisini ölçmek; beklenmeyen hatalara kontrollü cevap vermek.
6. **Güvenlik ve gizlilik:** Finansal şikâyetlerdeki kişisel verileri korumak; istek boyutu sınırları, erişim kontrolleri, güvenli loglama ve gerekli durumlarda rate limiting uygulamak.
7. **Dağıtım ve izleme:** Bağımlılık sürümlerini sabitlemek, model sürümünü takip etmek, servis sağlığını izlemek, hata oranlarını kaydetmek ve zamanla veri dağılımı değişirse (*data drift*) performansı yeniden değerlendirmek.

**Haftadan çıkardığım en önemli ders:** Bir ML projesi yalnızca yüksek accuracy elde etmekten ibaret değil. Modeli tekrar kullanılabilir hâle getirmek, doğru test etmek, API üzerinden sunmak ve hata durumlarını yönetmek de AI Engineering işinin önemli bir parçası.
