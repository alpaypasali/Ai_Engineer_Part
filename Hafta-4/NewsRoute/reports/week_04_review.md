# Hafta 4 — Teorik Tekrar

**Proje:** NewsRoute  
**Konu:** Transformer, Attention, Positional Encoding ve Model Değerlendirme

## 1.1. Transformer ve Attention

### Soru 1 — RNN vs Transformer
**Soru:** RNN ve LSTM metinleri nasıl işler? Transformer'ın Self-Attention mekanizması bu yaklaşımlardan hangi yönleriyle farklıdır?

**Cevabım:** RNN ve LSTM cümledeki kelimeleri sırayla işler, aynı anda işlemez. Hidden state önceki kelimelerden gelen bilgileri tutar ve sonraki adımlara aktarır. LSTM, klasik RNN'deki kaybolan gradyan problemini uzun süreli bellek ve kapılar (gate) kullanarak azaltmaya çalışır. Self-Attention ise kelimeleri tek tek sırayla işlemek zorunda değildir; bir kelimenin diğer kelimelerle ilişkilerini aynı katmanda paralel olarak hesaplayabilir. Transformer'ın avantajı hem bu ilişkileri daha doğrudan öğrenebilmesi hem de hesaplamaları paralel yapabilmesidir.

### Soru 2 — Query, Key, Value
**Soru:** Q = XW_Q, K = XW_K, V = XW_V ifadelerini açıkla. Q, K ve V neden ayrı projeksiyonlarla oluşturulur?

**Cevabım:** X, token'ların vektörlerini içeren giriş temsilidir. Query (Q), hangi bilgiyi aradığımı gösterir; yani bir token'ın diğer token'lardan ne öğrenmek istediğiyle ilgilidir. Key (K), Query ile hangi özelliklerin eşleşebileceğini gösterir ve uyum skorlarını hesaplamak için kullanılır. Value (V) ise bu eşleşmelerden sonra hangi bilgilerin aktarılacağını taşıyan vektördür. Q, K ve V için ayrı ağırlık matrisleri kullanırız çünkü modelin bilgi arama, eşleştirme ve aktarma işlerini farklı şekillerde öğrenmesini istiyoruz.

### Soru 3 — Scaled Dot-Product Attention
**Soru:** Attention(Q,K,V) = softmax(QKᵀ / √d_k)V formülündeki işlemleri ve √d_k kullanılma nedenini açıkla.

**Cevabım:** İlk olarak Q ile K'nin transpozu çarpılır ve token'lar arasındaki uyum skorları hesaplanır. Sonra bu skorları √d_k değerine böleriz; burada d_k, Query ve Key vektörlerinin boyutudur. Bu bölme, skorların çok büyümesini ve Softmax'ın aşırı keskin sonuçlar üretmesini azaltır. Softmax skorları 0 ile 1 arasında, toplamı 1 olan attention ağırlıklarına dönüştürür. Son olarak bu ağırlıkları Value ile çarparız ve token için yeni bir çıktı temsili elde ederiz.

### Soru 4 — Multi-Head Attention
**Soru:** d_model = 128 ve num_heads = 8 ise her head'in boyutu kaç olur? Head çıktıları neden birleştirilir ve Output Projection neden kullanılır?

**Cevabım:** Her head'in boyutu d_model / num_heads ile hesaplanır, yani 128 / 8 = 16 olur. Birden fazla head kullanıyoruz çünkü modelin aynı metindeki farklı ilişkileri farklı temsil alanlarında öğrenmesini istiyoruz. Her head önce kendi attention hesabını yapar ve kendi attention ağırlıklarını Value ile çarpar. Daha sonra sekiz head'in çıktısını birleştiririz; böylece farklı head'lerden gelen bilgiler aynı temsilde toplanır. Output Projection (W_o) ise bu bilgileri öğrenilebilir bir şekilde karıştırır ve modelin kullandığı çıktı boyutuna dönüştürür.

## 1.2. Positional Encoding ve Padding

### Soru 5 — Positional Encoding
**Soru:** Transformer neden token konumlarıyla ilgili bilgiye ihtiyaç duyar? Sinusoidal Positional Encoding nasıl çalışır?

**Cevabım:** Transformer'ın Self-Attention mekanizması tek başına kelimelerin sırasını bilmez. Bu yüzden her token'ın dizide hangi konumda olduğuna dair bilgi eklememiz gerekir. Embedding token'ın öğrenilebilir vektörünü oluştururken Positional Encoding token'ın konumunu temsil eder. Sinusoidal Positional Encoding, farklı frekanslarda sinüs ve kosinüs kullanarak her konum için bir vektör üretir. NewsRoute'ta Embedding ile Positional Encoding'i topladık; böylece model hem token bilgisini hem de konum bilgisini kullanabildi.

### Soru 6 — Padding Mask
**Soru:** src_key_padding_mask ile Masked Mean Pooling arasındaki fark nedir? İkisini neden birlikte kullanıyoruz?

**Cevabım:** Haberlerin token uzunlukları farklı olduğu için kısa olanların sonuna PAD ekleyerek hepsini aynı uzunluğa getiriyoruz. src_key_padding_mask, PAD konumlarını işaretler ve Self-Attention sırasında bu konumların Key olarak kullanılmasını engeller. Ama PAD konumları Query olarak işlem görebildiği için Transformer Encoder yine de bu konumlarda çıktı vektörü üretebilir. Masked Mean Pooling ise haberin tek bir vektörünü hesaplarken PAD konumlarını toplama ve ortalamaya dahil etmez. Yani ilk maske attention hesabında, ikincisi son haber temsilinde PAD etkisini engellemek için kullanılır.

## 1.3. Model Değerlendirme

### Soru 7 — TF-IDF vs Transformer
**Soru:** NewsRoute'ta TF-IDF neden daha başarılı olmuş olabilir? Bu sonuç Transformer mimarisinin genel olarak başarısız olduğunu kanıtlar mı?

**Cevabım:** Önemli olan modelin sadece karmaşık olması değil, veri miktarı ve problemin yapısıdır. Biz 3.200 haberle çalıştık ve TF-IDF kelime ve kelime ikililerinden kategorileri ayırmak için yararlı bilgiler çıkarabildi. Mini Transformer ise önceden eğitilmiş değildi; embedding ve attention ağırlıklarını sıfırdan, sadece 3 epoch içinde öğrenmeye çalıştı. Daha fazla veri veya eğitim farklı sonuçlar verebilir ama daha fazla epoch'un kesin olarak başarıyı artıracağını söyleyemeyiz. Bu deney Transformer'ın başarısız olduğunu kanıtlamaz; yalnızca bizim koşullarımızda TF-IDF'nin daha iyi sonuç verdiğini gösterir.

### Soru 8 — Macro-F1
**Soru:** Macro-F1 neden haber sınıflandırma projemizde önemli? Accuracy yüksek olmasına rağmen belirli bir kategoride Recall düşük olabilir mi?

**Cevabım:** Accuracy, bütün haberlerin kaç tanesini doğru tahmin ettiğimizi gösterir. Macro-F1 ise her kategorinin Precision ve Recall değerlerinden ayrı ayrı F1 hesaplar, sonra bu değerlerin ortalamasını alır. Böylece bütün sınıfları eşit şekilde değerlendirebiliriz. Accuracy yüksek olsa bile belirli bir kategoride Recall düşük olabilir; örneğin bizim Transformer'da Accuracy yaklaşık %71,38 iken Business Recall %55,5 oldu. Haberleri doğru ekiplere yönlendirmek istediğimiz için yalnızca genel Accuracy'ye değil, Macro-F1 ve sınıf bazlı hatalara da bakmalıyız.
