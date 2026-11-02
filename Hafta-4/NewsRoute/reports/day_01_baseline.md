# NewsRoute — Gün 1: TF-IDF + Logistic Regression Baseline

## 1. AG News nedir?

AG News, İngilizce haber metinlerinden oluşan ve haberleri dört konu sınıfından birine atamayı amaçlayan bir metin sınıflandırma veri setidir. Bu projede Hugging Face üzerindeki `fancyzhx/ag_news` sürümü kullanıldı: https://huggingface.co/datasets/fancyzhx/ag_news

Orijinal veri setinde 120.000 eğitim ve 7.600 test haberi bulunuyor. Bu çalışmada yalnızca orijinal eğitim bölümü kullanıldı; orijinal test bölümü haftanın final değerlendirmesi için ayrıldı.

## 2. Sınıflar nelerdir?

| Etiket | Kategori |
|---:|---|
| 0 | World |
| 1 | Sports |
| 2 | Business |
| 3 | Sci/Tech |

## 3. Veri örneklemi nasıl seçildi?

Orijinal `train` bölümünden `groupby("label").sample(n=1000, random_state=42)` yaklaşımıyla her sınıftan rastgele 1.000 haber seçildi. Böylece toplam 4.000 haberden oluşan, her sınıfın %25 paya sahip olduğu dengeli bir çalışma örneklemi elde edildi.

Örneklem `train_test_split(test_size=0.20, stratify=sample_df["label"], random_state=42)` ile **3.200 eğitim** ve **800 validation** haberine ayrıldı. Her sınıfın eğitimde 800, validation'da 200 örneği bulunuyor. Aynı ayrımı daha sonraki Transformer karşılaştırmasında kullanabilmek için `data/train_split.csv` ve `data/val_split.csv` dosyaları oluşturuldu.

Örneklemde eksik, tamamen boş veya yalnızca boşluklardan oluşan metin bulunmadı; örneklemde birebir aynı metin de tespit edilmedi. Training–validation arasında yapılan birebir metin kontrolünün sonucu **0 ortak haber** oldu. Normalize edilmiş metin ve indeks çakışması için otomatik test kodu da hazırlandı; bu testlerin çalıştırılma çıktıları ayrıca doğrulanmalıdır.

## 4. Metin uzunlukları nasıl dağılıyor?

4.000 haberlik örneklemde, metinler boşluklara göre kelimelere ayrılarak `word_count` hesaplandı.

| İstatistik | Kelime sayısı |
|---|---:|
| Minimum | 2 |
| Ortalama | 22,7695 |
| Medyan (P50) | 22 |
| Maksimum | 76 |
| P90 | 30 |
| P95 | 32 |

Örneklemdeki haberlerin en az %90'ı 30 kelime veya daha kısa, en az %95'i 32 kelime veya daha kısadır. Bu değerler **boşluk temelli kelime sayılarıdır**; Transformer tokenizer'ının üreteceği subword token sayılarıyla aynı değildir. Kategorilerin boxplot karşılaştırmasında Sports haberleri görece kısa; Sci/Tech kategorisinde daha belirgin uç uzunluklar gözlendi.

## 5. TF-IDF nasıl çalışır?

TF-IDF (*Term Frequency–Inverse Document Frequency*), bir kelimenin veya kelime grubunun bir metindeki önemini sayısal olarak temsil eder. Metinde daha fazla görülen ancak tüm belgelerde yaygın olmayan terimler daha ayırt edici ağırlıklar alabilir. Böylece haber metni, Logistic Regression'ın kullanabileceği sayısal bir özellik vektörüne dönüştürülür.

Bu baseline için `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)` kullanıldı. Model en fazla 5.000 özellik kullanır; hem tek kelimeler (*unigram*) hem ardışık ikili kelime grupları (*bigram*) dikkate alınır.

## 6. Pipeline neden kullanıldı?

`Pipeline([("tfidf", tfidf), ("classifier", LogisticRegression(max_iter=1000))])` ile metinleri sayısal özelliklere dönüştürme ve sınıflandırma adımları tek akışta birleştirildi. `pipeline.fit()` yalnızca eğitim verisinde çalıştırıldı. Böylece TF-IDF kelime dağarcığı ve IDF ağırlıkları validation verisinden öğrenilmedi; validation aşamasında yalnızca `predict()` kullanıldı. Bu yaklaşım, veri sızıntısı (*data leakage*) riskini azaltır.

## 7. Validation performansı

| Metrik | Sonuç |
|---|---:|
| Accuracy | 0,8712 |
| Macro Precision | 0,8719 |
| Macro Recall | 0,8712 |
| **Macro-F1 (ana metrik)** | **0,8711** |

Bu değerler **800 haberlik validation kümesinde ölçüldü**. Macro-F1, dört sınıfın F1 skorlarını eşit ağırlıkla ortalar. Sonuç, perşembe günkü Transformer modeli için baseline olarak kullanılacaktır.

## 8. En çok hangi sınıflar karıştı?

Confusion Matrix'te satırlar gerçek, sütunlar tahmin edilen sınıfları gösterir:

| Gerçek / Tahmin | World | Sports | Business | Sci/Tech |
|---|---:|---:|---:|---:|
| World | 172 | 10 | 13 | 5 |
| Sports | 2 | 193 | 2 | 3 |
| Business | 7 | 2 | 161 | 30 |
| Sci/Tech | 5 | 4 | 20 | 171 |

En belirgin karışma **Business ↔ Sci/Tech** arasındadır: 30 Business haberi Sci/Tech, 20 Sci/Tech haberi Business olarak tahmin edildi (**toplam 50 hata**). Sports sınıfı **193/200 = %96,5 recall** ile en iyi, Business sınıfı **161/200 = %80,5 recall** ile en düşük recall sonucunu verdi. Olası neden, iş dünyası ve teknoloji haberlerinin ortak kelime dağarcığına sahip olmasıdır; bu açıklama henüz yanlış sınıflandırılan metinler tek tek incelenerek doğrulanmadı.

## 9. Modelin sınırlamaları

- TF-IDF metnin bağlamsal anlamını Transformer modelleri gibi temsil etmez; yalnızca unigram ve bigram örüntülerinden yararlanır.
- Özellikle Business ve Sci/Tech gibi ortak kelime kullanan kategoriler arasında hatalar oluşabilir.
- Eğitim yalnızca 4.000 haberlik dengeli bir örnekleme dayanır; tüm veri dağılımına genelleme kesin değildir.
- Veri İngilizce haberlerden oluşur; başka dillere ve güncel haber akışına genelleme ayrıca test edilmelidir.
- Raporlanan metrikler validation kümesine aittir; **orijinal test kümesinde henüz final değerlendirme yapılmadı**.

**Baseline referansı:** `TFIDF_LogisticRegression` — **Macro-F1: 0,8711**.
