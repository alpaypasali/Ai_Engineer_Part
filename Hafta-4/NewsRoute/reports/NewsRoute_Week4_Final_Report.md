# NewsRoute — Hafta 4 Final Raporu

**Proje:** NewsRoute  
**Veri seti:** AG News  
**Hafta:** 4  
**Amaç:** Haberleri dört konu ekibine otomatik yönlendirmek ve sistemi kayıtlı modellerle tekrar çalıştırabilmek.

## 1. Project Overview

NewsRoute, haber metnini okuyup ilgili kategoriye yönlendiren bir sınıflandırma projesidir. Bu hafta önce Attention ve Transformer bileşenlerini öğrendim, sonra TF-IDF + Logistic Regression ile sıfırdan eğitilmiş Mini Transformer'ı aynı veriler üzerinde değerlendirdim. Son aşamada kaydettiğim modelleri Jupyter oturumu olmadan yükleyip komut satırından tahmin aldım. Model performansını artırmak için yeni denemeler yapmadım; mevcut sonuçları inceledim ve uygulamanın güvenilirliğini geliştirdim.

## 2. Dataset

**AG News** veri setini kullandım. Orijinal eğitim bölümünden her sınıftan 1.000 haber seçerek toplam **4.000 haber** oluşturdum. Bunları **3.200 eğitim** ve **800 validation** haberi olarak ayırdım; validation kümesinde her sınıftan 200 haber vardı. Hafta boyunca model geliştirme ve karşılaştırmada yalnızca bu ayrımları kullandım. Modelleri seçip geliştirmeyi bitirdikten sonra orijinal, **7.600 haberlik test kümesini** ilk kez final değerlendirme için kullandım; test kümesinde dört sınıfın her birinden 1.900 örnek vardı.

Metin hazırlığında küçük harfe dönüştürme, noktalama ve sayı temizleme, boşlukları düzenleme, İngilizce stopword kaldırma, nadir kelime filtreleme ve lemmatization uygulandı. Nadir kelime filtresi başlangıçtaki 4.000 haber üzerinden belirlendi; Transformer vocabulary'si ise yalnızca **3.200 eğitim haberi** kullanılarak oluşturuldu. Vocabulary boyutu **7.787**, `[PAD] = 0`, `[UNK] = 1` ve maksimum dizi uzunluğu **64** idi.

## 3. Problem

Bu proje, **World (0), Sports (1), Business (2), Sci/Tech (3)** olmak üzere dört sınıflı haber sınıflandırma problemidir. Sistem her haber için tek bir kategori üretir. Haber yanlış sınıflandırılırsa yanlış konu ekibine gidebilir; bu nedenle yalnızca genel Accuracy'ye değil, her sınıfın hatalarına da bakmak gerekir.

## 4. Models

### TF-IDF + Logistic Regression

- `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)`
- `LogisticRegression(max_iter=1000)`
- Kelime ve bigram ağırlıklarına dayanır.
- Eğitilmiş Pipeline `models/tfidf_pipeline.joblib` olarak kaydedildi.

### Mini Transformer

- Token embedding: `d_model=64`
- Sinusoidal Positional Encoding: maksimum uzunluk `64`
- Transformer Encoder: `1` katman, `4` attention head, feedforward boyutu `128`, dropout `0.1`
- Padding mask ve Masked Mean Pooling
- Son katman: `Linear(64, 4)` → dört sınıf için logits
- Eğitim: `CrossEntropyLoss`, `AdamW(lr=0.001)`, batch size `32`, **3 epoch**
- Training Loss: **1.2868 → 0.8547 → 0.5855**
- Model ağırlıkları `models/newsroute_transformer.pt`, vocabulary `models/vocab.json` olarak kaydedildi.

Mini Transformer önceden eğitilmiş BERT gibi bir model değildir; ağırlıkları sıfırdan öğrenildi.

## 5. Validation Results

İki modeli **aynı sıradaki 800 validation haberi** üzerinde değerlendirdim.

| Metrik | TF-IDF + Logistic Regression | Mini Transformer |
|---|---:|---:|
| Validation Samples | 800 | 800 |
| Accuracy | 0.8712 | 0.7138 |
| Macro Precision | 0.8719 | 0.7143 |
| Macro Recall | 0.8712 | 0.7138 |
| **Macro-F1** | **0.8711** | **0.7108** |
| World Recall | 0.8600 | 0.7700 |
| Sports Recall | 0.9650 | 0.7950 |
| Business Recall | 0.8050 | 0.5550 |
| Sci/Tech Recall | 0.8550 | 0.7350 |

Ana metriğim **Macro-F1** oldu; çünkü her sınıfın F1 değerine eşit ağırlık verir. Bu sonuçlarda TF-IDF bütün sınıflarda daha yüksek Recall elde etti.

## 6. Final Test Results

Model seçimini validation sonuçlarına göre yaptıktan sonra, **orijinal AG News test kümesindeki 7.600 haber** üzerinde iki kayıtlı modeli yeniden eğitmeden çalıştırdım. Eğitimdeki metin hazırlama işlemlerini aynı sırayla uyguladım. Aşağıdaki metrikler terminalde hesaplanan **gerçek sonuçlardır** ve `reports/final_test_metrics.csv` dosyasına kaydedildi.

| Metrik | TF-IDF + Logistic Regression | Mini Transformer |
|---|---:|---:|
| Test Samples | 7.600 | 7.600 |
| Accuracy | 0.8705 | 0.7361 |
| Macro Precision | 0.8699 | 0.7374 |
| Macro Recall | 0.8705 | 0.7361 |
| **Macro-F1** | **0.8701** | **0.7360** |

### İstenen Validation / Test Karşılaştırması

| Model | Validation Macro-F1 | Test Macro-F1 |
|---|---:|---:|
| **TF-IDF + LR** | **0.8711** | **0.8701** |
| **Mini Transformer** | **0.7108** | **0.7360** |

TF-IDF'nin test başarısı validation sonucuna çok yakındı. Transformer'ın test Macro-F1 değeri validation'dan biraz daha yüksek çıktı; tek başına bu farkın nedenini kesin olarak söyleyemem.

**Önemli kontrol:** İlk test denemesinde ham metinlerle Transformer Macro-F1 **0.5046** idi. Eğitimdeki preprocessing testte de uygulandığında **0.7360** oldu. Bu fark, kayıtlı ağırlıkları yüklemenin tek başına yeterli olmadığını; metin hazırlama sürecini de tutarlı yürütmek gerektiğini gösterdi. Bu kontrol sırasında test verisinden yeni vocabulary veya nadir kelime filtresi öğrenmedim.

## 7. Model Selection

**Mevcut validation sonuçlarına göre TF-IDF + Logistic Regression'ı seçtim.** Çünkü Macro-F1 değeri daha yüksekti (**0.8711'e karşı 0.7108**) ve özellikle Business haberlerinde Recall farkı büyüktü (**0.8050'a karşı 0.5550**). TF-IDF daha basit bir model olduğu için kullanımı ve bakımı da daha kolay olabilir. Nihai test sonuçları da önceden verdiğim bu kararı destekledi; test sonuçlarına bakarak yeni model seçmedim veya hiperparametre değiştirmedim.

## 8. Attention Architecture

RNN/LSTM metni sırayla işlerken Transformer, Self-Attention ile token'lar arasındaki ilişkileri aynı katmanda paralel hesaplayabilir. Attention'da **Query** hangi bilgiyi aradığımızı, **Key** hangi bilginin eşleştiğini, **Value** ise aktarılacak içeriği temsil eder. Önce `QKᵀ` ile uyum skorları hesaplanır, `√d_k` ile ölçeklendirilir ve `softmax` ile toplamı 1 olan attention ağırlıklarına dönüştürülür. Bu ağırlıklar `V` ile çarpılarak yeni token temsilleri elde edilir.

Multi-Head Attention, farklı öğrenilmiş temsil uzaylarında birden fazla ilişkiyi inceleme imkânı sağlar. NewsRoute modelinde `d_model=64` ve `4` head olduğu için her head'in boyutu **16** idi. Head sonuçları birleştirilir, ardından Output Projection farklı head bilgilerinin öğrenilebilir biçimde karıştırılmasını sağlar. Positional Encoding token sırasıyla ilgili bilgi ekler. **`src_key_padding_mask`** PAD konumlarının attention'da Key olarak kullanılmasını engeller; **Masked Mean Pooling** ise PAD konumlarını haber vektörünün ortalamasından çıkarır.

## 9. Error Analysis

Cuma günü Transformer'ın yanlış sınıflandırdığı **10 farklı validation haberi** seçtim. Seçtiğim bu 10 haberin **tamamını TF-IDF doğru**, Transformer yanlış tahmin etti. Bu örnekler Transformer'ın hataları arasından özellikle seçildiği için genel başarı oranı olarak yorumlanmamalıdır.

| Validation row_id | Gerçek sınıf | Transformer tahmini | Metne dayalı kısa gözlem |
|---:|---|---|---|
| 67 | Business | World | Santander teklifi Avrupa Birliği bağlamında anlatılıyor. |
| 124 | World | Sports | Kasırga ve can kaybı haberi; Sports tahmininin açık nedeni görünmüyor. |
| 132 | Business | World | Şirket yöneticisi ve mahkeme kararı birlikte geçiyor. |
| 227 | Sports | Sci/Tech | Formula 1 sürücüsünün emekliliği, Toyota ve test sürücülüğüyle anlatılıyor. |
| 554 | Business | Sports | Petrol fiyatı ve Nijerya'daki tehdit haberi. |
| 626 | World | Sci/Tech | CBS, Başkan Bush ve belgelerle ilgili siyasi haber. |
| 628 | Sci/Tech | Sports | Çok kısa HP, Unix ve Opteron haberi. |
| 648 | Sci/Tech | Sports | Sony Walkman ve iPod karşılaştırması. |
| 757 | Sci/Tech | Sports | NASA'nın hipersonik uçuş/hız rekoru denemesi. |
| 771 | Sports | Business | Bir patencinin yarışmadaki düşüşüyle ilgili haber. |

Confusion Matrix'te Transformer **48 Business haberini Sci/Tech** olarak sınıflandırdı; TF-IDF'de bu sayı **30** idi. Transformer için en düşük Recall **Business (%55,5)**, en yüksek Recall ise **Sports (%79,5)** oldu. Bazı haberler iki sınıfla ilgili kelimeler taşıyor olabilir; ancak yalnızca tahmin ve metin üzerinden hatanın kesin nedenini veya attention'ın hangi kelimeye odaklandığını kanıtlayamam.

## 10. Inference Demo

Kayıtlı TF-IDF Pipeline'ını kullanan bir **`predict.py`** hazırladım. Komut satırında:

```bash
python predict.py "NASA announces new satellite"
```

Çalıştırılan örneğin gerçek çıktısı:

```text
Input: NASA announces new satellite
Predicted Class: Sci/Tech
```

Demo metni eğitimdeki gibi temizler, kayıtlı nadir kelime listesini kullanır, modelin sınıf ID'sini kategori adına çevirir. Bu işlem Jupyter notebook'a gerek kalmadan çalıştı. Transformer için de kaydedilmiş ağırlıkları ve vocabulary'yi ayrı bir Python oturumunda yükleyip tahmin ürettim.

## 11. Tests

`tests/test_prediction.py` içerisinde önce üç temel hata senaryosunu, ardından Input Validation iyileştirmesiyle toplam **altı testi** çalıştırdım. Ekran görüntüsünde altı testin de **PASSED** olduğu görüldü.

| Test | Beklenen davranış | Sonuç |
|---|---|---|
| Empty Input (`""`) | Anlamlı `ValueError` | PASSED |
| Whitespace Input (`"    "`) | Anlamlı `ValueError` | PASSED |
| Missing Model | Model dosyası yokken hata, uydurma tahmin yok | PASSED |
| None Input | `TypeError` | PASSED |
| Integer Input (`123`) | `TypeError` | PASSED |
| Valid Input | Geçerli kategori adı döndürür | PASSED |

Eksik model testi gerçek `tfidf_pipeline.joblib` dosyasını silmeden, geçici bir dizinde uygulandı.

## 12. Improvement

Haftanın son iyileştirmesi **Input Validation** oldu. `validate_text(text)` fonksiyonu girdinin string olup olmadığını ve boş/yalnızca boşluk içerip içermediğini kontrol ediyor. `None` ve `123` gibi yanlış türlerde anlamlı hata veriyor; geçerli metinlerde ise mevcut modeli kullanarak tahmin üretmeye devam ediyor. **Bu iyileştirme modelin Accuracy değerini yükseltmiyor; uygulamanın güvenilirliğini artırıyor.** Yeni model eğitmedim.

## 13. Limitations

- Eğitimde yalnızca **3.200 haber** vardı; bu, sıfırdan eğitilen Transformer için sınırlı olabilir.
- Mini Transformer **önceden eğitilmedi** ve sadece **3 epoch** kullanıldı. Daha fazla epoch'un mutlaka daha iyi sonuç vereceği söylenemez.
- AG News, gerçek bir haber platformunun bütün güncel konu çeşitliliğini ve dağılım değişikliklerini temsil etmeyebilir.
- Eğitim/validation bölünmesinden **önce** 4.000 haber üzerinde nadir kelime filtresi hesaplanmış olması validation açısından metodolojik bir sınırlamadır. Bu hafta geçmiş deneyle tutarlılık için aynı işlem yeniden üretildi.
- İlk ham test sonucu ile doğru temizlenmiş test sonucu arasındaki fark preprocessing bağımlılığını gösterdi; bu bağımlılığın kalıcı olarak belgelenip korunması gerekir.
- Yanlış yönlendirmenin gerçek operasyonel maliyeti ölçülmedi. İnsan incelemesi, izleme ve gerçek kullanım koşullarında ek değerlendirme gerekebilir.
- Notebook'tan bağımsız yükleme **mevcut `myenv` ortamında** doğrulandı; tamamen yeni bir sanal ortamda kurulum test edilmedi.
- Bunlar bir üretim sistemi için yeterli bütün kontroller değildir; henüz API, Docker veya web uygulaması geliştirilmedi.

Bu nedenle **NewsRoute demosu çalışıyor**, ancak mevcut sonuçlarla gerçek bir haber platformunda denetimsiz kullanıma tamamen hazır olduğunu söyleyemem.

## 14. What I Learned

Bu hafta Transformer'ın yalnızca bir hazır model olmadığını, Embedding, Positional Encoding, Q/K/V, Multi-Head Attention, Padding Mask ve Pooling gibi parçaların birlikte çalıştığı bir yapı olduğunu öğrendim. Q ve K ile uyum skorlarının hesaplandığını, Value ile yeni temsil üretildiğini ve her head'in farklı ilişkileri öğrenebileceğini anladım. Ayrıca daha karmaşık modelin her zaman daha iyi olmadığını gördüm; bizim veri ve eğitim koşullarımızda TF-IDF daha başarılı oldu. Macro-F1 ve Confusion Matrix ile sadece toplam başarıya değil, sınıf bazlı hatalara da bakmam gerektiğini öğrendim. En önemli mühendislik dersim ise **modeli eğitmenin yetmediği**: Aynı preprocessing'i korumak, ağırlıkları tekrar yüklemek, terminalden tahmin almak ve hatalı girdiler için test yazmak da projenin önemli parçaları.

---

**Sonuç:** NewsRoute'un dört sınıflı haber yönlendirme demosunu bitirdim. Validation sonuçlarına dayanarak seçtiğim **TF-IDF + Logistic Regression**, bağımsız final testte **Macro-F1 = 0.8701** elde etti; Mini Transformer'ın final test Macro-F1 değeri **0.7360** oldu. Modelleri testte yeniden eğitmedim veya test sonucuna göre ayarlamadım.
