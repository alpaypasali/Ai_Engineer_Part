# Baseline Model Report — Consumer Complaint Classification

**Proje:** Data_ComplainRouter  
**Kaynak notebook:** `02_baseline_model.ipynb`  
**Yaklaşım:** TF-IDF + Logistic Regression  
**Rapor kapsamı:** Notebook'ta çalıştırılmış iki baseline deneyinin sonuçları

## 1. Problem Definition

Bu projenin amacı, tüketicilerin yazdığı **şikâyet metinlerinden ilgili finansal ürün kategorisini otomatik tahmin etmek**. Problem, dört sınıflı bir **metin sınıflandırma (multiclass text classification)** problemidir.

- **Girdi (`X`):** `Consumer complaint narrative` sütunundaki İngilizce, serbest biçimli şikâyet metni.
- **Çıktı (`y`):** `Product` sütununa karşılık gelen ürün kategorisi (model eğitiminde sayısal sınıf etiketi).

| Etiket | Ürün kategorisi |
|---|---|
| 1 | Credit reporting, credit repair services, or other personal consumer reports (raporda **Credit Repair Services** olarak kısaltılmıştır) |
| 2 | Debt collection |
| 3 | Credit reporting |
| 4 | Mortgage |

**Neden dört kategori?** Kullanılan `complaints_train_5000.csv` dosyasında bu dört kategori bulunuyor ve her biri eşit sayıda örnek içeriyor. Bu kapsam, temel bir sınıflandırma modeli kurup sınıf bazındaki hataları karşılaştırmak için elverişli. Bu dört kategorinin daha geniş bir veri evreninden nasıl seçildiği notebook'ta belirtilmiyor.

## 2. Dataset

- **Toplam örnek:** 5.000.
- **Sınıf dağılımı:** Her kategori için 1.250 örnek (%25).
- **Bölme yöntemi:** `train_test_split(test_size=0.2, random_state=42, stratify=y)`.
- **Eğitim kümesi:** 4.000 örnek (sınıf başına 1.000).
- **Test kümesi:** 1.000 örnek (sınıf başına 250).

> **Önemli:** Notebook'ta **ayrı bir validation kümesi yoktur**. Veri yalnızca eğitim ve test olarak ayrılmıştır. İki özellik sayısının performansı aynı test kümesinde karşılaştırıldığı için, nihai model seçimini doğrulamak üzere ayrıca validation/cross-validation ve dokunulmamış son test kümesi kullanılması daha doğru olur.

**Veri hazırlama kararları:**

1. `Product` kategorileri, manuel oluşturulan `label_map` sözlüğüyle 1–4 arasındaki sayılara dönüştürüldü.
2. Şikâyet metinleri küçük harfe çevrildi; `XXXX` gibi anonimleştirilmiş alanlar, noktalama işaretleri ve fazla boşluklar temizlendi. Eksik metin görülürse `clean_text` fonksiyonu boş metin döndürüyor.
3. `Complaint ID` modele verilmedi; kayıt kimliği metnin finansal kategorisini açıklayan anlamlı bir özellik olarak değerlendirilmedi.
4. Kaynak metin sütununda **0 birebir yinelenen metin** bulundu. Bölme işleminden sonra eğitim ve test metinlerinin kesişimi tekrar kontrol edildi ve **0 ortak metin** görüldü.
5. `stratify=y` ile her sınıfın eğitim ve test kümelerindeki oranı korundu.

## 3. Feature Engineering

**Neden TF-IDF?** Serbest metni Logistic Regression'ın kullanabileceği sayısal, seyrek (sparse) özelliklere dönüştürmek için TF-IDF tercih edildi. TF-IDF, bir kelimenin ilgili belgede ne kadar belirleyici olduğunu, diğer belgelerde ne kadar sık geçtiğini de dikkate alarak ölçer. Metin sınıflandırmasında hızlı, yorumlanabilir ve güçlü bir baseline sağlar.

**İki deneyde kullanılan ayarlar:**

| Ayar | Deney 1 | Deney 2 |
|---|---|---|
| Vectorizer | `TfidfVectorizer` | `TfidfVectorizer` |
| `max_features` | **1500** | **3000** |
| `ngram_range` | Varsayılan `(1, 1)` | Varsayılan `(1, 1)` |
| Diğer TF-IDF ayarları | scikit-learn varsayılanları | scikit-learn varsayılanları |

İlk deneyde TF-IDF eğitim matrisinin boyutu **(4000, 1500)**, test matrisinin boyutu **(1000, 1500)** olarak doğrulandı. İkinci deneyde özellik üst sınırı 3.000'e çıkarıldı. Notebook'ta stop-word filtresi, lemmatization veya bigram ayarı ayrıca tanımlanmadı.

**Data Leakage nasıl önlendi?** Önce eğitim/test ayrımı yapıldı. TF-IDF yalnızca eğitim metinlerinde `fit_transform(X_train)` ile öğrenildi; test metinlerine yalnızca `transform(X_test)` uygulandı. Böylece test kümesinin kelime istatistikleri TF-IDF sözlüğünün öğrenilmesine katılmadı. Ayrıca eğitim/test arasında birebir aynı metin bulunmadığı kontrol edildi. **Bununla birlikte**, anlamsal olarak neredeyse aynı şikâyetler veya aynı olayla ilişkili farklı kayıtlar ayrıca kontrol edilmediği için tüm sızıntı riskleri ortadan kalkmış sayılmaz.

## 4. Model Training

**Neden Logistic Regression?** TF-IDF'nin ürettiği yüksek boyutlu, sparse veriler üzerinde hızlı eğitilebilen, güçlü bir başlangıç (baseline) sınıflandırıcısıdır. Sonuçları daha karmaşık modellere karşı kıyaslamak kolaydır; modelin kelime ağırlıkları da yorumlanabilir.

**Notebook'taki model tanımı:**

```python
model = LogisticRegression(
    max_iter=1000,
    random_state=42
)
```

- **`max_iter=1000`:** Optimizasyonun yakınsaması için yeterli iterasyon sınırı tanımak amacıyla kullanıldı.
- **`random_state=42`:** Tekrar üretilebilirlik için ayarlandı.
- **Diğer hiperparametreler:** Kodda açıkça değiştirilmedi; kullanılan scikit-learn sürümünün varsayılanları geçerli (ör. `C=1.0`, L2 regularization, `solver='lbfgs'`).

Her iki deneyde aynı algoritma ve eğitim/test ayrımı kullanıldı. Değiştirilen tek açık deney parametresi TF-IDF'nin `max_features` değeridir. İkinci deneyde `model.fit(...)` yeniden çalıştırılarak model tekrar eğitildi.

## 5. Evaluation

Ana değerlendirme metrikleri **Accuracy** ve **Macro-F1**. Macro-F1, dört sınıfın F1 skoruna eşit ağırlık verir; bu nedenle her sınıfın başarısını birlikte değerlendirmeyi sağlar.

**İlk baseline (1.500 TF-IDF özelliği):**

- **Accuracy:** %72,30 (**723 / 1000** doğru tahmin).
- **Macro-F1:** **%71,97**.
- **Macro Precision:** %71,76.
- **Macro Recall:** %72,30.
- **Yanlış tahmin:** 277.

**En problemli kategori:** **Credit Repair Services (etiket 1)**. İlk deneyde recall **%52,4** (131/250), F1 yaklaşık **0,55**. Model bu sınıftaki şikâyetlerin neredeyse yarısını kaçırıyor. En başarılı kategori **Mortgage**: ilk deneyde recall **%92,8** (232/250).

İlk deneyin confusion matrix'ine göre en sık karıştırılan iki kategori **Credit Repair Services** ile **Credit Reporting**: bir yönde **80**, diğer yönde **64**, toplam **144** yanlış sınıflandırma. Bu iki kategori kredi raporu hataları ve düzeltme talepleri gibi benzer ifadeler içerebilir; ancak gerçek metin benzerliği ayrıca analiz edilmelidir.

**İkinci deney (3.000 TF-IDF özelliği):** Accuracy **%72,60**, Macro-F1 **%72,31**. Bu deneyde de en düşük sınıf bazlı F1, yaklaşık **0,56** ile Credit Repair Services kategorisindedir.

## 6. Experiments

| Metrik | Deney 1: 1.500 özellik | Deney 2: 3.000 özellik | Değişim |
|---|---:|---:|---:|
| Accuracy | %72,30 | **%72,60** | **+0,30 yüzde puan** |
| Macro Precision | %71,76 | **%72,09** | +0,33 yüzde puan |
| Macro Recall | %72,30 | **%72,60** | +0,30 yüzde puan |
| Macro-F1 | %71,97 | **%72,31** | **+0,34 yüzde puan** |
| Test örneği sayısı | 1.000 | 1.000 | Aynı |

**Hangi model daha iyi?** Kaydedilmiş test metriklerine göre **3.000 özellikli ikinci deney az farkla daha başarılı**: 726 doğru tahminle ilk deneyden 3 fazla örneği doğru sınıflandırıyor. Macro-F1 de 0,34 yüzde puan artıyor.

**Neden?** Artan kelime dağarcığı modele ek ayırt edici kelimeler sağlamış olabilir. Ancak bu açıklama bir hipotezdir; notebook'ta kelime önemleri veya anlamlılık testiyle doğrulanmadı. Özellik üst sınırı iki katına çıkmasına karşın kazanım küçük. Eğitim süresi ve bellek tüketimi ölçülmediğinden kaynak maliyeti hakkında kesin karşılaştırma yapılamaz.

**Tercih:** Sadece mevcut tahmin başarısına göre **Deney 2** seçilebilir. Ancak gerçek kullanım için maliyet/performans kıyaslaması ve bağımsız validation sonuçları görülmeden bunun daha iyi genelleyen model olduğu söylenemez. Özellikle kaynak kısıtı varsa Deney 1 de güçlü bir başlangıç seçeneğidir.

## 7. Limitations

**Veri setinin sınırlamaları:**

- Veri **5.000 örnekle** sınırlı ve dört sınıfa yapay olarak eşit dağılmış görünüyor; gerçek dünyadaki sınıf sıklıkları farklı olabilir.
- Yalnızca dört finansal ürün kategorisi değerlendirildi. Diğer ürünler için tahmin üretemez; model, tanımadığı bir metni de mevcut dört sınıftan birine atar.
- İki kategori (Credit Repair Services ve Credit Reporting) kavramsal olarak birbirine yakın; sınıf tanımlarındaki örtüşme hatayı artırabilir.
- `XXXX` gibi maskelenmiş ifadeler ve metin temizleme aşaması bazı bağlamsal ipuçlarını kaldırabilir.
- Veri tarihine, etiketlerin güvenilirliğine, şikâyet kaynaklarının çeşitliliğine ve gerçek üretim dağılımına ilişkin notebook'ta ayrıntılı analiz yok.
- **Ayrı validation kümesi yok.** İki deneyi aynı test kümesi üzerinde karşılaştırmak, model seçimine bağlı değerlendirme yanlılığı oluşturabilir.

**Hangi durumlarda hata yapılabilir?**

- Kredi raporundaki yanlış kayıtların düzeltilmesini anlatan ve iki benzer sınıfın kelimelerini birlikte içeren metinler.
- Çok kısa, belirsiz veya birden fazla finansal sorundan söz eden şikâyetler.
- Eğitimde az görülen terimler, yeni şirket/ürün adları, yazım hataları ve farklı anlatım biçimleri.
- Eğitim verisinin dışındaki kategorilere ait veya alan dışı (out-of-distribution) metinler.

**Gerçek bir şirkette kullanmadan önce:**

1. **Daha temsil edici ve güncel veri** toplayıp etiket kalitesini denetlemek; yakın kopyaları ve aynı olaydan gelen kayıtları kontrol etmek.
2. **Train / validation / test** ayrımını kurmak veya eğitim verisinde çapraz doğrulama uygulamak; en sonunda hiç kullanılmamış test kümesiyle ölçmek.
3. Özellikle **Credit Repair Services ↔ Credit Reporting** karışıklığında yanlış tahminleri ve açıklayıcı kelimeleri incelemek; gerekiyorsa iş kurallarıyla sınıf tanımlarını netleştirmek.
4. `ngram_range`, `min_df`, stop-word yaklaşımı, `max_features` ve Logistic Regression'ın `C` parametresini sistematik olarak aramak; alternatif olarak farklı doğrusal modeller veya dil modelleriyle karşılaştırmak.
5. **Sınıf bazlı precision, recall, F1**, hata maliyeti, olasılık kalibrasyonu ve düşük güvenli tahminleri **insan incelemesine yönlendirme** mekanizmasını değerlendirmek.
6. **Gizlilik ve güvenlik** önlemlerini, veri kayması (drift) izlemeyi, yeniden eğitim planını ve çıkarım gecikmesini üretime geçmeden önce doğrulamak.

---

**Kısa sonuç:** TF-IDF + Logistic Regression, dört sınıflı şikâyet yönlendirmesi için makul bir baseline sunuyor. Notebook'taki en yüksek sonuç **%72,60 Accuracy / %72,31 Macro-F1** (3.000 TF-IDF özelliği). Geliştirmenin önceliği birbirine yakın kredi raporlama kategorilerinin ayrımını iyileştirmek ve nihai performansı bağımsız bir test kümesinde doğrulamak olmalı.
