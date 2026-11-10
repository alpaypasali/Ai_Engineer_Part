# ReviewLens — Day 01: Pipeline Pilot Teknik Raporu

**Proje:** ReviewLens · **Hafta:** 5 · **Tarih:** 9 Kasım 2026  
**Kapsam:** Veri analizi, iki hazır Hugging Face pipeline'ı, küçük pilot deneyler ve otomatik veri kontrolleri. Model eğitimi veya fine-tuning yapılmadı.

## 1. Problem

ReviewLens, İngilizce müşteri yorumlarını analiz ederek 1–5 yıldız değerlendirmesiyle ilişkili metinsel sinyalleri anlamayı hedefleyen bir NLP öğrenme projesidir. Bu aşamada amaç yeni bir model geliştirmek veya en yüksek skoru bulmak değil, hazır modellerin sınıflandırma davranışlarını gerçek yorumlarda incelemektir. Bir yorumdaki olumlu ve olumsuz ifadeler birlikte bulunabildiği için yıldız tahmini, yalnızca positive/negative ayrımından daha ayrıntılıdır.

## 2. Dataset

- **Ad:** Yelp Review Full
- **Kaynak:** Hugging Face Datasets — `Yelp/yelp_review_full`
- **Dil:** Ağırlıklı olarak İngilizce
- **Orijinal eğitim bölümü:** 650.000 yorum
- **Orijinal test bölümü:** 50.000 yorum
- **Sınıflar:** Beş yıldız kategorisi; orijinal veri setinin her kategorisinde 130.000 train ve 10.000 test yorumu bulunuyor.
- **Alanlar:** `text`, `label`
- **Bu gün kullanılan bölüm:** Yalnızca `train`. Orijinal test bölümüne dokunulmadı.

## 3. Data Source & License

- Veri kartı: https://huggingface.co/datasets/Yelp/yelp_review_full/blob/main/README.md
- Dataset sayfası: https://huggingface.co/datasets/Yelp/yelp_review_full
- Kartın işaret ettiği Yelp Dataset Terms of Use: https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf

Hugging Face sayfasında lisans `other` olarak işaretlenir ve veri kartı Yelp'in özel sözleşmesine yönlendirir. Bağlantı verilen sözleşmede akademik/ticari olmayan kullanım çerçevesi ile ham yorumların paylaşımı ve kamuya açık sonuçların yayımlanması konusunda kısıtlamalar vardır. Bu yüzden örnek yorumlar ve `data/yelp_sample_1000.csv` dosyası, gerekli kullanım hakkı doğrulanmadan herkese açık GitHub deposuna eklenmemelidir. Herkese açık bir sonuç raporu yayımlanmadan önce ilgili sözleşme hükümleri kontrol edilmelidir.

## 4. Label Mapping

Veri setindeki sınıf etiketi 0–4 aralığındadır; kullanıcının gördüğü yıldız sayısı 1–5'tir.

| `label` | `stars` |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

Dönüşüm: **`stars = label + 1`**. Orijinal `label` alanı korundu; insan tarafından okunabilir `stars` sütunu ayrıca oluşturuldu.

## 5. Sampling

`train` bölümü streaming yöntemiyle okundu. `shuffle(seed=42, buffer_size=10000)` kullanıldı; her `label` grubundan 200 yorum toplanınca döngü durduruldu. Toplam **1.000 dengeli yorum** elde edildi ve yerel çalışma alanında `data/yelp_sample_1000.csv` dosyasına kaydedildi. Streaming `buffer_size` karıştırması, 650.000 kaydın tümünü tek seferde tamamen rastgele karıştırmakla eşdeğer değildir.

| Yıldız | Örnek sayısı |
|---|---:|
| 1 | 200 |
| 2 | 200 |
| 3 | 200 |
| 4 | 200 |
| 5 | 200 |
| **Toplam** | **1.000** |

Train/validation ayrımı yapılmadı.

## 6. Data Quality

Kalite kontrolleri 1.000 yorumluk yerel örneklemde gerçekleştirildi.

| Kontrol | Bulgular |
|---|---:|
| Toplam yorum | 1.000 |
| Boş veya yalnızca boşluk içeren metin | 0 |
| Eksik etiket | 0 |
| Birebir aynı metnin tekrarı | 0 |
| Ortalama kelime sayısı | 131,20 |
| Minimum kelime sayısı | 2 |
| Maksimum kelime sayısı | 847 |

Kelime sayısı `str.split().str.len()` ile hesaplandı. Duplicate kontrolü aynı `text` değerlerine göre yapıldı; benzer anlamlı, ancak farklı yazılmış yorumlar ayrıca araştırılmadı. Her yıldızdan üçer olmak üzere 15 gerçek yorum özel çalışma ortamında incelendi. Yıldız dağılımı bar grafiğiyle doğrulandı.

## 7. Hugging Face

- **Model Hub:** Hazır modellerin ve model kartlarının bulunduğu platform.
- **Datasets:** Veri setlerini indirme, inceleme ve streaming ile okuma araçları.
- **Transformers:** Tokenizer ve model kullanımını kolaylaştıran kütüphane; `pipeline()` ile hazır inference yapılabilir.

Pipeline; tokenization, modelin çalıştırılması ve çıktıların etiket/skora dönüştürülmesi gibi adımları basitleştirir. BERT encoder tabanlıdır; bir encoder–decoder modeli değildir.

## 8. Text Classification

- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Pipeline:** `text-classification`
- **Amaç:** Ürün değerlendirmelerine yönelik önceden eğitilmiş modelle 1–5 yıldız tahmini.
- **İlk örnek:** `The food was delicious and the staff was very friendly.`
- **Gözlenen ilk çıktı:** `4 stars`, `score = 0.4579159`.

Dengeli örneklemden her yıldız grubu için 5 yorum seçildi (**25 yorum**). Pilot karşılaştırmada **11 doğru, 14 yanlış** yıldız tahmini gözlendi; **3 yıldızlı beş yorumun hiçbiri tam isabetle tahmin edilmedi**. Bunlar yalnızca küçük pilot gözlemlerdir; güvenilir genel performans veya dağıtıma hazır bir doğruluk sonucu değildir.

## 9. Zero-Shot Classification

- **Model:** `typeform/distilbert-base-uncased-mnli`
- **Pipeline:** `zero-shot-classification`
- **Temel yaklaşım:** MNLI/NLI ile eğitilmiş model, inference sırasında verilen aday açıklamaları yorumla karşılaştırır.
- **Ayar:** `multi_label=False`; adaylar alternatif seçenekler olarak değerlendirilir.
- **Veri:** Önceki 25 yorumdan her yıldız kategorisine ait 2 yorum, toplam **10 yorum**.
- **Lisans notu:** Model kartında lisans **Unknown / belirlenmemiş** görünüyor. Deney yalnızca eğitim amaçlı yürütüldü.

Kullanılan aday açıklamalar:

1. `a very negative customer review`
2. `a negative customer review`
3. `a mixed or neutral customer review`
4. `a positive customer review`
5. `a very positive customer review`

Her yorum için `labels[0]` ve `scores[0]` kaydedildi:

| Yelp stars | En yüksek skorlu Zero-Shot etiketi | Score |
|---:|---|---:|
| 1 | a positive customer review | 0.201998 |
| 1 | a mixed or neutral customer review | 0.659484 |
| 2 | a very negative customer review | 0.235896 |
| 2 | a mixed or neutral customer review | 0.269301 |
| 3 | a positive customer review | 0.222450 |
| 3 | a very positive customer review | 0.407212 |
| 4 | a very positive customer review | 0.594614 |
| 4 | a mixed or neutral customer review | 0.219253 |
| 5 | a positive customer review | 0.318961 |
| 5 | a very negative customer review | 0.201915 |

**Önemli:** Zero-shot aday açıklamaları Yelp'in 1–5 yıldızlarının birebir karşılığı değildir. Bu yüzden bu tablodan yıldız Accuracy/F1 metriği çıkarılmadı.

### Text Classification ve Zero-Shot karşılaştırması

| Özellik | Text Classification | Zero-Shot |
|---|---|---|
| Model | `nlptown/bert-base-multilingual-uncased-sentiment` | `typeform/distilbert-base-uncased-mnli` |
| Önceden eğitilmiş mi? | Evet, yıldız/sentiment sınıflandırması için | Evet, MNLI / NLI için |
| Çalışma sırasında aday etiket veriliyor mu? | Hayır, sabit sınıfları var | Evet, `candidate_labels` veriliyor |
| Çıktı biçimi | `[{'label': '4 stars', 'score': ...}]` | `{'sequence': ..., 'labels': [...], 'scores': [...]}` |
| Yelp'e özel fine-tuning yapıldı mı? | Hayır | Hayır |

## 10. Pilot Observations

- Hazır yıldız modeli, beş örnekle sınırlı 3-yıldız grubunda zorlandı; bu durum hem olumlu hem olumsuz ifadelerin aynı yorumda bulunmasıyla ilişkili olabilir.
- Bazı hazır model tahminleri görece yüksek skorla yanlış çıktı. `score`, genel doğruluk yüzdesi veya kalibre edilmiş güven garantisi değildir.
- Zero-shot pilotunda bir 1-yıldız yoruma `positive`, bir 5-yıldız yoruma `very negative` etiketi verildi. Bununla birlikte bu etiketler doğrudan yıldız sınıfları sayılmamalıdır.
- Bazı zero-shot skorları beş aday arasında yaklaşık `0.20` düzeyindeydi; bu, o örneklerde güçlü bir ayrım görülmediğine işaret ediyor.
- Aday etiketlerin ifadesi, metnin uzunluğu ve eğitim alanı ile Yelp yorumu arasındaki farklar sonuçları etkileyebilir.

## 11. Tests

`assert` ile üç otomatik kontrol çalıştırıldı. Notebook çıktısı:

```text
Test 1 — Label Mapping: PASSED
Test 2 — Balanced Sample: PASSED
Test 3 — Valid Text: PASSED
```

- **Test 1:** Etiket kümesi `{0,1,2,3,4}`; `stars = label + 1`; yıldızlar 1–5 aralığında.
- **Test 2:** Her yıldız kategorisinde 200 yorum.
- **Test 3:** Bütün metinler string ve `strip()` sonrasında boş değil.

## 12. What I Learned

Bu çalışmada `streaming` kullanarak büyük bir veri setinin tamamını belleğe almadan dengeli pilot örneklem oluşturmayı öğrendim. `label` ile gerçek yıldızların aynı şey olmadığını doğruladım; eksik kayıt, tekrar ve yorum uzunluğu kontrolleri yaptım. Hugging Face pipeline'ın tokenizer, inference ve çıktı hazırlama işlemlerini kolaylaştırdığını gördüm. Sabit etiketlerle çalışan sentiment modeli ile aday etiketleri çalışma sırasında verdiğim zero-shot modeli arasındaki farkı uygulamalı olarak anladım. Yüksek bir `score` değerinin tahminin kesin doğru olduğu anlamına gelmediğini ve sınırlı pilot örneklerinden genelleme yapmamam gerektiğini öğrendim. Modellerin ağırlıklarını değiştirmedim; fine-tuning ve train/validation bölünmesi yapmadım.

## Kaynaklar

- Yelp Review Full veri kartı: https://huggingface.co/datasets/Yelp/yelp_review_full/blob/main/README.md
- Yelp kullanım sözleşmesi (veri kartındaki bağlantı): https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf
- Sentiment model kartı: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
- Zero-shot model kartı: https://huggingface.co/typeform/distilbert-base-uncased-mnli
