# ReviewLens — Data Card

**Veri seti:** Yelp Review Full · **Yerel pilot dosya:** `data/yelp_sample_1000.csv` · **Tarih:** 9 Kasım 2026

## 1. Veri kaynağı ve amaç

Yelp Review Full, Yelp yorumlarından derlenen beş sınıflı, ağırlıklı olarak İngilizce bir metin sınıflandırma veri setidir. Bu projede amaç, yorum metninden yıldız değerlendirmesi tahmin etmeye yönelik NLP araçlarını eğitim amaçlı denemektir.

- Veri kartı: https://huggingface.co/datasets/Yelp/yelp_review_full/blob/main/README.md
- Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full
- Orijinal `train`: **650.000 yorum**
- Orijinal `test`: **50.000 yorum**
- Örneklem: Yalnızca **train** bölümünden **1.000 yorum**. Test bölümüne dokunulmadı.

## 2. Veri yapısı ve etiketler

| Alan | Tür | Açıklama |
|---|---|---|
| `text` | String | Müşteri yorumunun metni |
| `label` | Integer | Modelleme etiketi: 0, 1, 2, 3, 4 |
| `stars` | Integer | Kullanıcı dostu yıldız: 1, 2, 3, 4, 5 |

**Dönüşüm:** `stars = label + 1`

| `label` | `stars` |
|---:|---:|
| 0 | 1 |
| 1 | 2 |
| 2 | 3 |
| 3 | 4 |
| 4 | 5 |

## 3. Örneklem yöntemi

Veri, Hugging Face `datasets` ile **streaming** olarak okundu. Yalnızca eğitim bölümü `shuffle(seed=42, buffer_size=10000)` kullanılarak tampon üzerinden karıştırıldı. Her etiketten ilk 200 uygun kayıt alındı; bütün sayaçlar 200 olunca işlem sonlandırıldı. Tam küresel shuffle yapılmadığı için bu yöntem tam rastgele örnekleme ile aynı değildir.

| Yıldız | Sayı | Örneklem payı |
|---|---:|---:|
| 1 | 200 | %20 |
| 2 | 200 | %20 |
| 3 | 200 | %20 |
| 4 | 200 | %20 |
| 5 | 200 | %20 |
| **Toplam** | **1.000** | **%100** |

Yerel çalışma dosyası `data/yelp_sample_1000.csv` olarak kaydedildi. Henüz train/validation ayrımı yapılmadı.

## 4. Veri kalitesi

| Ölçüm | Sonuç |
|---|---:|
| Toplam kayıt | 1.000 |
| Boş/whitespace-only metin | 0 |
| Eksik `label` | 0 |
| Birebir tekrarlanan `text` | 0 |
| Ortalama kelime sayısı | 131,20 |
| Minimum kelime sayısı | 2 |
| Maksimum kelime sayısı | 847 |

Kelime sayıları boşluk bazlı `str.split()` ile bulundu. Duplicate kontrolü metnin birebir aynı olup olmadığına bakar; farklı yazılmış benzer yorumları dışlamaz. `assert` tabanlı label mapping, sınıf dengesi ve geçerli metin testlerinin üçü de başarıyla geçti.

## 5. Lisans ve kullanım sınırları

Hugging Face'te lisans etiketi `other` olarak görünür; veri kartı Yelp Dataset Terms of Use sözleşmesine yönlendirir:

https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf

Bu bağlantıdaki sözleşme akademik/ticari olmayan kullanımı ve çeşitli paylaşım kısıtlarını tanımlar. Özellikle ham Yelp yorumlarının üçüncü taraflarla paylaşımı ve kamuya açık sonuçların yayımlanması konusunda sözleşmedeki şartlar kontrol edilmelidir. **Lisansın izin verdiği açıkça doğrulanmadan `data/yelp_sample_1000.csv` veya yorum içeren notebook çıktıları herkese açık GitHub deposuna yüklenmemelidir.** Bu kart lisans izni sağlamaz; bağlayıcı metin ilgili sözleşmedir.

## 6. Olası sınırlamalar ve riskler

- **Küçük pilot:** 1.000 kayıt büyük veri setinin küçük bir bölümüdür; güvenilir üretim performansı göstermez.
- **Yapay sınıf dengesi:** Her yıldız %20 seçildi; gerçek kullanıcı yorumlarının dağılımını temsil etmeyebilir.
- **Streaming shuffle:** `buffer_size=10000` bütün eğitim verisini tam rastgele karıştırmaz.
- **Tarih ve alan değişimi:** Kaynak, eski bir Yelp veri kümesinden türetilmiştir; güncel yorumlara ve farklı alanlara genelleme belirsizdir.
- **Öznel etiketler:** Benzer içerikli yorumlar farklı yıldız puanları alabilir; özellikle orta yıldızlarda anlam örtüşmesi vardır.
- **Yorum uzunluğu:** 2–847 kelime aralığı, modellerin token sınırlarında kesilmeye neden olabilir.
- **Tekrar kontrolü kapsamı:** Yalnızca birebir metin tekrarları incelendi; tüm orijinal veri ile kesişim veya anlamsal tekrar analizi yapılmadı.
- **Veri gizliliği ve telif:** Yorum metinleri kullanıcı içeriğidir; paylaşımda kaynak lisansı ve ilgili kısıtlar dikkate alınmalıdır.

## 7. Bu aşamada yapılmayanlar

Orijinal test bölümünün açılması, train/validation ayrımı, model eğitimi, fine-tuning, hyperparameter tuning ve üretim performansı değerlendirmesi yapılmadı.

## Kaynaklar

- Dataset card: https://huggingface.co/datasets/Yelp/yelp_review_full/blob/main/README.md
- Yelp Dataset Terms of Use: https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf
