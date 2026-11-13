# ReviewLens — Model Candidate Cards

**Tarih:** 13 Kasım 2026  
**Proje:** ReviewLens / Hafta 5  
**Karşılaştırma görevi:** Yelp yorumlarını 1–5 yıldız arasında sınıflandırma  
**Değerlendirme:** Aynı sırada, her yıldızdan 40 örnek olmak üzere 200 dengeli Yelp Review Full yorumu.  
**İnference:** Önceden eğitilmiş hazır modeller; ek fine-tuning yapılmadı.

> **Kaynak ayrımı:** Lisans, mimari, eğitim amacı ve dil bilgileri Hugging Face model kartlarından; performans ve latency değerleri ReviewLens notebook'unda kullanıcının ölçtüğü sonuçlardan alınmıştır. Aşağıdaki 200 örneklik ölçümler genel ürün/üretim performansını kanıtlamaz.

## Kısa karşılaştırma

| Özellik | Model A — NLP Town | Model B — TabularisAI |
| --- | --- | --- |
| Model ID | `nlptown/bert-base-multilingual-uncased-sentiment` | `tabularisai/multilingual-sentiment-analysis` |
| Architecture | Multilingual BERT (uncased) | Multilingual DistilBERT (cased) |
| Task | Ürün yorumlarında 1–5 yıldız tahmini | Beş seviyeli duygu sınıflandırması |
| Languages | 6 dil (EN, NL, DE, FR, ES, IT) | 23 dil (İngilizce ve Türkçe dahil) |
| Output Labels | `1 star`, `2 stars`, `3 stars`, `4 stars`, `5 stars` | `Very Negative`, `Negative`, `Neutral`, `Positive`, `Very Positive` |
| Label Mapping | 1–5 yıldız etiketi doğrudan 1–5 sayısına çevriliyor | Very Negative→1, Negative→2, Neutral→3, Positive→4, Very Positive→5 (**değerlendirme varsayımı**) |
| Evaluation Samples | 200; her sınıfta 40 | 200; **aynı örnekler ve aynı sıra** |
| Accuracy | **0.5550** | 0.4200 |
| Macro-F1 | **0.5532** | 0.3827 |
| 1-Star Recall | **0.6000** | 0.5000 |
| 1-Star Error Rate | **0.4000** | 0.5000 |
| Mean Latency | 16.76 ms | **7.16 ms** |
| P95 Latency | 32.26 ms | **14.61 ms** |
| License (model kartı) | **MIT** | **CC BY-NC 4.0** |
| Ticari kullanım | MIT koşulları altında genel olarak mümkün | CC BY-NC koşulları kapsamında ticari kullanım izni verilmez; ticari senaryoda ayrıca izin/lisans gerekebilir |

## Model A — NLP Town Multilingual BERT

### Kimlik ve amaç

- **Model ID:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Architecture:** `bert-base-multilingual-uncased` tabanlı BERT.
- **Task:** Ürün değerlendirme metinlerinden 1–5 yıldız sınıflandırması.
- **Languages:** İngilizce, Felemenkçe, Almanca, Fransızca, İspanyolca, İtalyanca.
- **Output Labels:** `1 star`, `2 stars`, `3 stars`, `4 stars`, `5 stars`.
- **Label Mapping:** Model etiketindeki yıldız sayısı 1–5 aralığında tamsayıya çevrilir. Bu modelde etiketler doğrudan yıldız kategorilerini ifade eder.

### ReviewLens değerlendirmesi

- **Evaluation Samples:** 200 dengeli Yelp yorumu; her yıldızdan 40 örnek.
- **Accuracy:** 0.5550.
- **Macro-F1:** **0.5532**.
- **1-Star Recall:** 0.6000 (40 gerçek 1 yıldızdan 24 doğru).
- **1-Star Error Rate:** **0.4000** (40 gerçek 1 yıldızdan 16 yanlış).
- **Mean Latency:** **16.76 ms**.
- **P95 Latency:** **32.26 ms**.
- **License:** **MIT**. Ticari kullanım genel olarak mümkündür; telif ve lisans bildirimlerinin korunması dahil lisans şartlarına uyulmalıdır. Model lisansı, Yelp veri seti için otomatik bir kullanım/paylaşım hakkı vermez.

### Limitations (sınırlamalar)

1. Model ürün değerlendirmeleri üzerinde fine-tune edilmiştir; doğrudan Yelp yorumları için uyarlanmış değildir. Alan/dil/üslup farklılığı performansı etkileyebilir.
2. Özellikle orta yıldız kategorilerinde hatalar görülmüştür; örneğin 3 yıldız Recall yaklaşık 0.375'tir.
3. `truncation=True`, `max_length=256` nedeniyle uzun yorumların bir bölümü kesilebilir; kritik ifadeler kaybolabilir.
4. Sadece 200 dengeli örnek üzerinde değerlendirilmiştir; gerçek kullanım dağılımı, bağımsız test ve alt grup analizleri yapılmamıştır.
5. `score` çıktısı kalibre edilmiş doğruluk olasılığı kabul edilmemelidir.

### Risks (riskler)

- Olumsuz müşteri deneyimlerini olduğundan olumlu gösteren yıldız tahminleri kullanıcı güvenini zedeleyebilir.
- 1 yıldızlı yorumlarda %40 hata, olumsuz deneyimlerin önemli kısmının gözden kaçmasına neden olabilir.
- Yanlış sınıflandırma, restoran kalite izleme ve önceliklendirme kararlarını etkileyebilir.
- Üretim kullanımından önce veri işleme, lisans ve gerçek dağılım performansı ayrıca denetlenmelidir.

## Model B — TabularisAI Multilingual Sentiment

### Kimlik ve amaç

- **Model ID:** `tabularisai/multilingual-sentiment-analysis`
- **Architecture:** `distilbert/distilbert-base-multilingual-cased` tabanlı DistilBERT.
- **Task:** Beş sınıflı çok dilli duygu analizi; model kartında müşteri geri bildirimleri, ürün yorumları ve sosyal medya gibi kullanım alanları listeleniyor.
- **Languages:** Model kartına göre 23 dil, İngilizce ve Türkçe dahil.
- **Output Labels:** `Very Negative`, `Negative`, `Neutral`, `Positive`, `Very Positive`.
- **Label Mapping:** Very Negative→1; Negative→2; Neutral→3; Positive→4; Very Positive→5.
- **Mapping uyarısı:** Bu eşleştirme **matematiksel veya anlamsal bir eşdeğerlik değil**, Yelp'in 1–5 yıldız etiketleriyle karşılaştırma için kullanılan bir değerlendirme varsayımıdır.

### ReviewLens değerlendirmesi

- **Evaluation Samples:** Model A ile aynı 200 dengeli Yelp yorumu, aynı `row_id` ve etiket sırası doğrulanmış.
- **Accuracy:** 0.4200.
- **Macro-F1:** **0.3827**.
- **1-Star Recall:** 0.5000 (40 gerçek 1 yıldızdan 20 doğru).
- **1-Star Error Rate:** **0.5000** (40 gerçek 1 yıldızdan 20 yanlış).
- **Mean Latency:** **7.16 ms**.
- **P95 Latency:** **14.61 ms**.
- **License:** **CC BY-NC 4.0**. Lisans ticari olmayan kullanımları kapsar ve atıf koşulları içerir. Ticari bir restoran platformunda kullanım için hak sahibinden ayrıca uygun izin/lisans alınması gerekebilir.

### Limitations (sınırlamalar)

1. Duygu etiketleri doğrudan yıldız değildir; Very Negative→1 vb. varsayımlar Yelp gerçek etiketleriyle her zaman örtüşmeyebilir. Hataların tümü yalnızca model mimarisine bağlanamaz.
2. **CC BY-NC 4.0** nedeniyle ticari kullanım lisans açısından kısıtlıdır.
3. Yelp dağılımındaki performans, yalnızca 200 örnekli geliştirme değerlendirmesinde incelenmiştir; bağımsız ve temsili değerlendirme gerekir.
4. `max_length=256` ve `truncation=True` uzun yorumlarda bilgi kesilmesine yol açabilir. İki modelin tokenizer'ları farklı olduğu için aynı token sınırı tam olarak aynı metin parçalarının görüldüğü anlamına gelmez.
5. Çıktı `score` değeri kalibre edilmiş güven olasılığı olarak kabul edilmemelidir.

### Risks (riskler)

- 1 yıldızlı yorumların %50'si yanlış sınıflandırıldığı için olumsuz müşteri deneyimleri gözden kaçabilir.
- Duygu etiketleriyle yıldızların zorunlu eşleştirilmesi, Yelp'e özgü beklentilerde yanıltıcı sonuç üretebilir.
- Daha hızlı inference, düşük sınıflandırma kalitesini veya ticari lisans kısıtını tek başına telafi etmez.
- Ticari kullanıma gerekli izni almadan başlamak hukuki uyum riski oluşturabilir.

## Inference ölçüm koşulları ve sınırlamalar

- **Değerlendirme ayarları:** Her model için 200 örnek, `batch_size=8`, `max_length=256`, `truncation=True`, fine-tuning yok.
- **Latency ölçümü:** Her sınıftan 4 yorum (toplam 20); iki modelde aynı metinler ve aynı sıra; `batch_size=1`; GPU `cuda:0`; önceden yüklenen modeller; ısınma çalıştırması ve CUDA senkronizasyonu kullanıldı.
- **Ölçülen metrikler:** Her yorum için süre; bunlardan ortalama, medyan ve P95 hesaplandı.
- **Önemli sınırlama:** 20 tekil ölçüm üretim ortamının güvenilir P95 değerini temsil etmez. Donanım ve sistem yükü değiştikçe hızlar değişebilir. Model indirme/yükleme süresi tekil inference latency'sine dahil edilmedi.

## Lisans ve veri paylaşımı kararı

- **Model A (MIT):** Ticari kullanım için daha elverişli; yine de lisans bildirimleri ve diğer yasal yükümlülükler kontrol edilmeli.
- **Model B (CC BY-NC 4.0):** Ticari restoran platformunda doğrudan kullanım varsayılmamalı. Hak sahibiyle ayrı lisans/izin görüşmesi gerekebilir; yalnızca yüksek performans ticari kullanım hakkı sağlamaz.
- **Yelp Review Full:** Hugging Face veri kartı lisansı `other` olarak işaretliyor ve resmi Yelp veri seti anlaşmasına yönlendiriyor. Model lisansları Yelp metinlerinin kopyalanmasına, dağıtılmasına veya herkese açık GitHub'a yüklenmesine otomatik izin vermez. Ham yorumlar ve paylaşım yetkisi doğrulanmamış türetilmiş veri çıktıları public repoya yüklenmemeli. Notebook kayıtlı hücre çıktıları da gözden geçirilmeli.

## Model seçimi — şu anki karar

**Ön değerlendirme: Model A (NLP Town BERT).** Aynı 200 Yelp yorumunda daha yüksek Macro-F1 (`0.5532` vs. `0.3827`) ve daha düşük 1-Star Error Rate (`%40` vs. `%50`) elde etti. Model B daha hızlıdır (ortalama `7.16` vs. `16.76` ms), ancak bu hız avantajı ReviewLens'in olumsuz deneyimleri kaçırmama önceliğini tek başına karşılamaz. Ayrıca Model B'nin CC BY-NC 4.0 lisansı ticari senaryo açısından önemli bir engeldir.

**Karar koşulludur:** Gerçek ürün performansı, hız gereksinimleri, bağımsız test, veri seti izinleri ve yasal inceleme tamamlanmadan üretime alma onayı verilmemelidir.

## Kaynaklar

- NLP Town Hugging Face model kartı: https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment
- TabularisAI Hugging Face model kartı: https://huggingface.co/tabularisai/multilingual-sentiment-analysis
- Yelp Review Full veri kartı: https://huggingface.co/datasets/Yelp/yelp_review_full
- Creative Commons CC BY-NC 4.0 açıklaması: https://creativecommons.org/licenses/by-nc/4.0/

> Bu belge hukuki görüş değildir; lisans izinleri ve kapsamı kullanım senaryosuna göre doğrulanmalıdır.
