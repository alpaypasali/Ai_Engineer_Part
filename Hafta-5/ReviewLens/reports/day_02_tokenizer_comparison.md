# ReviewLens — Day 02: Tokenizer Comparison

**Proje:** ReviewLens  
**Hafta:** 5  
**Çalışma:** İki hazır Hugging Face tokenizer'ının Yelp yorumları üzerinde karşılaştırılması  
**Veri:** `data/yelp_sample_1000.csv` (1.000 dengeli yorum, yalnızca orijinal `train` bölümünden)

## 1. Problem

ReviewLens, İngilizce Yelp müşteri yorumlarını 1–5 yıldızlı değerlendirmeler bağlamında inceleyen bir NLP öğrenme projesidir. Bugünkü amaç bir model eğitmek veya tahmin başarısını artırmak değil; farklı tokenizer'ların aynı metinleri nasıl tokenlara dönüştürdüğünü, token uzunluklarının truncation, padding ve teorik attention maliyeti üzerindeki etkilerini anlamaktır. **Model inference veya fine-tuning yapılmadı.**

## 2. Tokenizer A

- **Model:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Mimari ailesi:** Multilingual BERT
- **Tokenization yaklaşımı:** WordPiece
- **Vocabulary size:** 105.879
- **Model maksimum giriş uzunluğu:** 512 token
- **Rolü:** Önceki gün 1–5 yıldız tahmini için kullanılan hazır sentiment modelinin tokenizer'ı.

## 3. Tokenizer B

- **Model:** `typeform/distilbert-base-uncased-mnli`
- **Mimari ailesi:** DistilBERT
- **Tokenization yaklaşımı:** WordPiece
- **Vocabulary size:** 30.522
- **Model maksimum giriş uzunluğu:** 512 token
- **Rolü:** Önceki gün aday etiketlerle zero-shot classification için kullanılan NLI modelinin tokenizer'ı.

İki tokenizer farklı algoritmalar arasında değil, **WordPiece kullanan iki farklı vocabulary/model yapılandırması** arasında karşılaştırma yapmamızı sağladı.

## 4. Vocabulary

| Özellik | Tokenizer A | Tokenizer B |
|---|---:|---:|
| Vocabulary size | 105.879 | 30.522 |
| Model maximum length | 512 | 512 |

Vocabulary büyüklüğü tek bir yorumda üretilecek token sayısını doğrudan belirlemez. Metindeki kelimelerin o tokenizer'ın sözlüğünde nasıl temsil edildiği önemlidir. Örnek cümlede Tokenizer A **12**, Tokenizer B **10** normal token üretti; `[CLS]` ve `[SEP]` eklendiğinde giriş uzunlukları sırasıyla **14** ve **12** oldu. Aynı sözcük farklı vocabulary'lerde farklı alt parçalara ayrılabilir. Token ID'leri de tokenizer'a özgüdür; iki modelin tokenizer'larını doğrulama olmadan birbirinin yerine kullanmak doğru değildir.

## 5. Special Tokens

| Token | İşlevi | A | B |
|---|---|---|---|
| `[CLS]` | BERT tabanlı sınıflandırma girdisinin başlangıcı | Var | Var |
| `[SEP]` | Metin bölümlerini ayırma / dizinin sonunu belirtme | Var | Var |
| `[PAD]` | Kısa dizileri gerekli uzunluğa tamamlama | Var | Var |
| `[UNK]` | Vocabulary ile temsil edilemeyen girdi | Var | Var |

Token uzunlukları **özel tokenlar dâhil**, `truncation=False` ile hesaplandı.

## 6. Token Length Analysis

`df["text"]` içindeki 1.000 yoruma iki tokenizer uygulandı; hiçbir yorum bu ölçüm sırasında kesilmedi. Kelime sayısı boşluğa göre bölme (`str.split().str.len()`) ile bulundu.

| Metrik | Tokenizer A | Tokenizer B |
|---|---:|---:|
| Minimum token | 5 | 5 |
| Ortalama token | 184,50 | 175,48 |
| Medyan token | 137 | 130,50 |
| Maksimum token | 1.230 | 1.163 |
| P90 | 383 | 367,40 |
| P95 | 500,15 | 467,15 |
| P99 | 858 | 801,20 |

**Ortalama kelime sayısı:** 131,20. Tokenizer A, yorum başına ortalama yaklaşık **9,02 token** daha fazla üretti. Token sayısı, kelime sayısından farklıdır; özellikle alt kelime parçalama nedeniyle kısa görünen ifadeler birden fazla tokendan oluşabilir.

## 7. Token Distribution

Notebook'ta iki grafik oluşturuldu:

1. **Token Length Distribution:** İki tokenizer'ın token uzunluğu histogramları aynı grafikte karşılaştırıldı. Dağılımlar sağa çarpık; yorumların çoğu görece kısa, az sayıdaki yorum ise çok uzun.
2. **Word Count vs Token Count:** Kelime sayısı ile token sayısı arasında belirgin pozitif ilişki gözlendi. Aynı kelime sayısı her iki tokenizer'da aynı token sayısını garanti etmiyor.

Grafikler `notebooks/03_tokenizer_comparison.ipynb` içindeki ilgili hücrelerde görüntülendi. Ayrı PNG dosyaları kaydedildiyse `figures/` klasörüne eklenebilir. **Ham yorum içeren grafik veya çıktı yayımlanmamalıdır.**

## 8. Truncation

`truncation=True`, `max_length=128` ve `max_length=256` seçeneklerinin etkileri, 1.000 yoruma ait orijinal token uzunluklarından hesaplandı. Kesilen token sayısı `max(orijinal_uzunluk - sınır, 0)` formülüne dayanıyor.

| Metrik | Tokenizer A | Tokenizer B |
|---|---:|---:|
| 128'i aşan yorum sayısı | 532 (%53,2) | 509 (%50,9) |
| 256'yı aşan yorum sayısı | 227 (%22,7) | 206 (%20,6) |
| 128 sınırında ortalama kesilen token | 83,96 | 76,65 |
| 256 sınırında ortalama kesilen token | 37,77 | 33,10 |

Ortalama kayıp **1.000 yorumun tamamı** üzerinden hesaplandı; kesilmeyen yorumların kaybı sıfır alındı. Seçilen kısa, orta ve uzun birer yorumda 128 sınırı da gerçek tokenization ile doğrulandı. 128 token daha fazla içeriği keserken 256 token daha fazla içeriği korudu. Özellikle yorum sonundaki olumsuz ifadelerin kesilmesi yıldız tahmini için anlam kaybı yaratabilir; **bu etki model skorlarıyla ölçülmedi**.

## 9. Padding

- **Fixed Padding:** `padding="max_length"`; örneğin 128 uzunluğuna kadar kısa girdiler `[PAD]` ile tamamlanır.
- **Dynamic Padding:** `padding=True`; bir batch içindeki en uzun girdiye göre tamamlama yapılır.

128'e sabit padding deneyinde seçilen kısa yorum için sonuçlar:

| Ölçüm | Tokenizer A | Tokenizer B |
|---|---:|---:|
| Gerçek token | 22 | 19 |
| `[PAD]` sayısı | 106 | 109 |
| Toplam `input_ids` uzunluğu | 128 | 128 |

256'ya sabit padding uygulansaydı aynı kısa yorumlarda padding sayıları sırasıyla **234** ve **237** olurdu. Dynamic Padding, değişken uzunluklu batch'lerde gereksiz padding miktarını azaltabilir.

## 10. Attention Mask

Hugging Face tokenizer çıktısında `attention_mask` genellikle `1 = gerçek token`, `0 = padding` anlamına gelir. Her iki tokenizer için 128 uzunluğunda `input_ids` ve `attention_mask` üretildi. `assert` ile tüm padding konumlarının maske değeri `0`, gerçek token konumlarının `1` olduğu doğrulandı.

NewsRoute projesindeki PyTorch `src_key_padding_mask` gösterimi farklıdır: `True = maskelenecek PAD`. **Kavram aynı, API gösterimi farklıdır.**

## 11. Memory Cost

Standart self-attention'da her head için attention skor matrisi `L × L` boyutundadır:

| Özellik | 128 token | 256 token |
|---|---:|---:|
| Attention skor matrisi | 128 × 128 | 256 × 256 |
| Matris eleman sayısı | 16.384 | 65.536 |
| Göreli teorik skor-matrisi maliyeti | 1× | 4× |
| Truncation riski | Daha yüksek | Daha düşük |
| Sabit padding miktarı | Genellikle daha az | Genellikle daha fazla |

Giriş uzunluğu iki katına çıktığında attention skor matrisindeki eleman sayısı **4 katına** çıkar: `65.536 / 16.384 = 4`. Bu, **tüm model belleğinin veya gerçek çalışma süresinin mutlaka 4 katına çıkacağı** anlamına gelmez. GPU benchmark'ı yapılmadı.

## 12. Tests

`03_tokenizer_comparison.ipynb` sonunda, iki tokenizer'a uygulanan üç `assert` tabanlı kontrol çalıştırıldı. Kullanıcının çalıştırma sonucuna göre **üç test de geçti**.

1. **Tokenization — PASSED:** 1.000 yorumda boş olmayan, geçerli token ID listeleri.
2. **Truncation — PASSED:** Her iki tokenizer için `max_length=128` sonrası giriş uzunlukları ≤ 128 ve en az bir uzun yorumda kesilme.
3. **Padding Mask — PASSED:** 128 tokenlık dizilerde `[PAD]` ile attention mask eşleşmesi.

Bu testler tokenization ve giriş hazırlama davranışını doğrular; model tahmini veya gerçek dünya başarısını ölçmez.

## 13. What I Learned

Bu çalışmada metindeki kelime sayısının modelin gördüğü token sayısına eşit olmadığını uygulamalı olarak gördüm. İki WordPiece tokenizer aynı İngilizce yorumları farklı parçalara ayırdı; daha büyük vocabulary, otomatik olarak daha kısa token dizisi üretmedi. Token ID'lerinin modele ait vocabulary ile uyumlu olması gerektiğini anladım.

Truncation ile Padding'in farklı amaçları olduğunu öğrendim: Truncation uzun girdiyi keserken Padding kısa girdiyi tamamlıyor. Attention Mask'in hangi konumların gerçek metin, hangilerinin PAD olduğunu bildirdiğini testlerle doğruladım. Ayrıca 128 ve 256 token arasında yalnızca hız değil, **korunan bilgi miktarıyla teorik hesaplama maliyeti arasında bir denge** bulunduğunu gördüm. Sonuçları model eğitmeden ve inference yapmadan, veri hazırlama düzeyinde inceledim.

## Data Source & Sharing Note

- Dataset kartı: https://huggingface.co/datasets/Yelp/yelp_review_full/blob/main/README.md
- Dataset: https://huggingface.co/datasets/Yelp/yelp_review_full
- İlgili lisans: Yelp Dataset Terms of Use (dataset kartındaki `other / yelp-licence` bilgisi).

Ham Yelp yorumları ve bu yorumların notebook içindeki metin çıktıları kamuya açık GitHub'a eklenmemelidir. Türetilmiş CSV'de ham metin yer almaz; **ancak sayısal türetilmiş sonuçların ve raporların kamuya açık paylaşım izni de sözleşme hükümleri açısından ayrıca kontrol edilmelidir.** Gerekli izin teyidi olmadan raporu veya CSV'yi yayımlamayın.
