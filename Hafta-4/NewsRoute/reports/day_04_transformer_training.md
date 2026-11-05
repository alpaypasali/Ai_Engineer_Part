# NewsRoute — Gün 4: Transformer Classifier Eğitimi

## Dataset

- **Veri kümesi:** AG News, dört sınıflı haber sınıflandırması
- **Eğitim (train):** 3.200 haber (her sınıftan 800)
- **Doğrulama (validation):** 800 haber (her sınıftan 200)
- **Sınıflar:** `0=World`, `1=Sports`, `2=Business`, `3=Sci/Tech`
- Mevcut train/validation ayrımı korundu; orijinal AG News test kümesi kullanılmadı.

## Tokenization

Haber metinleri küçük harfe dönüştürülüp boşluk karakterlerine göre ayrıldı:

```python
def tokenize(text):
    return text.lower().split()
```

Bu basit whitespace tokenizer noktalama işaretlerini ayrıca temizlemez. Örneğin `changing.` ve `changing` farklı token'lardır.

## Vocabulary

- Vocabulary yalnızca **training** metinlerinden oluşturuldu; validation verisinden kelime eklenmedi.
- **Toplam vocabulary boyutu:** **7.787** (özel token'lar dahil).
- **Özel token'lar:** `[PAD] = 0`, `[UNK] = 1`.
- Training sözlüğünde olmayan token'lar `[UNK]` ID'sine dönüştürüldü.

## Padding

- **`MAX_LENGTH = 64`**.
- Uzun haberler 64 token'a kısaltıldı (*truncation*).
- Kısa haberler `[PAD]` ile 64 uzunluğa tamamlandı (*padding*).
- Giriş ve hedef Tensor'ları: `X_train (3200,64)`, `y_train (3200,)`, `X_val (800,64)`, `y_val (800,)`; veri tipi `torch.int64`.

## Model Architecture

`NewsRouteTransformer(nn.Module)` mimarisi:

`Token IDs → Embedding → Sinusoidal Positional Encoding → Transformer Encoder → Masked Mean Pooling → Linear Classifier → 4 logits`

| Model parametresi | Değer |
|---|---:|
| `max_length` | 64 |
| `d_model` | 64 |
| `num_heads` | 4 |
| `num_layers` | 1 |
| `dim_feedforward` | 128 |
| `dropout` | 0.1 |
| `num_classes` | 4 |
| `batch_size` | 32 |
| `epochs` | 3 |
| `learning_rate` | 0.001 |
| `seed` | 42 |

### Mimari ve Tensor şekilleri

| Katman | Tensor Shape |
|---|---|
| Input IDs | `(32, 64)` |
| Token Embedding | `(32, 64, 64)` |
| Positional Encoding eklenmiş giriş | `(32, 64, 64)` |
| Transformer Encoder Output | `(32, 64, 64)` |
| Masked Pooling | `(32, 64)` |
| Classifier Logits | `(32, 4)` |

> İlk boyut batch içindeki haber sayısıdır (32); son logit boyutu dört sınıfa karşılık gelir. Logits henüz olasılık değildir.

## Embedding

`nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)` ile her token ID'si **64 boyutlu öğrenilebilir bir vektöre** dönüştürüldü. Embedding ağırlık matrisi `(7787, 64)` boyutundadır. `padding_idx=0`, PAD embedding'inin öğrenilmesini engeller; attention ve pooling maskelerinin yerini tutmaz.

## Positional Encoding

Çarşamba günü çalışılan sinüs/kosinüs tabanlı **sabit positional encoding** kullanıldı. `(64,64)` boyutlu PE, `register_buffer("pe", ...)` ile öğrenilebilir parametre olmadan modele kaydedildi ve token embedding'lerine eklendi. Bu bilgi token'ların sıra konumlarını temsil eder.

## Attention Mask

- `padding_mask = input_ids == PAD_ID` ile maske üretildi.
- `padding_mask` bir Boolean Tensor'dır ve şekli `(batch_size, seq_len)` biçimindedir.
- PyTorch `src_key_padding_mask` için **`True = maskelenecek PAD konumu`** anlamına gelir.
- Maske `self.encoder(src=x, src_key_padding_mask=padding_mask)` çağrısına verildi.

Kontrol edilen örnek batch'te:

| Kontrol | Sonuç |
|---|---|
| Batch Shape | `(32, 64)` |
| Toplam token konumu | 2.048 |
| Padding Token Count | 1.272 |
| Valid Token Count | 776 |
| Padding Mask Shape | `(32, 64)` |
| PAD ID | `0` |
| Boolean / şekil / PAD işaretleme kontrolleri | `True` |

## Masked Pooling

Encoder çıktısı her token için bir vektör içerdiğinden `(32,64,64)` boyutundadır. Haber başına tek vektör elde etmek için geçerli token'ların ortalaması alındı:

```python
valid_mask = input_ids != PAD_ID
mask = valid_mask.unsqueeze(-1).float()
masked_x = x * mask
token_sum = masked_x.sum(dim=1)
token_count = mask.sum(dim=1).clamp(min=1)
pooled = token_sum / token_count
```

PAD konumları toplama katılmaz; `.clamp(min=1)` sıfıra bölünmeyi önler. Son pooling çıktısı `(32,64)` boyutundadır.

## Training

- **Loss:** `nn.CrossEntropyLoss()`; loss öncesinde `softmax()` uygulanmadı.
- **Optimizer:** `torch.optim.AdamW(model.parameters(), lr=0.001)`.
- **3 epoch**, her biri **100 batch** (`3200 / 32`).
- Her batch'te: `zero_grad → forward → loss → backward → optimizer.step`.
- Model eğitim sonuçlarına göre hiperparametreler değiştirilmedi.

| Epoch | Train Loss |
|---|---:|
| 1 | 1.2868 |
| 2 | 0.8547 |
| 3 | 0.5855 |

Train Loss üç epoch boyunca azaldı. Bu, eğitim verisindeki optimizasyonun ilerlediğini gösterir; tek başına genelleme performansını kanıtlamaz.

## Training Time

**Donanım:** CPU. Zaman ölçümü, eğitilmiş ve değerlendirilmiş asıl modelden ayrı tutulan aynı ayarlardaki bir eğitim koşusunda yapıldı; model ağırlıklarını değiştirmek amacıyla kullanılmadı.

| Epoch | Train Loss | Süre (saniye) |
|---|---:|---:|
| 1 | 1.2868 | 1.0514 |
| 2 | 0.8547 | 1.0007 |
| 3 | 0.5855 | 0.9952 |
| **Toplam** | — | **3.05** |

Epoch süreleri birbirine yakındı; ilk epoch az miktarda daha uzun sürdü. Süreler ölçüm koşullarına bağlıdır.

## Validation

Eğitim sonunda model `model.eval()` ve `torch.no_grad()` kullanılarak **800 validation haberi** üzerinde değerlendirildi.

| Metrik | Sonuç |
|---|---:|
| Validation Accuracy | **0.7137** |
| **Validation Macro-F1 (ana metrik)** | **0.7108** |
| Validation Loss | **0.7518** |

Son epoch Train Loss `0.5855`, Validation Loss `0.7518` oldu. Ayrı validation verisinde kayıp daha yüksek; ancak tek başına kesin bir aşırı öğrenme teşhisi yapılmadı. Pazartesi günkü TF-IDF modeli ile detaylı karşılaştırma ve confusion matrix **cuma gününe** bırakıldı.

## Tests

Üç otomatik test notebook'ta çalıştırıldı ve başarılı oldu:

| Test | Sonuç |
|---|---|
| Test 1 — Forward Shape: `(batch_size, 4)` | **PASSED** |
| Test 2 — Padding Mask: Boolean, doğru şekil, PAD konumları `True` | **PASSED** |
| Test 3 — Finite Loss: `torch.isfinite(loss)` | **PASSED** |

İlk forward pass'te de logit boyutu `(32,4)` bulundu; NaN ve Inf gözlenmedi.

## What I Learned — Öğrendiklerim

- Token ID'leri, Transformer'ın işleyebileceği yoğun vektörlere **Embedding** aracılığıyla dönüştürülüyor.
- **Positional Encoding**, attention hesaplarına sıra konumu bilgisini katıyor.
- **Transformer Encoder**, token sayısını ve `d_model` boyutunu korurken token temsillerini bağlamla zenginleştirebiliyor.
- PyTorch'ta `src_key_padding_mask` için `True` değerleri **padding konumlarını** gösteriyor; geçerli-token maskesinin tersidir.
- Sadece Encoder'da maskeleme yeterli değil; **Masked Mean Pooling** sırasında da PAD token'ları ortalamadan çıkarılmalı.
- **CrossEntropyLoss** ham logits kullanır; `softmax()` önceden çağrılmaz.
- **Backpropagation** ve **AdamW** ile modelin öğrenilebilir parametreleri güncellenir.
- Training Loss'taki azalma ile validation başarısı farklı şeylerdir; ana metrik olarak **Macro-F1** kaydedildi.

## Çıktıların Konumu

- `reports/transformer_training_metrics.csv` — epoch / training loss (ilk ölçümde epoch süreleri kaydedilmemişti).
- `reports/transformer_timing_check.csv` — ayrı süre ölçümündeki epoch kayıpları ve süreleri.
- `reports/transformer_validation.csv` — validation metrikleri.
- `models/newsroute_transformer.pt` — eğitilmiş modelin ağırlıkları için belirlenen yol (**kaydedildiği ayrıca doğrulanmalı**).
- `notebooks/05_transformer_training.ipynb` — model, eğitim, değerlendirme ve test kodları.
