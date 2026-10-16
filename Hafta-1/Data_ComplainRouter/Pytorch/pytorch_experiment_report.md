# PyTorch Experiment Report — Consumer Complaint Classification

**Kaynak notebook:** [`03_pytorch_mlp.ipynb`](03_pytorch_mlp.ipynb)  
**Kapsam:** Dört sınıflı İngilizce finansal şikâyet sınıflandırması · 5.000 metin  
**Karşılaştırma:** TF-IDF + PyTorch MLP (128/256 nöron) vs. Logistic Regression baseline  
**Veri bölümü:** 4.000 train / 1.000 holdout (notebook'ta `val` olarak adlandırılmış)  
**Temel sonuç:** MLP-128 **0,6739 Macro-F1**, MLP-256 **0,6730 Macro-F1**; 1.500 özellikli Logistic Regression **0,7197 Macro-F1**

> **Sonuçların statüsü:** Bu rapor yüklenen notebook'taki **kaydedilmiş kod ve çıktılar** ile önceki `baseline_report.md` kaydına dayanır. Veri CSV'si pakette bulunmadığı için deneyler bu aşamada yeniden çalıştırılmadı. Kayıtlı deneyde bağımsız validation ayrımı yoktur; karşılaştırma yapılan %20'lik holdout'u final test sonucu olarak sunmuyoruz.

## Deney özeti

| Deney | TF-IDF özellik | Mimari / yöntem | Parametre | Son train loss | Holdout accuracy | Holdout Macro-F1 |
|---|---:|---|---:|---:|---:|---:|
| LR baseline 1 | 1500 | Logistic Regression | — | — | **%72,30** | **0,7197** |
| LR baseline 2 | 3000 | Logistic Regression | — | — | **%72,60** | **0,7231** |
| MLP-128 | 1500 | 1500→128→32→4 | **196.388** | **0,1226** | %67,30 | 0,6739 |
| MLP-256 | 1500 | 1500→256→32→4 | **392.612** | **0,0406** | Notebook'ta ölçülmedi | 0,6730 |

## 1. PyTorch modelimin mimarisi nedir?

Model `torch.nn.Module` üzerinden yazılan **feed-forward MLP** sınıflandırıcısıdır. Metin özellikleri `TfidfVectorizer(max_features=1500)` ile hazırlanır. İki gizli Linear katmandan sonra dört sınıfa karşılık gelen dört ham logit üretilir.

```text
Input: TF-IDF · 1500 özellik
    ↓
fc1: Linear(1500 → 128)    [ikinci deneyde 256]
    ↓ ReLU
fc2: Linear(128 → 32)      [ikinci deneyde 256 → 32]
    ↓ ReLU
fc3: Linear(32 → 4)
    ↓
Output: (batch_size, 4) logits
```

`Flatten` gereksizdir çünkü TF-IDF verileri zaten `(batch, features)` şeklindedir. Son katmanda Softmax kullanılmaz; sınıf seçimi değerlendirmede `argmax(dim=1)` ile yapılır.

## 2. Kaç eğitilebilir parametrem var?

Her Linear için `(in_features × out_features) + out_features (bias)` formülü kullanılır.

**MLP-128:**

| Katman | Weight | Bias | Toplam |
|---|---:|---:|---:|
| fc1 · 1500→128 | 192.000 | 128 | 192.128 |
| fc2 · 128→32 | 4.096 | 32 | 4.128 |
| fc3 · 32→4 | 128 | 4 | 132 |
| **Toplam** | **196.224** | **164** | **196.388** |

**MLP-256:** `1500×256+256 = 384.256`, `256×32+32 = 8.224`, `32×4+4 = 132`, toplam **392.612**.

`sum(p.numel() for p in model.parameters() if p.requires_grad)` ile doğrulanabilir. ReLU'nun öğrenilebilir parametresi yoktur. Batch size toplam parametre sayısını değiştirmez.

## 3. Hangi loss fonksiyonunu kullandım?

**`nn.CrossEntropyLoss()`** kullandım. Her örnekte model dört **logit** üretir; gerçek sınıf etiketleri `torch.long` biçiminde `0,1,2,3` değerleridir. `CrossEntropyLoss`, LogSoftmax ve negatif log-likelihood hesabını birleştirir; bu nedenle çıktılara ayrıca Softmax uygulamak doğru değildir.

## 4. Neden Adam optimizer tercih ettim?

**`torch.optim.Adam(lr=0.001)`** kullandım. Adam, gradyanların birinci ve ikinci moment tahminleri üzerinden parametre başına adaptif öğrenme adımları uygular. Başlangıç deneylerinde kullanımı kolay, iyi bir standart başlangıç seçimidir. Ancak **bu projede Adam'ı SGD/AdamW ile karşılaştırmadım**; dolayısıyla burada diğerlerinden daha iyi olduğu gösterilmiş değil.

Eğitim adımı: `zero_grad()` → forward → loss → `backward()` → `optimizer.step()`. Yeni bir model örneği için ayrıca yeni optimizer tanımladım. `batch_size=32`, `epochs=10`; 4.000 eğitim kaydı için epoch başına 125 batch kullanıldı.

## 5. Training loss nasıl değişti?

| Epoch | MLP-128 | MLP-256 |
|---:|---:|---:|
| 1 | 1,0802 | 1,0498 |
| 2 | 0,5873 | 0,5717 |
| 3 | 0,4785 | 0,4420 |
| 4 | 0,3974 | 0,3458 |
| 5 | 0,3299 | 0,2674 |
| 6 | 0,2761 | 0,2080 |
| 7 | 0,2300 | 0,1512 |
| 8 | 0,1905 | 0,1101 |
| 9 | 0,1560 | 0,0687 |
| 10 | **0,1226** | **0,0406** |

İki modelde de eğitim kaybı düzenli azaldı. 256 nöronlu model eğitim verisinde daha düşük loss elde etti; **ama bu, görmediği verilerde daha iyi sonuç verdiği anlamına gelmiyor**.

![Training loss karşılaştırması](figures/training_loss_comparison.png)

## 6. Validation Macro-F1 kaç?

Notebook'ta `val_loader` olarak geçen fakat başlangıçta `X_test` üzerinden oluşturulan **aynı holdout veri bölümü** için:

- **MLP-128:** `0,673871420434641` → **0,6739**
- **MLP-256:** `0,6729960801458406` → **0,6730**

MLP-128'in holdout **accuracy** değeri **0,673 (673 / 1.000 doğru)**. MLP-256'nın accuracy çıktısı yüklenen notebook'ta saklanmamış olduğundan raporda **bilinmiyor** olarak gösterildi.

Macro-F1, dört sınıfın F1 değerlerine eşit ağırlık verir. Tüm sınıflarda 250'şer holdout örneği olduğundan sınıflar eşit dağılıyor; yine de Macro-F1 hata tiplerini ayrı izlemek açısından değerlidir.

## 7. Logistic Regression ile karşılaştırıldığında nasıl bir sonuç elde ettim?

Önceki `baseline_report.md` dosyasındaki Logistic Regression deneyleri aynı dört kategoriyi temel alır, ancak baseline kodlaması `1–4`, bu notebook kodlaması `0–3` olarak düzenlenmiştir. **Karşılaştırma kategori adları düzeyindedir.**

- Aynı **1.500 TF-IDF özellik** ile LR: **0,7197 Macro-F1 / %72,30 accuracy**; MLP-128: **0,6739 Macro-F1 / %67,30 accuracy**.
- Aradaki Macro-F1 farkı yaklaşık **0,0458 (4,58 yüzde puan)**, LR lehinedir.
- LR'nin **3.000 özellikli** deneyinde Macro-F1 **0,7231**, accuracy **%72,60** ile daha yüksektir; ancak bunun hem farklı model hem de iki kat özellik sayısı içerdiği unutulmamalıdır.
- MLP-256, MLP-128'den daha düşük training loss elde etmesine rağmen holdout Macro-F1 yaklaşık **0,0009** azalmıştır.

**Yorum:** Küçük ve TF-IDF tabanlı bu problemde doğrusal baseline daha güçlü görünmektedir. Ancak bu tek/az sayıda koşudaki küçük farkların istatistiksel güvenilirliği test edilmemiştir.

## 8. Model hangi kategorilerde başarısız oldu?

Notebook'ta sınıf bazlı metrikler **MLP-128** için kaydedilmiştir:

| Etiket | Kategori | Precision | Recall | F1 | Support |
|---:|---|---:|---:|---:|---:|
| 0 | Credit reporting, credit repair services, or other personal consumer reports | 0,52 | 0,54 | **0,53** | 250 |
| 1 | Debt collection | 0,72 | 0,71 | 0,72 | 250 |
| 2 | Credit reporting | 0,56 | 0,56 | **0,56** | 250 |
| 3 | Mortgage | 0,90 | 0,89 | **0,89** | 250 |

**En zor iki kategori 0 ve 2:** Confusion matrix'te gerçek `0` etiketli **76** şikâyet `2` tahmin edilmiş; gerçek `2` etiketli **79** şikâyet `0` tahmin edilmiş; bu çiftte toplam **155** hata var. Kategorilerin finansal raporlama ve düzeltme dilinde örtüşmesi olası bir neden; doğrulamak için ham yanlış tahmin örnekleri incelenmeli. **Mortgage** en yüksek F1 skoruna sahiptir.

```text
               Tahmin 0  Tahmin 1  Tahmin 2  Tahmin 3
Gerçek 0            135        28        76        11
Gerçek 1             34       177        30         9
Gerçek 2             79        26       139         6
Gerçek 3             12        14         2       222
```

![Confusion matrix](figures/confusion_matrix_mlp128.png)

## 9. Sonraki deneyde neyi değiştirmeyi öneriyorum?

**Birinci öncelik: Daha doğru bir karşılaştırma protokolü oluşturmak.**

1. Başlangıçta **train / validation / test** ayrımı yap; hiperparametre seçiminde yalnızca validation kullan, bağımsız testi en sona sakla.
2. 128 ve 256 nöronlu modelleri aynı seed, aynı split ve aynı DataLoader shuffle dizisiyle **sıfırdan** yeniden eğit; mümkünse birden fazla seed ile ortalama ve değişkenliği bildir.
3. **Her epoch sonunda validation loss ve Macro-F1** kaydet. Eğitim loss'u azalıp validation loss'u artmaya başlıyorsa **early stopping** dene.
4. Düzenlileştirme deneyi yap: **Dropout**, Adam `weight_decay`, veya daha küçük gizli katman. Bu seçeneklerin etkisini kontrollü kıyasla.
5. Özellik tarafında **TF-IDF max_features=3000** ve **bigram (`ngram_range=(1,2)`)** ile Logistic Regression dahil her modeli aynı giriş koşullarında yeniden değerlendir.
6. 0 ↔ 2 hatalarının metinlerini manuel incele; sınıf tanımları, kısa metinler ve veri etiketlerini denetle.

### Sınırlamalar ve çıkarımlar

- **Ayrı validation kümesi yok:** Kodda `val_...` isimleri görünse de sonuçlar %20 holdout/test bölümündendir. Holdout'u tuning için kullanmak nihai test yorumunu zayıflatır.
- **Tam tekrarlanabilirlik doğrulanmadı:** Seed yazılmış olsa da notebook'un arşivlenen koşularında her deney için generator/shuffle sırası sabit ve bağımsız tutulmamıştır.
- **Veri / model kapsamı:** 5.000 dengeli örnek ve yalnızca dört kategori, gerçek dünyadaki sınıf dengesizliği veya yeni ürün kategorilerini temsil etmeyebilir.
- **Dense TF-IDF maliyeti:** 5.000×1.500 float32 matris ≈30 MB; 1 milyon×1.500 ≈6 GB ham veri. Dönüşüm sırasında bellek tepe değeri artar.
- **Sonuç:** Daha düşük training loss, daha iyi genelleme garantisi vermez. Güçlü ve hafif bir Logistic Regression baseline'ı, bu deneyde daha karmaşık MLP'den iyi sonuç verdi.

## Öğrendiklerim

Bu projede TF-IDF'yi PyTorch Tensor'a dönüştürmeyi; `nn.Module`, Linear, ReLU ve logits tasarımını; ağırlık/bias parametre sayımını; `TensorDataset` + `DataLoader` ile mini-batch eğitimini; `zero_grad → forward → loss → backward → step` akışını; GPU/CPU cihaz eşlemesini; `eval + no_grad + argmax` ile çıkarımı; accuracy, Macro-F1 ve confusion matrix yorumlamayı öğrendim. **En önemli ders:** Karar yalnızca training loss ile verilmez; düzgün ayrılmış veri ve basit baseline karşılaştırması şarttır.

---

**Kaynaklar:** `03_pytorch_mlp.ipynb` içindeki kaydedilmiş çıktılar; önceki çalışmanın `baseline_report.md` sonuçları. [Proje README](README.md).
