# NewsRoute — Day 02: Self-Attention, Q/K/V ve Padding Mask

**Proje:** NewsRoute  
**Notebook:** `notebooks/03_attention_qkv_and_mask.ipynb`  
**Çalışmanın amacı:** Öğrenilmemiş, küçük PyTorch Tensor'larıyla Scaled Dot-Product Attention ve Key Padding Mask mekanizmasını anlamak. Bu çalışmada gerçek AG News metinleri işlenmedi ve model eğitilmedi.

## 1. Self-Attention

Self-Attention, dizideki her token'ın diğer erişilebilir token'lardan ne kadar bilgi alacağını hesaplayan mekanizmadır. Her token için oluşturulan Query (Q), diğer token'ların Key (K) vektörleriyle karşılaştırılır. Elde edilen ağırlıklar Value (V) vektörlerini birleştirerek bağlama duyarlı yeni token temsilleri üretir. RNN'den farklı olarak token'lar arasındaki ilişkiler matris işlemleriyle paralel hesaplanabilir.

## 2. Input Tensor

Çalışmada `torch.manual_seed(42)` kullanıldı. Rastgele giriş Tensor'ı şu parametrelerle oluşturuldu:

```python
batch_size = 2
seq_len = 4
d_model = 4
X = torch.randn(batch_size, seq_len, d_model)
```

`X.shape = (2, 4, 4)` biçimi sırasıyla **iki metin**, **metin başına dört token konumu** ve **token başına dört sayısal özellik** anlamına gelir. İlk metin dört gerçek token içerir. İkinci metnin dördüncü token konumu `[PAD]` olarak kabul edilerek `X[1, -1] = 0` yapıldı. Bu sıfırlama tek başına Attention Mask değildir.

## 3. Q/K/V

Aynı giriş Tensor'ından üç ayrı, bias içermeyen doğrusal projeksiyon üretildi:

```python
W_q = nn.Linear(d_model, d_k, bias=False)
W_k = nn.Linear(d_model, d_k, bias=False)
W_v = nn.Linear(d_model, d_v, bias=False)

Q = W_q(X)
K = W_k(X)
V = W_v(X)
```

Burada `d_k = 4` ve `d_v = 4` kullanıldı. Query, aranacak bilgiyle ilgili özellikleri; Key, eşleşmede kullanılacak özellikleri; Value ise ağırlıklandırılarak aktarılacak bilgiyi temsil eder. Üç katmanın ağırlıkları birbirinden bağımsızdır ve henüz eğitilmemiştir. `W_q.weight`, `W_k.weight`, `W_v.weight` şekillerinin her biri `(4, 4)` olur; PyTorch `nn.Linear` ağırlıklarını `(out_features, in_features)` sırasıyla saklar.

## 4. Attention Scores

İlk aşamada her Query konumu ile bütün Key konumlarının noktasal çarpımı hesaplandı:

```python
scores = torch.matmul(Q, K.transpose(-2, -1))
```

Formül: **Scores = QKᵀ**. Skor Tensor'ı `(2, 4, 4)` boyutundadır: iki metnin her birinde dört Query konumu, dört Key konumuyla karşılaştırılır. `scores[0, 1, 3]`, birinci metnin ikinci Query konumu ile dördüncü Key konumu arasındaki skordur.

## 5. Scaling

Skorlar `sqrt(d_k)` ile ölçeklendirildi:

```python
scaled_scores = scores / math.sqrt(d_k)
```

`d_k = 4` olduğundan bütün skorlar **2'ye bölündü**. Büyük özellik boyutlarında dot-product skorlarının büyümesi, Softmax'ın aşırı keskin dağılımlar üretmesine neden olabilir. Ölçekleme, sayısal büyüklüğü kontrol ederek daha kararlı bir Attention hesabına yardımcı olur. Tensor şekli değişmez.

## 6. Softmax

```python
attention_weights = F.softmax(scaled_scores, dim=-1)
```

Softmax **Key boyutu** (`dim=-1`) üzerinde uygulandı. Böylece her Query satırının Attention ağırlıkları toplamı yaklaşık 1 olur. `dim=1` kullanılsaydı normalizasyon Query konumları arasında, yani sütunlar boyunca yapılacaktı; bu, istenen Attention normalizasyonu değildir.

İlk metnin paylaşılan maskesiz Attention matrisi:

```text
[[0.0908, 0.1926, 0.2918, 0.4249],
 [0.2079, 0.2339, 0.2624, 0.2957],
 [0.2757, 0.2591, 0.2464, 0.2188],
 [0.4204, 0.2396, 0.1844, 0.1557]]
```

Paylaşılan notebook çıktısında satır toplamları `tensor([1.0000, 1.0000, 1.0000, 1.0000])` olarak görüldü. Sayılar dört basamağa yuvarlanmıştır.

## 7. Attention Output

```python
attention_without_mask = torch.matmul(attention_weights, V)
```

Formül: **Output = AttentionWeights × V**. Her Query'nin çıkışı yalnızca kendi Value vektöründen değil, izin verilen Key konumlarına karşılık gelen Value vektörlerinin ağırlıklı toplamından oluşur. Çıktı şekli `(2, 4, 4)` olarak gözlemlendi. Çıktı vektörlerinin elemanları negatif olabilir; 0–1 aralığı kuralı Attention **ağırlıkları** içindir, çıktı vektörleri için değil.

## 8. Padding Mask

İkinci metnin son konumu `[PAD]` olduğundan aşağıdaki Boolean maske kullanıldı:

```python
valid_mask = torch.tensor([
    [True, True, True, True],
    [True, True, True, False]
])

key_mask = valid_mask.unsqueeze(1)
masked_scores = scaled_scores.masked_fill(~key_mask, float('-inf'))
masked_weights = F.softmax(masked_scores, dim=-1)
attention_with_mask = torch.matmul(masked_weights, V)
```

`valid_mask` şekli `(2, 4)`, genişletilmiş `key_mask` şekli `(2, 1, 4)` olur. Ortadaki tek elemanlı eksen, maskenin bütün Query satırlarına *broadcast* edilmesini sağlar. `masked_fill` yalnızca maskesi `True` olan konumları değiştirdiğinden `~key_mask` ile padding konumları seçildi.

**Maske Softmax'tan önce uygulanır.** `-inf` değerinin Softmax ağırlığı sıfırdır; oysa sıfır skorun Softmax ağırlığı pozitif olabilir. Padding vektörünü sıfırlamak tek başına yeterli değildir. Buradaki işlem **Key Padding Mask**'tir: `[PAD]` konumuna Key olarak verilen ağırlıkları sıfırlar; padding konumu Query olarak yine çıktı üretebilir.

## 9. Mask Experiment

Aynı Q, K ve V kullanılarak iki deney karşılaştırıldı. Yeni rastgele projeksiyon oluşturulmadı.

| Kontrol | Maskesiz | Maskeli |
|---|---:|---:|
| Attention Shape | `(2, 4, 4)` | `(2, 4, 4)` |
| Output Shape | `(2, 4, 4)` | `(2, 4, 4)` |
| İkinci metin, ilk Query → PAD Key | `0.2535` | `0.0000` |
| İlk Query satırı toplamı | `≈1.0000` | `≈1.0000` |

**İkinci metin, ilk Query konumu için paylaşılan Attention ağırlıkları:**

| Key konumu | Maskesiz | Maskeli |
|---|---:|---:|
| 1 — Markets | 0.2495 | 0.3342 |
| 2 — fall | 0.2672 | 0.3579 |
| 3 — today | 0.2299 | 0.3079 |
| 4 — `[PAD]` | **0.2535** | **0.0000** |

Bu ağırlıklar notebook ekran görüntüsünden dört ondalık basamakla aktarılmıştır. Maskeyle gerçek token'ların **skorları** değişmez; Softmax normalizasyonu PAD konumunu dışladığı için gerçek token'ların **ağırlıkları** değişir. Ağırlıkları Softmax'tan *sonra* sıfırlamak, kalan ağırlıkları yeniden normalize etmez.

Aynı Query için raporlanan Attention çıktı vektörleri:

```text
Maskesiz: [-0.1482, -0.4233,  0.4374, -0.4790]
Maskeli : [-0.1985, -0.5670,  0.5859, -0.6416]
```

## 10. Tensor Shapes — Notebook Bulguları

| Tensor | Gerçek Shape |
|---|---|
| X | `(2, 4, 4)` |
| Q | `(2, 4, 4)` |
| K | `(2, 4, 4)` |
| V | `(2, 4, 4)` |
| Attention Scores (`scores`) | `(2, 4, 4)` |
| Attention Weights (`attention_weights`) | `(2, 4, 4)` |
| Attention Output (`attention_without_mask`) | `(2, 4, 4)` |
| Padding Mask (`valid_mask`) | `(2, 4)` |
| Genişletilmiş Key Mask (`key_mask`) | `(2, 1, 4)` |
| Maskelenmiş Attention Weights (`masked_weights`) | `(2, 4, 4)` |
| Maskeli Attention Output (`attention_with_mask`) | `(2, 4, 4)` |

**Kaynak notu:** Bu şekiller çalışma sırasında paylaşılan kodlar ve notebook ekran görüntülerindeki Tensor boyutlarına dayanır. Notebook dosyasının tamamı doğrudan incelenmedi; bazı ara Tensor şekilleri aynı kodun bilinen matris boyutlarından da doğrulanabilir.

## 11. Tests

Üç otomatik kontrol için `assert` tabanlı kod hazırlandı:

| Test | Kontrol | Doğrulama durumu |
|---|---|---|
| Test 1 — Q/K/V Shapes | Q, K, V ve Attention çıktıları `(2,4,4)` olmalı | Kod hazır; toplu test çalıştırma çıktısı paylaşılmadı |
| Test 2 — Attention Weights Sum | Maskeli ve maskesiz her Query satırının toplamı `torch.allclose(..., 1)` ile kontrol edilmeli | Kod hazır; toplu test çalıştırma çıktısı paylaşılmadı |
| Test 3 — Padding Mask | `masked_weights[1, :, 3]` sıfır olmalı | Kod hazır; toplu test çalıştırma çıktısı paylaşılmadı |

İlgili notebook çıktılarında ilk metnin Attention satır toplamları 1 olarak, ikinci metnin ilk Query konumunda maskeli PAD ağırlığı 0 olarak görülmüştür. **Bu gözlemler, üç otomatik testin topluca çalıştırılıp `PASSED` çıktısı verdiğini tek başına kanıtlamaz.** Testler çalıştırıldığında durumlar güncellenmelidir.

## 12. What I Learned

- Aynı giriş Tensor'ından Q, K ve V projeksiyonları üretmenin amacı, **hangi bilgiden ne kadar yararlanılacağını** hesaplamak ile **aktarılacak bilgiyi** ayrı temsillerle ifade etmektir.
- Scaled Dot-Product Attention sırasıyla **QKᵀ → √dₖ ile bölme → (varsa) maskeleme → Softmax → Value vektörlerinin ağırlıklı toplamı** işlemlerini uygular.
- Softmax her Query satırındaki Key ağırlıklarını toplamı 1 olacak biçimde düzenler; bu nedenle `dim=-1` kullanılmalıdır.
- Maske, pad konumlarını Softmax'ın normalizasyonundan **önce** çıkarmalıdır. Pad'i sonradan sıfırlamak aynı sonucu vermez.
- Padding vektörü `[0, 0, 0, 0]` olsa bile sıfır attention skoru pozitif Softmax ağırlığına dönüşebilir; bu yüzden Key Padding Mask gereklidir.
- Bu deneyde sayılar, **eğitilmemiş** doğrusal projeksiyonlardan türetilmiştir. Gerçek dilsel ilişkileri öğrendiğimiz iddia edilemez; amaç Tensor boyutlarını ve mekanizmayı anlamaktır.
