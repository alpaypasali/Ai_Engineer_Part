# Day 03 — Multi-Head Attention ve Sinusoidal Positional Encoding

**Proje:** NewsRoute  
**Notebook:** `notebooks/04_multihead_positional_encoding.ipynb`  
**Kapsam:** Küçük PyTorch Tensor'larıyla Multi-Head Attention, Key Padding Mask ve sabit Sinusoidal Positional Encoding. Bu çalışmada AG News metinleri henüz modele verilmedi; projeksiyon katmanları eğitilmedi.

## 1. Multi-Head Attention nedir?

Multi-Head Attention, tek bir Attention hesabı yerine birden fazla Attention Head kullanarak aynı dizideki token'lar arasındaki ilişkileri farklı projeksiyon alt uzaylarında hesaplayan mekanizmadır. Her head kendi Q, K ve V bileşenlerini kullanır. Head çıktıları birleştirilir ve son bir doğrusal projeksiyondan geçirilir. Farklı head'lerin farklı dağılımlar üretmesi, özellikle bu eğitilmemiş deneyde, dilsel uzmanlaşma kazandıkları anlamına gelmez.

## 2. `d_model` ve `head_dim` nedir?

- **`d_model`:** Ana token temsilinin özellik boyutu. Bu deneyde `d_model = 8`.
- **`num_heads`:** Attention Head sayısı. Bu deneyde `num_heads = 2`.
- **`head_dim`:** Her head'in Q/K/V vektörlerindeki özellik sayısı. Eşit head boyutları için `head_dim = d_model // num_heads = 8 // 2 = 4`.

`d_model % num_heads == 0` koşulu kontrol edildi. Girişte iki örnek, her örnekte dört token konumu bulunduğundan `X.shape = (2, 4, 8)`.

## 3. Q/K/V projeksiyonları ve head'lere ayırma

Aynı giriş `X`, üç ayrı doğrusal katmandan geçirildi:

```python
W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(X)
K = W_k(X)
V = W_v(X)
```

Her projeksiyonun ilk şekli `(2, 4, 8)` oldu. Son özellik boyutu önce `(num_heads, head_dim) = (2, 4)` olarak ayrıldı; ardından head ekseni token ekseninin önüne taşındı:

```python
Q_heads = Q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
K_heads = K.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
V_heads = V.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
```

Dönüşüm: `(batch, seq_len, d_model) → (batch, seq_len, heads, head_dim) → (batch, heads, seq_len, head_dim)`. Sonuç her üç Tensor için `(2, 2, 4, 4)`. `transpose()` kullanımı aynı token'a ait farklı head özelliklerinin doğru eşleşmesini korur; yalnızca doğrudan `reshape(2, 2, 4, 4)` yapmak aynı eleman yerleşimini vermez.

## 4. Attention Scores ve Scaling

Her head için Query–Key skorları hesaplandı:

\[
S = QK^T,\qquad S_{scaled}=\frac{QK^T}{\sqrt{head\_dim}}
\]

```python
scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1))
scaled_scores = scores / math.sqrt(head_dim)
```

`scores.shape = (2, 2, 4, 4)`; son iki eksen sırasıyla Query ve Key konumlarını gösteriyor. Her head'in Key vektörü dört boyutlu olduğundan ölçekleme katsayısı `sqrt(4) = 2`. Ölçekleme, büyük Dot Product skorlarının Softmax dağılımını aşırı keskinleştirmesini azaltmaya yardımcı olur.

## 5. Attention Mask nasıl uygulandı?

İkinci örneğin dördüncü token konumu `[PAD]` olarak kabul edildi ve giriş vektörü sıfırlandı. Geçerli token maskesi:

```python
valid_mask = torch.tensor([
    [True, True, True, True],
    [True, True, True, False]
])

key_mask = valid_mask.unsqueeze(1).unsqueeze(1)
masked_scores = scaled_scores.masked_fill(~key_mask, float('-inf'))
attention_weights = F.softmax(masked_scores, dim=-1)
```

`valid_mask.shape = (2, 4)` ve `key_mask.shape = (2, 1, 1, 4)`; broadcasting sayesinde **her iki head'in bütün Query konumlarında** aynı PAD Key konumu maskelendi. Softmax öncesinde `-inf` verilen PAD Key ağırlıkları her iki head için de sıfır oldu. `dim=-1`, her Query'nin Key eksenindeki ağırlıklarının toplamını yaklaşık 1 yapar.

**Sınır:** Bu bir **Key Padding Mask**; PAD konumunun Query satırını tek başına sıfırlamaz.

## 6. Attention Head çıktıları, Concatenation ve Output Projection

```python
head_outputs = torch.matmul(attention_weights, V_heads)
concatenated_output = (
    head_outputs.transpose(1, 2)
    .contiguous()
    .reshape(batch_size, seq_len, d_model)
)
W_o = nn.Linear(d_model, d_model, bias=False)
multihead_output = W_o(concatenated_output)
```

Her head'in çıktısı `(2, 2, 4, 4)` biçimindeydi. **Concatenation**, aynı token'ın iki head'den gelen dörder özelliğini yan yana birleştirerek `(2, 4, 8)` şekline getirdi. Önce `transpose(1, 2)` yapılması token–head eşleşmesinin korunması için gerekliydi; `contiguous()` bellekte ardışık yerleşim sağlar.

**Output Projection (`W_o`)**, birleştirilen sekiz özelliği öğrenilebilir doğrusal dönüşümle karıştırır. `d_model → d_model` olduğu için şekil `(2, 4, 8)` kalır; ancak vektör değerleri değişebilir.

## 7. Positional Encoding nedir?

Standart Self-Attention, tek başına token konumlarını açıkça kodlamaz. Positional Encoding, token'ın dizideki pozisyonu hakkında modele bilgi ekler. Bu deneyde **sabit sinusoidal** yöntem kullanıldı; öğrenilebilir positional embedding uygulanmadı.

Orijinal sinusoidal formül:

\[
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

Çift boyut indekslerinde sinüs, tek indekslerde kosinüs uygulanır. Farklı boyut çiftlerindeki frekanslar farklı hızlarda değişerek birbirini tamamlayan konum bilgileri sağlar.

Kullanılan fonksiyon:

```python
def positional_encoding(seq_len, d_model):
    positions = torch.arange(seq_len).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float()
        * (-math.log(10000.0) / d_model)
    )
    angles = positions * div_term
    PE = torch.zeros(seq_len, d_model)
    PE[:, 0::2] = torch.sin(angles)
    PE[:, 1::2] = torch.cos(angles)
    return PE
```

`PE.shape = (4, 8)`. İlk pozisyon için `PE[0] = [0, 1, 0, 1, 0, 1, 0, 1]`. Dört pozisyonun vektörleri birbirinden farklı ve değerleri sonlu sayılardan oluşuyor.

## 8. Positional Encoding çıktıyı nasıl etkiledi? — Deney A / B

İki deneyde **aynı `W_q`, `W_k`, `W_v` ve `W_o` katmanları** kullanıldı; yeni rastgele Linear katmanlar oluşturulmadı.

- **Deney A:** Giriş `X`; Positional Encoding yok.
- **Deney B:** Giriş `X_pos = X + PE.unsqueeze(0)`; Positional Encoding var.

İki deneyin giriş şekli `(2, 4, 8)`; nihai çıktı şekli de `(2, 4, 8)`. PE eklendiğinde ikinci metnin PAD vektörü sıfır kalmadı; buna rağmen **aynı Key Padding Mask iki deneyde de uygulandı**.

### Attention ağırlıkları — ikinci metin, Head 1, Query 1

| Key konumu | Deney A (PE yok) | Deney B (PE var) |
|---|---:|---:|
| Token 1 | 0.4217 | 0.3540 |
| Token 2 | 0.2606 | 0.2896 |
| Token 3 | 0.3178 | 0.3564 |
| Token 4 `[PAD]` | **0.0000** | **0.0000** |

`torch.allclose(weights_A, weights_B)` sonucu **`False`** oldu.

### Nihai çıktı — birinci metin, ilk token

**Deney A:** `[0.0928, -0.0312, 0.0803, -0.1119, 0.0724, -0.0335, 0.0509, 0.0183]`

**Deney B:** `[0.1567, 0.2253, -0.0919, 0.1568, -0.0110, -0.0073, -0.0113, -0.0148]`

`torch.allclose(output_A, output_B)` sonucu **`False`** oldu. Farklılık, model ağırlıklarının yeniden öğrenilmesinden değil, girişe PE eklenmesiyle Q/K/V temsillerinin ve bunlardan türetilen Attention hesaplarının değişmesinden kaynaklanıyor. Bu gözlem, **eğitilmemiş modelin cümle anlamını öğrendiğini göstermez**.

## 9. Head karşılaştırması — Deney B

Birinci metnin ilk Query konumu için iki head'in ağırlıkları:

| Key konumu | Head 1 | Head 2 |
|---|---:|---:|
| Token 1 | 0.2825 | 0.3183 |
| Token 2 | 0.0860 | 0.2679 |
| Token 3 | 0.3881 | 0.1587 |
| Token 4 | 0.2435 | 0.2551 |

Head 1, bu örnekte Token 3'e; Head 2 ise Token 1'e daha yüksek ağırlık verdi. Head'ler farklı Q/K projeksiyon bileşenleri kullandığı için dağılımları farklı olabilir. **Bu deneyde katmanlar eğitilmediğinden farkları anlamsal uzmanlaşma kanıtı olarak yorumlamıyorum.**

## 10. Tensor boyutları — notebook çıktıları

| Tensor | Gerçek Shape |
|---|---|
| `X` | `(2, 4, 8)` |
| `Q` | `(2, 4, 8)` |
| `K` | `(2, 4, 8)` |
| `V` | `(2, 4, 8)` |
| `Q_heads` | `(2, 2, 4, 4)` |
| `K_heads` | `(2, 2, 4, 4)` |
| `V_heads` | `(2, 2, 4, 4)` |
| Attention Scores (`scores`) | `(2, 2, 4, 4)` |
| Attention Weights (`attention_weights`) | `(2, 2, 4, 4)` |
| Head Output (`head_outputs`) | `(2, 2, 4, 4)` |
| Concatenated Output (`concatenated_output`) | `(2, 4, 8)` |
| Final Output (`multihead_output`) | `(2, 4, 8)` |
| Positional Encoding (`PE`) | `(4, 8)` |
| `valid_mask` | `(2, 4)` |
| `key_mask` | `(2, 1, 1, 4)` |
| `X_pos` | `(2, 4, 8)` |
| Final Output A / B | `(2, 4, 8)` / `(2, 4, 8)` |

Tablodaki boyutlar, bu çalışmada paylaşılan notebook çıktılarına ve çalıştırılan testlere dayanır. `Q`, `K`, `V` satırları **head ayrımı öncesindeki** projeksiyonları gösterir.

## 11. Otomatik test sonuçları

Notebook'ta çalıştırılan üç testin görülen çıktısı:

```text
Test 1 — Multi-Head Shapes: PASSED
Test 2 — Padding Mask: PASSED
Test 3 — Positional Encoding: PASSED
```

- **Test 1:** Giriş, Q/K/V head boyutları, skorlar, birleştirilmiş ve nihai çıktı boyutları kontrol edildi.
- **Test 2:** Her iki head'de PAD Key ağırlıklarının sıfır olması ve her Query'nin ağırlık toplamının yaklaşık 1 olması `torch.allclose(..., atol=1e-6)` ile kontrol edildi. Hem ilk hesaplama hem A/B karşılaştırması kapsandı.
- **Test 3:** `PE.shape == (4, 8)`, tüm PE değerlerinin sonlu olması ve farklı pozisyonların vektörlerinin birbirinden farklı olması kontrol edildi.

Bu testler temel yapısal/matematiksel davranışı doğrular; model doğruluğu veya dilsel öğrenme performansı için değerlendirme değildir.

## 12. Bugün ne öğrendim?

Bu çalışmada **`d_model` ile `head_dim` arasındaki farkı** ve iki head kullanıldığında sekiz özelliğin head başına dört özelliğe ayrıldığını netleştirdim. `reshape()` ile `transpose()` işlemlerinin neden aynı olmadığını; head ve token eksenlerini doğru sırada tutmazsam aynı token'ın farklı head'lerdeki bilgilerini yanlış birleştirebileceğimi gördüm.

Her head'in kendi Query–Key ilişkilerini hesapladığını, **Padding Mask'in bütün head'lerde** uygulanması gerektiğini ve `dim=-1` ile Softmax'ın Key konumları üzerinde çalıştığını öğrendim. İki head'in çıktılarını birleştirip `W_o` ile dönüştürdüm.

Son olarak, **Positional Encoding'in sabit matematiksel bir vektör olduğunu** ve aynı model parametreleriyle girişe eklendiğinde Attention ağırlıkları ile çıktıların değişebildiğini deneyde gözlemledim. Head dağılımlarının farklılığı veya PE'li/PE'siz çıktı farkı, eğitim yapılmadan dilsel anlamın öğrenildiği anlamına gelmiyor.
