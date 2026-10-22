# ShopperIntent — Gün 3: Gradient Stability / Teknik Deney Raporu

**Proje:** Hafta 2 · ShopperIntent  
**Tarih:** 22 Ekim 2026  
**İlgili dosya:** `notebooks/04_gradient_stability_tamamlandi.ipynb`  
**Metrik tablosu:** `reports/gradient_experiments.csv`  
**Durum:** Deney tanımları doğrulandı; gerçek veri üzerinde tamamlanmış A/B/C eğitimlerinin sayısal çıktıları paylaşılan sürümde bulunmadığı için **sonuç metrikleri henüz doğrulanmadı**.

> **Veri bütünlüğü uyarısı:** `04_gradient_stability.ipynb` dosyasının ilk sürümünde Clipping OFF ve ON deneyleri aynı model ve optimizer üzerinde ardışık gerçekleştirilmişti. Bu çıktıların A/B nedensel karşılaştırmasında kullanılması doğru değildir. Düzeltilmiş notebook, bağımsız modelleri ve optimizer'ları aynı ilk ağırlıklarla ve aynı batch sırasıyla çalıştıracak şekilde hazırlanmıştır; ancak yüklenen düzeltilmiş kopyada hücre çıktıları boş. Yerel ortamda gerçek `online_shoppers_intention.csv` veri dosyası bulunmadığından sayılar yeniden üretilemedi. Ayrı sentetik test verileriyle elde edilmiş değerler **gerçek sonuç olarak kullanılmadı**.

## Deney tasarımı ve değerlendirme yöntemi

- **Veri:** UCI Online Shoppers Purchasing Intention, 12.330 oturum; hedef `Revenue`, pozitif sınıf `True / 1`. Kaynak: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- **Ön işleme:** `PageValues` olası sızıntı riski nedeniyle kullanılmadı. Train/validation/test ayrımı %70/%15/%15, stratified, `random_state=42`; scaler ve encoder yalnızca train üzerinde `fit` ediliyor. Bağımsız test bölümü bugünkü karşılaştırmada kullanılmıyor.
- **Ağ:** 73 → 64 → 32 → 2, ReLU, `CrossEntropyLoss`, batch size 32, Adam. Aynı modelin farklı eğitim aşamaları değil, üç bağımsız eğitim deneyi karşılaştırılıyor.
- **A / Clipping OFF:** Adam, LR 0.001, `clip_max_norm=None`, 10 epoch hedefi.
- **B / Clipping ON:** Adam, LR 0.001, `clip_max_norm=1.0`, 10 epoch hedefi.
- **C / Yüksek LR:** Adam, LR 0.1, clipping kapalı, **en fazla** 10 epoch; non-finite loss veya gradyanda deney durduruluyor.
- **Adil A/B testi:** Sabit seed 42, aynı başlangıç ağırlıkları, veri bölünmesi ve batch sıralaması. CUDA ve bazı işlem kütüphaneleri açısından seed tek başına bit düzeyinde tam determinizm garantilemez.

### CSV alanlarının yorumu

| Alanlar | Tanım |
|---|---|
| `train_loss`, `validation_loss` | Tamamlanan **son epoch** sonunda hesaplanan örnek ağırlıklı ortalama kayıp |
| `mean_gradient_norm` | Tamamlanan bütün epoch'lardaki **batch başına clipping öncesi** toplam L2 normlarının ortalaması |
| `max_gradient_norm` | Tamamlanan bütün epoch'lardaki **clipping öncesi** en yüksek batch L2 normu |
| `accuracy`, `precision`, `recall`, `f1` | Son modelin **validation** verisindeki sınıflandırma metrikleri; precision/recall/F1 pozitif sınıf (`Revenue=1`) içindir |
| `average_precision` | `Revenue=1` olasılıkları kullanılarak hesaplanan validation Average Precision (`average_precision_score`) |
| `epochs` | CSV ilk sürümünde **planlanan üst sınır 10**; gerçek çalışmada erken durma varsa tamamlanan epoch sayısıyla değiştirilmelidir |

**Ölçüm zamanı:** `loss.backward()` → `calculate_gradient_norm(model)` → (B için) `clip_grad_norm_(..., max_norm=1.0)` → `optimizer.step()`. Dolayısıyla tabloda B'nin maksimum normunun **1.0'dan büyük olması hata değildir**: tablodaki değer clipping *öncesine* aittir. Clipping sonrasındaki normlar (`post_clip_gradient_norm` / `max_post_clip_gradient_norm`) notebook'ta ayrıca tutulur.

### Sonuç tablosu — teyit bekleyen ölçümler

| Metrik | A — Clipping OFF | B — Clipping ON | C — LR 0.1, OFF |
|---|---|---|---|
| Son Training Loss | Henüz doğrulanmadı | Henüz doğrulanmadı | Henüz doğrulanmadı |
| Son Validation Loss | Henüz doğrulanmadı | Henüz doğrulanmadı | Henüz doğrulanmadı |
| Validation Accuracy / Precision / Recall / F1 / AP | Henüz doğrulanmadı | Henüz doğrulanmadı | Henüz doğrulanmadı |
| Ortalama / maksimum Gradient Norm (clipping öncesi) | Henüz doğrulanmadı | Henüz doğrulanmadı | Henüz doğrulanmadı |
| NaN/Inf nedeniyle erken durdu mu? | Bilinmiyor | Bilinmiyor | Bilinmiyor |

Sayılar uydurulmadı; ayarlar CSV'ye girildi, sayısal metrik alanları bilinçli olarak **boş** bırakıldı. Çalışan notebook'ta `history_A`, `history_B`, `history_C` üretildikten sonra aşağıdaki dışa aktarma hücresi bu boşlukları doldurur.

## Araştırma soruları

### 1) Momentum, Adagrad ve RMSProp arasındaki temel farklar nelerdir?

**Momentum**, geçmiş gradyanların üstel ağırlıklı toplamını kullanarak güncelleme yönünde bir hız terimi tutar; tutarlı yöndeki ilerlemeyi destekleyip yön değiştiren gradyanları yumuşatabilir. **Adagrad**, parametre başına geçmiş *tüm* gradyan karelerini biriktirir, böylece sık güncellenen parametrelerin efektif öğrenme oranı zamanla küçülür; seyrek özelliklerde yararlı olabilir ama öğrenme oranı aşırı düşebilir. **RMSProp**, Adagrad'ın sınırsız büyüyen toplamı yerine gradyan karelerinin **üstel hareketli ortalamasını** kullanır; daha yakın geçmişe ağırlık verir. Çalışmamızdaki **Adam**, birinci moment (momentum benzeri ortalama) ile ikinci moment (RMSProp benzeri ölçekleme) bilgisini birleştirir.

### 2) Gradient Norm nedir?

Model parametrelerinin `.grad` Tensor'larındaki tüm gradyanları tek bir vektöre açtığımızda bu vektörün L2 uzunluğudur:

\[ \|G\|_2=\sqrt{\sum_{p}\sum_i (\nabla_p L_i)^2} \]

Örneğin `[3,4]` vektörünün normu `5` olur. Norm, tek bir skalerle gradyanların büyüklüğünü izlememizi sağlar; tek başına modelin iyi öğrendiğini veya güncelleme büyüklüğünü garanti etmez (optimizer türü de etkilidir).

### 3) Modelimin gradyanlarında NaN veya Inf tespit ettim mi?

`check_gradients()` bütün `param.grad` Tensor'larında `torch.isfinite()` kontrolü yapıyor ve sorunlu parametreleri bildiriyor. Küçük bir **bilerek NaN eklenmiş test gradyanı** ile algılama mantığı doğrulanacak biçimde notebook'a dahil edildi. **Gerçek A/B/C eğitim koşuları için kaydedilmiş tamamlanmış health-check çıktısı bulunmadığından, modelde NaN/Inf görüldü veya görülmedi iddiasında bulunamam.** Gerçek çalışmada sorun görülürse epoch/batch numarası raporlanmalı; özellikle C'de durma koşulu etkin.

### 4) Gradient Clipping uygulanınca eğitim davranışı değişti mi?

**Henüz sonuçla doğrulanamıyor.** Teoride B'de toplam gradyan normu 1.0'ı geçtiğinde gradyanlar orantılı olarak küçültülür; geçmediğinde pratikte değişiklik olmaz. Kararı vermek için A/B validation loss ve F1/AP eğrilerini, B'de `clipped_batches` sayısını, clipping sonrası normları ve son metrikleri birlikte incelemeliyim. A/B'nin eski ardışık eğitim çıktıları adil kanıt değildir.

### 5) Hangi deneyin validation loss değeri daha düşük?

**Henüz belirlenemiyor.** Bağımsız koşuların son epoch kayıtlarındaki `history_A[-1]['val_loss']`, `history_B[-1]['val_loss']`, `history_C[-1]['val_loss']` karşılaştırılmalı; erken durma halinde son tamamlanan epoch not edilmelidir. En düşük *tek epoch* değeri ile son epoch değerini birbirine karıştırmıyorum.

### 6) Hangi model satın alma gerçekleştiren kullanıcıları daha iyi yakalıyor?

**Henüz belirlenemiyor.** Pozitif sınıf `Revenue=True` için **Recall**, gerçekten satın alanların ne kadarının bulunduğunu en doğrudan gösterir. İki modelin Recall değerini karşılaştıracağım; daha yüksek Recall gereksiz pozitifleri artırabileceğinden **Precision**, **F1** ve olasılık sıralamasını değerlendiren **Average Precision** ile beraber okuyacağım. Accuracy sınıf dengesizliği nedeniyle tek başına yeterli değil.

### 7) Learning rate artırıldığında nasıl bir sonuç gözlemledim?

LR, **0.001'den 0.1'e (100 kat)** çıkarıldı; ancak gerçek Deney C sonuçları bu dosya kopyasında olmadığı için training/validation loss ya da Gradient Norm'daki değişimi sayısal olarak ileri süremiyorum. Çalışmada kayıpların dalgalanması, büyümesi veya sonlu kalması; NaN/Inf tespit edilmesi ve tamamlanan epoch sayısı gözlemlenmeli. **Yüksek LR her zaman NaN/Inf üretmez.**

### 8) Bu deneylerden sonra model eğitiminde neyi değiştirmeyi öneririm?

**Öncelikli önerim, üç deneyi doğrulanabilir ve karşılaştırılabilir şekilde çalıştırıp mevcut ölçümleri kaydetmek.** Öğrenme oranını artırmayı otomatik iyileştirme saymam: ilk kontrollü referans olarak **Adam + LR 0.001** kullanılabilir. Clipping'i ancak gradyanların sık sık 1.0'ı aştığı ve validation sonuçlarının bunu desteklediği durumda etkinleştiririm; normlar küçükse katkısı sınırlı olabilir. Pozitif sınıfın azlığı nedeniyle yalnızca train loss yerine **Recall, F1 ve AP** ile karar veririm. Yeni bir veri sızıntısı oluşturmamak için aynı validation seti kullanır, bağımsız test setini son değerlendirmeye saklarım. Bu öneriler **deney sonuçlarının yerine geçen kanıt değil**, doğrulama tamamlanana kadar ihtiyatlı çalışma planıdır.

## GitHub'a aktarmak ve CSV'yi gerçek sonuçlarla doldurmak

Dosyaları projenin `reports/` klasörüne koy. Önce düzeltilmiş notebook'u gerçek veriyle **Restart & Run All** çalıştır. Aşağıdaki hücreyi notebook'un en altına ekle: `history_A/B/C`, `model_A/B/C`, `dataloader_val` değişkenlerinden istenen **16 sütunu** çıkarır ve CSV'yi doğrulanmış verilerle günceller. Yeniden model eğitmez.

```python
from pathlib import Path
import csv
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score

FIELDS = [
    'experiment_id', 'optimizer', 'learning_rate', 'clipping', 'max_norm',
    'seed', 'epochs', 'train_loss', 'validation_loss', 'mean_gradient_norm',
    'max_gradient_norm', 'accuracy', 'precision', 'recall', 'f1', 'average_precision'
]

def validation_scores(model, loader):
    model.eval()
    device = next(model.parameters()).device
    labels, predictions, probabilities = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device))
            labels.extend(yb.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            probabilities.extend(torch.softmax(logits, dim=1)[:, 1].cpu().tolist())
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, pos_label=1, zero_division=0),
        'recall': recall_score(labels, predictions, pos_label=1, zero_division=0),
        'f1': f1_score(labels, predictions, pos_label=1, zero_division=0),
        'average_precision': average_precision_score(labels, probabilities),
    }

def export_row(experiment_id, history, model, lr, clipping, max_norm):
    row = dict.fromkeys(FIELDS, '')
    row.update({
        'experiment_id': experiment_id,
        'optimizer': 'Adam',
        'learning_rate': lr,
        'clipping': clipping,
        'max_norm': max_norm if max_norm is not None else '',
        'seed': 42,
        'epochs': len(history),  # gerçek tamamlanan epoch sayısı
    })
    if not history:
        return row
    row['train_loss'] = history[-1]['loss']
    row['validation_loss'] = history[-1]['val_loss']
    row['mean_gradient_norm'] = sum(h['gradient_norm'] for h in history) / len(history)
    row['max_gradient_norm'] = max(h['max_gradient_norm'] for h in history)

    # Kısmi epoch sırasında erken durduysa mevcut model ağırlıkları,
    # history'deki son tamamlanan epoch ile örtüşmeyebilir.
    # Bu yüzden tamamlanmayan deneyin sınıflandırma metriklerini boş bırak.
    if len(history) == 10:
        row.update(validation_scores(model, dataloader_val))
    return row

rows = [
    export_row('A', history_A, model_A, 0.001, 'OFF', None),
    export_row('B', history_B, model_B, 0.001, 'ON', 1.0),
    export_row('C', history_C, model_C, 0.1, 'OFF', None),
]
reports_path = Path('../reports') if Path.cwd().name == 'notebooks' else Path('reports')
reports_path.mkdir(parents=True, exist_ok=True)
with (reports_path / 'gradient_experiments.csv').open('w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)
print('Kaydedildi:', reports_path / 'gradient_experiments.csv')
```

> **Çalıştırma notu:** A/B/C `history` listelerinden biri boşsa, o deneyin metrikleri boş kalır. Kısmi epoch'ta erken durduysa modelin güncel parametreleri son tamamlanan epoch'a ait olmayabileceği için classification metrikleri güvenilir olmayabilir ve boş bırakılır. CSV dolduktan sonra bu rapordaki **4–7. soruların sonuç temelli cevapları** gerçek metriklerle güncellenmelidir.

Önerilen Git adımları:

```bash
git add reports/gradient_experiments.csv reports/day_03_gradient_stability.md notebooks/04_gradient_stability_tamamlandi.ipynb
git commit -m "Document gradient clipping and LR stress-test experiments"
```

**Kaynaklar:** UCI Online Shoppers Purchasing Intention Dataset (veri kaynağı); PyTorch `torch.nn.utils.clip_grad_norm_` (clipping öncesi toplam normu döndürür). https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html
