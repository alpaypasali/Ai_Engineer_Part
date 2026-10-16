# Consumer Complaint Classifier · TF-IDF + PyTorch MLP

**Türkçe proje dokümantasyonu | NLP · PyTorch · Scikit-learn · Model Evaluation**

> 5.000 finansal tüketici şikâyetini dört ürün kategorisine ayırma deneyi. TF-IDF + Logistic Regression baseline'ı ile **128 ve 256 gizli nöronlu PyTorch MLP** modellerini karşılaştırıyorum.

[Notebook'u aç](03_pytorch_mlp.ipynb) · [Ayrıntılı deney raporu](pytorch_experiment_report.md)

## Sonuçlar

| Deney | Özellik sayısı | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Logistic Regression (baseline) | 1.500 | **%72,30** | **0,7197** |
| Logistic Regression (baseline) | 3.000 | **%72,60** | **0,7231** |
| PyTorch MLP · 128 nöron | 1.500 | %67,30 | 0,6739 |
| PyTorch MLP · 256 nöron | 1.500 | Ölçülmedi | 0,6730 |

**Ana bulgu:** Karmaşıklık artışı bu örnekte genelleme başarısını yükseltmedi. MLP (256) daha düşük son eğitim loss'u (**0,0406**) elde etse de holdout Macro-F1 (**0,6730**), MLP (128) değerinden (**0,6739**) biraz daha düşüktür. **Logistic Regression baseline, kaydedilmiş sonuçlarda daha başarılıdır.**

![128 ve 256 nöronlu MLP için training loss eğrileri](figures/training_loss_comparison.png)

## Problem ve veri

Amaç, İngilizce `Consumer complaint narrative` metninden ilgili `Product` kategorisini tahmin etmek. Kaynak çalışma dosyası **`complaints_train_5000.csv`** (5.000 kayıt). `Complaint ID` model girdisi değildir.

| Etiket (MLP) | Kategori |
|---:|---|
| 0 | Credit reporting, credit repair services, or other personal consumer reports |
| 1 | Debt collection |
| 2 | Credit reporting |
| 3 | Mortgage |

`train_test_split(test_size=0.2, random_state=42, stratify=y)` ile **4.000 eğitim / 1.000 holdout** örneği ayrıldı. Holdout setinde her sınıftan 250 örnek bulunuyor. TF-IDF yalnızca eğitim verisinde öğrenildi (`fit_transform`); holdout için sadece `transform` uygulandı.

> **Değerlendirme notu:** Notebook kodu `val_loader` ismini kullansa da, bu veri `X_test` bölümünden geliyor; bağımsız validation kümesi yok. Aynı holdout kümesi hiperparametre karşılaştırmasında da kullanıldığı için metrikler nihai, tarafsız test sonucu olarak yorumlanmamalıdır. Bir sonraki deneyde ayrı **train / validation / test** uygulanacak.

## Uygulama akışı

```text
complaints_train_5000.csv
     │
     ├─ Metin temizleme + 4 kategoriye eşleme
     ├─ Stratified split: 4.000 train / 1.000 holdout
     ├─ TF-IDF, 1.500 özellik (train'de fit)
     ├─ Dense float32 Tensor + long etiketler
     ├─ MLP: 1500 → 128/256 → 32 → 4
     │      ReLU       ReLU     logits
     ├─ CrossEntropyLoss + Adam (lr=0.001)
     ├─ DataLoader(batch_size=32, shuffle=True)
     ├─ 10 epoch · forward/backward/optimizer.step
     └─ Accuracy, Macro-F1, sınıf raporu, confusion matrix
```

## Model mimarisi ve parametreler

| Katman | MLP-128 | MLP-256 |
|---|---:|---:|
| Giriş boyutu | 1500 | 1500 |
| İlk gizli katman | 128 | 256 |
| İkinci gizli katman | 32 | 32 |
| Çıkış / sınıf sayısı | 4 | 4 |
| **Eğitilebilir parametre** | **196.388** | **392.612** |
| Loss | CrossEntropyLoss | CrossEntropyLoss |
| Optimizer | Adam, lr=0.001 | Adam, lr=0.001 |
| Epoch | 10 | 10 |

Son çıkış **logits** üretir; ayrıca Softmax kullanılmaz, çünkü `CrossEntropyLoss` bunu kendi içinde uygun biçimde hesaplar.

## Hata analizi

128 nöronlu modelin en düşük F1 skorları **sınıf 0 (0,53)** ve **sınıf 2 (0,56)** için görülüyor. Bu sınıflar arasında **155 yanlış tahmin** yapıldı: 0→2 için 76, 2→0 için 79. En başarılı kategori **Mortgage (sınıf 3, F1 0,89)**.

![128 nöronlu MLP için confusion matrix](figures/confusion_matrix_mlp128.png)

Bu karışıklığın nedeni benzer kredi raporlama terimleri olabilir; kesinleştirmek için yanlış tahmin edilen gerçek metinler incelenmelidir.

## Proje dosyaları

```text
.
├── README.md
├── 03_pytorch_mlp.ipynb
├── pytorch_experiment_report.md
├── requirements.txt
└── figures/
    ├── training_loss_comparison.png
    └── confusion_matrix_mlp128.png
```

`complaints_train_5000.csv` kullanıcı tarafından ayrıca sağlanmalıdır; pakete dahil edilmemiştir. Veri seti kişisel bilgi veya lisans kısıtlamaları içeriyorsa ham şikâyetleri herkese açık repoya yüklemeyin.

## Kurulum ve çalıştırma

1. Python **3.10+** ile yeni bir sanal ortam oluşturun (PyTorch kurulumu için desteklenen sürümü kontrol edin).
2. Bağımlılıkları kurun: `pip install -r requirements.txt`.
3. `complaints_train_5000.csv` dosyasını notebook ile **aynı klasöre**, **bir üst klasöre** veya `data/` klasörüne yerleştirin.
4. `jupyter notebook` veya VS Code içinde **`03_pytorch_mlp.ipynb`** dosyasını açıp hücreleri yukarıdan aşağıya çalıştırın.
5. Notebook `cuda` mevcutsa GPU'yu, değilse CPU'yu seçer.

**Tekrarlanabilirlik uyarısı:** Notebook'ta seed tanımlıdır; ancak arşivlenmiş 128/256 karşılaştırmasında DataLoader karıştırmasının her deney için ayrı aynı seed ile yeniden başlatıldığı doğrulanamaz. GPU/kütüphane farkları ve çalışma sırası da sonuçları etkileyebilir. Notebook'taki sonuçlar kaydedilmiş bir çalışma anına aittir; CSV bu pakette olmadığından burada yeniden çalıştırılmadı. Kontrol edilmiş yeniden karşılaştırma için seed, veri ayrımı ve DataLoader generator'ını her model için eşitleyin.

## Bu projede öğrendiklerim

- Sparse TF-IDF ile dense PyTorch Tensor arasındaki farkı ve belleğe etkisini.
- `nn.Module`, `nn.Linear`, ReLU, logits, öğrenilebilir ağırlık/bias parametrelerini.
- `TensorDataset` / `DataLoader`, batch ve epoch kavramlarını.
- `zero_grad() → forward → loss → backward() → step()` eğitim döngüsünü.
- `model.eval()`, `torch.no_grad()`, `argmax(dim=1)` ve değerlendirme metriklerini.
- Macro-F1'in sınıf bazlı hataları görünür kıldığını ve bir **baseline** modelin neden vazgeçilmez olduğunu.

## Sıradaki deneyler

1. Doğru **train/validation/test** ayrımı ve her koşul için sabit random seed / veri sıralaması.
2. Epoch bazında **validation loss ve Macro-F1**, ardından early stopping.
3. `max_features` (1500/3000), `ngram_range` (unigram/bigram), TF-IDF temizleme kararlarının kıyaslanması.
4. Dropout, weight decay ve farklı `hidden_dim` değerleriyle düzenlileştirme deneyleri.
5. Sınıf 0 ↔ 2 yanlış örneklerini okuyup etiket/özellik ayrımını iyileştirme.
6. Son model kararı verildikten sonra **hiç kullanılmamış test kümesinde** tek seferlik raporlama.

**Ayrıntılı yöntem ve sınırlamalar:** [`pytorch_experiment_report.md`](pytorch_experiment_report.md). Baseline sonuçları önceki `baseline_report.md` kaydından alınmıştır; isterseniz aynı depoya bu kaynağı da ekleyebilirsiniz.
