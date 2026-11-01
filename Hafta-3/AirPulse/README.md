# AirPulse — Next-Hour PM2.5 Forecasting

**Time Series Regression · Naive Forecast · PyTorch RNN/LSTM · Input Validation · Pytest**

**Durum:** Eğitim ve araştırma amaçlı çalışan prototip. **Production-ready değildir; hava kalitesi uyarı veya sağlık kararlarında kullanılmamalıdır.**

AirPulse, **Aotizhongxin** istasyonunda geçmiş **24 ardışık saatlik PM2.5** ölçümünü kullanarak **bir sonraki saatin PM2.5 yoğunluğunu (µg/m³)** tahmin etmeyi amaçlar. Naive Forecast, vanilla RNN ve LSTM karşılaştırılmış; eğitilmiş modeller kaydedilmiş, bağımsız bir kronolojik test döneminde değerlendirilmiş ve geçersiz girişleri reddeden bir tahmin demosu hazırlanmıştır.

> **Son doğrulanan terminal çalıştırması:** `python demo_predict.py` başarılı; `python -m pytest -v tests/test_inference.py` → **3 passed**. Demo, veri setindeki son geçerli tarihsel pencereyi kullanır; canlı tahmin hizmeti değildir.

## 1. Veri kaynağı, atıf ve lisans

- **Veri seti:** [Beijing Multi-Site Air Quality — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data)
- **Veri seti hazırlayan:** Song Chen
- **Kaynak gösterimi:** Chen, S. (2017). *Beijing Multi-Site Air Quality* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5RK5G
- **Veri lisansı:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- **UCI kapsamı:** Pekin'deki 12 istasyondan 2013-03-01–2017-02-28 saatlik kirlilik ve meteoroloji ölçümleri.
- **Bu projede kullanılan alt küme:** **Aotizhongxin** istasyonu, `PRSA_Data_Aotizhongxin_20130301-20170228.csv`.

**Kullanım koşulları:** CC BY 4.0 kapsamında veri paylaşılabilir ve uyarlanabilir; kaynağa, yazara, lisansa atıf yapılmalı ve değişiklik yapıldığı belirtilmelidir. AirPulse bu veri setini **tek istasyona filtreleyerek**, PM2.5 için **eksik ölçümlü/ardışık olmayan pencereleri çıkararak** ve **24 saatlik dizilere dönüştürerek** kullanır. Bunlar projede uygulanan dönüşümlerdir. Veri setinin CC BY 4.0 lisansı **proje kodunun otomatik olarak aynı lisansa sahip olduğu anlamına gelmez**; kod için ayrıca bir lisans belirtilmediyse izinleri varsaymayın.

## 2. Problem ve modeller

**Input:** Geçmiş 24 ardışık saatin PM2.5 ölçümleri (her saat 1 özellik); RNN/LSTM giriş şekli `(batch, 24, 1)`.  
**Target:** Bir sonraki saatin `PM2.5(t+1)` değeri, **µg/m³**.  
**Problem türü:** Tek değişkenli zaman serisi regresyonu.

| Model | Mantık / mimari | Öğrenilebilir parametre |
|---|---|---:|
| **Naive Forecast** | `ŷ(t+1)=PM2.5(t)`; son ölçümü tekrarlar | 0 |
| **Vanilla RNN** | `nn.RNN(1, 32, num_layers=1, batch_first=True)` → son adım → `Linear(32,1)` | 1.153 |
| **LSTM** | `nn.LSTM(1, 32, num_layers=1, batch_first=True)` → son hidden state → `Linear(32,1)` | 4.513 |

RNN/LSTM için kullanılan sabit eğitim ayarları: **10 epoch**, **Adam (learning rate=0.001)**, **MSELoss**, **batch size=32**, **torch seed=42**. Final test öncesinde özgün ağırlıklar mevcut olmadığından aynı protokolle **yeniden eğitilip kaydedilen** modeller kullanıldı. Yeni ve eski modellerin validation sonuçları birbirine karıştırılmamıştır.

## 3. Proje yapısı

```text
AirPulse/
├── data/
│   └── PRSA_Data_Aotizhongxin_20130301-20170228.csv
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_naive_forecast.ipynb
│   ├── 03_temporal_split_and_lags.ipynb
│   ├── 04_rnn_baseline.ipynb
│   ├── 05_lstm_comparison.ipynb
│   ├── 06_model_comparison_audit.ipynb
│   └── 07_final_evaluation.ipynb
├── models/
│   ├── rnn_state_dict.pth
│   ├── lstm_state_dict.pth
│   ├── input_scaler.joblib
│   └── target_scaler.joblib
├── src/
│   └── validation.py
├── tests/
│   └── test_inference.py
├── reports/
│   ├── final_test_results.csv
│   ├── week_03_final_report.md
│   └── ...
├── demo_predict.py
├── AirPulse_model_card.md
└── README.md
```

Bu ağaç ana dosyaları gösterir; `reports/` ve `figures/` altındaki diğer çıktıların tamamını listelemez. İsteğe bağlı `requirements.txt` dosyası kullanılabilir.

## 4. Kurulum

**Gereksinim:** Python, PyTorch, NumPy, pandas, scikit-learn, matplotlib, joblib, pytest. Jupyter notebook'ları çalıştırmak için Jupyter/IPython da gerekebilir.

```bash
# Proje ana dizininde:
cd Hafta-3/AirPulse

# Kullanılan Conda ortamı zaten mevcutsa:
conda activate myenv

# Gereken paketler (ortamda yoksa):
python -m pip install torch numpy pandas scikit-learn matplotlib joblib pytest jupyter ipykernel
```

Alternatif olarak projenizde `requirements.txt` mevcutsa `python -m pip install -r requirements.txt` kullanabilirsiniz. Bağımlılıkların birebir sürümlerini sabitleyen bir ortam kilit dosyası henüz doğrulanmadığı için **farklı makinelerde tamamen bit düzeyinde aynı yeniden eğitim sonucu garanti edilmez**.

Veri setini UCI kaynağından indirip Aotizhongxin CSV'sini `data/` klasörüne koyun. Yeniden eğitim yapmadan demo için `models/` klasöründeki **LSTM ağırlıkları ve iki scaler** de gereklidir; demo bunları üretmez.

## 5. Veri hazırlama ve kronolojik ayrım

1. `year`, `month`, `day`, `hour` sütunlarından `timestamp` oluştur; kronolojik sırala.
2. Her örnek için geçmiş **24 saatlik PM2.5** ölçümlerini `X`, sonraki saati `y` olarak al.
3. `NaN` içeren giriş/hedef pencerelerini veya **1 saat aralıklı olmayan** pencereleri çıkar.
4. Pencereleri **hedef zamanına göre** Train / Validation / Test olarak ayır.
5. `input_scaler` ve `target_scaler` (`StandardScaler`) yalnızca training verisine `fit()` edilir; diğer bölümlere sadece `transform()` uygulanır.
6. Metrikler için model tahminlerini `target_scaler.inverse_transform()` ile orijinal **µg/m³** birimine döndür.

| Bölüm | Hedef zaman sınırı | Geçerli pencere sayısı |
|---|---|---:|
| Train | `< 2015-12-18 16:00` | **21.177** |
| Validation | `2015-12-18 16:00 ≤ t < 2016-07-24 20:00` | **4.266** |
| Test | `t ≥ 2016-07-24 20:00` | **4.619** |

Ham Aotizhongxin CSV: **35.064 satır**, PM2.5 sütununda **925 eksik değer**; toplam **30.062 geçerli pencere**. İlk validation/test pencerelerinin önceki döneme ait geçmiş ölçümleri içermesi, ölçümler tahmin anında erişilebiliyorsa tek başına leakage değildir.

## 6. Validation ve bağımsız final test

**Aşağıdaki validation ve test sonuçları aynı yeniden eğitilmiş model sürümlerine aittir.** Her model validation'da **4.266**, testte **4.619** aynı hedef saatinde değerlendirilmiştir; MAE/RMSE birimi **µg/m³**.

| Model | Validation MAE ↓ | Validation RMSE ↓ | Test MAE ↓ | Test RMSE ↓ |
|---|---:|---:|---:|---:|
| Naive | 10.3748 | 21.5890 | 11.1121 | 21.2815 |
| RNN (kaydedilmiş yeni sürüm) | 10.5165 | 21.1358 | 11.2278 | 20.7872 |
| **LSTM (kaydedilmiş yeni sürüm)** | **10.3180** | **20.5135** | **11.0416** | **20.1592** |

Test döneminde en düşük MAE ve RMSE **LSTM** tarafından elde edildi. RNN, Naive'e göre RMSE'de daha iyi; MAE'de daha kötü. Validation'dan teste geçişte **üç modelde de MAE yükseldi, RMSE düştü**. Önceden seçilen **RMSE önceliği**, test sonuçları görüldükten sonra değiştirilmedi.

**Önceki sürümlerle ayrım:** İlk validation karşılaştırmasında Naive ≈ `10.37/21.59`, RNN ≈ `10.66/21.14`, LSTM ≈ `10.47/20.78` (MAE/RMSE) idi. `rnn_predictions.npz` dosyasının yanlışlıkla LSTM tahminleri içermesi düzeltildi. Bu `.npz` dosyaları yalnızca **eski validation tahminleri** içindir; model ağırlıkları değildir. Yeni checkpoint sonuçları `reports/final_test_results.csv` içindedir.

## 7. Demo — notebook dışında tahmin

Projenin ana dizininden:

```bash
python demo_predict.py
```

Demo, veri setindeki en son geçerli **24 ardışık saatlik** pencereyi bulur, `src/validation.py` ile doğrular, **training üzerinde fit edilmiş** scaler'ları ve `models/lstm_state_dict.pth` dosyasını yükler. Sonraki saatin tahminini orijinal birime çevirir.

**Gerçek çalıştırma örneği:**

```text
Model           : LSTM
Station         : Aotizhongxin
Sequence Length : 24 hours
Input End Time  : 2017-02-28 23:00:00
Forecast Time   : 2017-03-01 00:00:00
Predicted PM2.5 : 22.26 µg/m³
Status          : Prediction completed
Usage           : Educational demo only
```

Bu örnek, **2017 tarihli verilerden** elde edilen tarihsel demodur; güncel veya canlı hava kalitesi tahmini değildir.

## 8. Otomatik testler ve ölçülebilir iyileştirme

```bash
python -m pytest -v tests/test_inference.py
```

**Son doğrulanmış sonuç:** `3 passed in 0.30s` (yerel terminal çalıştırması).

| Hatalı girdi | Doğrulama öncesi | Doğrulama sonrası |
|---|---|---|
| 23 saat (24 yerine) | Tahmin üretildi | `ValueError` ile reddediliyor |
| `NaN`/`Inf` | Geçersiz tahmin üretilebiliyordu | `ValueError` ile reddediliyor |
| Bir saatlik zaman boşluğu | Tahmin üretildi | `ValueError` ile reddediliyor |

**Measurable improvement:** Üç hatalı girdinin doğru reddedilmesi **0/3 → 3/3**. Bu bir **güvenilirlik iyileştirmesidir**; MAE/RMSE iyileştirmesi değildir.

## 9. Sınırlamalar ve güvenli kullanım

- Yalnızca **Aotizhongxin** istasyonu değerlendirilmiştir; başka istasyonlara aktarım doğrulanmadı.
- Model girdisi sadece PM2.5 geçmişidir; sıcaklık, rüzgâr, vb. meteorolojik veriler dahil değildir.
- Kesintisiz **24 sonlu PM2.5 değeri** gerektirir; eksik saat veya `NaN`/`Inf` varsa tahmin reddedilir.
- Ani kirlilik yükseliş ve düşüşlerinde büyük hatalar gözlemlenmiştir. Sağlık açısından kritik yükselişleri düşük tahmin etme riski vardır.
- Tek bağımsız kronolojik test dönemi gerçek zamanlı genelleme, farklı mevsimler, sensör arızaları veya veri dağılımı kaymasına karşı yeterli garanti sağlamaz.
- Güven aralığı, canlı veri izleme, sensör doğrulama, drift takibi ve hata eşikleri üretim için doğrulanmadı.

**Production Status: NOT VALIDATED FOR PRODUCTION.** Bu projeyi **canlı hava kalitesi uyarısı, sağlık veya kamu güvenliği kararı** için kullanmayın.

## 10. İlgili dokümanlar

- [`AirPulse_model_card.md`](AirPulse_model_card.md) — modelin kullanım sınırları, metrikler, riskler.
- [`reports/week_03_final_report.md`](reports/week_03_final_report.md) — haftalık teknik rapor.
- [`reports/final_test_results.csv`](reports/final_test_results.csv) — yeniden eğitilmiş modellerin validation/test sonuçları.
- [`reports/day_05_model_audit.md`](reports/day_05_model_audit.md) — duplicate ve yanlış tahmin kaynağı denetimi.

**Veri lisansı ve atıf bilgisi:** UCI Machine Learning Repository, Dataset ID 501, CC BY 4.0; bağlantılar yukarıdadır.
