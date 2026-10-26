# AirPulse — Data Card

## 1. Problem

Aotizhongxin istasyonunda **bir sonraki saatin PM2.5 değerini (µg/m³)** tahmin etmek.

## 2. Veri kaynağı

- **Veri seti:** UCI Beijing Multi-Site Air Quality
- **Kaynak:** https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data
- **İstasyon:** Aotizhongxin
- **CSV:** `PRSA_Data_Aotizhongxin_20130301-20170228.csv`
- **Sıklık:** Saatlik

## 3. Veri dönemi ve toplam kayıt

| Alan | Değer |
|---|---|
| İlk tarih | 2013-03-01 00:00 (kaynakta belirtilen dönem başlangıcı) |
| Son tarih | 2017-02-28 23:00 (kaynakta belirtilen dönem sonu) |
| Toplam kayıt | 35.064 (orijinal Aotizhongxin veri setinin kayıt sayısı) |

*Bu değerler kaynak dosyaya ait referans bilgilerdir; çalışılan CSV ile doğrulanmalıdır.*

## 4. Hedef

- **Hedef:** `PM2.5(t+1)`
- **Yeni sütun:** `target_pm25_next_hour`
- **Yöntem:** `shift(-1)` ile sonraki gözlemin değerini hedefe taşıma.
- **Geçerlilik:** Sadece ardışık zaman damgaları tam 1 saat aralıklıysa ve iki PM2.5 ölçümü de mevcutsa kullanılır.

## 5. Eksik değerler

| Sütun | Eksik kayıt | Eksik oranı |
|---|---:|---:|
| `PM2.5` | CSV ile doğrulanacak | CSV ile doğrulanacak |
| `PM10` | CSV ile doğrulanacak | CSV ile doğrulanacak |
| `TEMP` | CSV ile doğrulanacak | CSV ile doğrulanacak |
| Diğer sütunlar | CSV ile doğrulanacak | CSV ile doğrulanacak |

Orijinal CSV bu sohbette bulunmadığı için eksik ölçüm sayıları tahmin edilmedi. Sayıları ve tüm sütunları rapora eklemek için aşağıdaki komut çalıştırılır; **eksik değerler doldurulmaz**:

```bash
python reports/update_data_card.py
```

## 6. Zaman bütünlüğü

| Kontrol | Sonuç |
|---|---|
| Aynı saatte tekrarlanan kayıt | CSV ile doğrulanacak |
| İlk ve son tarih arasında kaydı eksik saat | CSV ile doğrulanacak |
| Hedeflenen ölçüm sıklığı | 1 saat |

Sensör ölçümünün `NaN` olması ile zaman damgasının tamamen eksik olması farklı durumlardır.

## 7. Naive Forecast

Kural: **Bir sonraki saatin tahmini, mevcut saatin PM2.5 değerine eşittir.**

`ŷ(t+1) = PM2.5(t)`

Bu yöntem RNN ve LSTM modelleri için başlangıç referansıdır. Bugünkü analiz keşif amaçlıdır; henüz kronolojik train/validation/test değerlendirmesi yapılmadı.

## 8. Veri sızıntısı riskleri

- Gelecekteki PM2.5 değerinin girdi olarak kullanılması.
- Veriyi `shuffle=True` ile rastgele bölerek gelecekteki örnekleri eğitime taşımak.
- Eksik değerleri gelecek ölçümleri kullanarak doldurmak.
- Eksik satırları sildikten sonra `shift(-1)` ile yanlış saatleri eşlemek.

## 9. Sınırlamalar

- Yalnızca Aotizhongxin istasyonu inceleniyor.
- Bazı saatlerde sensör ölçümleri eksik olabilir.
- Naive Forecast ani yükseliş ve düşüşleri öngöremez.
- Nihai değerlendirme henüz yapılmadı.
