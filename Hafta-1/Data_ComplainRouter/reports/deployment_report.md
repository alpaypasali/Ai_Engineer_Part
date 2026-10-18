# ComplaintRouter AI — Teknik Dağıtım Raporu

## 1. Neden ilk uygulamada Logistic Regression tercih edildi?

Bu projede amaç, müşteri şikâyetlerini dört finansal kategoriye ayıran çalışan bir başlangıç modeli oluşturmaktı. **TF-IDF + Logistic Regression**, metin sınıflandırma için hızlı eğitilebilen, nispeten az kaynak tüketen ve sonuçları kolay değerlendirilebilen bir temel yöntemdir. Böylece daha karmaşık NLP modellerine geçmeden önce tüm uygulama akışını kurabildim.

Mevcut değerlendirme çıktısında **accuracy %72,60**, **macro F1 %72,31** elde ettim. Performans sınıflara göre farklılaşıyor; bu nedenle bu sonuçlar modelin üretim için yeterli olduğunu tek başına kanıtlamıyor.

## 2. Modeli yeniden eğitmeden nasıl kullandım?

Eğittiğim Pipeline'ı `joblib.dump()` ile `models/complaintrouter_lr.joblib` dosyasına kaydettim. `predict.py` açıldığında `joblib.load()` ile bu dosyayı yükleyip `predict()` ve `predict_proba()` metotlarını kullandım. Uygulamayı Jupyter Notebook açmadan terminalden çalıştırabiliyorum:

```bash
python predict.py --text "My mortgage payment is incorrect"
```

Bu işlemde modelin öğrenilmiş parametreleri dosyadan gelir; yeniden `fit()` çağrılmaz. Terminalden farklı çalıştırmalarda tahmin alabildim.

## 3. TF-IDF ile modelin birlikte saklanması neden önemli?

Logistic Regression, ham metin değil TF-IDF'nin ürettiği sayısal özellikleri bekler. Modelin ağırlıkları, eğitimde öğrenilen **kelime sözlüğü, IDF değerleri ve özellik sırası** ile ilişkilidir. Farklı bir TF-IDF oluşturmak özelliklerin anlamını veya sırasını değiştirerek hatalı tahminlere neden olabilir.

Bu nedenle `Pipeline` içinde metin temizleme, `TfidfVectorizer(max_features=3000)` ve `LogisticRegression` adımlarını birlikte sakladım. Böylece tahmin sırasında eğitimde öğrenilen TF-IDF dönüşümü kullanılıyor.

## 4. Eğitim ve tahmin aşamalarında neden aynı preprocessing kullanılmalı?

Metin temizleme değişirse aynı cümle eğitimde ve tahminde farklı özelliklere dönüşebilir. Bu, modelin performansını ve tutarlılığını bozabilir.

Projede `clean_text()` fonksiyonu metni küçük harfe dönüştürüyor, gizlenmiş `XXXX` benzeri ifadeleri ve noktalama işaretlerini kaldırıyor, fazla boşlukları temizliyor. Fonksiyonları `preprocessing.py` içinde tuttum. Pipeline'daki `FunctionTransformer` bu temizlemeyi tahmin sırasında da uyguluyor. `predict.py` içinde ayrıca yalnızca **temizleme sonrası metnin boş kalıp kalmadığını** kontrol ediyorum; ham metni Pipeline'a veriyorum.

## 5. İnsan incelemesi mekanizması ne zaman faydalı olabilir?

Tahmin edilen sınıfın olasılığı **0,60'ın altındaysa** sonucu `Human Review Required` olarak işaretleyen deneysel bir iş kuralı ekledim. Böylece modelin daha belirsiz bulduğu şikâyetler doğrudan otomatik yönlendirilmek yerine incelenebilir.

Örneğin `My laptop battery is not working.` metni finansal kategorilerin dışında olmasına rağmen model bir sınıf seçti; **0,32** olasılık verdiği için insan incelemesine yönlendirildi. Ayrıca en yüksek iki sınıf olasılığının birbirine yakın olması da belirsizlik işareti olabilir.

**Sınırlılık:** 0,60 doğrulanmış bir güven eşiği değildir. Yüksek olasılık her zaman doğru tahmin anlamına gelmez; mevcut mekanizma bütün alan dışı (OOD) girdileri yakalayamaz.

## 6. Production ortamına çıkmadan önce neler eksik?

- **Daha kapsamlı testler:** Geçerli/boş/çok uzun metinler, eksik model dosyası ve kaydet-yükle tutarlılığı için otomatik `pytest` testleri; bu testlerin tamamı henüz doğrulanmadı.
- **Kalibrasyon ve eşik doğrulama:** `predict_proba()` değerlerinin güvenilirliği ölçülmeli, %60 eşiği validation sonuçları ve yanlış yönlendirme maliyetlerine göre seçilmeli.
- **Alan dışı girdiler:** Finans dışı şikâyetler için özel OOD veya reddetme yaklaşımı geliştirilmeli; yalnızca düşük olasılığa güvenilmemeli.
- **Model kalitesi:** Özellikle daha düşük F1 alan kategoriler incelenmeli; benzer kredi raporlama etiketleri arasındaki karışıklık analiz edilmeli ve daha temsilî veriyle değerlendirme yapılmalı.
- **Operasyonel hazırlık:** Bağımlılıklar ve Python sürümü sabitlenmeli; model sürümlendirme, loglama, izleme, hata yönetimi ve gerektiğinde geri alma süreci hazırlanmalı.
- **Güvenlik ve gizlilik:** Şikâyetlerdeki kişisel bilgiler korunmalı; yalnızca güvenilir `joblib` dosyaları yüklenmeli ve modele erişim kuralları tanımlanmalı.

## Sonuç

ComplaintRouter AI bugün terminalden çalışan bir **prototiptir**. Kaydedilmiş TF-IDF + Logistic Regression Pipeline'ı ile dört finansal kategoriden birini ve olasılığını döndürüyor; düşük olasılıklı durumlarda manuel inceleme öneriyor. Gerçek müşterilerle production kullanımından önce test, kalibrasyon, OOD, güvenlik ve izleme çalışmalarının tamamlanması gerekiyor.
