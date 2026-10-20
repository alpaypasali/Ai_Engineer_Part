ShopperIntent — Veri Kartı ve Veri Sızıntısı Analizi
Veri setine ilk bakış
Bu projede bir alışveriş oturumunun satın almayla sonuçlanıp sonuçlanmayacağını tahmin etmeye çalışıyorum. Hedef değişkenim Revenue; True satın alma gerçekleştiğini, False gerçekleşmediğini gösteriyor. Çalıştığım veri setinde 12.330 oturum ve 17 giriş özelliği bulunuyor. Veri dengesiz: satın alma gerçekleşen oturumların oranı yaklaşık %15,5, gerçekleşmeyenlerin oranı yaklaşık %84,5.

Sütun tiplerini incelerken OperatingSystems, Browser, Region ve TrafficType alanlarının int olmasına rağmen sürekli sayısal değerler değil, kategorik kodlar olduğunu fark ettim. Month ve VisitorType da kategorik; Weekend ikili bir özellik. Administrative, Informational, ProductRelated gibi alanlar ise ziyaret edilen sayfa sayılarını, _Duration alanları bu sayfalarda geçirilen süreyi temsil ediyor.

Tekrar eden kayıtları da kontrol ettim. Aynı özelliklere ve aynı hedefe sahip bazı satırlar birden fazla kez görünüyor; paylaştığım örneklerde en sık tekrarlanan satır 14 kez bulunuyor. Bu tekrarların önemli bir kısmında ProductRelated=1, ProductRelated_Duration=0, BounceRates=0.2, ExitRates=0.2, PageValues=0 ve Revenue=False. Bu örüntü düşük etkileşimli oturumlarla uyumlu. Yine de elimde benzersiz oturum kimliği olmadığı için bunların veri hatası mı, farklı kişilerin benzer oturumları mı olduğunu kesinleştiremiyorum. Bu nedenle şimdilik otomatik olarak silmedim.
A) Oturum başında tahmin yapacaksam hangi bilgileri kullanabilirim?
Burada ilk fark ettiğim şey şu oldu: CSV'deki bir sütunun var olması, o değerin oturum başında bilindiği anlamına gelmiyor.

Örneğin ProductRelated_Duration oturum boyunca ürün sayfalarında geçirilen toplam süreyi gösteriyorsa, bunu ziyaretçi siteye girdiği anda bilemem. Aynı durum ProductRelated, Administrative ve Informational gibi sayfa sayaçlarının oturum sonunda kesinleşen toplamları için de geçerli. Bu değerleri başlangıçtaki tahmine verirsem model, karar anında henüz oluşmamış davranışlardan faydalanmış olur. Bu da zamansal veri sızıntısı yaratır.

Buna karşılık Month ve Weekend gibi bilgiler giriş anında elde edilebilir. Browser, OperatingSystems ve benzeri alanlar da ilk istekte tespit edilebiliyorsa kullanılabilir; fakat bunun uygulamada doğrulanması gerekir.

Kararım: Önce tahmini hangi anda yapacağımı netleştirmeliyim. Oturum başlangıcında tahmin yapacaksam yalnızca o ana kadar bilinebilen özellikleri kullanmalıyım. Oturum ortasında tahmin yaparsam süre ve sayfa sayılarının sadece o ana kadar birikmiş değerlerini kullanabilirim.
B) PageValues alanında ne fark ettim?
Tekrar eden satırları incelerken PageValues değerinin gösterdiğim düşük etkileşimli oturumların tamamında 0.0 olduğunu gördüm. Ancak bu gözlem, bütün veri setinde değerin sıfır olduğu ya da PageValues ile satın alma arasında kesin bir ilişki bulunduğu anlamına gelmez. Önce dağılımını ve Revenue ile ilişkisini ölçmem gerekiyor.

Bu sütun sayfalara atanan analitik bir değer gibi görünüyor. Fakat CSV tek başına bu değerin hangi formülle ve hangi tarihte hesaplandığını söylemiyor. Özellikle satın alma verilerinden yararlanılarak oluşturulduysa önemli bir soru ortaya çıkıyor: Bu değer, tahmin etmek istediğim satın alma gerçekleştikten sonra mı hesaplandı, yoksa oturum başlamadan önceki geçmiş verilerden mi elde edildi?

İlk durumda veri sızıntısı olabilir. İkinci durumda, değer tahmin anında erişilebilir durumdaysa güvenle kullanılabilir. Bunu sadece Revenue ile yüksek korelasyon gösterip göstermediğine bakarak kanıtlayamam.

Notebook'ta aşağıdaki kontrolleri yapabilirim:
# PageValues değerlerinin dağılımı
print(df["PageValues"].describe())
print("PageValues = 0 oranı:", (df["PageValues"] == 0).mean())

# Satın alan / almayan gruplarda nasıl değişiyor?
print(df.groupby("Revenue")["PageValues"].agg(["count", "mean", "median"]))

# Sıfır olan ve olmayan grupların satın alma oranları
print(df.groupby(df["PageValues"] > 0)["Revenue"].mean())
Bu analizler ilişkinin gücünü gösterir, hesaplamanın ne zaman yapıldığını göstermez. Canlı kullanım için veriyi üreten sistemde PageValues değerinin oluşma zamanını kontrol etmeli, tahmin zamanında mevcut olup olmadığını test etmeliyim. Ayrıca modeli PageValues ile ve olmadan karşılaştırırım; beklenmedik derecede büyük performans farkı varsa olası sızıntıyı ayrıntılı incelerim.

Şimdilik kararım: Bu sütunun üretim zamanı doğrulanana kadar oturum başı tahmin modelimde PageValues kullanmayacağım. Bunun sızıntı olduğunu kanıtladığım için değil, güvenli olduğundan henüz emin olmadığım için yapıyorum.
C) Rastgele train/test ayrımı yeterli mi?
Veri setinde Month bilgisi var fakat tam oturum tarihi, saat bilgisi ya da yıl bilgisi görünmüyor. Bu nedenle satırları gerçek zaman sırasına güvenilir biçimde koyamıyorum.

Rastgele train/test ayrımı, aynı türden ve benzer dönemlerden gelen oturumlar üzerinde başlangıç performansını ölçmek için kullanılabilir. Fakat modeli geçmiş verilerle eğitip ileride gelecek ziyaretçilere uygulayacaksam tek başına yeterli olmayabilir. Ziyaretçi davranışları, kampanyalar veya satın alma oranı dönemlere göre değişebilir. Rastgele ayırım bu farkları gizleyebilir.

Ayrıca veri setinde aynı özelliklere sahip tekrar grupları bulunuyor. Bu kayıtlar hem train hem test tarafına düşerse test sonucu bazı senaryolarda olduğundan iyimser görünebilir. Tekrarların gerçekten aynı oturumu temsil edip etmediğini bilmediğim için bu riski ayrı olarak incelemem gerekiyor.

İzleyeceğim yol:
1. İlk aşamada Revenue dağılımını koruyan stratified train/test ayrımıyla başlangıç modeli kurarım.
2. Modelleme sırasında encoder, scaler ve benzeri öğrenilen dönüşümleri sadece train verisinde fit ederim; cross-validation kullanırsam bu adımları Pipeline içine alırım.
3. Tekrar eden özellik gruplarının train ve test arasında ne kadar örtüştüğünü ayrıca kontrol ederim.
4. Gerçek zaman damgalı kayıtlar elde edersem geçmiş dönemlerde eğitim, sonraki dönemlerde test yaparak zamana dayalı değerlendirme uygularım.
5. Veri dengesiz olduğu için yalnızca Accuracy'ye güvenmem; özellikle Average Precision (AP), Recall, Precision ve F1 metriklerine bakarım.
Sonuç
Bu incelemede benim için en önemli konu, özelliğin ne anlattığı kadar ne zaman kullanılabilir hâle geldiği oldu. Oturum sonunda ortaya çıkan bilgileri başlangıç tahmininde kullanırsam gerçekçi olmayan bir model elde edebilirim. PageValues bu açıdan özellikle kontrol edilmesi gereken bir özellik. Ayrıca tarih bilgisi eksik olduğundan mevcut CSV ile gerçek bir gelecek dönem testi yaptığımı söyleyemem.

Not: Buradaki sayısal gözlemler, şimdiye kadar incelediğim sütunlar ve paylaştığım örnek kayıtlarla sınırlıdır. PageValues için tüm dağılım sonuçlarını veya canlı sistemdeki hesaplanma zamanını henüz doğrulamadım; bunları kesin bulgu gibi yazmıyorum.