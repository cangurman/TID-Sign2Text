# Araştırma Raporu — TİD Sign-to-Text (11 Eylül 2026)

İki paralel derin araştırmanın sentezi: (A) TİD veri setleri, (B) MediaPipe +
dizi modeli tabanlı SLR sistemleri ve en iyi pratikler. Mimari kararlarımızın
gerekçeleri burada.

## 1. TİD Veri Setleri

| Veri seti | İçerik | Erişim | Bizim için |
|-----------|--------|--------|------------|
| **AUTSL** (Ankara Ünv.) | 226 izole işaret, 43 işaretçi, 38.336 video (Kinect v2, RGB+derinlik+iskelet) | ChaLearn kaydı + şifreli dosyalar; Kaggle aynaları var. Yalnız araştırma lisansı | Hemen indirilebilir tek büyük set; hastane kelimesi yok ama **ön eğitim / benchmark** için ideal |
| **BosphorusSign22k** (Boğaziçi) | 744 gloss, **428'i sağlık alanı**, 22.542 video, 6 işaretçi | E-posta ile EULA (ogulcan.ozdemir@boun.edu.tr) | Kelime dağarcığı olarak en iyi eşleşme; 6 işaretçi az |
| **HospiSign** | **33 hastane cümlesi** (BosphorusSign alt kümesi), 6 işaretçi | BosphorusSign sitesi şu an erişilemez; yazarlardan istenebilir | Projemizin birebir akademik öncülü — DTW ile real-time yapmışlar |
| **ERUSLR** (Erciyes Ünv.) | **25 acil servis kelimesi**, 49 işaretçi, 13.186 örnek | İndirme kanalı bulunamadı; yazarlarla iletişim (T. Özcan, A. Baştürk) | Alan + işaretçi çeşitliliği açısından en değerli; erişim belirsiz |
| **E-TSL** | Sürekli TİD, 24 saat, 11 işaretçi (ders videoları) | Yazarlarla iletişim (Hacettepe) | Sürekli çeviri fazı için (MVP sonrası) |
| **tidsozluk.aile.gov.tr** | Resmi sözlük: 2.000 kelime, 11.428 video | Ücretsiz erişim; ML kullanım lisansı BELİRSİZ | İşaret formu **referansı** olarak kullan; eğitim verisi olarak indirme (telif) |

**Sonuç:** Kısa vadede kendi verimizi `collect_data.py` ile toplamak (+ kullanıcının
bulacağı demo videoları `extract_landmarks.py`'den geçirmek) en hızlı yol.
Paralelde HospiSign/ERUSLR için yazarlara e-posta atmaya değer; AUTSL ön eğitim
için indirilebilir.

## 2. Mimari kararların gerekçeleri (araştırma bulguları)

- **258 boyutlu özellik vektörü** (pose 33×4 + eller 21×3×2): Standart tutorial
  hattı (Nick Renotte, github.com/nicknochnack/ActionDetectionforSignLanguage)
  1662 boyut kullanır ama bunun 1404'ü yüz mesh'i ve az sinyal taşır; ciddi
  takipçilerin hepsi yüzü atar. Kaggle GISLR kazananları da el+kol+dudak alt
  kümesi seçmenin en önemli faktör olduğunu gösterdi.
- **Normalizasyon** (omuz ortası merkezleme + omuz genişliği ölçekleme):
  SPOTER (WACV 2022) ve TransSLR'nin standardı; işaretçiye özgü vücut ölçülerini
  dilsel hareketten ayırır. Eksik el = sıfır dolgu (normalizasyondan sonra
  anlamlı "yok" tokeni — Kaggle 2. çözümüyle aynı).
- **Sabit uzunluğa yeniden örnekleme** (60 kare): kırpma/padding yerine
  interpolasyonlu resampling — Kaggle kazananlarının ortak tercihi.
- **Augmentasyon** (ayna + zaman kırpma + gürültü): landmark tabanlı SLR'de
  genellemenin en büyük kaldıracı. Ayna augmentasyonu sol/sağ landmark
  değişimiyle birlikte yapılmalı (features.py'de doğru uygulandı).
- **Bi-LSTM önce, Transformer yedek**: 30–50 sınıf × 30–100 örnek ölçeğinde
  ikisi benzer doğruluğa ulaşıyor; Bi-LSTM hiperparametreye daha az duyarlı.
  Veri büyüyünce Transformer'ı dene (`--model transformer` hazır).
- **Kayan pencere + oylama + eşik**: her karede değil her 5 karede tahmin,
  son 8 tahmin üzerinde çoğunluk oyu, ~0.6-0.7 güven eşiği. Renotte tarzı
  demoların 1 numaralı hatası: eller dinlenirken tahmin üretmek —
  **çözüm: `idle` (işaret yok) sınıfı eğitime dahil edilecek** (PHRASES.md).
- **MediaPipe sürümü**: 1.0.x (Tem 2026) eski `mp.solutions` API'sini kaldırdı
  (kendi makinemizde doğrulandı) → `mediapipe==0.10.21` sabitlendi (Py 3.9–3.12;
  sistemdeki Python 3.14 desteklenmiyor, venv 3.12 ile kuruldu).
- **Eğitim/servis tutarlılığı**: Web/Flutter ön kamerası aynalıdır; canlı demo ve
  veri toplama aracı görüntüyü flip'liyor, web istemcisi aynı 258 vektörü üretiyor.
  FPS uyuşmazlığı sessiz katildir — kayıt ve servis aynı ~30fps hedefler.

## 3. Benzer sistemler

- **SignAll SDK**: MediaPipe el takibi üzerine kurulu tek kameralı ASL tanıma —
  landmark yaklaşımının ticari kanıtı.
- **PopSign (Google/Georgia Tech)**: Kaggle GISLR modellerinin üretimdeki hali;
  cihaz üstü TFLite landmark modeli (bizim Flutter fazımızın emsali).
- **Signapse**: ters yön (metin→işaret videosu), emsal değil.
- **HospiSign**: hastane senaryolu TİD tanıma platformu — projemizin birebir
  akademik öncülü; 33 cümleyle DTW kullanmışlar, biz öğrenilmiş dizi modeliyle
  daha iyisini hedefliyoruz.
- 2026 itibarıyla genel dağıtımda büyük ölçekli kamera-tabanlı işaret→metin
  ürünü yok; 30–50 izole cümle ölçeği tam olarak çalışan sistemlerin olduğu yer.

## 4. İkinci araştırma turu (11 Eylül 2026): işaretçiden-bağımsızlık + sürekli işaretleme

### 4a. İşaretçiden-bağımsızlık — %8 tavan değil, veri sorunu
- **ChaLearn 2021 AUTSL yarışması** (43 işaretçi, işaretçi-ayrık test): kazanan
  SAM-SLR %98.4; kritik bulgu — **yalnız iskelet/landmark girdisiyle SL-GCN
  çok-akışlı model %95.45** (arxiv.org/abs/2103.08833). Yani landmark yaklaşımı
  işaretçiden-bağımsızlıkta tavana sahip değil; 31 işaretçiyle eğitim + doğru
  mimariyle çözülüyor. Bizim %8, tek işaretçili eğitimin doğal sonucu.
- Kazanan reçete: 27 üst-gövde noktası grafiği, **4 akış: eklem + kemik (bone)
  + eklem-hareket + kemik-hareket**, ST-GCN türevi, ağır augmentasyon, ensembl.
  Bone ve motion akışları başlı başına birer "işaretçi-bağımsızlaştırma"
  normalizasyonudur (mutlak vücut geometrisini düşürür) — bizim mimariye
  eklenebilir ilk şey.
- **OpenHands** (AI4Bharat, arxiv.org/abs/2110.05877): MediaPipe pozu üzerinde
  önceden eğitilmiş, **AUTSL/Türkçe dahil** hazır checkpoint'ler; bizim boru
  hattıyla doğrudan uyumlu — transfer öğrenme için ilk aday.
- Transfer kanıtı: Logos ön-eğitimi AUTSL'yi 96.58→97.83'e taşıyor
  (arxiv.org/abs/2505.10481). Az-örnekli kişiselleştirme literatürü zayıf;
  5-10 örnekle meta-öğrenme öneriliyor ama standart benchmark yok.
- Öncelik sırası (kanıta göre): çok işaretçi > bone+motion akışları >
  augmentasyon > ön-eğitimli ağırlık > işaretçi-düşmanıl (adversarial) baş.

### 4b. Sürekli işaretleme — hayalet kelimelerin bilimsel adı ve çözümü
- Bizim "geçiş penceresi" problemimizin klasik adı: **movement epenthesis**
  (işaretler arası ara hareket). HMM çağında açık geçiş modelleriyle çözülürdü;
  modern çözüm **CTC'nin blank etiketi** — hizalamayı marjinalize eder, geçiş
  kareleri blank'e emilir, hiçbir pencere kelimeye zorlanmaz. Bizim `_gecis`
  sınıfı elle yazılmış bir blank'ti; CTC bunu hizalamayla birlikte kendisi
  öğrenir. Sıradaki mimari adım: mevcut kodlayıcının üstüne CTC başı.
- Salt-iskelet CTC çalışıyor: **CoSign** (ICCV 2023) PHOENIX14-T'de ~%19.5 WER
  ile RGB SOTA'sının 1 puan yakınında. TwoStream-SLR'nin keypoint akışı tek
  başına %27 WER.
- **Birleştirilmiş izole klipler** literatürde tanınan bir teknik; bilinen
  sınırları tam bizim yaşadıklarımız: coarticulation yok, sınırlar keskin.
  İyileştirme: sınırda enterpolasyonlu yumuşak geçiş sentezi
  (arxiv.org/html/2506.09643v1) — make_sentences'a eklenebilir.
- **TİD'de olumsuzluk/soru yüzle kodlanır** (baş sallama, kaş; Gökgöz 2011).
  TİD özel çalışması: yüzden soru/olumsuzluk/ağrı tanıma %78.5
  (ieeexplore.ieee.org/document/8936081 — "ağrı" sınıfı hastane senaryomuz
  için dikkat çekici). Kaş+dudak+baş pozu noktaları vektöre eklenmeli.
- Gloss-free LLM çevirisi (Sign2GPT, SignLLM, MMSLT) yüzlerce saat eşli veri
  istiyor — bizim ölçek için henüz değil; gloss üzerinden CTC doğru ilk adım.
- Alternatif: açık sınır tespiti (Moryossef 2023, BIO-etiketleme, poz girdili)
  → segmentle + izole sınıflandırıcıyı segment başına çalıştır. CTC'den basit
  ama sınır hatası zincirleme hata üretir; yedek plan olarak not edildi.

## 5. Dürüst değerlendirme notu
AUTSL'de rastgele bölme %95.9'a karşı işaretçiden bağımsız bölme %62 (baseline) —
aradaki uçurum, aynı kişinin örnekleriyle test etmenin ne kadar yanıltıcı
olduğunu gösterir. Demo sonrası mutlaka farklı kişilerle test edilmeli.
