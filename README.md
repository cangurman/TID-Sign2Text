# TID-Sign2Text

> Real-time Turkish Sign Language (TİD) → Turkish text translation.
> Camera → MediaPipe landmarks → Transformer → live subtitles. Hospital-domain
> MVP with a confirmation-based communication flow. Code only — see
> "Veri Edinme" for obtaining the AUTSL dataset (research-only license).

Gerçek zamanlı Türk İşaret Dili (TİD) → Türkçe metin çeviri demosu.
Kamera → MediaPipe landmark (258 boyut/kare) → Bi-LSTM/Transformer → canlı altyazı.

## Veri Edinme (repo veri içermez)
Bu repo **yalnız kod** içerir. AUTSL veri seti (Sincan & Keles, 2020,
[arXiv:2008.00932](https://arxiv.org/abs/2008.00932)) akademik/araştırma
lisanslıdır ve yeniden dağıtılamaz; eğitilmiş ağırlıklar da bu nedenle repoda
yoktur. Kendi kopyanız için:
1. Resmi yol: [ChaLearn](https://chalearnlap.cvc.uab.cat/dataset/40/description/)
   üzerinden kayıt olup indirin, veya araştırma amaçlı aynalardan edinin.
2. Videoları `backend/data/autsl/` altına açın; `SignList_ClassId_TR_EN.csv`
   ve `train_labels.csv` aynı klasöre.
3. `prepare_autsl.py` → `extract_landmarks.py` → `train.py` (Hızlı başlangıç
   bölümündeki komutlar) — boru hattı gerisini üretir.

## Ürün Vizyonu
Hedef senaryo: hastaneye gelen, konuşamayan ve yalnız TİD bilen bir birey,
ortamda işaret dili bilen kimse yokken kameraya işaret ederek derdini anlatır;
sistem bunu görevlinin okuyacağı canlı altyazıya çevirir (HospiSign'ın modern,
öğrenilmiş-model halefi). Gerçekçi çerçeve: **kapalı alanlı (50-100 cümle)
"hastane iletişim asistanı"** — serbest/açık dağarcıklı TİD çevirisi henüz
çözülmemiş bir araştırma problemidir ve vaat edilmez.
Ürünleşme için kritik üç gereksinim (yol haritasında):
1. **Onaylı iletişim UX'i**: çeviri hastaya gösterilip evet/hayır işaretiyle
   onaylatılmadan görevliye iletilmez; düşük güvende "tekrar eder misiniz".
   Tıbbi bağlamda sessiz otomatik çeviri kabul edilemez.
2. **Yüz özellikleri**: TİD'de olumsuzluk/soru büyük ölçüde yüzle kodlanır;
   özellik vektörüne dudak+kaş noktaları eklenmeli (var/yok ayrımı için şart).
3. **İşaretçiden bağımsızlık**: ürünün gerçek metriği görülmemiş kişideki
   doğruluktur; çok işaretçili eğitim bu yüzden ana yatırımdır.

MVP kapsamı: Acil Durum / Hastane senaryosuna ait 30–50 cümle kalıbının
**izole cümle tanıması** (sürekli çeviri değil). Cümle listesi: `docs/PHRASES.md`.
Araştırma ve mimari gerekçeler: `docs/RESEARCH.md`.

## Kurulum durumu

- [x] Python venv: `backend/venv` (sistemdeki 3.14 MediaPipe'la uyumsuz; 24 Eyl
      kurulumu Python 3.11.9 ile yapıldı ve çalıştı)
- [x] Bağımlılıklar kurulu (`mediapipe==0.10.21` — 1.0.x eski API'yi kaldırdı).
      Sıfırdan kurulumda `pip install -r requirements.txt` bağımlılık çözücüsü
      takıldı (24 Eyl); çalışan yol: `numpy==1.26.4`, `mediapipe==0.10.21 --no-deps`,
      `protobuf==4.25.3`, `opencv-contrib-python==4.10.0.84`, torch cu124; jax/jaxlib
      kurulmadı (`mp.solutions.holistic` onsuz çalışıyor, `pip check` uyarısı zararsız).
      Her kurulumdan sonra numpy'yi tekrar 1.26.4'e sabitle.
- [x] Uçtan uca hat sentetik veriyle test edildi (extract → train → stream predict → API)
- [x] Gerçek TİD verisiyle kanıt: AUTSL shard 1 (HuggingFace `aipieces/AUTSL`, 500 video)
- [x] **31 sınıf × 4-6 video (toplam 131)** ile eğitim — model karşılaştırması:
      Bi-LSTM val %90.3 (28/31) / **Transformer val %93.5 (29/31)** → aktif model
      Transformer (`best.pt`; Bi-LSTM yedeği `bilstm_903.pt`).
      Hatalar: `ben→acikmak`, `itmek→hafif` (görsel olarak benzer işaretler).
- [x] Demo galerisi: `http://localhost:8080/demo` — her video oynatılabilir,
      yanında modelin çevirisi, güveni ve eğitim/doğrulama rozeti
- [x] **Canlı altyazı sayfası: `http://localhost:8080/watch`** — video oynarken
      kayan-pencere motorunun (`StreamingPredictor`, 1 sn pencere / 3 kare stride)
      kare kare tahminleri altta altyazı olarak akar. Zaman çizelgeleri
      `pipeline/build_subtitles.py` ile üretilir (8 doğrulama videosu, 7/8
      doğru kararlı sonuç; `anne` videosunda kısa pencere `ben`e takılıyor).
- [x] Video oynatma düzeltmesi: AUTSL MP4'leri FMP4 codec'li (tarayıcı oynatmaz);
      ffmpeg (imageio-ffmpeg) ile 131 video H.264'e çevrildi → `data/web_videos/`
      (orijinaller `raw_videos/`te bozulmadan duruyor; sunucu web_videos'u servis eder)
- [x] **Cümle çevirisi: `http://localhost:8080/sentence`** — doğrulama kliplerinden
      birleştirilen sürekli işaret dizileri kayan pencere + kelime emisyon
      kuralıyla (kararlı×3, eşik 0.55, pencere 45) çözülüp kalıp tablosuyla
      akıcı Türkçeye çevrilir (`pipeline/make_sentences.py`).
      Sonuç: 4 cümleden 3'ü kusursuz — "Ben acıktım." / "Eczane uzak." /
      "Tehlike var, kaçın!". 3 kelimeli zor örnekte polis→catal hatası
      bilerek sayfada bırakıldı (model polis işaretinde zayıf).
- [x] `_gecis` (geçiş) sınıfı deneyleri: hayalet kelimelerin kökü, pencerenin iki
      işaret arasındaki sınırı görmesi. 200 örnek/24'er kare → sınıf dengesizliği
      (33x) modele "kararsızsan geçiş" önyargısı verdi, gerçek kelimeleri yuttu
      (3/8). 60 örnek/8-15 kare dengeli sürüm → 4/8. Ders: tek işaretçili küçük
      veride emisyon ayarıyla çözüm tükendi; kalıcı çözüm çok işaretçili veri +
      (sonraki aşama) CTC tarzı sınır öğrenimi.
- [x] **Çok işaretçili eğitim + A/B ablasyonu (11 Eyl 2026)**: 16 işaretçi /
      1790 video, 64 sınıf; signer10 eğitimden tamamen dışlanıp test seti yapıldı.
      | Model | Aynı-işaretçi val | signer10 (görülmemiş) | Top-3 |
      |---|---|---|---|
      | Tek işaretçili (eski) | ~%99 | %7.9 | %11 |
      | A: salt eklem 258 | %87.9 | %69.8 | %90.5 |
      | **B: +kemik+hareket 660 (aktif)** | **%93.5** | **%79.4** | **%95.2** |
      İki ölçülmüş bulgu: işaretçi çeşitliliği %7.9→%69.8; bone+motion akışları
      (SAM-SLR reçetesi) +9.6 puan. Sonraki basamak: kalan işaretçiler (issue #2).
      (Bu tablo önceki oturumdan; o veri/checkpoint bu makinede yoktu, yeniden
      üretilemedi. Aşağıdaki 24 Eyl koşusu bağımsız bir yeniden kuruluştur.)
- [x] **Val + test işaretçileriyle yeniden kurulum (23-24 Eyl 2026)**: AUTSL aynasının
      val (işaretçi 1,11,16,18,25,35) ve test (6,14,27,30,34,39) bölmeleri indirildi;
      signer0 (train shard 1) ile birlikte 64 sınıf. **signer35 tamamen eğitim dışı
      tutuldu** (test seti, 188 örnek); diğer 12 işaretçi eğitimde. Transformer +
      bone/motion akışları (660), 65 sınıf (64 + `_gecis`), 80 epoch:
      | Eğitim işaretçisi | Karışık val | signer35 (görülmemiş) | Top-3 |
      |---|---|---|---|
      | 6 (signer0,1,11,16,18,25) | %93.9 | %78.2 (147/188) | %89.9 |
      | 12 (+6,14,27,30,34,39), eski sızıntılı split | %96.6 | %89.4 (168/188) | %97.9 |
      | **12, sızıntısı giderilmiş split — aktif** | **%95.3** | **%87.2 (164/188)** | **%96.8** |
      Aynı held-out işaretçi ve aynı ölçüm betiği: 6→12 işaretçi signer35 doğruluğunu
      +9.0 puan artırdı (%78.2→%87.2), hatalar 41→24, tamamen kaçırılan sınıf 9→3
      (`carsamba`, `dakika`→`saat`, `odun`). `_gecis` gerçek kelime yutması 5→2 hata
      (`pamuk`, `saat`); sıfırlanmadı. Yedekler: `best_6signers.pt`,
      `best_12signers_oldsplit.pt`. Not: 6 işaretçili satır da eski bölmeyle eğitildi.
      Bu koşuda sınıf listesi en sık örneklenen 64 sınıftır (7 işaretçide ortak 198
      sınıftan, liste: `docs/CLASSES.md`); önceki 64 sınıfla birebir aynı olduğu
      doğrulanamadı.
- [x] **İşaretçi-gruplu çapraz doğrulama (24 Eyl 2026)**: `pipeline/cv_signer_independent.py`.
      13 işaretçi 4 gruba bölündü (4/3/3/3); her grup sırayla test edildi, model kalan
      9-10 işaretçiyle eğitildi (`_gecis` de yalnız onların kliplerinden), 2 seed,
      son epoch değerlendirildi (held-out veriyle checkpoint seçimi yok). 2565 örnek:
      | Ölçüt | Sonuç |
      |---|---|
      | Toplam doğruluk (top-1 / top-3) | **%78.5 / %90.6** (seed 0: %78.5, seed 1: %78.6) |
      | Seed'ler arası sapma | ~0 (toplam), fold başına ≤1.9 puan |
      | İşaretçi bazında | ort. %78.3, std 9.3, aralık %54.7 (signer16) – %91.2 (signer0) |
      | %95 güven aralığı (işaretçiler üzerinde bootstrap) | [%73.8, %83.0] |
      | Gerçek işaretin `_gecis`'e düşmesi | 36 / 5130 (%0.7) |
      Yorum: değişkenliğin ana kaynağı eğitim rastgeleliği değil **işaretçidir**. Yeni bir
      kişide beklenti ~%78 (aralık ~%74-83), tek kişide %55-91 arası çıkabilir. Fold
      modelleri 9-10 işaretçiyle eğitildiğinden 12 işaretçili aktif modelin gerçek
      performansı bundan biraz yüksek olabilir (ölçülmedi). En zayıf sınıflar:
      `akilli`↔`carsamba`, `ezberlemek`→`ayni`, `aglamak`→`bakmak`, `tatli`↔`kiz`,
      `kotu`→`onlar`, `biz`→`hayir` (sınıf recall'u %34-60).
- [x] **Hata kök neden analizi (24 Eyl 2026)**: `pipeline/analyze_errors.py`, çapraz
      doğrulama tahminlerinden (2565 örnek × 2 seed). Ölçümler:
      1. **Yakın sınıf çiftleri**: en sık 15 karışıklığın 9'u, ortalama landmark
         uzayında tüm sınıf çiftlerinin en yakın %6'sında (`aglamak→bakmak` %0,
         `tatli↔kiz` %0, `dakika→saat` %0, `dede→dolu` %0, `akilli→ezberlemek` %1,
         `polis→cumartesi` %2). Üç çiftin (tatli/kiz, aglamak/bakmak, dakika/saat) 5'er
         karesine baktım: ayırt edici ipucunu gözle seçemedim. Ayırt edici bilginin
         (parmak şekli, yüz/ağız vb.) 258 boyutlu özellikte eksik olması bir *tahmin*,
         doğrulanmadı.
      2. **"Hub" sınıflar**: `ayni` 162 kez tahmin edildi, 80 kez gerçekte var (precision
         %44); `onlar`, `pantolon`, `ogretmen` benzer. Karışıklıklar tek yönlü ve
         landmark mesafeleri özellikle yakın değil (%13-39 yüzdelik), yani benzerlikle
         açıklanmıyor. Bu 7 sınıfa (`ayni`, `onlar`, `pantolon`, `ogretmen`, `hep`,
         `catal`, `hayir`) düşen yanlış tahmin ~296 / 1102 (%27; bir kısmı gerçek yakın
         çiftler), beklenenin üstündeki fazla tahmin ise en az 234 (%21). Nedeni
         doğrulanmadı (tahmin: görülmemiş işaretçi girdisi birkaç "genel" sınıfa kayıyor).
      3. **El algılama hatası küçük payda**: örneklerin %4'ünde (107) el karelerin
         <%80'inde algılanmış, bunlarda doğruluk %64 (diğerleri %77-86). Tahmini katkı
         ~16/551 hata (%3, seed başına). signer16 için bu **çürütüldü**: düşük el algılamalı örnekleri %56.4,
         yüksek olanlar %52.0. Bu işaretçi tek örnek sınıf (ben) için baktığım karelerde
         ayakta ve kameradan uzak görünüyor (signer0 ve signer11 oturuyor, kadraj
         yakın); bunun nedeni olduğu doğrulanmadı.
      4. İşaretçi düzeyinde kare sayısı ile doğruluk arasında ρ=-0.62 (p=0.03, n=13),
         örnek düzeyinde ilişki yok (ρ=-0.02); 4 ilişki denendiği için zayıf kanıt.
      5. **Güvenle reddetme işe yarıyor**: yanlışların ortalama güveni 0.50, doğruların
         0.78. Eşik 0.7: örneklerin %67'si kabul, kabul edilenlerde doğruluk %92.3;
         0.8: %55 kabul, %95.0; 0.9: %16 kabul, %98.3. (Kalibrasyon ayrıca doğrulanmadı.)
      Hatalar birkaç sınıfa toplanmıyor: en kötü 10 sınıf hataların %36'sını taşıyor.
- [x] **Hub düzeltme denemesi: sınıf başına logit ayarı — ETKİSİZ (24 Eyl 2026)**:
      `pipeline/logit_adjust_cv.py`. Her fold'da eğitim işaretçilerinden 2'si kalibrasyona
      ayrıldı, model kalanlarla eğitildi, sınıf başına bias kalibrasyon işaretçilerinin
      logit'lerinden öğrenildi (test işaretçileri hiç kullanılmadı; `_gecis` bias'ı 0'da
      sabit; birincil λ=0.1 sonuçlardan önce belirlendi). Aynı model, bias'sız vs bias'lı
      (8 koşu, 2565×2 tahmin): doğruluk **%73.5 → %73.4 (-0.08 puan, işaretçi düzeyinde
      %95 GA [-0.50, +0.30])**; λ=0.01: -0.10, λ=1.0: +0.10. 13 işaretçiden 4'ü iyileşti,
      7'si kötüleşti. Hub'lar biraz azaldı (`ayni` 2.12→1.91 tahmin/gerçek, sınıflar arası
      std 0.226→0.201) ama doğruluğa dönüşmedi; güvenle reddetme eğrisi de değişmedi
      (eşik 0.8: %93.5 → %93.4). Sonuç: hub eğilimi sınıf-sabit toplamsal bir sapma gibi
      görünmüyor; girdiye (işaretçiye) bağlı olması bir *tahmin*, doğrulanmadı.
      Benimsenmedi. Ek gözlem: bu modeller 7-8 işaretçiyle eğitildi ve tabanları %73.5;
      9-10 işaretçili CV'de %78.5 idi (aynı fold/epoch; kalibrasyon işaretçileri seed'e
      göre değişiyor, o yüzden karışık): işaretçi sayısı hâlâ en güçlü kaldıraç.
- [ ] Özellik vektörüne yüz (dudak+kaş) eklenmesi — TİD'de olumsuzluk/soru
      yüzle kodlanır; "var/yok" ayrımı için gerekli (bkz. Ürün Vizyonu)
- [ ] Kendi webcam verisi (`idle` dahil) + kişiselleştirme
- [ ] Flutter istemci (şimdilik web istemci var)

### Teknik ders (tekrarlanabilirlik)
MediaPipe Holistic video modunda takip durumu videolar arasında sızar: aynı
örnekle art arda video işlemek, sonucu işleme SIRASINA bağımlı yapar.
`make_sentences.py` bu yüzden her videoya taze Holistic örneği açar.

### Dürüstlük notları (önemli)
1. **Gerçek metrik görülmemiş-işaretçi sütunudur**, karışık val değil: karışık val
   bölmesi örnek bazlıdır, aynı işaretçinin videoları hem eğitimde hem val'da bulunur.
   Tek işaretçiyle (signer35, 188 örnek) ölçmek yanıltıcıdır: çapraz doğrulamada
   işaretçiler %55-91 arasında değişti (std 9.3 puan); en güvenilir sayı çapraz
   doğrulamanın %78.5'idir (aralık %74-83). signer35 orada %82.4 çıktı, yani ortalamanın
   biraz üstünde bir işaretçi (ve aktif modelde 3 işaretçi daha fazla eğitim vardı).
   Eski tek işaretçili (signer0) %93.5, yalnız aynı kişide geçerliydi; görülmemiş
   işaretçide %7.9 çıkmıştı.
2. **Val/test bölmelerinin işaretçileri artık eğitimde**; aynadan bağımsız yeni bir
   görülmemiş işaretçi havuzu kalmadı (signer35 dışında). Yeni ölçüm için diğer
   train shard'larından işaretçi indirilmeli.
3. Eğitim doğruluğu %100 = model bu ölçekte ezberliyor; genelleme iddiası yalnızca
   görülmemiş-işaretçi sütunundan okunmalı.
4. **`_gecis` veri sızıntısı (düzeltildi, 24 Eyl)**: `stratified_split` tek RNG akışı
   kullanıyordu; `_gecis` varken/yokken bölme kayıyor, `make_transitions.py`'nin
   "yalnız train klipleri" varsayımı tutmuyordu (ölçüldü: gerçek sınıf val'ı iki
   görünümde ortak değildi). Şimdi her sınıf kendi adından türeyen tohumla bölünüyor
   (`dataset.py`); gerçek sınıfların bölmesi `_gecis`'ten bağımsız (467 val örneği
   iki görünümde birebir aynı, ölçüldü). `_gecis` yeniden üretildi, model yeniden
   eğitildi: karışık val %96.6 → %95.3. Signer35 %89.4 → %87.2 ise sızıntıdan
   kaynaklanmış olamaz (signer35 hiçbir aşamada eğitimde/`_gecis`te yoktu); tek
   eğitim koşusu olduğundan bu fark eğitim rastgeleliği ile ayırt edilemedi.
5. **`evaluate_videos.py` split hatası (düzeltildi)**: split'i `_gecis`'i çıkardıktan
   sonra hesaplıyordu; ölçüm: "val" denen 467 örnekten 373'ü aslında eğitimdeydi.
   Artık `train.py` ile aynı split kullanılıyor (val %94.9, n=467, `_gecis` hariç;
   tablodaki %95.3 `train.py`'nin 507 örnekli, `_gecis` dahil val'ıdır).
6. `signer1` örnek sayısı diğerlerinin ~2 katı (384; AUTSL val etiketlerinde 1344 örnek,
   diğer val işaretçilerinde ~650-670). Kopya değil (byte-identical dosya yalnızca 2
   çift, aynı işaretçi ve sınıf içinde); dengelenmedi. Çapraz doğrulamada signer1
   %69.4 çıktı (ortalamanın altında); fazla örneğinin eğitime etkisi ayrıca ölçülmedi.
7. `idle` (işaret yok) sınıfı yok: AUTSL'de bu veri bulunmuyor, webcam kaydı gerekir.

AUTSL not: `backend/data/autsl/shard_001/` içinde 474 kullanılmayan video daha
var (215 sınıf); `prepare_autsl.py --top N` ile daha fazla sınıf eklenebilir.
Diğer 56 shard HuggingFace'ten indirilebilir.

## Hızlı başlangıç (PowerShell, repo kökünden)

```powershell
$py = ".\backend\venv\Scripts\python.exe"

# 1a. Webcam ile veri topla (sınıf başına ~30 örnek + bol 'idle')
& $py backend\demo\collect_data.py --label yardim_istiyorum --samples 30
& $py backend\demo\collect_data.py --label idle --samples 40

# 1b. VEYA hazır videolardan çıkar (data\raw_videos\<etiket>\*.mp4)
& $py backend\preprocessing\extract_landmarks.py

# 2. Eğit (checkpoint: backend\models\checkpoints\best.pt)
& $py backend\pipeline\train.py

# 3a. Masaüstü canlı demo (OpenCV penceresi)
& $py backend\demo\live_demo.py

# 3b. VEYA sunucu + tarayıcı demoları
& $py backend\pipeline\evaluate_videos.py    # video başına tahmin raporu üret
& $py -m uvicorn main:app --app-dir backend --port 8000
# http://localhost:8000/demo  -> indirilen videolar + model çevirileri galerisi
# http://localhost:8000       -> canlı webcam istemcisi (kamera izni ver)
```

## Mimari

```
webcam ──> MediaPipe Holistic ──> 258-float/kare (features.py)
                                        │
             eğitim: .npy dizileri ──> normalize + resample(60) + augment
                                        │
                              Bi-LSTM / Transformer (model.py)
                                        │
   canlı: kayan pencere(60) + her 5 karede tahmin + çoğunluk oyu + eşik
                                        │
                    OpenCV altyazı  /  WebSocket /ws/stream  /  web istemci
```

Tüm bileşenler aynı özellik tanımını `backend/preprocessing/features.py`'den
kullanır — burayı değiştirirsen veri ve model geçersiz olur.
