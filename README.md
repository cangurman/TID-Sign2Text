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

- [x] Python 3.12 venv: `backend/venv` (sistemdeki 3.14 MediaPipe'la uyumsuz)
- [x] Bağımlılıklar kurulu (`mediapipe==0.10.21` — 1.0.x eski API'yi kaldırdı)
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
- [~] **Çok işaretçili eğitim (sürüyor)**: 10 yeni shard indirildi → eğitim
      havuzu 16 işaretçi / 1790 video (64 sınıf, sınıf başına ~28). signer10
      eğitimden tamamen dışlandı, dokunulmamış test seti. Landmark çıkarımı
      arka planda; ardından yeniden eğitim + signer10'da dürüst genelleme ölçümü
      (tek işaretçili modelin taban çizgisi: %7.9-9.5).
- [ ] Özellik vektörüne yüz (dudak+kaş) eklenmesi — TİD'de olumsuzluk/soru
      yüzle kodlanır; "var/yok" ayrımı için gerekli (bkz. Ürün Vizyonu)
- [ ] Kendi webcam verisi (`idle` dahil) + kişiselleştirme
- [ ] Flutter istemci (şimdilik web istemci var)

### Teknik ders (tekrarlanabilirlik)
MediaPipe Holistic video modunda takip durumu videolar arasında sızar: aynı
örnekle art arda video işlemek, sonucu işleme SIRASINA bağımlı yapar.
`make_sentences.py` bu yüzden her videoya taze Holistic örneği açar.

### Dürüstlük notları (önemli)
1. **Tüm veri tek işaretçiden** (signer0 — shard 1 sadece onu içeriyor). %93.5,
   *aynı kişinin görülmemiş videoları* üzerindeki doğruluktur. Farklı kişide
   doğruluk belirgin düşecektir (AUTSL literatürü: rastgele bölmede ~%96'ya
   karşı işaretçiden bağımsız bölmede ~%62 baseline). Çözüm: diğer shard'lardan
   farklı işaretçiler indirilip test edilmeli.
2. Doğrulama kümesi sınıf başına 1 video (n=31) — Transformer ile Bi-LSTM
   arasındaki fark tek video, istatistiksel olarak zayıf. İkisi de tutuluyor.
3. Eğitim doğruluğu %100 = model bu ölçekte ezberliyor; genelleme iddiası
   yalnızca val sütunundan okunmalı.

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
