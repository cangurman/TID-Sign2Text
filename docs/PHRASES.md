# MVP Cümle Listesi — Acil Durum / Hastane Bankosu Senaryosu

Klasör adı = sınıf etiketi (ASCII slug). Veri toplarken/videoları yerleştirirken
`backend/data/raw_videos/<slug>/` veya `backend/data/landmarks/<slug>/` kullanılır.

Referans işaret formları için resmi sözlük: https://tidsozluk.aile.gov.tr/
(2.000 kelime, 11.428 video — telif durumu belirsiz olduğundan videoları eğitim
verisi olarak indirmeyin; işaretleri doğru yapmak için referans olarak kullanın.)

## Faz 1 — Çekirdek 15 (önce bunlarla demo çıkar)

| # | Slug | Türkçe cümle |
|---|------|--------------|
| 1 | merhaba | Merhaba |
| 2 | yardim_istiyorum | Yardım istiyorum |
| 3 | acil | Acil! / Acil durum |
| 4 | agrim_var | Ağrım var |
| 5 | basim_agriyor | Başım ağrıyor |
| 6 | karnim_agriyor | Karnım ağrıyor |
| 7 | nefes_alamiyorum | Nefes alamıyorum |
| 8 | ilac_istiyorum | İlaç istiyorum |
| 9 | doktor_nerede | Doktor nerede? |
| 10 | randevu_almak_istiyorum | Randevu almak istiyorum |
| 11 | evet | Evet |
| 12 | hayir | Hayır |
| 13 | tesekkur_ederim | Teşekkür ederim |
| 14 | anlamadim | Anlamadım |
| 15 | tuvalet_nerede | Tuvalet nerede? |

## Faz 2 — Genişleme (+25)

| # | Slug | Türkçe cümle |
|---|------|--------------|
| 16 | iyi_gunler | İyi günler |
| 17 | adim_ne | Adınız ne? / Adım... |
| 18 | kimlik | Kimlik (kartı) |
| 19 | sigorta | Sigorta |
| 20 | atesim_var | Ateşim var |
| 21 | usuyorum | Üşüyorum |
| 22 | bas_donmesi | Başım dönüyor |
| 23 | midem_bulaniyor | Midem bulanıyor |
| 24 | kustum | Kustum |
| 25 | dusdum | Düştüm |
| 26 | kirik_olabilir | Kırık olabilir |
| 27 | kanama_var | Kanama var |
| 28 | alerjim_var | Alerjim var |
| 29 | hamileyim | Hamileyim |
| 30 | seker_hastasiyim | Şeker hastasıyım |
| 31 | kalp_hastasiyim | Kalp hastasıyım |
| 32 | tansiyonum_var | Tansiyonum var |
| 33 | ne_zaman | Ne zaman? |
| 34 | nerede | Nerede? |
| 35 | bekleyin | Bekleyin / Bekliyorum |
| 36 | rapor_istiyorum | Rapor istiyorum |
| 37 | recete | Reçete |
| 38 | eczane_nerede | Eczane nerede? |
| 39 | ambulans | Ambulans |
| 40 | polis | Polis |

## Özel sınıf (zorunlu)

| Slug | Açıklama |
|------|----------|
| idle | İşaret YOK: eller aşağıda dinlenme, işaretler arası geçişler. Araştırma bulgusu: bu sınıf olmadan model eller hareketsizken bile sürekli tahmin üretir — demoların 1 numaralı hata modu. Her kayıt oturumunda bolca `idle` örneği toplayın. |

## Veri toplama hedefi
- Sınıf başına **en az 30 örnek** (ideal 50+), mümkünse **birden fazla kişi**
  işaretlesin (işaretçiden bağımsız genelleme için).
- Örnek uzunluğu ~2 sn (60 kare @ 30fps) — `collect_data.py` varsayılanı.
- Değerlendirmede dürüst ölçüm: mümkünse bir kişiyi tamamen validation'a ayırın
  (signer-independent split).
