# 64 sınıf listesi (24 Eyl 2026 koşusu)

Bu liste `backend/data/` altında (gitignore'da) duruyordu; tekrarlanabilirlik için buraya
alındı. **Listeyi yeniden hesaplama, aşağıdaki satırı kullan** (`prepare_autsl.py --classes`).

## Seçim yöntemi
- Aday havuz: AUTSL'de signer0 (train shard 1) + val işaretçileri 1,11,16,18,25,35'in
  hepsinde bulunan sınıflar: **198 sınıf**.
- Bu 198 içinden, 7 işaretçideki toplam örnek sayısına göre en çok örneklenen 64 sınıf.
- Eşit sayıda örneği olan sınıflar arasındaki sıralama deterministik değildi (küme
  yinelemesi); bu yüzden listeyi yeniden üretmeye çalışma, dosyadan al.
- Önceki oturumun 64 sınıflı listesiyle aynı olduğu doğrulanamadı (o liste kayıtlı değil).
- Sınıf adları `SignList_ClassId_TR_EN.csv`'deki `TR` sütunudur (ASCII slug).

## Liste (virgülle ayrılmış, doğrudan `--classes` argümanı)
```
abla,acikmak,aglamak,akilli,akilsiz,anahtar,anne,ayni,baba,bakmak,bebek,bekar,ben,benzin,bilgi_vermek,biz,calismak,carsamba,catal,cekic,cumartesi,dakika,dede,dolu,dugun,dun,duvar,ezberlemek,fotograf,hastane,hayir,hep,isik,itmek,kacmak,kavsak,kemer,kiz,koku,kopek,kotu,leke,melek,memnun_olmak,nasil,nerede,oda,odun,ogretmen,onlar,ozur_dilemek,pamuk,pantolon,pazar,piknik,polis,saat,selam,serbest,tatli,tehlike,uzak,yatak,yildiz
```
