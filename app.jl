using HTTP, Flux, Images, FileIO, BSON, Sockets

const KELAS  = ["Healthy", "Common_Rust", "Gray_Leaf_Spot", "Northern_Leaf_Blight"]
const LABEL  = ["Daun sehat", "Karat daun (Common Rust)",
                "Bercak daun abu-abu (Gray Leaf Spot)", "Hawar daun (Northern Leaf Blight)"]
const UKURAN = (128, 128)
const MODEL_PATH = joinpath(@__DIR__, "model_jagung.bson")

# true  = bisa dibuka dari HP di Wi-Fi yang sama
# false = hanya bisa dibuka dari komputer ini
const BISA_DARI_HP = true
const PORT = 8080

println("Memuat model...")
BSON.@load MODEL_PATH model
Flux.testmode!(model)   # matikan Dropout saat memprediksi

function prediksi(path::String)
    img = load(path)
    img = imresize(img, UKURAN)
    img = RGB.(img)
    arr = permutedims(Float32.(channelview(img)), (3, 2, 1))
    arr = reshape(arr, UKURAN..., 3, 1)
    return Float64.(vec(model(arr)))   # model sudah berakhir dengan softmax
end

const HALAMAN = raw"""
<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deteksi Penyakit Daun Jagung</title>
<style>
:root{--bg:#f7f3e8;--kertas:#fffdf6;--ink:#2a2418;--muted:#7a705c;--rule:#ddd5bf;--acc:#55702f;--hero:#2f3a1f;--herotx:#f4efdc;--warn:#8a4b12;--warnbg:#f6e8d2;--err:#a32a22}
@media(prefers-color-scheme:dark){:root{--bg:#1a1812;--kertas:#232018;--ink:#ece6d2;--muted:#a39b84;--rule:#3a3629;--acc:#a3c46a;--hero:#14170d;--herotx:#ece6d2;--warn:#e2ab72;--warnbg:#33281a;--err:#e58a82}}
*{box-sizing:border-box}
html{scroll-behavior:smooth;scroll-padding-top:70px}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
body.kunci{overflow:hidden}
h1,h2,h3{font-family:Georgia,"Times New Roman",serif;font-weight:600;margin:0}
svg{display:block}.ik{width:20px;height:20px;flex:none}
@media(prefers-reduced-motion:reduce){*{animation-duration:.01s!important;animation-delay:0s!important;transition-duration:.01s!important;scroll-behavior:auto!important}}

/* splash */
#splash{position:fixed;inset:0;z-index:100;background:var(--hero);color:var(--herotx);display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;text-align:center;padding:24px;transition:opacity .6s,visibility .6s}
#splash.selesai{opacity:0;visibility:hidden}
.sp-daun{width:min(270px,62vw);margin-bottom:12px}
.gam{fill:none;stroke:#a9cf6e;stroke-width:2.4;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:1;stroke-dashoffset:1;animation:gambar 1.5s ease forwards}
.gam.urat{stroke-width:1.8;animation-delay:.5s}
.gisi{fill:#a9cf6e;opacity:0;animation:isi .9s ease 1.1s forwards}
.sp-judul{font:600 clamp(1.5rem,5vw,2.2rem)/1.15 Georgia,serif;opacity:0;animation:muncul .7s ease .9s forwards}
.sp-sub{opacity:0;font-size:.95rem;animation:muncul .7s ease 1.2s forwards;color:#cdd5b3}
.sp-bar{width:170px;height:3px;background:rgba(255,255,255,.16);border-radius:2px;overflow:hidden;margin-top:18px}
.sp-bar i{display:block;height:100%;width:0;background:#a9cf6e;animation:muat 2.4s linear forwards}
#lewati{margin-top:14px;background:none;border:0;color:#cdd5b3;font:inherit;font-size:.82rem;cursor:pointer;text-decoration:underline;opacity:.8}
@keyframes gambar{to{stroke-dashoffset:0}}
@keyframes isi{to{opacity:.2}}
@keyframes muncul{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes muat{to{width:100%}}

/* header */
.top{position:sticky;top:0;z-index:10;background:var(--bg);border-bottom:1px solid var(--rule)}
.top-in{max-width:920px;margin:0 auto;padding:0 22px;height:56px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.brand{display:flex;align-items:center;gap:8px;font:600 1.02rem Georgia,serif;color:var(--ink);text-decoration:none;white-space:nowrap}
.brand .ik{color:var(--acc);width:24px;height:24px}
nav{display:flex;gap:2px;overflow-x:auto;scrollbar-width:none}nav::-webkit-scrollbar{display:none}
nav a{padding:7px 12px;border-radius:6px;color:var(--muted);text-decoration:none;font-size:.88rem;white-space:nowrap}
nav a:hover{color:var(--ink)}nav a.aktif{background:var(--kertas);color:var(--ink);box-shadow:0 0 0 1px var(--rule)}
@media(max-width:700px){.brand .tx{display:none}}

.btn{display:inline-flex;align-items:center;gap:8px;padding:10px 20px;border:1px solid var(--acc);border-radius:8px;background:var(--acc);color:#fff;font:inherit;font-size:.92rem;cursor:pointer;text-decoration:none}
@media(prefers-color-scheme:dark){.btn{color:#12160b}}
.btn.terang{background:var(--herotx);border-color:var(--herotx);color:var(--hero)}
.btn.garis{background:transparent;color:var(--ink);border-color:var(--rule)}
.btn.kecil{padding:6px 12px;font-size:.82rem}

/* banner */
.heroS{background:var(--hero);color:var(--herotx);padding:54px 22px 56px}
.hero{max-width:920px;margin:0 auto;display:grid;grid-template-columns:1.1fr 1fr;gap:38px;align-items:center}
.hero h1{font-size:clamp(2.1rem,6vw,3.3rem);line-height:1.08;margin-bottom:12px}
.hero p{margin:0 0 22px;opacity:.86;font-size:1.05rem;max-width:380px}
.stat{display:flex;gap:26px;margin-top:30px;padding-top:18px;border-top:1px solid rgba(255,255,255,.16)}
.stat b{display:block;font:600 1.5rem Georgia,serif}.stat span{font-size:.76rem;opacity:.75}
.ubin{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.ubin div{background:rgba(255,255,255,.08);border-radius:12px;padding:10px 10px 8px}
.ubin span{display:flex;align-items:center;margin-top:5px;font-size:.78rem;opacity:.88}
.ubin i{width:8px;height:8px;border-radius:2px;margin-right:6px;background:var(--c)}
@media(max-width:700px){.hero{grid-template-columns:1fr}}

section.b{max-width:920px;margin:0 auto;padding:54px 22px 4px}
.b h2{font-size:1.6rem;margin-bottom:4px}
.sub{color:var(--muted);margin:0 0 16px}

/* periksa */
#zona{display:flex;flex-direction:column;align-items:center;gap:4px;text-align:center;padding:36px 14px;border:2px dashed var(--rule);border-radius:14px;cursor:pointer;color:var(--muted);background:var(--kertas)}
#zona.aktif{border-color:var(--acc);color:var(--ink)}
#zona .ik{width:40px;height:40px;color:var(--acc);margin-bottom:4px}
#zona b{font:600 1.15rem Georgia,serif;color:var(--ink)}#zona small{font-size:.82rem}
input[type=file]{display:none}
.aksi{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
#status{margin-top:10px;color:var(--muted);font-size:.88rem;min-height:1.3em}
.tips3{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-top:6px}
.tips3 div{display:flex;flex-direction:column;align-items:center;gap:4px;padding:12px 6px;background:var(--kertas);border:1px solid var(--rule);border-radius:12px;font-size:.84rem;color:var(--muted)}
.tips3 .ik{width:26px;height:26px;color:var(--acc)}

/* hasil */
.kertas{background:var(--kertas);border:1px solid var(--rule);border-radius:14px;padding:20px}
.kosong{display:flex;flex-direction:column;align-items:center;gap:6px;color:var(--muted);padding:6px 0}
.fantom{width:min(240px,70%);opacity:.3;filter:grayscale(1)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:26px}
@media(max-width:700px){.grid{grid-template-columns:1fr}}
.pasang{display:grid;grid-template-columns:1fr 1fr;gap:10px;align-content:start}
.pasang figure{margin:0}
.kotak{aspect-ratio:1;background:var(--bg);border-radius:10px;overflow:hidden;display:grid;place-items:center;border:1px solid var(--rule)}
.kotak img{max-width:100%;max-height:100%;object-fit:contain}.kotak svg{width:100%}
.pasang figcaption{margin-top:5px;font-size:.76rem;color:var(--muted);text-align:center;word-break:break-all}
.kecilteks{font-size:.78rem;color:var(--muted)}
.nama-hasil{font:600 1.5rem/1.2 Georgia,serif;margin-top:2px}
.en{font-size:.8rem;color:var(--muted)}
.angka{font:600 3.3rem/1.05 Georgia,serif;margin:4px 0 8px}
.br{display:grid;grid-template-columns:48px 1fr 52px;gap:10px;align-items:center;margin-top:8px}
.br .nm{font-size:.84rem}
.trek{height:7px;background:var(--rule);border-radius:4px;margin-top:3px;overflow:hidden}
.isi{height:100%;background:var(--muted);opacity:.5}.isi.atas{opacity:1}
.br .pc{text-align:right;font-size:.86rem;font-variant-numeric:tabular-nums}
.pw{display:flex;gap:8px;align-items:flex-start;margin-top:12px;padding:9px 12px;border-radius:8px;background:var(--warnbg);color:var(--warn);font-size:.86rem}
.pw .ik{margin-top:1px}.galat{color:var(--err)}

/* riwayat */
.galeri{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px}
.it{cursor:pointer;border-radius:10px;overflow:hidden;background:var(--kertas);border:1px solid var(--rule)}
.it.pilih{outline:2px solid var(--acc);outline-offset:1px}
.it img,.it .nol{display:block;width:100%;aspect-ratio:1;object-fit:cover;background:var(--rule)}
.it .pita{height:5px;background:var(--c,var(--rule))}
.it .t{padding:6px 8px;font-size:.78rem;line-height:1.35}
.it .n{color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}

/* kondisi daun */
.kp{display:grid;gap:16px}
.kk{display:grid;grid-template-columns:210px 1fr;gap:22px;background:var(--kertas);border:1px solid var(--rule);border-radius:14px;padding:16px}
@media(max-width:640px){.kk{grid-template-columns:1fr}}
.kk .gb{background:var(--bg);border-radius:10px;padding:10px;border-bottom:4px solid var(--c);align-self:start}
.kk h3{font-size:1.25rem}.kk .desk{margin:6px 0 0;color:var(--muted);font-size:.93rem}
.kk dl{margin:12px 0 0;display:grid;grid-template-columns:116px 1fr;gap:8px 14px;font-size:.88rem;border-top:1px solid var(--rule);padding-top:12px}
.kk dt{color:var(--muted)}.kk dd{margin:0}
@media(max-width:460px){.kk dl{grid-template-columns:1fr;gap:2px}.kk dt{margin-top:6px}}
.panel{margin-top:26px}.panel h3{font-size:1.15rem;margin-bottom:10px}
.jitu .br{grid-template-columns:110px 1fr 52px}
@media(max-width:460px){.jitu .br{grid-template-columns:84px 1fr 48px}}
footer{max-width:920px;margin:0 auto;padding:32px 22px 60px;color:var(--muted);font-size:.8rem}
</style></head>
<body class="kunci">

<div id="splash" role="dialog" aria-label="Pembuka">
  <svg class="sp-daun" viewBox="0 0 200 120" aria-hidden="true">
    <path class="gisi" d="M8 100 C50 20 140 10 194 22 C160 80 80 112 8 100 Z"/>
    <path class="gam" pathLength="1" d="M8 100 C50 20 140 10 194 22 C160 80 80 112 8 100 Z"/>
    <path class="gam urat" pathLength="1" d="M8 100Q100 52 194 22"/>
  </svg>
  <div class="sp-judul">Deteksi Penyakit Daun Jagung</div>
  <div class="sp-sub">Analisis daun dengan model CNN</div>
  <div class="sp-bar"><i></i></div>
  <button id="lewati">Lewati</button>
</div>

<header class="top"><div class="top-in">
  <a class="brand" href="#beranda"><span data-ikon="daun"></span><span class="tx">Deteksi Daun Jagung</span></a>
  <nav><a href="#beranda">Beranda</a><a href="#periksa">Periksa</a><a href="#hasil-s">Hasil</a><a href="#riwayat-s">Riwayat</a><a href="#penyakit">Kondisi daun</a></nav>
</div></header>

<section class="heroS" id="beranda"><div class="hero">
  <div>
    <h1>Cek daun jagungmu</h1>
    <p>Foto daunnya, lihat tebakan AI lengkap dengan tingkat keyakinannya.</p>
    <a class="btn terang" href="#periksa">Mulai periksa</a>
    <div class="stat">
      <div><b>4.187</b><span>foto dataset</span></div>
      <div><b>4</b><span>kondisi daun</span></div>
      <div><b>88,9%</b><span>akurasi uji</span></div>
    </div>
  </div>
  <div class="ubin" id="ubin"></div>
</div></section>

<section class="b" id="periksa">
  <h2>Periksa</h2>
  <p class="sub">Unggah satu atau beberapa foto daun jagung.</p>
  <label id="zona" for="file"><span data-ikon="unggah"></span><b>Letakkan foto di sini</b><small>atau klik, atau tempel dengan Ctrl+V</small></label>
  <input type="file" id="file" accept="image/*" multiple>
  <input type="file" id="kamera" accept="image/*" capture="environment">
  <div class="aksi">
    <button class="btn" id="bPilih"><span data-ikon="unggah"></span>Pilih foto</button>
    <button class="btn garis" id="bKamera"><span data-ikon="kamera"></span>Kamera HP</button>
  </div>
  <div id="status"></div>
  <div class="tips3">
    <div><span data-ikon="daun"></span>Satu daun</div>
    <div><span data-ikon="kaca"></span>Dekat</div>
    <div><span data-ikon="matahari"></span>Terang</div>
  </div>
</section>

<section class="b" id="hasil-s">
  <h2>Hasil</h2>
  <div class="kertas" id="hasil"></div>
</section>

<section class="b" id="riwayat-s">
  <h2>Riwayat (<span id="jml">0</span>)</h2>
  <div class="galeri" id="galeri"></div>
  <div class="aksi"><button class="btn garis kecil" id="bHapus">Kosongkan</button></div>
</section>

<section class="b" id="penyakit">
  <h2>Empat kondisi daun</h2>
  <p class="sub">Yang dikenali model, beserta penjelasan umumnya.</p>
  <div class="kp" id="kp"></div>
  <div class="panel jitu">
    <h3>Seberapa jitu model</h3>
    <div class="kertas" id="jitu"></div>
    <p class="kecilteks">Persentase foto tiap kelas yang dikenali benar pada 837 foto uji.</p>
  </div>
</section>

<footer>Proyek belajar, bukan diagnosis resmi. Rentang suhu pada penjelasan bersifat perkiraan umum. Untuk keputusan penanganan, konsultasikan dengan penyuluh pertanian setempat.</footer>

<script>
/* splash dipisah supaya selalu bisa menutup */
(function(){
  const sp=document.getElementById('splash');let selesai=false;
  function tutup(){if(selesai)return;selesai=true;sp.classList.add('selesai');document.body.classList.remove('kunci');setTimeout(()=>sp.remove(),700)}
  setTimeout(tutup,2600);sp.addEventListener('click',tutup);
  document.getElementById('lewati').addEventListener('click',tutup);
  document.addEventListener('keydown',tutup,{once:true});
})();
</script>

<script>
const NAMA=['Sehat','Karat daun','Bercak abu-abu','Hawar daun'];
const EN=['Healthy','Common Rust','Gray Leaf Spot','Northern Leaf Blight'];
const WARNA=['#5f8a3c','#b5651d','#6f8190','#7a5236'];
// Recall per kelas dari data uji (837 foto). Perbarui bila model dilatih ulang.
const RECALL=[99.6,92.5,81.3,77.1];
const KOND=[
 {desk:'Daun tidak menunjukkan gejala penyakit yang dikenali model.',
  d:[['Ciri di daun','Hijau merata, permukaan bersih tanpa bercak atau bintik.'],['Dampak','Tidak ada gangguan yang terlihat.'],['Langkah umum','Tetap pantau tanaman secara rutin, karena penyakit bisa muncul belakangan.']]},
 {desk:'Penyakit jamur yang menimbulkan bintik-bintik seperti karat pada permukaan daun.',
  d:[['Ciri di daun','Bintik kecil (pustul) bulat sampai lonjong, cokelat kemerahan, sedikit menonjol, tersebar di kedua sisi daun.'],['Penyebab','Jamur Puccinia sorghi.'],['Cuaca pendukung','Sejuk (sekitar 16-23 °C) dan lembap.'],['Dampak','Pada serangan berat, daun menguning dan mengering lebih awal sehingga hasil panen bisa turun.'],['Langkah umum','Pilih varietas yang tahan dan pantau sejak tanaman masih muda.']]},
 {desk:'Penyakit jamur dengan bercak persegi panjang yang dibatasi tulang daun.',
  d:[['Ciri di daun','Bercak persegi panjang abu-abu sampai cokelat muda, sejajar tulang daun, dan relatif kecil.'],['Penyebab','Jamur Cercospora zeae-maydis.'],['Cuaca pendukung','Hangat (sekitar 25-30 °C), sangat lembap, daun lama basah.'],['Dampak','Bercak menyatu dan mematikan jaringan daun sehingga pengisian biji terganggu.'],['Langkah umum','Rotasi tanaman, kelola sisa tanaman yang terinfeksi, dan pilih varietas yang tahan.']]},
 {desk:'Penyakit jamur dengan bercak panjang berbentuk seperti cerutu.',
  d:[['Ciri di daun','Bercak memanjang runcing, abu-abu kehijauan sampai cokelat, umumnya jauh lebih besar daripada bercak Gray Leaf Spot.'],['Penyebab','Jamur Exserohilum turcicum.'],['Cuaca pendukung','Sedang (sekitar 18-27 °C), berembun dan lembap.'],['Dampak','Bercak yang meluas mengurangi luas daun yang berfotosintesis sehingga hasil panen turun.'],['Langkah umum','Pilih varietas yang tahan, rotasi tanaman, dan kelola sisa tanaman.']]}
];

const $=id=>document.getElementById(id);
function el(t,c,x){const e=document.createElement(t);if(c)e.className=c;if(x!==undefined)e.textContent=x;return e}
const fmt=n=>String(n).replace('.',',');

/* ikon */
const IK={
 unggah:'<path d="M12 16V5M7 9l5-5 5 5M4 19h16"/>',
 kamera:'<path d="M3 8h4l2-3h6l2 3h4v11H3z"/><circle cx="12" cy="13" r="3.5"/>',
 matahari:'<circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2"/>',
 kaca:'<circle cx="10.5" cy="10.5" r="6.5"/><path d="M15.5 15.5L21 21"/>',
 daun:'<path d="M5 19C5 9 11 4 20 4c0 9-5 15-15 15zM5 19L14 10"/>',
 peringatan:'<path d="M12 3L2 20h20L12 3zM12 10v5M12 17.5v.5"/>'
};
function ikon(n){return '<svg class="ik" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'+IK[n]+'</svg>'}

/* ilustrasi daun (digambar dengan kode) */
let uid=0;
function acak(s){return()=>{s=(s*1664525+1013904223)%4294967296;return s/4294967296}}
const BLADE='M8 100 C50 20 140 10 194 22 C160 80 80 112 8 100 Z';
function daun(t){
  const id='cp'+(uid++),r=acak(11+t*29);let v='',l='';
  for(let k=1;k<10;k++){const s=k/10,a=(1-s)*(1-s),b=2*(1-s)*s,c=s*s;
    const x=(a*8+b*100+c*194).toFixed(1),y=(a*100+b*52+c*22).toFixed(1);
    v+='<path d="M'+x+' '+y+'l16 -22M'+x+' '+y+'l12 24"/>';}
  if(t===1){for(let i=0;i<36;i++){const x=(20+r()*170).toFixed(1),y=(22+r()*80).toFixed(1),q=1.5+r()*1.5;
    l+='<circle cx="'+x+'" cy="'+y+'" r="'+(q+1.4).toFixed(1)+'" fill="#e2b46c" opacity=".55"/><circle cx="'+x+'" cy="'+y+'" r="'+q.toFixed(1)+'" fill="#a8501b"/>'}}
  if(t===2){for(let i=0;i<10;i++){const x=(28+r()*150).toFixed(1),y=(30+r()*62).toFixed(1),w=12+r()*9,h=3.6+r()*1.6;
    l+='<rect x="'+(-w/2).toFixed(1)+'" y="'+(-h/2).toFixed(1)+'" width="'+w.toFixed(1)+'" height="'+h.toFixed(1)+'" rx=".8" transform="translate('+x+' '+y+') rotate(-22)" fill="#c4b894" stroke="#8d8566" stroke-width=".8"/>'}}
  if(t===3){[[66,70,36,7],[118,50,32,6.5],[150,33,20,4.5],[102,84,22,4.5]].forEach(a=>{
    const x=a[0],y=a[1],L=a[2],W=a[3];
    l+='<path d="M'+(-L)+' 0Q0 '+(-W*1.6)+' '+L+' 0Q0 '+(W*1.6)+' '+(-L)+' 0Z" transform="translate('+x+' '+y+') rotate(-22)" fill="#b39a72" stroke="#7d6845" stroke-width="1"/>'})}
  return '<svg viewBox="0 0 200 120" role="img" aria-label="Ilustrasi daun: '+EN[t]+'"><defs><clipPath id="'+id+'"><path d="'+BLADE+'"/></clipPath></defs>'
   +'<path d="'+BLADE+'" fill="#79a34a"/><g clip-path="url(#'+id+')"><g stroke="#e8f1c8" stroke-opacity=".28" stroke-width="1" fill="none">'+v+'</g>'+l+'</g>'
   +'<path d="M8 100Q100 52 194 22" stroke="#e3edbd" stroke-width="2" fill="none" stroke-linecap="round"/>'
   +'<path d="'+BLADE+'" fill="none" stroke="#3f5f26" stroke-opacity=".5" stroke-width="1.2"/></svg>';
}
document.querySelectorAll('[data-ikon]').forEach(e=>e.outerHTML=ikon(e.dataset.ikon));

/* isi statis: banner, kartu kondisi, kinerja */
NAMA.forEach((n,i)=>{
  const u=el('div');u.innerHTML=daun(i);
  const s=el('span');s.innerHTML='<i style="--c:'+WARNA[i]+'"></i>';s.appendChild(document.createTextNode(n));
  u.appendChild(s);$('ubin').appendChild(u);

  const k=el('div','kk');k.id='k'+i;
  const g=el('div','gb');g.style.setProperty('--c',WARNA[i]);g.innerHTML=daun(i);
  const isi=el('div');isi.append(el('h3','',n),el('div','en',EN[i]),el('p','desk',KOND[i].desk));
  const dl=el('dl');KOND[i].d.forEach(([a,b])=>dl.append(el('dt','',a),el('dd','',b)));
  isi.appendChild(dl);k.append(g,isi);$('kp').appendChild(k);

  const b=el('div','br'),tr=el('div','trek'),bar=el('div','isi atas');
  bar.style.width=RECALL[i]+'%';bar.style.background=WARNA[i];tr.appendChild(bar);
  b.append(el('div','nm',n),tr,el('div','pc',fmt(RECALL[i])+'%'));$('jitu').appendChild(b);
});

/* menu: tandai bagian yang sedang dilihat */
const tautan=[...document.querySelectorAll('nav a')];
const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting){
  tautan.forEach(a=>a.classList.toggle('aktif',a.getAttribute('href')==='#'+e.target.id))}}),{rootMargin:'-35% 0px -60% 0px'});
['beranda','periksa','hasil-s','riwayat-s','penyakit'].forEach(id=>io.observe($(id)));

/* menerima foto */
const riwayat=[],antrian=[];let sibuk=false,terpilih=null;
const zona=$('zona'),inp=$('file'),kam=$('kamera');
$('bPilih').onclick=()=>inp.click();$('bKamera').onclick=()=>kam.click();
inp.onchange=()=>{tambah(inp.files);inp.value=''};kam.onchange=()=>{tambah(kam.files);kam.value=''};
['dragenter','dragover'].forEach(n=>zona.addEventListener(n,e=>{e.preventDefault();zona.classList.add('aktif')}));
['dragleave','drop'].forEach(n=>zona.addEventListener(n,e=>{e.preventDefault();zona.classList.remove('aktif')}));
zona.addEventListener('drop',e=>tambah(e.dataTransfer.files));
window.addEventListener('dragover',e=>e.preventDefault());window.addEventListener('drop',e=>e.preventDefault());
document.addEventListener('paste',e=>{const f=e.clipboardData?e.clipboardData.files:[];if(f.length)tambah(f)});
function tambah(d){for(const f of d){if(f.type.startsWith('image/')||/\.(jpe?g|png|webp|bmp)$/i.test(f.name))antrian.push(f)}jalan()}

function siapkan(file){return new Promise(ok=>{const u=URL.createObjectURL(file),im=new Image();
im.onload=()=>{const w=im.naturalWidth,h=im.naturalHeight,s=Math.min(1,640/Math.max(w,h));
const c=document.createElement('canvas');c.width=Math.round(w*s);c.height=Math.round(h*s);c.getContext('2d').drawImage(im,0,0,c.width,c.height);
const t=document.createElement('canvas');t.width=t.height=120;const m=Math.min(w,h);
t.getContext('2d').drawImage(im,(w-m)/2,(h-m)/2,m,m,0,0,120,120);
c.toBlob(b=>{URL.revokeObjectURL(u);ok({blob:b||file,thumb:t.toDataURL('image/jpeg',.75)})},'image/jpeg',.9)};
im.onerror=()=>{URL.revokeObjectURL(u);ok({blob:file,thumb:null})};im.src=u})}

async function jalan(){if(sibuk)return;sibuk=true;
while(antrian.length){const f=antrian.shift();
$('status').textContent='Memeriksa '+f.name+(antrian.length?' (+'+antrian.length+')':'')+'...';
const it={nama:f.name,url:null,thumb:null,persen:null,error:null};
try{const {blob,thumb}=await siapkan(f);it.url=URL.createObjectURL(blob);it.thumb=thumb;
const fd=new FormData();fd.append('foto',blob,'foto.jpg');
const r=await fetch('/prediksi',{method:'POST',body:fd}),d=await r.json();
if(!r.ok)throw new Error(d.error);it.persen=d.persen}catch(e){it.error=e.message}
riwayat.unshift(it);pilih(it)}
sibuk=false;$('status').textContent='Selesai.';$('hasil-s').scrollIntoView({behavior:'smooth'})}

/* hasil */
function peringatan(teks){const p=el('div','pw');p.innerHTML=ikon('peringatan');p.appendChild(el('span','',teks));return p}
function kosongHasil(){$('hasil').innerHTML='<div class="kosong"><div class="fantom">'+daun(0)+'</div><span>Belum ada foto</span></div>'}

function pilih(it){
  terpilih=it;const h=$('hasil');h.innerHTML='';
  if(it.error){h.appendChild(peringatan('Foto tidak bisa diperiksa: '+it.error));renderR();return}
  const u=it.persen.map((p,i)=>[p,i]).sort((a,b)=>b[0]-a[0]),[p1,i1]=u[0],[p2,i2]=u[1];
  const g=el('div','grid'),kiri=el('div','pasang'),ka=el('div');

  const f1=el('figure'),k1=el('div','kotak');
  if(it.url){const im=el('img');im.src=it.url;im.alt='Foto yang diperiksa';k1.appendChild(im)}
  f1.append(k1,el('figcaption','','Fotomu'));
  const f2=el('figure'),k2=el('div','kotak');k2.innerHTML=daun(i1);
  f2.append(k2,el('figcaption','','Contoh: '+NAMA[i1]));
  kiri.append(f1,f2);

  ka.append(el('div','kecilteks','Tebakan'),el('div','nama-hasil',NAMA[i1]),el('div','en',EN[i1]),el('div','angka',fmt(p1)+'%'));
  NAMA.forEach((n,i)=>{
    const b=el('div','br'),sv=el('div');sv.innerHTML=daun(i);
    const tengah=el('div'),tr=el('div','trek'),isi=el('div','isi'+(i===i1?' atas':''));
    isi.style.width=it.persen[i]+'%';if(i===i1)isi.style.background=WARNA[i];
    tr.appendChild(isi);tengah.append(el('div','nm',n),tr);
    b.append(sv,tengah,el('div','pc',fmt(it.persen[i])+'%'));ka.appendChild(b);
  });
  if(p1<60)ka.appendChild(peringatan('Model kurang yakin. Coba foto yang lebih dekat dan jelas.'));
  else if(p1-p2<25)ka.appendChild(peringatan('Hampir sama kuat dengan '+NAMA[i2]+' ('+fmt(p2)+'%).'));
  if(i1>=2)ka.appendChild(peringatan('Sering tertukar dengan '+NAMA[i1===2?3:2]+'. Bandingkan bentuk bercak.'));

  const aksi=el('div','aksi'),a=el('a','btn garis kecil','Pelajari '+NAMA[i1]);a.href='#k'+i1;aksi.appendChild(a);ka.appendChild(aksi);
  g.append(kiri,ka);h.appendChild(g);renderR();
}

/* riwayat */
function renderR(){const gl=$('galeri');$('jml').textContent=riwayat.length;gl.innerHTML='';
if(!riwayat.length){gl.appendChild(el('div','kecilteks','Belum ada.'));return}
riwayat.forEach(it=>{const k=el('div','it'+(it===terpilih?' pilih':''));
if(it.thumb){const im=el('img');im.src=it.thumb;im.alt='';k.appendChild(im)}else k.appendChild(el('div','nol'));
const pita=el('div','pita');let hs='Gagal';
if(!it.error){const mx=Math.max(...it.persen),i=it.persen.indexOf(mx);hs=NAMA[i]+' '+fmt(mx)+'%';pita.style.setProperty('--c',WARNA[i])}
const t=el('div','t');t.append(el('div','n',it.nama),el('div','',hs));k.append(pita,t);
k.onclick=()=>{pilih(it);$('hasil-s').scrollIntoView({behavior:'smooth'})};gl.appendChild(k)})}
$('bHapus').onclick=()=>{riwayat.length=0;terpilih=null;kosongHasil();renderR()};

kosongHasil();renderR();
</script></body></html>
"""

function handler(req::HTTP.Request)
    if req.method == "GET" && req.target == "/"
        return HTTP.Response(200, ["Content-Type" => "text/html; charset=utf-8"], HALAMAN)
    elseif req.method == "POST" && req.target == "/prediksi"
        path = ""
        try
            bagian = HTTP.parse_multipart_form(req)
            (bagian === nothing || isempty(bagian)) && error("Tidak ada file yang diterima")
            data = read(bagian[1].data)
            ext = lowercase(splitext(bagian[1].filename)[2])
            isempty(ext) && (ext = ".jpg")
            path = tempname() * ext
            write(path, data)
            p = prediksi(path)
            persen = join(round.(100 .* p; digits=1), ",")
            nama = join(("\"" * l * "\"" for l in LABEL), ",")
            json = "{\"nama\":[" * nama * "],\"persen\":[" * persen * "]}"
            return HTTP.Response(200, ["Content-Type" => "application/json"], json)
        catch e
            pesan = replace(first(split(sprint(showerror, e), '\n')), r"[\"\\]" => "'")
            return HTTP.Response(400, ["Content-Type" => "application/json"],
                                 "{\"error\":\"" * pesan * "\"}")
        finally
            isempty(path) || rm(path; force=true)
        end
    end
    return HTTP.Response(404, "Tidak ditemukan")
end

host = BISA_DARI_HP ? "0.0.0.0" : "127.0.0.1"
println("Siap! Buka di komputer ini: http://localhost:$PORT")
if BISA_DARI_HP
    ip = try string(getipaddr()) catch; "(cek dengan perintah ipconfig)" end
    println("Buka dari HP (Wi-Fi yang sama): http://$ip:$PORT")
end
println("Tekan Ctrl+C untuk berhenti.")
HTTP.serve(handler, host, PORT)