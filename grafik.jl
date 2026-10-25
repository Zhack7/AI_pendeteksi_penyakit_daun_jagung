include(joinpath(@__DIR__, "praktek.jl"))
using Plots.PlotMeasures

# ===== Pengaturan tampilan =====
gelap = true   # ubah ke false kalau ingin latar putih (lebih hemat tinta saat dicetak)

bg         = gelap ? RGB(0.118, 0.129, 0.094) : RGB(1, 1, 1)
teks       = gelap ? RGB(0.929, 0.906, 0.824) : RGB(0.10, 0.10, 0.10)
w_loss     = gelap ? RGB(0.851, 0.478, 0.314) : RGB(0.71, 0.29, 0.15)
w_acc      = gelap ? RGB(0.498, 0.722, 0.580) : RGB(0.24, 0.42, 0.31)
w_diag     = gelap ? RGB(0.19, 0.29, 0.22)    : RGB(0.75, 0.88, 0.78)
w_salah    = gelap ? RGB(0.23, 0.16, 0.13)    : RGB(0.95, 0.85, 0.78)
w_garis    = gelap ? RGB(0.22, 0.23, 0.18)    : RGB(0.80, 0.80, 0.75)
w_header   = w_acc 

# ===== Grafik 1: kurva training (dua sumbu) =====
epochs = 1:EPOCHS
persen = ["0%", "20%", "40%", "60%", "80%", "100%"]

p = plot(epochs, history.loss, color=w_loss, lw=2.5, label="Loss (train)",
         xlabel="Epoch", ylabel="Loss", ylims=(0, maximum(history.loss) * 1.1),
         title="Kurva training: loss & akurasi", titlefontsize=12,
         background_color=bg, foreground_color=teks, grid=false,
         legend=:outerbottom, legend_column=2, size=(900, 450),
         left_margin=8mm, right_margin=8mm, bottom_margin=6mm)
plot!(p, [NaN], [NaN], color=w_acc, lw=2.5, label="Akurasi (train)")   # hanya untuk legenda
p = plot!(twinx(), epochs, history.acc, color=w_acc, lw=2.5, label="",
          ylabel="Akurasi", ylims=(0, 1), yticks=(0:0.2:1, persen),
          grid=false, legend=false, foreground_color=teks)
savefig(p, "kurva_training.png")

# ===== Grafik 2: confusion matrix (tampilan tabel) =====
pred   = Flux.onecold(model(Xtest))
aktual = Flux.onecold(ytest)
n = length(KELAS)
cm = zeros(Int, n, n)
for (a, b) in zip(aktual, pred)
    cm[a, b] += 1
end

nama = ["Healthy", "Common Rust", "Gray Leaf Spot", "N. Leaf Blight"]
lebar_label = 1.9
lebar_total = lebar_label + n
kotak(x0, x1, y0, y1) = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])

t = plot(xlims=(0, lebar_total), ylims=(-1.0, n + 1), yflip=true,
         framestyle=:none, ticks=false, legend=false, grid=false,
         background_color=bg, size=(850, 380),
         left_margin=6mm, right_margin=6mm, top_margin=4mm, bottom_margin=4mm)

# Judul rata kiri
annotate!(t, 0.05, -0.45, text("Confusion matrix (data uji)", 13, w_header, :left))

# Warna sel: diagonal hijau, salah tebak cokelat, nol dibiarkan kosong
for i in 1:n, j in 1:n
    if cm[i, j] > 0
        x0 = lebar_label + j - 1
        warna = i == j ? w_diag : w_salah
        plot!(t, kotak(x0, x0 + 1, i, i + 1), fillcolor=warna, lw=0, linealpha=0)
    end
end

# Garis pemisah horizontal
for y in 1:(n + 1)
    plot!(t, [0, lebar_total], [y, y], color=w_garis, lw=1)
end

# Teks header
annotate!(t, 0.1, 0.5, text("Aktual \\ Prediksi", 9, w_header, :left))
for j in 1:n
    annotate!(t, lebar_label + j - 0.5, 0.5, text(nama[j], 9, w_header, :center))
end

# Nama baris dan angka (diagonal dicetak tebal dengan menimpa teks sedikit bergeser)
for i in 1:n
    annotate!(t, 0.1, i + 0.5, text(nama[i], 10, teks, :left))
    for j in 1:n
        x = lebar_label + j - 0.5
        s = string(cm[i, j])
        annotate!(t, x, i + 0.5, text(s, 12, teks, :center))
        if i == j
            annotate!(t, x + 0.015, i + 0.5, text(s, 12, teks, :center))
        end
    end
end

savefig(t, "confusion_matrix.png")

nama = ["Healthy", "Common Rust", "Gray Leaf Spot", "N. Leaf Blight"]
lebar_label = 1.8

kotak(x0, x1, y0, y1) = Shape([x0, x1, x1, x0], [y0, y0, y1, y1])

t = plot(xlims=(0, lebar_label + n), ylims=(0, n + 1), yflip=true,
         framestyle=:none, ticks=false, legend=false, grid=false,
         background_color=bg, size=(850, 450), left_margin=4mm, right_margin=4mm,
         title="Confusion matrix (data uji)", titlefontsize=12, foreground_color=teks)

annotate!(t, 0.1, 0.5, text("Aktual \\ Prediksi", 9, w_header, :left))
for j in 1:n
    annotate!(t, lebar_label + j - 0.5, 0.5, text(nama[j], 9, w_header, :center))
end
for i in 1:n
    annotate!(t, 0.1, i + 0.5, text(nama[i], 10, teks, :left))
    for j in 1:n
        x0 = lebar_label + j - 1
        warna = i == j ? w_diag : (cm[i, j] > 0 ? w_salah : bg)
        plot!(t, kotak(x0, x0 + 1, i, i + 1), fillcolor=warna, linecolor=w_garis, lw=1)
        annotate!(t, x0 + 0.5, i + 0.5, text(string(cm[i, j]), 12, teks, :center))
    end
end
savefig(t, "confusion_matrix.png")

println("Selesai. Lihat kurva_training.png dan confusion_matrix.png")