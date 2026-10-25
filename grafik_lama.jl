# Memanggil praktek.jl: memuat data, melatih model, dan menyimpannya
include(joinpath(@__DIR__, "praktek.jl"))

# Grafik kurva training
p1 = plot(1:EPOCHS, history.loss, label="Loss", xlabel="Epoch", ylabel="Loss")
p2 = plot(1:EPOCHS, history.acc, label="Akurasi", xlabel="Epoch", ylabel="Akurasi")
plot(p1, p2, layout=(1,2), size=(900,350))
savefig("kurva_training.png")

# Confusion matrix
pred   = Flux.onecold(model(Xtest))
aktual = Flux.onecold(ytest)
n = length(KELAS)
cm = zeros(Int, n, n)
for (a, p) in zip(aktual, pred)
    cm[a, p] += 1
end
println("Confusion matrix (baris = aktual, kolom = prediksi):")
println("Urutan kelas: ", KELAS)
for i in 1:n
    println(rpad(KELAS[i], 22), cm[i, :])
end
