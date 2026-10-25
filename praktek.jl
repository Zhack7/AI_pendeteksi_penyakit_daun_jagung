using Flux
using Images, FileIO
using MLUtils: splitobs, DataLoader
using Statistics: mean
using Plots

const KELAS = ["Healthy", "Common_Rust", "Gray_Leaf_Spot", "Northern_Leaf_Blight"]
const UKURAN = (128, 128)

function load_dataset(root::String)
    images = Array{Float32,4}[]
    labels = Int[]
    gagal = 0
    for (i, kelas) in enumerate(KELAS)
        folder = joinpath(root, kelas)
        for file in readdir(folder)
            try
                img = load(joinpath(folder, file))
                img = imresize(img, UKURAN)
                img = RGB.(img)
                arr = Float32.(channelview(img))
                arr = permutedims(arr, (3,2,1))
                push!(images, reshape(arr, UKURAN..., 3, 1))
                push!(labels, i)
            catch
                gagal += 1
            end
        end
    end
    println("Foto yang dilewati (rusak/tidak didukung): ", gagal)
    X = cat(images...; dims=4)
    y = Flux.onehotbatch(labels, 1:length(KELAS))
    return X, y
end

X, y = load_dataset("data/")

# Normalisasi piksel ke [0,1] sudah dilakukan oleh load(); split train/test 80:20
(Xtrain, ytrain), (Xtest, ytest) = splitobs((X, y); at=0.8, shuffle=true)

train_loader = DataLoader((Xtrain, ytrain); batchsize=32, shuffle=true)

model = Chain(
    Conv((3,3), 3=>16, relu; pad=1), MaxPool((2,2)),
    Conv((3,3), 16=>32, relu; pad=1), MaxPool((2,2)),
    Conv((3,3), 32=>64, relu; pad=1), MaxPool((2,2)),
    Flux.flatten,
    Dense(64 * 16 * 16, 128, relu),
    Dropout(0.4),
    Dense(128, length(KELAS)),
    softmax
)

opt_state = Flux.setup(Adam(1e-3), model)
loss_fn(m, x, y) = Flux.crossentropy(m(x), y)

EPOCHS = 20
history = (loss=Float64[], acc=Float64[])

for epoch in 1:EPOCHS
    for (xb, yb) in train_loader
        grads = Flux.gradient(m -> loss_fn(m, xb, yb), model)
        Flux.update!(opt_state, model, grads[1])
    end
    train_loss = loss_fn(model, Xtrain, ytrain)
    train_acc  = mean(Flux.onecold(model(Xtrain)) .== Flux.onecold(ytrain))
    push!(history.loss, train_loss)
    push!(history.acc, train_acc)
    println("Epoch $epoch: loss=$(round(train_loss, digits=4)), acc=$(round(train_acc, digits=4))")
end

test_acc = mean(Flux.onecold(model(Xtest)) .== Flux.onecold(ytest))
println("Akurasi data uji: $(round(test_acc * 100, digits=2))%")

using BSON: @save
@save "model_jagung.bson" model

   println("Ukuran data X: ", size(X))
   println("Ukuran label y: ", size(y))
