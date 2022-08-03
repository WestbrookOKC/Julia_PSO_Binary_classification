using CSV, DataFrames,Flux,Plots,Random,Distributions
real_money = DataFrame(CSV.File("/Users/fengkaiqi/Desktop/Julia/hw2-fkq-real_or_fake_money/Data/real-money-traindata.csv",normalizenames=true))#533*5
fack_money =DataFrame(CSV.File("/Users/fengkaiqi/Desktop/Julia/hw2-fkq-real_or_fake_money/Data/notgood-traindata.csv",normalizenames=true))#427*5
x_real_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(real_money) ]#533
x_fake_money = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(fack_money) ]#427
xs = vcat(x_real_money,x_fake_money)
using Flux:onehot
ys = vcat( fill(onehot(0,0:1), size(x_real_money)),
           fill(onehot(1,0:1), size(x_fake_money)))

real_money_test = DataFrame(CSV.File("/Users/fengkaiqi/Desktop/Julia/hw2-fkq-real_or_fake_money/Data/real-money-testdata.csv",normalizenames=true))
fack_money_test =DataFrame(CSV.File("/Users/fengkaiqi/Desktop/Julia/hw2-fkq-real_or_fake_money/Data/notgood-testdata.csv",normalizenames=true))
x_real_money_test = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(real_money_test) ]#229
x_fake_money_test = [ [row."V1", row."V2",row."V3",row."V4"] for row in eachrow(fack_money_test) ]#183
xss = vcat(x_real_money_test,x_fake_money_test)
using Flux:onehot
yss = vcat( fill(onehot(0,0:1), size(x_real_money_test)),
           fill(onehot(1,0:1), size(x_fake_money_test)))
testbatch = (Flux.batch(xss), Flux.batch(yss))

using Flux
layer1 = Dense(4,8,σ)
layer2 = Dense(8,2,identity)
using Statistics
using Printf

Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))]
#prediction(i) = findmax(model(Flux.batch(xs[i])))[2] - 1
function Confusion_Martix!(model)
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = findmax(model(Flux.batch(xs[i])))[2] - 1
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end
    end
    print([TP FN
            FP TN])
end

function getmodel()
    model = Chain(Dense(4,8,σ),Dense(8,2,identity),softmax)
end
function population(size)
    list = []
    for i in 1:size
        rand_model = getmodel()
        append!(list,[rand_model])
    end
    return list
end
function accuracy_each_model(rand_model)
    Actual = [fill(0,size(x_real_money)); fill(1,size(x_fake_money))]
    prediction(i) = findmax(rand_model(xs[i]))[2] - 1
    TP,FP,TN,FN = 0,0,0,0
    for i in 1:960
        predict_value = prediction(i)
        if predict_value == 0 && Actual[i] ==  0
            TP += 1
        end
        if predict_value == 0 && Actual[i] ==  1
            FP += 1
        end
        if predict_value == 1 && Actual[i] ==  1
            TN += 1
        end
        if predict_value == 1 && Actual[i] ==  0
            FN += 1
        end
    end
    accuracy = (TN + TP) / (TP+TN+FN+FP)
    return accuracy
end

global test = []
global dict = Dict()
global pbest = Matrix{Float64}(undef, 100, 101)
for i in 1:10000# make a matrix for gbest
    pbest[i] = 0
end
global gbest = []
global p =[]
global p1 = []
a = population(100)
acc = zeros(100)
#dict = Dict()
acc_list = []
final_list = zeros(100)
for i in 1:100# dict to link them
    acc[i] = accuracy_each_model(a[i])
    key = acc[i]
    value = a[i]
    push!(dict,key =>value)
end
x = sort(acc,rev = true)
append!(acc_list,x)# save the acc array
w = 0.6
vmax = 0.2
c1 = 1
c2 =2
v = zeros(100)
for i in 1:100# randly initialize v
    v[i] = rand()
end
for i in 1:100#for layer2
    append!(p,dict[acc_list[i]][2].weight)
end
for i in 1:100#for layer1
    append!(p1,dict[acc_list[i]][1].weight)
end
for i in 1:100# initialize pbest
    pbest[i,1] = acc_list[i]
end
acc_list

pbest[1,1]
pbest[1,2]
pbest[1][1]

#main loop for try
for i in 1:100
    if acc_list[1]>=0.98
        break
    end
    #for layer2
    for k in 1:16
        b = 0
        b = dict[pbest[i,1]][2].weight[k]
        acc_list = sort(acc_list,rev = true)
        gbest = dict[acc_list[1]][2].weight[k]
        v[i]=w*v[i]+c1*rand()*(b-p[i])+c2*rand()*(gbest-p[i])
        if v[i] > vmax
            v[i] = vmax
        end
        dict[acc_list[i]][2].weight[k] = dict[acc_list[i]][2].weight[k] + v[i]
        
    end
    #for layer1
    for j in 1:32
        q = 0
        q = dict[pbest[i,1]][1].weight[j]
        acc_list = sort(acc_list,rev = true)
        gbest = dict[acc_list[1]][1].weight[j]
        v[i]=w*v[i]+c1*rand()*(q-p1[i])+c2*rand()*(gbest-p1[i])
        if v[i] > vmax
            v[i] = vmax
        end
        dict[acc_list[i]][1].weight[j] = dict[acc_list[i]][1].weight[j] + v[i]
        
    end
    
    append!(acc_list,accuracy_each_model(dict[acc_list[i]]))
    push!(dict,acc_list[i+100]=>dict[acc_list[i]])
    acc_list = sort(acc_list,rev = true)
    #key = accuracy_each_model(dict[acc_list[i]])
    #value = a[i]
    #push!(dict,key =>value)
    pbest[i,i+1] = accuracy_each_model(dict[acc_list[i]])
    pbest = sort(pbest,dims = 2,rev=true)
end 


acc_list
pbest











import Plots
using Plots
acc_list = sort(acc_list)

scatter(acc_list,1:length(acc_list))

