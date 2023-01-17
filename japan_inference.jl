include("../src/SOT.jl")
using .SOT, Printf, LinearAlgebra, HDF5, DataFrames, CSV, Random, Distributions, PyPlot,SparseArrays
station = "WAKE"
nexclude = 71
h5file = @sprintf("data/japan_%s_x2y_%dex.h5",station,nexclude)
ppfile = @sprintf("../results/pairs/japan_%s_ppairs_2.5a3hz_%dex.csv",station,nexclude)
tpvfile = @sprintf("../results/pairs/japan_%s_tpairs_2.5a3hz_%dex.csv",station,nexclude)
θ = h5read(h5file, "θ")[:]
y = h5read(h5file, "y")[:]
l = 1
# manually exclude pairs
tpairs = CSV.read(tpvfile, DataFrame)
ppairs = CSV.read(ppfile, DataFrame)

# collect delays into data vector
#y = [tpairs.Δτ; ppairs.Δτ]

t,E = SOT.getE(tpairs,ppairs)
#tj = t[θ .≤ 0.9]
#tpairsj = tpairs[in.(tpairs.event1, [Set(tj)]), :]
#ppairsj = innerjoin(ppairs, select(tpairsj, [:event1, :event2]), on=[:event1, :event2])
#yj = [y[1:size(tpairs,1)][in.(tpairs.event1, [Set(tj)])]; ppairsj.Δτ]
#tj,Ej = getE(tpairsj,ppairsj)
#θj = θ[indexin(tj,t)]

# wotc~20it,
# correlation time (days)
# WAKE 37 [27,50] w otc; 38 [29,49] 
λt = 37

# correlation azimuth (degrees)
# WAKE 0.69 [0.55, 0.85]; H11 0.95 [0.75,1.19]
λθ = 0.69

# solution standard deviation for travel time anomalies (s)
# WAKE 0.30 [0.27,0.33]; H11 0.29 [0.26,0.31]
στ = 0.30

# noise (s)
# WAKE 0.0064 [0.0062,0.0066]; H11 0.0055 [5.4e-3,5.7e-3]
σn = 0.0064

# origin time correction standard deviation (s)
# WAKE 0.91 [0.85,0.98]; H11 1 [0.94,1.06]
σp = 0.91

# trend prior for coefficients of singular vectors (s/year)
# WAKE 0.009 [0.001, 0.06]; H11 0.05 [0.01,0.23]
σtrend = 0.009

## BFGS
# nootc: ML~9.40e3 vs 9.41e3
x0 = log.([λt,λθ,στ,σtrend,σp,σn])
bounds = [20 60;0.04 2;0.1 0.5;1e-3 0.1;0.1 2;3e-3 8e-3]
# small font
rc("font", size=8)
rc("axes", titlesize="medium")
fig, ax = subplots(2, 3, figsize=(6.4, 5.2), sharey=true)
ax = reshape(ax,(6,))
xlabels = ["\$\\lambda_t\$ (day)","\$\\lambda_a\$ (deg)","\$\\sigma_\\tau\$ (s)","\$\\sigma_t\$ (s/y)","\$\\sigma_p\$ (s)","\$\\sigma_n\$ (s)"]
#xlabels = ["\$\\ln(\\lambda_t)\$","\$\\ln(\\lambda_a)\$","\$\\ln(\\sigma_\\tau)\$","\$\\ln(\\sigma_t)\$","\$\\ln(\\sigma_p)\$","\$\\ln(\\sigma_n)\$"]
for i = 1:6
    @printf("parameter %d...\n",i)
    gridi = LinRange((bounds[i,1]), (bounds[i,2]), 50)
    #Δ = gridi[2]-gridi[1]
    loglikei = similar(gridi)
    xg = x0[:]
    for (j,g) in enumerate(gridi)
        xg[i] = log(g)
        loglikei[j] = SOT.loglikelihood(xg, y, t, θ, E;grad=false)
    end
    wi = SOT.exp_and_normalise(loglikei)
    ax[i].scatter(gridi,wi,s=10)
    ax[i].set_xlabel(xlabels[i])
end
ax[1].set_ylabel("weight")
ax[2].set_ylabel("weight")
ax = reshape(ax,(2,3))
for i = 1:2, j=1:3
    ax[i,j].set_title("($(('a':'z')[(i-1)*3+j]))", loc="left")
end
fig.tight_layout()
fig.savefig(@sprintf("results/marginalike_%s.pdf",station))


#f(x) = -1 .* SOT.loglikelihood(x, y, t, θ, E)
#@time SOT.BFGS(f,x0,1e-5,100)
xk,Hk = SOT.BFGS(f,x0,1e-5,100)
@printf("MLE: %s\n",exp.(xk))
@printf("lb: %s\n ub: %s\n",exp.(xk.-2*sqrt.(diag(Hk))),exp.(xk.+2*sqrt.(diag(Hk))))
#invH = inv(SOT.hess(f,xk))
#@printf("lb: %s\n ub: %s\n",exp.(xk.-2*sqrt.(diag(invH))),exp.(xk.+2*sqrt.(diag(invH))))
##

## Sequential Monte Carlo
Random.seed!(123)
θsep = [-10 -5.5 -1.25 0.9 10]
#H = inv(SOT.hess(f,x0))
#σ0 = [0.1 0.1 0.05 0.1 0.05 0.05]
δ = 2.38^2/size(H,1)
Σ = Array(spdiagm(0 => diag(H,0)))
π0 = MvNormal(x0, δ*Σ)
J,N,K = size(tpairs,1),100,5
ESSmin = N/2
v,W = SOT.IBIS(J,N,K,π0,δ,Σ,t,θ,y,tpairs,ppairs,ESSmin)

fig, ax = subplots(2, 2)
for i = 1:2
    for j = 1:2
        if j+2*(i-1)≤J
            ax[i,j].scatter(exp.(v[1,:,j+2*(i-1)]),exp.(v[2,:,j+2*(i-1)]),c=(w[:,j+2*(i-1)]))
            ax[i,j].set_xlabel("\$\\lambda_t\$ (day)")
            ax[i,j].set_ylabel("\$\\lambda_\\alpha \$ (deg)")
        end
    end
end
fig.tight_layout()
fig.savefig("results/scales.pdf")

fig, ax = subplots(2, 2)
for i = 1:2
    for j = 1:2
        if j+2*(i-1)≤J
            ax[i,j].scatter(exp.(v[3,:,j+2*(i-1)]),exp.(v[4,:,j+2*(i-1)]),c=(w[:,j+2*(i-1)]))
            ax[i,j].set_xlabel("\$\\sigma_\\tau \$ (s)")
            ax[i,j].set_ylabel("\$\\sigma_t \$ (s)")
        end
    end
end
fig.tight_layout()
fig.savefig("results/sigma.pdf")

fig, ax = subplots(2, 2)
for i = 1:2
    for j = 1:2
        if j+2*(i-1)≤J
            ax[i,j].scatter(exp.(v[5,:,j+2*(i-1)]),exp.(v[6,:,j+2*(i-1)]),c=(w[:,j+2*(i-1)]))
            ax[i,j].set_xlabel("\$\\sigma_p \$ (s)")
            ax[i,j].set_ylabel("\$\\sigma_n \$ (s)")
        end
    end
end
fig.tight_layout()
fig.savefig("results/sigma0.pdf")