# TODO:
# - Use sparse QR decomposition if inversion matrix becomes too big (qr([E; S])\y)

# reference time
global const tref = DateTime(2000, 1, 1, 0, 0, 0)

# mean year in days
global const meanyear = 365.2425

function loglikelihood(x, y, t, θ, E; D=I,grad=true)
    λt,λθ,στ,σtrend,σp,σn = exp.(x)
    σtrend /= meanyear

    l = 1
    m = length(t)
    # real time (days)
    trd = Dates.value.(t - DateTime(2000, 1, 1, 12, 0, 0))/1000/3600/24

    # solution covariance in time
    C = στ^2*exp.(-abs.(trd.-trd')/λt-(θ.-θ').^2/λθ.^2/2)

    # covariance matrix assuming no correlation between singular vector expansion coefficients
    R = [C zeros(l*m, m); zeros(m, (l+1)*m)] + σp^2*kron(sparse(ones(l+1, l+1)), I(m))
    R = [R zeros(size(R, 1), l); zeros(l, size(R, 2)) (σtrend.^2)]

    # noise covariance
    N = σn^2*I
  
  #try
    Ryy = E*R*E' + N
    #Ryy = D*Ryy*D'
    iRyy = inv(Ryy)
    ll = -0.5*(logdet(Ryy)+y'*iRyy*y+length(y)*log(2π))
    if grad
        d = length(x)
        Δll = zeros(d)
        α = iRyy*y
        δll(α,iRyy,δRyy) = 0.5*tr((α*α'-iRyy)*δRyy)
        #δll(α,iRyy,δRyy) = 0.5*tr((α*α'-iRyy)*D*δRyy*D')
        δR1 = E*[(abs.(trd.-trd')/λt).*C zeros(l*m, m+1); zeros(m+1, (l+1)*m+1)]*E'
        Δll[1] = δll(α,iRyy,δR1)
        δR2 = E*[((θ.-θ').^2/λθ^2).*C zeros(l*m, m+1); zeros(m+1, (l+1)*m+1)]*E'
        Δll[2] = δll(α,iRyy,δR2)
        δR3 = E*[2*C zeros(l*m, m+1); zeros(m+1, (l+1)*m+1)]*E'
        Δll[3] = δll(α,iRyy,δR3)
        δR4 = E*spdiagm(0 => [zeros((l+1)*m); 2*(σtrend^2)])*E'
        Δll[4] = δll(α,iRyy,δR4)
        δR5 = E*[2*σp^2*kron(sparse(ones(l+1, l+1)), I(m)) zeros((l+1)*m, l); zeros(l, (l+1)*m+1)]*E'
        Δll[5] = δll(α,iRyy,δR5)
        δR6 = 2*N
        Δll[6] = δll(α,iRyy,δR6)
        return ll,Δll
    else
        return ll
    end
  #catch y
  #  @printf("singular covariance\n")
  #  rethrow(y)
  #  return NaN
  #end
end

# CENTRAL FINITE DIFFERENCE CALCULATION
function grad(f,x)
    h = cbrt(eps())
    d = length(x)
    nabla = zeros(d)
    for i = 1:d 
        x_for = copy(x) 
        x_back = copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        nabla[i] = (f(x_for) - f(x_back))/(2*h) 
        if isnan(nabla[i])
            @printf("NaN gradient\n")
            return nabla*NaN
        end
    end
    return nabla 
end

# CENTRAL FINITE DIFFERENCE CALCULATION
function hess(f,x)
    h = cbrt(eps())
    d = length(x)
    hess = zeros(d,d)
    for i = 1:d 
        x_for = copy(x) 
        x_back = copy(x)
        x_for[i] += h 
        x_back[i] -= h 
        _,∇f_for = f(x_for)
        _,∇f_back = f(x_back)
        hess[:,i] = (∇f_for - ∇f_back)/(2*h)
        if isnan(hess[i,i])
            @printf("NaN hessian\n")
            return hess*NaN
        end
    end
    return 0.5*(hess + hess') 
end

# BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
function line_search(f,∇f,x,p,∇)
    αi = 1
    ϕ(α) = f(x+α*p)
    dϕ(α) = ∇f(x+α*p)⋅p
    ϕ0 = ϕ(0)
    dϕ0 = ∇⋅p
    ϕi,dϕi = ϕ(αi),dϕ(αi)
    c1 = 1e-4 
    c2 = 0.9 

    while ϕi > ϕ0 + (c1*αi*dϕ0) || dϕi < c2*dϕ0
        αi *= 0.5
        ϕi,dϕi = ϕ(αi),dϕ(αi)
    end
    return αi
end
    
function line_search_swolfe(f, xk, pk; ∇fk=NaN, old_fval=NaN, old_old_fval=NaN, c1=1e-4, c2=0.9, amax=NaN,maxiter=10)
    #fc = [0]
    #gc = [0]
    gval = NaN

    function ϕ(α)
        #fc[1] += 1
        fval,_ = f(xk + α * pk)
        return fval
    end

    function derϕ(α)
        #gc[1] += 1
        _,gval = f(xk + α * pk)
        return dot(gval, pk)
    end

    derϕ0 = dot(∇fk, pk)

    α_star, ϕ_star, old_fval, derϕ_star = scalar_search_swolfe(ϕ, derϕ; ϕ0=old_fval, old_ϕ0=old_old_fval, derϕ0, c1, c2, amax, maxiter=maxiter)

    if isnan(derϕ_star)
        @printf("The line search algorithm did not converge\n")
    else
        # derϕ_star is a number (derϕ) -- so use the most recently
        # calculated gradient used in computing it derϕ = ∇fk*pk
        # this is the gradient at the next step no need to compute it
        # again in the outer loop.
        derϕ_star = gval
    end

    return α_star, ϕ_star, old_fval, derϕ_star
end

function scalar_search_swolfe(ϕ, derϕ; ϕ0=NaN, old_ϕ0=NaN, derϕ0=NaN, c1=1e-4, c2=0.9, amax=NaN, maxiter=10)
    if isnan(ϕ0)
        ϕ0 = ϕ(0.)
    end

    if isnan(derϕ0)
        derϕ0 = derϕ(0.)
    end

    α0 = 0
    if !isnan(old_ϕ0) && (derϕ0 != 0)
        α1 = min(1.0, 1.01*2*(ϕ0 - old_ϕ0)/derϕ0)
    else
        α1 = 1.0
    end

    if α1 < 0
        α1 = 1.0
    end

    if !isnan(amax)
        α1 = min(α1, amax)
    end

    ϕ_α1 = ϕ(α1)

    ϕ_α0 = ϕ0
    derϕ_α0 = derϕ0

    extra_condition(α, ϕ) = true

    i = 1
    while i ≤ maxiter
        if α1 == 0 || (!isnan(amax) && α0 == amax)
            # α1 == 0: This shouldn't happen. Perhaps the increment has
            # slipped below machine precision?
            α_star = NaN
            ϕ_star = ϕ0
            ϕ0 = old_ϕ0
            derϕ_star = NaN

            if α1 == 0
                @printf("Rounding errors prevent the line search from converging\n")
            else
                @printf("The line search algorithm could not find a solution ≤ amax: %s\n", amax)
            end

            break
        end
        not_first_iteration = i > 1
        if (ϕ_α1 > ϕ0 + c1 * α1 * derϕ0) || ((ϕ_α1 >= ϕ_α0) && not_first_iteration)
            α_star, ϕ_star, derϕ_star = zoom(α0, α1, ϕ_α0, ϕ_α1, derϕ_α0, ϕ, derϕ, ϕ0, derϕ0, c1, c2, extra_condition)
            break
        end
        derϕ_α1 = derϕ(α1)
        if (abs(derϕ_α1) ≤ -c2*derϕ0)
            if extra_condition(α1, ϕ_α1)
                α_star = α1
                ϕ_star = ϕ_α1
                derϕ_star = derϕ_α1
                break
            end
        end
        if (derϕ_α1 ≥ 0)
            α_star, ϕ_star, derϕ_star = zoom(α1, α0, ϕ_α1,ϕ_α0, derϕ_α1, ϕ, derϕ, ϕ0, derϕ0, c1, c2, extra_condition)
            break
        end
        α2 = 2 * α1  # increase by factor of two on each iteration
        if !isnan(amax)
            α2 = min(α2, amax)
        end
        α0 = copy(α1)
        α1 = copy(α2)
        ϕ_α0 = copy(ϕ_α1)
        ϕ_α1 = ϕ(α1)
        derϕ_α0 = copy(derϕ_α1)

        i += 1
    end

    if i > maxiter
        # stopping test maxiter reached
        α_star = copy(α1)
        ϕ_star = copy(ϕ_α1)
        derϕ_star = NaN
        @printf("Maximum Wolfe iteration reached\n")
        @printf("The line search algorithm did not converge\n")
    end
    
    return α_star, ϕ_star, ϕ0, derϕ_star
end

function zoom(α_lo, α_hi, ϕ_lo, ϕ_hi, derϕ_lo, ϕ, derϕ, ϕ0, derϕ0, c1, c2, extra_condition)
    maxiter = 10
    i = 0
    delta1 = 0.2  # cubic interpolant check
    delta2 = 0.1  # quadratic interpolant check
    ϕ_rec = ϕ0
    a_rec = 0
    a_star, val_star, valprime_star = NaN,NaN,NaN
    while true
        # interpolate to find a trial step length between α_lo and
        # α_hi Need to choose interpolation here. Use cubic
        # interpolation and then if the result is within delta *
        # dalpha or outside of the interval bounded by α_lo or α_hi
        # then use quadratic interpolation, if the result is still too
        # close, then use bisection

        dalpha = α_hi - α_lo
        if dalpha < 0 
            a, b = α_hi, α_lo
        else 
            a, b = α_lo, α_hi
        end
        # minimizer of cubic interpolant
        # (uses ϕ_lo, derϕ_lo, ϕ_hi, and the most recent value of ϕ)
        #
        # if the result is too close to the end points (or out of the
        # interval), then use quadratic interpolation with ϕ_lo,
        # derϕ_lo and ϕ_hi if the result is still too close to the
        # end points (or out of the interval) then use bisection

        if (i > 0) 
            cchk = delta1 * dalpha
            αj = cubicmin(α_lo, ϕ_lo, derϕ_lo, α_hi, ϕ_hi, a_rec, ϕ_rec)
        end
        if (i == 0) || isnan(αj) || (αj > b - cchk) || (αj < a + cchk) 
            qchk = delta2 * dalpha
            αj = quadmin(α_lo, ϕ_lo, derϕ_lo, α_hi, ϕ_hi)
            if isnan(αj) || (αj > b-qchk) || (αj < a+qchk) 
                αj = α_lo + 0.5*dalpha
            end
        end
αj
        # Check new value of αj

        ϕ_αj = ϕ(αj)
        if (ϕ_αj > ϕ0 + c1*αj*derϕ0) || (ϕ_αj >= ϕ_lo) 
            ϕ_rec = copy(ϕ_hi)
            a_rec = copy(α_hi)
            α_hi = copy(αj)
            ϕ_hi = copy(ϕ_αj)
        else 
            derϕ_αj = derϕ(αj)
            if abs(derϕ_αj) ≤ -c2*derϕ0 && extra_condition(αj, ϕ_αj) 
                a_star = copy(αj)
                val_star = copy(ϕ_αj)
                valprime_star = copy(derϕ_αj)
                break
            end
            if derϕ_αj*(α_hi - α_lo) ≥ 0 
                ϕ_rec = copy(ϕ_hi)
                a_rec = copy(α_hi)
                α_hi = copy(α_lo)
                ϕ_hi = copy(ϕ_lo)
            else 
                ϕ_rec = copy(ϕ_lo)
                a_rec = copy(α_lo)
            end
            α_lo = copy(αj)
            ϕ_lo = copy(ϕ_αj)
            derϕ_lo = copy(derϕ_αj)
        end
        i += 1
        if (i > maxiter) 
            # Failed to find a conforming step size
            @printf("Maximum iteration reached\n")
            @printf("Failed to find a conforming step size\n")
            a_star = NaN
            val_star = NaN
            valprime_star = NaN
            break
        end
    end
    return a_star, val_star, valprime_star
end

function cubicmin(a, fa, fpa, b, fb, c, fc)
    xmin = NaN
    try
        C = fpa
        db = b - a
        dc = c - a
        denom = (db * dc)^2 * (db - dc)
        d1 = [dc^2 -db^2; -dc^3 db^3]
        A, B = d1*vec([fb - fa - C * db, fc - fa - C * dc])
        A /= denom
        B /= denom
        radical = B * B - 3 * A * C
        xmin = a + (-B + sqrt(radical)) / (3 * A)
    catch y
        return NaN
    end

    if !isfinite(xmin)
        return NaN
    end

    return xmin
end

function quadmin(a, fa, fpa, b, fb)
    xmin = NaN
    try
        D = fa
        C = fpa
        db = b - a * 1.0
        B = (fb - D - C * db) / (db * db)
        xmin = a - C / (2.0 * B)
    catch y
        return NaN
    end
    if !isfinite(xmin)
        return NaN
    end
    return xmin
end

function BFGS(f,x0,∇tol,maxiter; xrtol=0, safelog = false)
    d = length(x0) # dimension of problem 
    old_fval,∇fk = f(x0) # initial gradient 
    Hk = inv(hess(f,x0)) # I(d) # initial inverse hessian approx

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + norm(∇fk) / 2

    xk = x0[:]
    if safelog
        s = -Hk*∇fk
        x1 = x0 + s
        changex = 0
        while minimum(x1)<-7 || maximum(x1)>5 || norm(Hk*∇fk)>20
            changex = 1
            @printf("xmin = %.2e, xmax = %.2e, |H∇f| = %.2e\n", minimum(x1),maximum(x1),norm(H*∇f))
            s *= 0.5
            x1 = x0 + s
            _,∇fk = f(x1) # nabla = grad(f,x1)
            _,∇f1 = f(x0 + 0.5*s) # nabla2 = grad(f,x0 + 0.5*s)
            if !isnan(norm(∇fk)) && !isnan(norm(∇f1)) && norm(Hk*∇fk)<norm(Hk*∇f1) 
                break
            end
        end
        if changex>0
            xk = x1[]
            Hk = inv(hess(f,xk))
        end
    end

    k = 0 
    @printf("-H∇f = %s\n", -Hk*∇fk)
    ∇norm = norm(∇fk)

    while ∇norm > ∇tol && k < maxiter
        @printf("k = %d, ek = %.2e\n", k, ∇norm)
        @printf("xk = %s\n", xk)
        if k > maxiter
            @printf("Maximum iterations reached!\n")
            break
        end
        pk = -Hk*∇fk # search direction (Newton Method)
        
        αk, ∇fkp1 = NaN,∇fk[:]
        try
            αk, old_fval, old_old_fval, ∇fkp1 = line_search_swolfe(f, xk, pk; ∇fk,old_fval, old_old_fval, amax=1e100)
        catch y
            # Line search failed to find a better solution.
           break
        end

        #αk = line_search(f,∇f,xk,pk,∇fk) # line search 
        @printf("αk = %s\n", αk)
        if isnan(αk)
            break
        end
        sk = vec(αk * pk)
        xkp1 = xk .+ sk

        xk = xkp1[:]
        #∇fkp1 = ∇f(xkp1)
        yk = vec(∇fkp1 .- ∇fk)
        ∇fk = ∇fkp1[:] 
        
        k += 1
        ∇norm = norm(∇fk)
        if ∇norm ≤ ∇tol
            break
        end

        if αk*norm(pk) ≤ xrtol*(xrtol + norm(xk))
            break
        end

        ρk_inv = yk⋅sk
        if ρk_inv == 0.
            ρk = 1000.0
            @printf("Divide-by-zero encountered: ρk assumed large\n")
        else
            ρk = 1 / ρk_inv
        end

        A1 = I(d)-ρk*(sk*yk')
        A2 = I(d)-ρk*(yk*sk')
        if k < 1
             Hk = ρk_inv/(yk⋅yk)*I(d)
        end
        Hk = A1*Hk*A2 + ρk*(sk*sk') # BFGS Update
        
    end
    return xk,Hk
end

function getE(tpairs,ppairs)
    l = 1

    # number of T- and P-wave pairs
    nt = size(tpairs, 1)
    np = size(ppairs, 1)
    
    # find unique events
    t = sort(unique([tpairs.event1; tpairs.event2]))

    # number of unique events
    m = length(t)

    # real time (days)
    tr = Dates.value.(t - DateTime(2000, 1, 1, 12, 0, 0))/1000/3600/24

    # T-wave pair matrix
    tidx1 = indexin(tpairs.event1, t)
    tidx2 = indexin(tpairs.event2, t)
    Xt = sparse([1:nt; 1:nt], [tidx1; tidx2], [-ones(nt); ones(nt)])

    # P-wave pair matrix
    pidx1 = indexin(ppairs.event1, t)
    pidx2 = indexin(ppairs.event2, t)
    Xp = sparse([1:np; 1:np], [pidx1; pidx2], [-ones(np); ones(np)])

    # full design matrix
    E = blockdiag([Xt for i = 1:l]..., Xp)
    tm = tr[1]+(tr[m]-tr[1])/2
    E = [E [kron(I(l), Xt*(tr.-tm)); zeros(np, l)]]
    #D = [I(nt) -(E[1:nt,1:m]*pinv(Array(E[l*nt+1:l*nt+np,l*m+1:(l+1)*m])))]
    return t,E
end

function exp_and_normalise(lw)
    w = exp.(lw .- max(lw...))
    return w ./ sum(w)
end

function inverse_cdf(su,W)
    j=1
    s=W[1]
    M=size(su,1)
    A=Array{Int}(undef, M)
    for n = 1:M
      while su[n]>s
        j+=1
        s+=W[j]
      end
      A[n]=j
    end  
    return A
end
  
function systematic(M,W)
    su=(rand(1).+(0:1.0:M-1))/M
    return inverse_cdf(su,W)
end

ESS(W) = 1/sum(W.^2)

function importance_sampling(target, proposal, ϕ, N=1000)
    x = rand(proposal,N)
    lw = target.logpdf(x) - proposal.logpdf(x)
    W = exp_and_normalise(lw)
    return average(ϕ(x), weights=W)
end

function BPF(J,N,π0,Σ,t,θ,y,θsep,tpairs,ppairs,ESSmin)
    d = size(Σ,1)
    ntj = size(tpairs,1) ÷ J
    v = zeros(d,N,J+1)
    w = zeros(N,J+1)
    W = zeros(N,J+1)
    v[:,:,1] = v0
    tpairs0 = tpairs[1:ntj, :]
    ppairs0 = innerjoin(ppairs, select(tpairs0, [:event1, :event2]), on=[:event1, :event2])
    y0 = [y[1:ntj]; ppairs0.Δτ]
    t0,E0 = getE(tpairs0,ppairs0)
    θ0 = θ[indexin(t0,t)]
    for n = 1:N
        w[n,1] = loglikelihood(v0[n], y0, t0, θ0, E0; grad=false)
    end
    w[:,1] = exp.(w[:,1]-max(w[:,1]...))
    W[:,1] = w[:,1]./sum(w[:,1])

    for j = 1:J
        @printf("bootstrap particle filter j = %d\n", j)
        if ESS(W[:,j])<ESSmin
            A = systematic(N,W[:,j])
            what = ones(N)
        else
            A = 1:N
            what = w[:,j]
        end
        
        #θj,tj = θ[θsep[j+1] .<θ .<= θsep[j+2]],t[θsep[j+1] .<θ .<= θsep[j+2]]
        tpairsj = tpairs[ntj*j+1:ntj*(j+1),:]
        #tpairsj = tpairs[in.(tpairs.event1, [Set(tj)]), :]
        ppairsj = innerjoin(ppairs, select(tpairsj, [:event1, :event2]), on=[:event1, :event2])
        yj = [y[ntj*j+1:ntj*(j+1)]; ppairsj.Δτ]
        #yj = [y[1:size(tpairs,1)][in.(tpairs.event1, [Set(tj)])]; ppairsj.Δτ]
        tj,Ej = getE(tpairsj,ppairsj)
        θj = θ[indexin(tj,t)]
        for n = 1:N
            v[:,n,j+1] = v[:,A[n],j]+rand(MvNormal(zeros(d), Σ),1)
            w[n,j+1] = loglikelihood(v[:,n,j+1], yj, tj, θj, Ej; grad=false)
        end
        w[:,j+1] = what.*exp.(w[:,j+1].-max(w[:,j+1]...))
        W[:,j+1] = W[:,j+1]./sum(w[:,j+1])                 
    end
    return W, v
end

function IBIS(J,N,K,π0,δ,Σ,t,θ,y,tpairs,ppairs,ESSmin)
    d = size(Σ,1)
    ntj = size(tpairs,1) ÷ J
    v = zeros(d,N,J+1)
    w = zeros(N,J+1)
    #∇ = zeros(N)
    W = zeros(N,J+1)
    v[:,:,1] = rand(π0,N)
    tpairs0 = tpairs[1:ntj, :]
    ppairs0 = innerjoin(ppairs, select(tpairs0, [:event1, :event2]), on=[:event1, :event2])
    y0 = [y[1:ntj]; ppairs0.Δτ]
    t0,E0 = getE(tpairs0,ppairs0)
    θ0 = θ[indexin(t0,t)]
    for n = 1:N
        w[n,1] = loglikelihood(v[:,n,1], y0, t0, θ0, E0; grad=false)
    end
    γj_1 = w[:,1]
    w[:,1] = exp.(w[:,1].-max(w[:,1]...))
    W[:,1] = w[:,1]./sum(w[:,1])

    for j = 1:J
        @printf("IBIS j/J = %d/%d\n", j,J)
        tpairsj = tpairs[1:(1+j)*ntj, :]
        ppairsj = innerjoin(ppairs, select(tpairsj, [:event1, :event2]), on=[:event1, :event2])
        yj = [y[1:(1+j)*ntj]; ppairsj.Δτ]
        tj,Ej = getE(tpairsj,ppairsj)
        θj = θ[indexin(tj,t)]
        if ESS(W[:,j])<ESSmin
            A = systematic(N,W[:,j])
            what = ones(N)
            for n = 1:N
                v0 = v[:,A[n],j]
                lq0 = loglikelihood(v0, yj, tj, θj, Ej; grad=false)
                for k = 1:K
                    vh = rand(MvNormal(v0,δ*Σ),1)
                    u = rand()
                    lq1 = loglikelihood(vh, yj, tj, θj, Ej; grad=false)                   
                    lr = lq1 - lq0
                    if log(u)≤lr
                        v0 = vh[:]
                        lq0 = copy(lq1)
                    end                        
                end
                v[:,n,j+1] = v0[:]
                w[n,j+1] = copy(lq0)
            end
        else
            A = 1:N
            what = w[:,j]
            v[:,:,j+1] = v[:,:,j]
            for n = 1:N
                w[n,j+1] = loglikelihood(v[:,n,j+1], yj, tj, θj, Ej; grad=false)
            end
        end
        
        γj = w[:,j+1]
        w[:,j+1] = what.*exp.(γj.-γj_1)
        W[:,j+1] = w[:,j+1]./sum(w[:,j+1])
        γj_1 = γj[:]
    end
    return v,W
end