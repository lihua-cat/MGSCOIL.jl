function optical_extraction!(u, n_upper, n_lower, d, dt, line, sw = 1)
    gain = line.σ * (n_upper - line.gg * n_lower) ./ (1 + abs2.(u) ./ (n_upper * d)) * sw
    Δn_sp = n_upper * line.A * dt          
    Δpn_sp = Δn_sp * d
    @. u += sqrt(Δpn_sp)                 # no random phase noise
    Δn_st = gain .* abs2.(u)    
    @. u *= exp(1/2 * gain * d)
    n_upper .-= Δn_sp + Δn_st
    n_lower .+= Δn_sp + Δn_st
    nothing
end

function optical_extraction!(u::CuArray, n_upper::CuArray, n_lower::CuArray, d, dt, line, random = false, sw = 1)
    @uustrip d dt line
    s = size(u)
    threads = (64, 8)
    blocks = cld.(s, threads)
    @cuda threads = threads blocks = blocks kernel_oe(u, n_upper, n_lower, d, dt, line, s, random, sw)
    synchronize()
    threads = (32, 8)
    blocks = cld.(s, threads)
    @cuda threads = threads blocks = blocks kernel_fc(u, n_upper, n_lower, s)
    synchronize()
end

function kernel_oe(u, n_upper, n_lower, d, dt, line, s, random, sw)
    idi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y)

    for i in idi:stride[1]:s[1], j in idj:stride[2]:s[2]÷2
        nu = n_upper[i, j]
        nl = n_lower[i, j]
        uij = u[i, j]
        gain = line.σ * (nu - line.gg * nl) / (1 + abs2(uij) / (nu * d)) * sw
        Δn_sp = nu * line.A * dt
        Δpn_sp = Δn_sp * d / 40
        if random
            uij += sqrt(Δpn_sp) * exp((rand() - 1/2) * 2im * π)
        else
            uij += sqrt(Δpn_sp)
        end
        Δn_st = gain * abs2(uij)
        uij *= exp(1/2 * gain * d)
        Δn = Δn_sp + Δn_st
        nu -= Δn
        nl += Δn
        n_upper[i, j] = nu
        n_lower[i, j] = nl
        u[i, j] = uij
    end
    nothing
end

function kernel_fc(u, n_upper, n_lower, s)
    idi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y)

    for i in idi:stride[1]:s[1], j in idj:stride[2]:s[2]
       if j > s[2] ÷ 2
        jj = s[2] - j + 1
        u[i, j] = u[i, jj]
        n_upper[i, j] = n_upper[i, jj]
        n_lower[i, j] = n_lower[i, jj]
       end
    end
    nothing
end
