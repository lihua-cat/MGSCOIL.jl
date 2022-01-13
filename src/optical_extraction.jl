function optical_extraction!(u, n_upper, n_lower, d::Length, dt::Time, line, random = false, sw = 1)
    @uustrip d dt line
    gain = @. line.σ * (n_upper - line.gg * n_lower) / (1 + abs2(u) / (n_upper * d)) * sw
    Δn_sp = n_upper * line.A * dt
    Δpn_sp = Δn_sp * d
    if random
        s = size(u)
        p = cispi.((rand(s...) .- 1 // 2) * 2)
        @. u += sqrt(Δpn_sp) * p
    else
        @. u += sqrt(Δpn_sp)
    end
    Δn_st = gain .* abs2.(u)
    @. u *= exp(1 / 2 * gain * d)
    n_upper .-= Δn_sp + Δn_st
    n_lower .+= Δn_sp + Δn_st
    nothing
end

function optical_extraction!(u::CuArray, n_upper::CuArray, n_lower::CuArray, d::Length, dt::Time, line, random = false, sw = 1)
    @uustrip d dt line
    s = size(u)
    threads = (64, 4)
    blocks = cld.(s, threads)
    @cuda threads = threads blocks = blocks kernel_oe(u, n_upper, n_lower, d, dt, line, s, random, sw)
end

function kernel_oe(u, n_upper, n_lower, d, dt, line, s, random, sw)
    idi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y)

    for i = idi:stride[1]:s[1], j = idj:stride[2]:s[2]
        nu = n_upper[i, j]
        nl = n_lower[i, j]
        uij = u[i, j]
        gain = line.σ * (nu - line.gg * nl) / (1 + abs2(uij) / (nu * d)) * sw
        Δn_sp = nu * line.A * dt
        Δpn_sp = Δn_sp * d
        p = random ? exp((rand() - 0.5) * 2im * π) : 1
        uij += sqrt(Δpn_sp) * p
        Δn_st = gain * abs2(uij)
        uij *= exp(1 / 2 * gain * d)
        Δn = Δn_sp + Δn_st
        nu -= Δn
        nl += Δn
        n_upper[i, j] = nu
        n_lower[i, j] = nl
        u[i, j] = uij
    end
    nothing
end
