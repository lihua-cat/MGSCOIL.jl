flow_refresh!(fs, flow, dt) = flow_refresh!(fs, flow.react, flow.density, dt)

function flow_refresh!(fs::StructArray, react, density, dt)
    @uustrip react density dt
    @unpack O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31 = fs
    @unpack O₂, I, I₂, H₂O = density
    @unpack kf, kr, k1, k2 = react

    s = size(fs.Ie3)
    for i in s[1]:-1:2
        threads = (32, 5)
        blocks = cld.(s[2:3], threads)
        @cuda threads = threads blocks = blocks kernel_refresh!(s[2:3], i, fs, I₂, H₂O, k1, k2, dt)
    end
    synchronize()
end

function kernel_refresh!(s, i, fs, I₂, H₂O, k1, k2, dt)
    idj = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idk = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y)
    for j in idj:stride[1]:s[1], k in idk:stride[2]:s[2]
        quench3 = (k1 * fs.Ie3[i-1, j, k] * H₂O + k2 * fs.Ie3[i-1, j, k] * I₂) * dt
        quench2 = (k1 * fs.Ie2[i-1, j, k] * H₂O + k2 * fs.Ie2[i-1, j, k] * I₂) * dt
        fs.O₂g[i, j, k] = fs.O₂g[i-1, j, k]
        fs.O₂e[i, j, k] = fs.O₂e[i-1, j, k]
        fs.Ie3[i, j, k] = fs.Ie3[i-1, j, k] - quench3
        fs.Ie2[i, j, k] = fs.Ie2[i-1, j, k] - quench2
        fs.Ig4[i, j, k] = fs.Ig4[i-1, j, k] + 9/24 * (quench3 + quench2)
        fs.Ig2[i, j, k] = fs.Ig2[i-1, j, k] + 5/24 * (quench3 + quench2)
        fs.Ig31[i, j, k] = fs.Ig31[i-1, j, k] + 10/24 * (quench3 + quench2)
    end
    nothing
end

function flow_refresh_fast!(fs::StructArray, react, density, dt)
    @uustrip react density dt
    @unpack O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31 = fs
    @unpack O₂ = density
    @unpack kf, kr, k3, k4 = react
    ntot = sum(density)
    hfr_e = k4 * O₂
    hfr_g = k3 * ntot

    s = size(fs.Ie3)
    # kernel = @cuda launch=false kernel_refresh_fast(s, fs, hfr_e, hfr_g, kf, kr, dt)
    # config = launch_configuration(kernel.fun)
    # @show config.threads config.blocks
    threads = (64, 4, 1)
    blocks = cld.(s, threads)
    @cuda threads = threads blocks = blocks kernel_refresh_fast(s, fs, hfr_e, hfr_g, kf, kr, dt)
    synchronize()
end

function kernel_refresh_fast(s, fs, hfr_e, hfr_g, kf, kr, dt)
    idi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    idk = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y, blockDim().z * gridDim().z)
    for i in idi:stride[1]:s[1], j in idj:stride[2]:s[2], k in idk:stride[3]:s[3]
        if i > 1
            O₂g = fs.O₂g[i, j, k]
            O₂e = fs.O₂e[i, j, k]
            Ie3 = fs.Ie3[i, j, k]
            Ie2 = fs.Ie2[i, j, k]
            Ig4 = fs.Ig4[i, j, k]
            Ig2 = fs.Ig2[i, j, k]
            Ig31 = fs.Ig31[i, j, k]

            Ie = Ie3 + Ie2
            Ig = Ig4 + Ig2 + Ig31
            transfer_f = kf * O₂e * Ig * dt
            transfer_b = kr * O₂g * Ie * dt

            fs.O₂g[i, j, k] += transfer_f - transfer_b
            fs.O₂e[i, j, k] += transfer_b - transfer_f
            fs.Ie3[i, j, k] += 7/12 * transfer_f - kr * O₂g * Ie3 * dt - hfr_e * (Ie3 - 7/12 * Ie) * dt
            fs.Ie2[i, j, k] += 5/12 * transfer_f - kr * O₂g * Ie2 * dt - hfr_e * (Ie2 - 5/12 * Ie) * dt
            fs.Ig4[i, j, k] += 9/24 * transfer_b - kf * O₂e * Ig4 * dt - hfr_g * (Ig4 - 9/24 * Ig) * dt
            fs.Ig2[i, j, k] += 5/24 * transfer_b - kf * O₂e * Ig2  * dt - hfr_g * (Ig2 - 5/24 * Ig) * dt
            fs.Ig31[i, j, k] += 10/24 * transfer_b - kf * O₂e * Ig31 * dt - hfr_g * (Ig31 - 10/24 * Ig) * dt
        end
    end
    nothing
end