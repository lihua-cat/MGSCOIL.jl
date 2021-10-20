function xy2νxy(X, Y, Nx, Ny)
    νx = 1 / X * (collect(0:Nx-1) .- Nx ÷ 2)
    νy = 1 / Y * (collect(0:Ny-1) .- Ny ÷ 2)
    return νx, νy
end

function mat_as(νx, νy, λ, d, Nx, Ny)
    trans = Matrix{ComplexF64}(undef, Nx, Ny)
    νyt = transpose(νy)
    circ = @. ustrip(NoUnits, (λ * νx)^2 + (λ * νyt)^2)
    circ_mask = circ .< 1
    p = ustrip(NoUnits, d / λ)
    @. trans[circ_mask] = cispi(2 * p * √(1 - circ[circ_mask]))
    @. trans[!circ_mask] = exp(-2π * p * √(circ[!circ_mask] - 1))
    return ifftshift(trans)
end

function free_propagate!(u, trans, plan, iplan, Nx, Ny)
    plan * u
    u .*= trans
    iplan * u
    # s = (Nx, Ny)
    # threads = (64, 8)
    # blocks = cld.(s, threads)
    # @cuda threads = threads blocks = blocks kernel_flip(u, s)
    # synchronize()
end

function kernel_flip(u, s)
    idi = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    idj = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    stride = (blockDim().x * gridDim().x, blockDim().y * gridDim().y)

    for i in idi:stride[1]:s[1], j in idj:stride[2]:s[2]
        if j <= s[2]÷2
            u[i, j] = (u[i, j] + u[i, s[2]-j+1]) / 2
            u[i, s[2]-j+1] = u[i, j]
        end
    end
    nothing
end