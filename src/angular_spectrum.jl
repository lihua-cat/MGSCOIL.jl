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

function free_propagate!(u, trans, plan, iplan)
    plan * u
    u .*= trans
    iplan * u
    u .= (reverse(u, dims = 2) .+ u) ./ 2
    nothing
end