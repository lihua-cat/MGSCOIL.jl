##
u34 = zeros(ComplexF64, Nx, Ny)
u22 = zeros(ComplexF64, Nx, Ny)
plan, iplan = plan_as(u34)
(; d, r_oc, r_hr) = cavity
(; react) = flow
(; L, ap, radius, r_oc) = cavity
d_list = cavity.d
(; line34, line22) = line
(; Nx, Ny, Nz, X, Y, ratio) = grid
ap_mask, δps34, δps22, trans34, trans22 = angular_spectrum_paras(ap, radius, d_list, Nx, Ny, Nz, X, Y, line34.λ, line22.λ)

d_trip = (d..., reverse(d)...)
n_trip = 2Nz + 2
d_gs = Z / Nz
dt_gs = d_gs / 𝑐
fs = uustrip.(fs)

for j = 1:2Nz+2
    println("$j")
    dt = d_trip[j] / 𝑐
    #   reflector
    if j == Nz + 2
        u34 .*= δps34
        u22 .*= δps22
    end
    #   re-pump and hyperfine relaxation 
    flow_refresh_fast!(fs, react, flow.density, dt)
    #   free propagate
    tn = j in (1, n_trip) ? 1 :
            j in (Nz + 1, Nz + 2) ? 3 :
            2
    trans34v = view(trans34, :, :, tn)
    trans22v = view(trans22, :, :, tn)
    MGSCOIL.issymmetry(u34)
    MGSCOIL.issymmetry(u22)
    free_propagate!(u34, trans34v, plan, iplan)
    free_propagate!(u22, trans22v, plan, iplan)
    MGSCOIL.issymmetry(u34)
    MGSCOIL.issymmetry(u22)
    if j in (1:Nz..., Nz+2:2Nz+1...)
        k = j <= Nz ? j : 2Nz + 2 - j
        Ie3 = view(fs.Ie3, :, :, k)
        Ie2 = view(fs.Ie2, :, :, k)
        Ig4 = view(fs.Ig4, :, :, k)
        Ig2 = view(fs.Ig2, :, :, k)
        MGSCOIL.issymmetry(u34)
        MGSCOIL.issymmetry(u22)
        MGSCOIL.issymmetry(Ie3)
        MGSCOIL.issymmetry(Ie2)
        MGSCOIL.issymmetry(Ig4)
        MGSCOIL.issymmetry(Ig2)
        MGSCOIL.optical_extraction!(u34, Ie3, Ig4, d_gs, dt_gs, line34, false, 1)
        MGSCOIL.optical_extraction!(u22, Ie2, Ig2, d_gs, dt_gs, line22, false, 1)
        MGSCOIL.issymmetry(u34)
        MGSCOIL.issymmetry(u22)
        MGSCOIL.issymmetry(Ie3)
        MGSCOIL.issymmetry(Ie2)
        MGSCOIL.issymmetry(Ig4)
        MGSCOIL.issymmetry(Ig2)
    end
    #   aperture
    if j in (1, Nz, Nz + 2, 2Nz + 1)
        u34 .*= ap_mask
        u22 .*= ap_mask
    end
    #   reflectivity
    if j == Nz + 1
        u34 .*= sqrt(r_hr)
        u22 .*= sqrt(r_hr)
    elseif j == n_trip
        u34 .*= sqrt(r_oc)
        u22 .*= sqrt(r_oc)
    end
end

## 
using CUDA

function rand1!(a)
    @cuda threads = length(a) kernel(a)
    synchronize()
end

function kernel(a)
    i = threadIdx().x
    a[i] = (rand() - 0.5) * 2im * π |> exp
    nothing
end

a = CUDA.ones(ComplexF32, 10)
rand1!(a)
a