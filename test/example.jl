##
sleep(2)
using Dates
println("")
print("=========== PROGRAM START =========== ")
printstyled("$(now()) \n", color = :yellow)


println("Import packages and modules: Runing ...")
tic = time_ns()

import PhysicalConstants.CODATA2018: c_0 as 𝑐
using Unitful, StructArrays
using CUDA; CUDA.allowscalar(false)
using CUDA.CUFFT
using GLMakie
using ProgressLogging

using MGSCOIL
using MyShow

toc_import = time_ns()
let
    time = round((toc_import - tic)/1e9, digits = 4)
    print("Import packages and modules: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## --------------------------------------------------------------
println("Model Setup: Runing ...")
tic_setup = time_ns()

L = 1.5u"m"
Z = 25.0u"cm"
Y = 1.9u"cm"
gs = let
    num = 5
    d = Z / num
    d1 = 30.0u"cm"
    (;num, d, d1)
end
ap = (x = 6.0u"cm", y = 1.5u"cm")
cavity = let
    d = (gs.d1, ones(gs.num-1)*gs.d..., L-gs.d1-(gs.num-1)*gs.d) .|> u"cm"
    radius = 10.0u"m"
    r_hr = 0.9999
    r_oc = 0.84
    gth = -log(r_oc * r_hr) / 2Z |> u"cm^-1"
    Fresnel = Tuple([(a/2)^2 / L / 1.315u"μm" |> NoUnits for a in ap])
    (;L, ap, d, radius, r_hr, r_oc, gth, Fresnel)
end

inlet = let
    Cl₂ = 0.5u"mol/s"
    I₂ = 12u"mmol/s"
    He_P = 4 * Cl₂
    He_S = 1.1u"mol/s"
    He = He_P + He_S
    H₂O = 50u"mmol/s"
    (;Cl₂, I₂, He_P, He_S, H₂O)
end

T = 200u"K"
Mach = 1.8

N = Nx, Ny, Nz = (4096, 1024, gs.num)
f2c = 8192 ÷ Nx

g34_0 = 0.010u"cm^-1"

fs, flow, lines, grid = model_setup(L = L, Z = Z, Y = Y, 
                                    cavity = cavity, 
                                    inlet = inlet, 
                                    T = T, Mach = Mach, 
                                    N = N, f2c = f2c, 
                                    g34_0 = g34_0);

@myshow cavity grid flow;

toc_setup = time_ns()
let
    time = round((toc_setup - tic_setup)/1e9, digits = 4)
    print("Model Setup: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## ----------------------------------------------------
println("Simulation: Runing ...")
tic_running = time_ns()
begin
    u34 = zeros(ComplexF64, Nx, Ny)
    u22 = zeros(ComplexF64, Nx, Ny)
    power34, power22, u34n, u22n, fsn, ts = 
        propagate(u34, u22, fs, 20000; 
                  cavity = cavity, 
                  flow = flow, 
                  lines = lines, 
                  grid = grid, 
                  random = false)
end
toc_running = time_ns()
let
    time = round((toc_running - tic_running)/1e9, digits = 4)
    print("Model Setup: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## -----------------------------------------------------
begin
    PRECISION = Float32
    u34_d = CUDA.zeros(Complex{PRECISION}, Nx, Ny)
    u22_d = CUDA.zeros(Complex{PRECISION}, Nx, Ny)
    fs_d = replace_storage(CuArray, uustrip.(PRECISION, fs))
    ap_d, δps34_d, δps22_d, trans34_d, trans22_d = angular_spectrum_paras(cavity, grid, lines)
    plan_d, iplan_d = plan_fft!(u22_d), plan_ifft!(u22_d)
    AS_d = (ap = ap_d, δps34 = δps34_d, δps22 = δps22_d, trans34 = trans34_d, trans22 = trans22_d, p = plan_d, ip = iplan_d)

    t_trip = 2L / 𝑐
    dt_flush = grid.ratio * t_trip
    ds = grid.X * grid.Y / (grid.Nx * grid.Ny)
    out2inner = (1 - cavity.r_oc) / cavity.r_oc

    N = 2
    time_output = collect(0:N-1) * t_trip .|> u"μs"
    power34 = zeros(PRECISION, N) * u"W"
    power22 = zeros(PRECISION, N) * u"W"

    tic_main = time_ns()

    @progress for n in 2:N

        # sw = n % 2000 < 600 ? 1 : 0
        sw = 1

        bounce!(u34_d, u22_d, fs_d, true, sw; AS_d = AS_d, grid = grid, cavity = cavity, flow = flow, lines = lines)
        if n % grid.ratio == 0  
            flow_refresh!(fs_d, flow, dt_flush)
        end
        power34[n] = outpower(u34_d, ds, lines.line34.e, t_trip) * out2inner
        power22[n] = outpower(u22_d, ds, lines.line22.e, t_trip) * out2inner
    end

    toc_main = time_ns()
    let
        time = round((toc_main - tic_main)/1e9, digits = 4)
        print("Model Computation: Done ")
        printstyled("$(now()) [$time s] \n", color = :yellow)
    end
end