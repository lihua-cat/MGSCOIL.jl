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
using CUDA;
CUDA.allowscalar(false);
# using CUDA.CUFFT
using GLMakie
using ProgressLogging

import ZeemanSpectra: σr_ltp
import AngularSpectrum: plan_as, angular_spectrum_paras
using MGSCOIL
using MyShow

toc_import = time_ns()
let
    time = round((toc_import - tic) / 1e9, digits = 4)
    print("Import packages and modules: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## --------------------------------------------------------------
println("Model Setup: Runing ...")
tic_setup = time_ns()

L = 0.01u"μs" * 𝑐 / 2 |> u"m"   #   1.5u"m"
Z = 25.0u"cm"
Y = 1.9u"cm"
gs = let
    num = 5
    d = Z / num
    d1 = 30.0u"cm"
    (; num, d, d1)
end
ap = (x = 6.0u"cm", y = 1.5u"cm")
cavity = let
    d = (gs.d1, ones(gs.num - 1) * gs.d..., L - gs.d1 - (gs.num - 1) * gs.d) .|> u"cm"
    radius = Inf * u"m"
    r_hr = 0.9999
    r_oc = 0.9
    gth = -log(r_oc * r_hr) / 2Z |> u"cm^-1"
    Fresnel = Tuple([(a / 2)^2 / L / 1.315u"μm" |> NoUnits for a in ap])
    (; L, ap, d, radius, r_hr, r_oc, gth, Fresnel)
end

inlet = let
    Cl₂ = 0.5u"mol/s"
    I₂ = 12u"mmol/s"
    He_P = 4 * Cl₂
    He_S = 1.1u"mol/s"
    He = He_P + He_S
    H₂O = 50u"mmol/s"
    (; Cl₂, I₂, He_P, He_S, H₂O)
end

T = 200u"K"
Mach = 1.8

N = Nx, Ny, Nz = (1024, 256, gs.num)
f2c = 8192 ÷ Nx

g34_0 = 0.010u"cm^-1"

fs, flow, line, grid = model_setup(L = L, Z = Z, Y = Y,
    cavity = cavity,
    inlet = inlet,
    T = T, Mach = Mach,
    N = N, f2c = f2c,
    g34_0 = g34_0);

@myshow cavity grid flow line;

toc_setup = time_ns()
let
    time = round((toc_setup - tic_setup) / 1e9, digits = 4)
    print("Model Setup: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## ----------------------------------------------------
println("MGS interpolation: Runing ...")
tic = time_ns()

B_range = 0:10:600
interp_linear43 = σr_ltp(4, 3, B_range, "S", T = flow.T, P = flow.P, γ = flow.γₚ_mix)
interp_linear22 = σr_ltp(2, 2, B_range, "S", T = flow.T, P = flow.P, γ = flow.γₚ_mix)

toc = time_ns()
let
    time = round((toc - tic) / 1e9, digits = 4)
    print("MGS interpolation: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## ----------------------------------------------------
println("Simulation: Runing ...")
tic_running = time_ns()

waveform = (; period = 10000, cycle = 4000 / 10000, rising = 200, falling = 110,
        offset = 100, high = 360, low = 0, dc = 400)

begin
    u34 = zeros(ComplexF64, Nx, Ny)
    u22 = zeros(ComplexF64, Nx, Ny)
    power34, power22, u34n, u22n, fsn, time_output =
        propagate(u34, u22, fs, 5000;
            cavity = cavity,
            flow = flow,
            lines = line,
            grid = grid,
            waveform = waveform,
            interp = B -> (interp_linear43(B), interp_linear22(B)),
            random = false,
            PRECISION = Float32)
end
toc_running = time_ns()
let
    time = round((toc_running - tic_running) / 1e9, digits = 4)
    print("Model Setup: Done ")
    printstyled("$(now()) [$time s] \n", color = :yellow)
end

## -----------------------------------------------------
begin
    PRECISION = Float32
    u34_d = CUDA.zeros(Complex{PRECISION}, Nx, Ny)
    u22_d = CUDA.zeros(Complex{PRECISION}, Nx, Ny)
    fs_d = replace_storage(CuArray, uustrip.(PRECISION, fs))
    ap_d, δps34_d, δps22_d, trans34_d, trans22_d = angular_spectrum_paras(cavity, grid, line; gpu = true, PRECISION = PRECISION)
    plan_d, iplan_d = plan_as(u22_d)
    AS_d = (ap = ap_d, δps34 = δps34_d, δps22 = δps22_d, trans34 = trans34_d, trans22 = trans22_d, p = plan_d, ip = iplan_d)

    t_trip = 2L / 𝑐
    dt_flush = grid.ratio * t_trip
    ds = grid.X * grid.Y / (grid.Nx * grid.Ny)
    out2inner = (1 - cavity.r_oc) / cavity.r_oc

    waveform = (; period = 10000, cycle = 4000 / 10000, rising = 400, falling = 110,
        offset = 0, high = 360, low = 0, dc = 400)

    N = 500
    time_output = collect(0:N-1) * t_trip .|> u"μs"
    power34 = zeros(PRECISION, N) * u"W"
    power22 = zeros(PRECISION, N) * u"W"
    u34n = zeros(Complex{PRECISION}, Nx, Ny)
    u22n = zeros(Complex{PRECISION}, Nx, Ny)

    tic_main = time_ns()

    @progress for n in 2:N

        # sw = n % 500 < 150 ? 1 : 0
        # sw = 1
        field = MGSCOIL.pulse(n, waveform = waveform)
        sw34 = interp_linear43(field)
        sw22 = interp_linear22(field)

        bounce!(u34_d, u22_d, fs_d, true, sw34, sw22; AS_d = AS_d, grid = grid, cavity = cavity, flow = flow, lines = line)
        if n % grid.ratio == 0
            flow_refresh!(fs_d, flow, dt_flush)
        end
        power34[n] = outpower(u34_d, ds, line.line34.e, t_trip) * out2inner
        power22[n] = outpower(u22_d, ds, line.line22.e, t_trip) * out2inner

        Nv = 100
        if N > 1000 && n > N - Nv
            u34n .+= Array(abs2.(u34_d)) / Nv
            u22n .+= Array(abs2.(u22_d)) / Nv
        end
    end

    toc_main = time_ns()
    let
        time = round((toc_main - tic_main) / 1e9, digits = 4)
        print("Model Computation: Done ")
        printstyled("$(now()) [$time s] \n", color = :yellow)
    end
end

maximum(power34) / power34[end]
# plot1(u34n, grid, ap)
# plot2(power34, time_output)
# plot3(u34n, grid)