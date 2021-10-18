function model_setup(;L = L, Z = Z, Y = Y, 
                      cavity = cavity, 
                      inlet = inlet, 
                      T = T, Mach = Mach, 
                      N = N, f2c = f2c, 
                      g34_0 = g34_0)

    Nx, Ny, Nz = N
    flowrate0 = flowrate_0(inlet, diss_I₂ = 0.95, utl_Cl₂ = 0.95)
    #   gas mixture thermal properties
    gas_df = gas_mix(flowrate0)
    γ_mix, M_mix, γₚ_mix = coe_mix(gas_df)
    soundspeed = uconvert(u"cm/s", √(γ_mix * 𝑅 / M_mix * T))
    V = uconvert(u"cm/s", Mach * soundspeed)
    A = Z * Y     # flow cross-section
    P = uconvert.(u"Torr", sum(flowrate0) / (V * A) * 𝑅 * T)
    #   convert flow rate(mol/s) to partical density(/cm^3)
    density = flowrate2density(flowrate0, V, A)
    react = react_rate(T)
    flow = (;T, V, A, P, Mach, density, react)
    #   grid
    dt_trip = uconvert(u"μs", 2L / 𝑐)
    dt_flush = dt_trip * f2c
    t_flow = Nx * dt_flush
    X = uconvert(u"cm", V * t_flow)
    grid = (;Nx, Ny, Nz, X, Y, Z, ratio = f2c)
    #   transition lines
    line34, line22 = let
        l34 = line_I127(3, 4)
        l22 = line_I127(2, 2)
        A34 = l34.A
        A33 = A_I127(3, 3)
        A32 = A_I127(3, 2)
        A23 = A_I127(2, 3)
        A22 = l22.A
        A21 = A_I127(2, 1)
        τ3 = 1 / (A34 + A33 + A32)
        τ2 = 1 / (A23 + A22 + A21)
        σ34 = σ0_I127(3, 4, T, P, γₚ_mix)[2]
        σ22 = σ0_I127(2, 2, T, P, γₚ_mix)[2]
        merge(l34, (;σ = σ34, τ = τ3, gg = 7//9)), merge(l22, (;σ = σ22, τ = τ2, gg = 1))
    end
    lines = (;line34, line22)

    #   line broadenings
    νd = fwhm_doppler(line34.ν, ATOM_DATA[ATOM_DATA.Name .== "I127", :M][], T)
    νp = 2 * P * γₚ_mix
    linewidth = (;νd, νp)

    #   set intial SSG then compute yield
    σ34g_0 = line34.σ * 7 // 12
    yield_0 = gain2yield(g34_0, density, σ34g_0, react)

    #   initial flow field
    O2g_x, O2e_x, Ig_x, Ie_x = initial_flow(X, Nx, V, yield_0, g34_0, σ34g_0, density, react)
    flowsheet = initial_fs(Ny, Nz, O2g_x, O2e_x, Ie_x[:, 1], Ie_x[:, 2], Ig_x[:, 1], Ig_x[:, 3], Ig_x[:, 2]+Ig_x[:, 4])
    fs = StructArray(flowsheet)

    return fs, flow, lines, grid
end
