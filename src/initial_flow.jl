"reaction rate list, some depends on temperature."
function react_rate(T::Unitful.AbsoluteScaleTemperature)
    T = ustrip(u"K", T)
    kf = (2.33e-8/T)
    K = 0.74exp(401.4/T)
    kr = kf/K
    k1 = 2.0e-12       #   quench by water vapor
    k2 = 3.8e-11       #   quench by residual I₂
    k3 = 4.0e-10       #   ground state HFR
    k4 = 1.0e-10       #   excited state HFR
    kf, kr, k1, k2, k3, k4 = (kf, kr, k1, k2, k3, k4) .* u"cm^3/s"
    react = (;kf, kr, k1, k2, k3, k4)
    return react
end

function flowrate_0(fr_inlet; diss_I₂, utl_Cl₂)
    @unpack Cl₂, I₂, He_P, He_S, H₂O = fr_inlet
    He = He_P + He_S
    O₂ = Cl₂ * utl_Cl₂
    I = 2 * I₂ * diss_I₂
    Cl₂ = Cl₂ * (1 - utl_Cl₂)
    I₂ = I₂ * (1 - diss_I₂)
    flowrate0 = (;O₂, I, I₂, H₂O, Cl₂, He)
    return flowrate0
end

function gas_mix(flowrate)
    fr_tot = sum(flowrate)
    gas_df = DataFrame(
        Name = ["O₂", "I", "I₂", "H₂O", "Cl₂", "He"],
        γ = [7//5, 5//3, 7//5, 1.33, 7//5, 1.66],
        M = [31.9988, 126.90447, 253.80894, 18.0153, 70.906, 4.002602] * u"g/mol",
        γₚ = [5.7, 0, 0, 0, 0, 4.1] * u"MHz/Torr",
        flowrate = [values(flowrate)...],
        fraction = ustrip.(NoUnits, [values(flowrate)...] ./ fr_tot)
        )
    return gas_df
end

function coe_mix(gas_df::DataFrame)
    fraction = gas_df.fraction
    γ_list = gas_df.γ
    molar_mass = gas_df.M
    γ_mix = sum(@. fraction * γ_list / (γ_list - 1)) / sum(@. fraction / (γ_list - 1))
    M_mix = sum(fraction .* molar_mass)
    γₚ_mix = sum(fraction .* gas_df.γₚ)
    return γ_mix, M_mix, γₚ_mix
end

function flowrate2density(flowrate, V, A)
    k = keys(flowrate)
    v = [uconvert(u"cm^-3", fr / V / A * 𝑁) for fr in flowrate]
    density = (; zip(k, v)...)
    return density
end

function gain2yield(g34, density, σ34g, react)
    @unpack O₂, I, I₂, H₂O = density
    @unpack kf, kr, k1, k2 = react
    yield0 = (2g34 + I * σ34g) * (H₂O * k1 + I₂ * k2 + O₂ * kr) / O₂ / 
         (I * σ34g * (2kf + kr) - 2g34 * (kf - kr))
    return uconvert(NoUnits, yield0)
end

function initial_flow(X, Nx::Int, V, yield_0, g34_0, σ34g, density, react)
    @unpack O₂, I, I₂, H₂O = density
    @unpack kf, kr, k1, k2 = react
    
    dt = X / Nx / V

    O2g_0 = O₂ * (1 - yield_0)
    O2e_0 = O₂ * yield_0
    Ig_0 = 2//3 * (I - g34_0 / σ34g)
    Ie_0 = I - Ig_0

    O2g = Vector{typeof(O2g_0)}(undef, Nx)
    O2e = similar(O2g)
    Ig_t = similar(O2g)
    Ie_t = similar(O2g)

    O2g[1] = O2g_0
    O2e[1] = O2e_0
    Ig_t[1] = Ig_0
    Ie_t[1] = Ie_0

    for i in 1:Nx-1
        transfer = (kf * Ig_t[i] * O2e[i] - kr * Ie_t[i] * O2g[i]) * dt
        quench = (k1 * Ie_t[i] * H₂O + k2 * Ie_t[i] * I₂) * dt
        O2g[i+1] = O2g[i] + transfer
        O2e[i+1] = O₂ - O2g[i+1]
        Ig_t[i+1] = Ig_t[i] + (-transfer + quench)
        Ie_t[i+1] = I - Ig_t[i+1]
    end

    Fu = 3:-1:2
    Fl = 4:-1:1
    Ie = Matrix{typeof(O2g_0)}(undef, Nx, length(Fu))
    Ig = Matrix{typeof(O2g_0)}(undef, Nx, length(Fl))
    for i in 1:length(Fu)
        Ie[:, i] .= Ie_t * (Fu[i] * 2 + 1) / 12
    end
    for i in 1:length(Fl)
        Ig[:, i] .= Ig_t * (Fl[i] * 2 + 1) / 24
    end
    return O2g, O2e, Ig, Ie
end

function initial_fs(Ny::Int, Nz::Int, O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31)
    O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31 = 
    map((O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31)) do d
        d = repeat(d, 1, Ny, Nz)
    end
    (;O₂g, O₂e, Ie3, Ie2, Ig4, Ig2, Ig31)
end
