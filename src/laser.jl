dropmean(A; dims = :) = dropdims(mean(A; dims = dims); dims = dims)
mean_d(a) = sum(a) / length(a)

# function issymmetry(a)
#     len = size(a, 2)
#     # iseven(len) || error("not even")
#     au = selectdim(a, 2, 1:len÷2)
#     al = selectdim(a, 2, len:-1:(len-len÷2+1))
#     if eltype(a) <: Bool
#         symm = au == al
#     else
#         symm = all(au .≈ al)
#     end
#     if !symm
#         display(a)
#         error("not symmetry")
#     end
#     nothing
# end

"output power = total energy at oc plane / one round trip time"
function outpower(u, ds, e, t_trip)
    if eltype(u) <: Unitful.Quantity
        p = sum(abs2.(u)) * e * ds / t_trip
    else
        p = sum(abs2.(u)) * u"cm^-2" * e * ds / t_trip
    end
    return p |> u"W"
end

function angular_spectrum_paras(ap, radius, d_list, Nx, Ny, Nz, X, Y, λ34, λ22)
    nx = collect(0:Nx-1)
    ny = collect(0:Ny-1)
    x = nx * X / Nx
    y = ny * Y / Ny
    xs = x .- X / 2
    ys = y .- Y / 2
    # yts = transpose(ys)
    apx1 = findfirst(abs.(xs) .< ap.x / 2)
    apy1 = findfirst(abs.(ys) .< ap.y / 2)
    apx2 = Nx - apx1 + 1
    apy2 = Ny - apy1 + 1
    ap_mask = (apx1 .< nx .+ 1 .< apx2) .* transpose(apy1 .< ny .+ 1 .< apy2)
    # δ = isinf(radius) ? ones(Nx, Ny) .* zero(radius) : @. radius - √(radius^2 - xs^2 - yts^2)
    # δps34 = @. cispi(ustrip(NoUnits, -2 / λ34 * 2 * δ))
    # δps22 = @. cispi(ustrip(NoUnits, -2 / λ22 * 2 * δ))
    δps34 = phase_shift(xs, ys, radius, λ34)
    δps22 = phase_shift(xs, ys, radius, λ22)
    dz = d_list[[1, 2, Nz + 1]]
    # νx, νy = xy2νxy(X, Y, Nx, Ny)
    νx, νy = spatial_frequency(X, Y, Nx, Ny)
    trans34 = Array{ComplexF64}(undef, Nx, Ny, length(dz))
    trans22 = Array{ComplexF64}(undef, Nx, Ny, length(dz))
    for i = 1:length(dz)
        # trans34[:, :, i] .= mat_as(νx, νy, λ34, dz[i], Nx, Ny)
        # trans22[:, :, i] .= mat_as(νx, νy, λ22, dz[i], Nx, Ny)
        trans34[:, :, i] .= propagation_func(νx, νy, λ34, dz[i])
        trans22[:, :, i] .= propagation_func(νx, νy, λ22, dz[i])
    end
    return ap_mask, δps34, δps22, trans34, trans22
end

function angular_spectrum_paras(cavity, grid, lines; gpu = true, PRECISION = Float32)
    (; ap, radius, d) = cavity
    (; Nx, Ny, Nz, X, Y) = grid
    λ34, λ22 = lines.line34.λ, lines.line22.λ
    ap_mask, δps34, δps22, trans34, trans22 = angular_spectrum_paras(ap, radius, d, Nx, Ny, Nz, X, Y, λ34, λ22)
    if gpu
        ap_d = CuArray{Bool}(ap_mask)
        δps34_d = CuArray{Complex{PRECISION}}(δps34)
        δps22_d = CuArray{Complex{PRECISION}}(δps22)
        trans34_d = CuArray{Complex{PRECISION}}(trans34)
        trans22_d = CuArray{Complex{PRECISION}}(trans22)
        return ap_d, δps34_d, δps22_d, trans34_d, trans22_d
    else
        return ap_mask, δps34, δps22, trans34, trans22
    end
end

function pulse(i; waveform)
    (; period, cycle, rising, falling, offset, amplitude, residual) = waveform
    on = period * cycle
    i = i - offset >= 0 ? i - offset : period + (i - offset)
    i = i % period
    out = 0
    if i < rising
        out = (i / rising) * (amplitude - residual) + residual
    elseif i < rising + on
        out = amplitude
    elseif i < rising + on + falling
        out = amplitude - (i - rising - on) / falling * (amplitude - residual)
    else
        out = residual
    end
    return out
end

function bounce!(u34_d, u22_d, fs_d, random = false, sw34 = 1, sw22 = 1; AS_d, grid, cavity, flow, lines)
    (; d, r_oc, r_hr) = cavity
    (; react, density) = flow
    (; line34, line22) = lines
    (; Z, Nz) = grid
    ap_d, δps34_d, δps22_d, trans34_d, trans22_d, plan_d, iplan_d = AS_d

    d_trip = (d..., reverse(d)...)
    n_trip = 2Nz + 2
    d_gs = Z / Nz
    dt_gs = d_gs / 𝑐

    for j = 1:n_trip
        dt = d_trip[j] / 𝑐
        #   reflector
        if j == Nz + 2
            u34_d .*= δps34_d
            u22_d .*= δps22_d
        end
        #   re-pump and hyperfine relaxation 
        flow_refresh_fast!(fs_d, react, density, dt)
        #   free propagate
        tn = j in (1, n_trip) ? 1 :
             j in (Nz + 1, Nz + 2) ? 3 :
             2
        trans34 = view(trans34_d, :, :, tn)
        trans22 = view(trans22_d, :, :, tn)
        free_propagate!(u34_d, trans34, plan_d, iplan_d)
        free_propagate!(u22_d, trans22, plan_d, iplan_d)
        #   optical extraction
        if j in (1:Nz..., Nz+2:2Nz+1...)
            k = j <= Nz ? j : 2Nz + 2 - j
            Ie3 = view(fs_d.Ie3, :, :, k)
            Ie2 = view(fs_d.Ie2, :, :, k)
            Ig4 = view(fs_d.Ig4, :, :, k)
            Ig2 = view(fs_d.Ig2, :, :, k)
            optical_extraction!(u34_d, Ie3, Ig4, d_gs, dt_gs, line34, random, sw34)
            optical_extraction!(u22_d, Ie2, Ig2, d_gs, dt_gs, line22, random, sw22)
        end
        #   aperture
        if j in (1, Nz, Nz + 2, 2Nz + 1)
            u34_d .*= ap_d
            u22_d .*= ap_d
        end
        #   reflectivity
        if j == Nz + 1
            u34_d .*= sqrt(r_hr)
            u22_d .*= sqrt(r_hr)
        elseif j == n_trip
            u34_d .*= sqrt(r_oc)
            u22_d .*= sqrt(r_oc)
        end
    end
end

function propagate(u34, u22, fs, n; cavity, flow, lines, grid, waveform, random = false, PRECISION = Float32)
    #   unpack input namedtuples
    (; L, ap, radius, r_oc) = cavity
    d_list = cavity.d
    (; T, P, γₚ_mix, react, density) = flow
    (; line34, line22) = lines
    (; Nx, Ny, Nz, X, Y, ratio) = grid
    #   preparation 
    d_trip = (d_list..., reverse(d_list)...)
    n_trip = 2Nz + 2
    length(d_trip) == n_trip || error("gain sheet number not match")
    #   angular spectrum
    ap_mask, δps34, δps22, trans34, trans22 = angular_spectrum_paras(ap, radius, d_list, Nx, Ny, Nz, X, Y, line34.λ, line22.λ)
    #   submit to gpu device
    u34_d = CuArray{Complex{PRECISION}}(u34)
    u22_d = CuArray{Complex{PRECISION}}(u22)
    fs_d = replace_storage(CuArray{PRECISION}, uustrip.(fs))
    ap_d = CuArray{Bool}(ap_mask)
    δps34_d = CuArray{Complex{PRECISION}}(δps34)
    δps22_d = CuArray{Complex{PRECISION}}(δps22)
    trans34_d = CuArray{Complex{PRECISION}}(trans34)
    trans22_d = CuArray{Complex{PRECISION}}(trans22)
    plan_d, iplan_d = plan_as(u22_d)
    AS_d = (ap = ap_d, δps34 = δps34_d, δps22 = δps22_d, trans34 = trans34_d,
        trans22 = trans22_d, p = plan_d, ip = iplan_d)

    B_range = 0:1:600
    interp_linear43 = σr_ltp(4, 3, B_range, "S", T = T, P = P, γ = γₚ_mix)
    interp_linear22 = σr_ltp(2, 2, B_range, "S", T = T, P = P, γ = γₚ_mix)

    #   main loop
    t_trip = 2L / 𝑐
    dt_flush = ratio * t_trip
    timeseries = collect(1:n) * t_trip
    ds = X * Y / (Nx * Ny)
    power34 = zeros(typeof(outpower(u34, ds, line34.e, t_trip)), n)
    power22 = copy(power34)
    power34[1] = outpower(u34, ds, line34.e, t_trip)
    power22[1] = outpower(u22, ds, line22.e, t_trip)

    #   visualization
    ap_m = ap_makie(uustrip(ap), uustrip(X), uustrip(Y))
    x = collect(0:Nx-1) * uustrip(X) / Nx
    y = collect(0:Ny-1) * uustrip(Y) / Ny
    i_node = Node(1)
    u34_node = Node(abs2.(u34))
    u22_node = Node(abs2.(u22))
    y_node = Node(Array(dropmean(fs_d.O₂e, dims = 3) / uustrip(density.O₂)))
    g34_node = Node(Array(uustrip(line34.σ) * (dropmean(fs_d.Ie3 - line34.gg * fs_d.Ig4, dims = 3))))
    g22_node = Node(Array(uustrip(line22.σ) * (dropmean(fs_d.Ie2 - line22.gg * fs_d.Ig2, dims = 3))))
    title = ("3 -> 4", "2 -> 2", "yield", "g34", "g22")
    fig = Figure(resolution = (1200, 2100), fontsize = 24)
    ax = [Axis(fig[i, 1], aspect = AxisAspect(X / Y), title = title[i]) for i = 1:length(title)]
    heatmap!(ax[1], x, y, u34_node, colormap = :plasma)
    heatmap!(ax[2], x, y, u22_node, colormap = :plasma)
    heatmap!(ax[3], x, y, y_node, colormap = :plasma)
    heatmap!(ax[4], x, y, g34_node, colormap = :plasma)
    heatmap!(ax[5], x, y, g22_node, colormap = :plasma)
    for axis in ax
        poly!(axis, ap_m, color = :transparent, strokecolor = :cyan, strokewidth = 1)
        limits!(axis, 0, uustrip(X), 0, uustrip(Y))
    end
    Label(fig[0, :], text = @lift("trip = $($i_node)"), tellwidth = false)
    display(fig)

    @progress for i = 2:n
        field = pulse(i, waveform = waveform)
        sw34 = interp_linear43(field)
        sw22 = interp_linear22(field)
        # sw = i % 500 < 150 ? 1 : 0
        bounce!(u34_d, u22_d, fs_d, random, sw34, sw22; AS_d = AS_d, grid = grid,
            cavity = cavity, flow = flow, lines = lines)
        if i % ratio == 0
            flow_refresh!(fs_d, react, density, dt_flush)
        end
        power34[i] = outpower(u34_d, ds, line34.e, t_trip) * ((1 - r_oc) / r_oc)
        power22[i] = outpower(u22_d, ds, line22.e, t_trip) * ((1 - r_oc) / r_oc)

        Ie3 = view(fs_d.Ie3, ap_d, :)
        Ie2 = view(fs_d.Ie2, ap_d, :)
        Ig4 = view(fs_d.Ig4, ap_d, :)
        Ig2 = view(fs_d.Ig2, ap_d, :)
        g34 = mean_d(uustrip(line34.σ) * (Ie3 - line34.gg * Ig4)) * 100
        g22 = mean_d(uustrip(line22.σ) * (Ie2 - line22.gg * Ig2)) * 100
        yield = mean_d(fs_d.O₂e[ap_d, :]) / uustrip(density.O₂)
        ye3 = mean_d(Ie3)
        ye2 = mean_d(Ie2)
        p34 = uustrip(power34[i])
        p22 = uustrip(power22[i])


        i == 2 && @printf "%5s | %8s | %8s | %7s | %10s | %10s | %10s | %10s | %7s \n" "i" "g34" "g22" "yield" "Ie3" "Ie2" "power34" "power22" "t(μs)"
        if (i >= 50 && i % 5 == 0) || i < 50
            # if i % 50 == 0
            @printf "%5d | %8.6f | %8.6f | %7.5f | %10.4e | %10.4e | %10.4e | %10.4e | %7.3f \n" i g34 g22 yield ye3 ye2 p34 p22 ustrip(u"μs", i * t_trip)
        end

        if i < 100 || i % 10 == 0
            i_node[] = i
            u34_node[] = Array(abs2.(u34_d))
            u22_node[] = Array(abs2.(u22_d))
            y_node[] = Array(dropmean(fs_d.O₂e, dims = 3) / uustrip(density.O₂))
            g34_node[] = Array(uustrip(line34.σ) * (dropmean(fs_d.Ie3 - line34.gg * fs_d.Ig4, dims = 3)))
            g22_node[] = Array(uustrip(line22.σ) * (dropmean(fs_d.Ie2 - line22.gg * fs_d.Ig2, dims = 3)))
        end
    end
    return power34, power22, Array(u34_d), Array(u22_d), replace_storage(Array, fs_d), timeseries
end