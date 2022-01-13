function ap_makie(ap, X, Y)
    @uustrip X Y ap
    apx1, apy1 = @. ((X, Y) - (ap.x, ap.y)) ./ 2
    apx2, apy2 = @. (X, Y) .- (apx1, apy1)
    ap_makie = Point2f0[(apx1, apy1), (apx1, apy2), (apx2, apy2), (apx2, apy1)]
end

function plot1(u, grid, ap)
    @uustrip grid ap
    (; X, Y, Nx, Ny) = grid
    apx1, apy1 = @. ((X, Y) - (ap.x, ap.y)) ./ 2
    apx2, apy2 = @. (X, Y) .- (apx1, apy1)
    ap_poly = Point2f0[(apx1, apy1), (apx1, apy2), (apx2, apy2), (apx2, apy1)]
    x = collect(0:Nx-1) * X / Nx
    y = collect(0:Ny-1) * Y / Ny
    p = u
    # val, pos = findmax(p)

    fig = Figure(resolution = (1450, 600))
    colsize!(fig.layout, 1, Relative(3 / 4))
    rowsize!(fig.layout, 1, Relative(1 / 2))
    ax1 = Axis(fig[1, 1], aspect = AxisAspect(X / Y), title = "xy plane")
    ax2 = Axis(fig[2, 1], title = "along x", ylabel="Intensity (W/m²)")
    ax3 = Axis(fig[1, 2], title = "along y")
    h = heatmap!(ax1, x, y, p, colormap = :coolwarm, tellheight = true, interpolate = true)
    poly!(ax1, ap_poly, color = :transparent, strokecolor = :cyan, strokewidth = 1)
    # lines!(ax2, x, p[:, pos[2]], linewdith = 2)
    lines!(ax2, x, dropmean(p, dims = 2), linewdith = 2)
    # lines!(ax3, p[pos[1], :], y, linewdith = 2)
    lines!(ax3, dropmean(p, dims = 1), y, linewdith = 2)
    vlines!(ax2, [apx1, apx2], color = (:red, 0.4))
    hlines!(ax3, [apy1, apy2], color = (:red, 0.4))
    Colorbar(fig[2, 2], h, flipaxis = true, tellwidth = false)

    limits!(ax1, 0, X, 0, Y)
    xlims!(ax2, 0, X)
    ylims!(ax3, 0, Y)
    ylims!(ax2, 0, nothing)
    xlims!(ax3, 0, nothing)

    fig
end

function plot2(power, ts)
    @uustrip power ts

    fig = Figure(title = "output power")
    ax1 = Axis(fig[1, 1], xlabel = "time(μs)", ylabel = "power(kW)")
    [lines!(ax1, ts * 1e6, p * 1e-3) for p in power]
    fig
end

function plot3(u, grid)
    @uustrip grid
    (; X, Y, Nx, Ny) = grid
    x = collect(0:Nx-1) * X / Nx
    y = collect(0:Ny-1) * Y / Ny
    p = abs2.(u)

    fig = Figure()
    ax = fig[1, 1] = Axis3(fig, aspect = (1, Y / X, 0.5))
    cmap = :plasma
    zmin, zmax = minimum(p), maximum(p)
    surface!(ax, x, y, p, colormap = cmap, colorrange = (zmin, zmax))
    # wireframe!(ax, x, y, p)
    fig
end