### A Pluto.jl notebook ###
# v0.17.3

using Markdown
using InteractiveUtils

# ╔═╡ 8674c900-228a-4166-a280-ba161b3a96e9
begin
	import Pkg
	Pkg.activate("..")
	Base.Text(sprint(io->Pkg.status(io=io)))
end

# ╔═╡ f4d8a19a-ccd0-4fea-b80d-c7dc71db5bb6
push!(LOAD_PATH, "C:\\Users\\lihua\\.julia\\dev")

# ╔═╡ 28d0a2f5-e2b6-4ba8-9d9d-b95351154124
using Revise

# ╔═╡ a744b17d-d260-4f0b-92ca-ee19457722e6
using DataFrames

# ╔═╡ 7a8a8dcc-0cc4-484c-9aa4-241b99e27407
using StructArrays

# ╔═╡ cfbe01e2-a8b5-4a34-bd56-3cd9fb51bf14
using Unitful

# ╔═╡ 1912ddd3-2837-4400-8b8e-e0768d0a0cfe
using ZeemanSpectra

# ╔═╡ 59c363f1-6321-46df-a15b-487b6e1c23db
using MGSCOIL

# ╔═╡ a4a6e3d5-6dff-4967-b819-d59e5c11f8cb
using CairoMakie

# ╔═╡ 7bdcd8f0-3559-438e-8bbd-6f2f7b8fbc49
html"""<style>
main {
    max-width: 900px;
    align-self: flex-start;
    margin-left: 50px;
}
"""

# ╔═╡ dbb681e1-2961-48ba-8670-6a8223eb0c17
import PlutoUI

# ╔═╡ 273f49a4-8c75-444f-98d2-c3fd2d9ddeab
import PhysicalConstants.CODATA2018: R, N_A, c_0, h, ε_0

# ╔═╡ eb3c0bd5-4290-490e-badd-e7c6ebcdca1e
PlutoUI.TableOfContents()

# ╔═╡ f7efb4f2-9671-4765-abca-0be96343164a
md"# `MGSCOIL.jl`"

# ╔═╡ e1c9d6bd-f22d-48db-b0f1-87bd61cbaffb
begin
	L = 1.5u"m"
	Z = 25.0u"cm"
	Y = 1.9u"cm"
	T = 200u"K"
	Mach = 1.8
	g34_0 = 0.010u"cm^-1"
	gs = let
	    num = 5
	    d = Z / num
	    d1 = 30.0u"cm"
	    (;num, d, d1)
	end
	Nx, Ny, Nz = (1024, 256, gs.num)
end

# ╔═╡ 39c570bb-3fcf-4d97-a2d4-bdad2373dbe3
md"## 1. `unit_operations.jl`"

# ╔═╡ de41ffa2-7bff-49f6-99fb-d4e7ff4e18cd
md"### default units: `UNIT`"

# ╔═╡ b1a7f74e-cf69-4359-87ee-c4d451de96af
DataFrame(Quantity = [keys(UNIT)...], unit = [values(UNIT)...])

# ╔═╡ 2b572f72-a380-4839-ad35-6da00d81301f
md"### unit strip: `uustrip`, `@uustrip`"

# ╔═╡ fb8b99bf-ffd6-4697-a901-7d68f120b073
let
	a = 2u"m"
	b = 30.0u"ns"
	t = (a, b)
	nt = (; a, b)
	dict = Dict("a"=>a, "b"=>b)
	arr = [a, b]
	@uustrip a b t nt dict arr
	a, b, t, nt, dict, arr
end

# ╔═╡ bc7b1fb7-460e-4f58-b8c7-4804ea2864f8
let
	a = 2u"m"
	b = 30.0u"ns"
	arr = [a, b]
	uustrip(Float32, arr)
end

# ╔═╡ a23a9fcb-8339-48fe-99fe-36b5e48558f4
md"## 2. `initial_flow.jl`"

# ╔═╡ 4f8c3e7b-aa82-456e-a2ca-277a1996c238
md"### reaction rate constants: `react_rate`"

# ╔═╡ 3a5c1b51-1189-49d8-973c-a2bab8ae84b4
react = MGSCOIL.react_rate(T)

# ╔═╡ 12704438-0acb-43ed-a568-9df1bfd1ca67
md"### initial gas flowrate: `initial_flowrate`"

# ╔═╡ 5b4a51fd-935c-4afc-be5a-20ed79e644b9
inlet = let
    Cl₂ = 0.5u"mol/s"
    I₂ = 12u"mmol/s"
    He_P = 4 * Cl₂
    He_S = 1.1u"mol/s"
    He = He_P + He_S
    H₂O = 50u"mmol/s"
    (;Cl₂, I₂, He_P, He_S, H₂O)
end

# ╔═╡ cef550ba-4907-4a6d-9118-2f21b159418e
flowrate0 = MGSCOIL.initial_flowrate(inlet, diss_I₂ = 0.95, utl_Cl₂ = 0.95)

# ╔═╡ 21368d29-c163-42be-9c1c-3aa954a0f805
md"### gas mixtures: `gas_mix`, `coe_mix`"

# ╔═╡ adfbce97-db30-42e8-a7a0-a6e9a3a0d0a7
gas_df = MGSCOIL.gas_mix(flowrate0)

# ╔═╡ 0cf584b1-c693-4055-8ce6-7777fae814f1
γ_mix, M_mix, γₚ_mix = MGSCOIL.coe_mix(gas_df)

# ╔═╡ e3d495a5-3012-4bdf-8ab6-6da703e38809
soundspeed = uconvert(u"cm/s", √(γ_mix * R / M_mix * T))

# ╔═╡ d031e2ec-8de0-4f7f-b363-2b1d02760d39
V = uconvert(u"cm/s", Mach * soundspeed)

# ╔═╡ 4e0ed1ce-e503-4576-a384-31c320c3d874
A = Z * Y     # flow cross-section

# ╔═╡ 71cd8b11-fbc0-4f10-98cd-c516ecb4b172
P = uconvert.(u"Torr", sum(flowrate0) / (V * A) * R * T)

# ╔═╡ e518580e-77b1-4f53-8831-583f9e10af09
md"### gas density: `flowrate2density`"

# ╔═╡ f64aebb9-3b17-4145-9ac7-7ee98332bb86
density = MGSCOIL.flowrate2density(flowrate0, V, A)

# ╔═╡ d0d5a47b-b2ce-47b6-a37a-7cc832fd1d38
md"### initial yield( or SSG): `gain2yield`"

# ╔═╡ 86e5bf8b-74c4-4dc0-8cce-9db5482a6542
line34, line22 = let
        l34 = line_I127(4, 3)
        l22 = line_I127(2, 2)
        A34 = l34.A
        A33 = A_I127(3, 3)
        A32 = A_I127(2, 3)
        A23 = A_I127(3, 2)
        A22 = l22.A
        A21 = A_I127(1, 2)
        τ3 = 1 / (A34 + A33 + A32)
        τ2 = 1 / (A23 + A22 + A21)
        σ34 = σ0_I127(4, 3, T=T, P=P, γ=γₚ_mix)
        σ22 = σ0_I127(2, 2, T=T, P=P, γ=γₚ_mix)
        merge(l34, (;σ = σ34, τ = τ3, gg = 7//9)), merge(l22, (;σ = σ22, τ = τ2, gg = 1))
end

# ╔═╡ b91176e0-de63-45b1-bbbd-fd504e4602f7
σ34g_0 = line34.σ * 7 // 12

# ╔═╡ 694bae9c-e7d8-4880-8541-26b5d6ee79af
yield_0 = MGSCOIL.gain2yield(g34_0, density, σ34g_0, react)

# ╔═╡ 05130e2c-4336-4b9d-a2f5-968387e65c7c
md"### initial flow field: `initial_flow`, `initial_fs`"

# ╔═╡ e26f8690-e69d-417e-a5b1-d23d8a1b7c6e
 dt_trip = uconvert(u"μs", 2L / c_0)

# ╔═╡ 5f1042b7-da3e-4d42-ac3a-f8f2946d1282
f2c = 8192 ÷ Nx

# ╔═╡ 6d5c4ec3-6cd6-4604-be8f-2e957a27a403
dt_flush = dt_trip * f2c

# ╔═╡ 60f02c14-43e2-4845-88cd-33648f8edd4d
t_flow = Nx * dt_flush

# ╔═╡ 8fdff8a0-9655-4e11-979d-2df5ce982898
 X = uconvert(u"cm", V * t_flow)

# ╔═╡ 09dacd3f-6362-4fda-a07a-3ccd658e4482
O2g_x, O2e_x, Ig_x, Ie_x = MGSCOIL.initial_flow(X, Nx, V, yield_0, g34_0, σ34g_0, density, react);

# ╔═╡ e3efc9f2-0e5e-41b0-aa4a-8caf5b7ad3e1
flowsheet = MGSCOIL.initial_fs(Ny, Nz, O2g_x, O2e_x, Ie_x[:, 1], Ie_x[:, 2], Ig_x[:, 1], Ig_x[:, 3], Ig_x[:, 2]+Ig_x[:, 4]);

# ╔═╡ 5c8d9c58-94db-4364-8a5c-b59774be361f
fs = StructArray(flowsheet);

# ╔═╡ 811a6af3-a168-4e95-b406-a62bc90317d6
fs.O₂g[:, :, 1]

# ╔═╡ df4ebe72-1dd3-4c39-a1f7-5d44b6099f4f
propertynames(fs)

# ╔═╡ 7335a276-740c-4590-b499-9addaf55f399
md"## 3. `model_setup.jl`"

# ╔═╡ 445abac3-99a7-418c-9b2a-fb372dd8cf68
md"### build model: `model_setup`"

# ╔═╡ bcf1dece-43c5-4413-911d-31875efe7090
md"""
```julia
fs, flow, lines, grid = model_setup(L = L, Z = Z, Y = Y,
                                    cavity = cavity, 
                                    inlet = inlet, 
                                    T = T, Mach = Mach, 
                                    N = N, f2c = f2c, 
                                    g34_0 = g34_0);
```
"""

# ╔═╡ adb5b89c-3b2c-4c3b-918b-df47c46d2998
md"## 4. `angular_spectrum.jl`"

# ╔═╡ 1f0d0e7d-9ec4-4307-bca4-0c6e29aefa23
md"### `free_propagate!`"

# ╔═╡ c4f7a7fc-6875-401e-92cb-db13f4d31281
md"## 5. `flow_refresh.jl`"

# ╔═╡ 02897d5f-366c-45a0-a0bc-d7d8768c934e
md"### flow and quench: `kernel_refresh!`"

# ╔═╡ 227508ba-3b87-446f-b089-d69a977d8874
md"### reaction and relaxation: `kernel_refresh_fast!`"

# ╔═╡ 5ecd964b-7f3c-4a9a-b0c8-8fb3367ea242
md"## 6. `optical_extraction.jl`"

# ╔═╡ c9170980-2523-4c14-b995-fdf536ae9197
md"### `optical_extraction!`"

# ╔═╡ 3278d6e7-7f66-4f02-abeb-5743bd394c8a
md"## 7. `laser.jl`"

# ╔═╡ b3f23618-a305-4eb3-b345-00fe6dc633f1
md"### update optical field for one trip: `bounce!`"

# ╔═╡ c8070009-b96a-49cb-a257-0c89ff1e5670
md"### main function `propagate`"

# ╔═╡ 83d724b2-a6b8-4781-8f18-15240207f09f
md"## 8. `visualization.jl`"

# ╔═╡ e0ecb826-8749-414c-969e-efbb82064c20
waveform = (;period = 500, cycle = 0.4, rising = 100, falling = 10, offset = -100, amplitude = 400, residual = 50)

# ╔═╡ 8081ce03-8307-4341-9d63-96c92e06d48a
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

# ╔═╡ bddf165b-f0f0-4429-8170-720f47f3d456
pulse(10, waveform = waveform)

# ╔═╡ 9155d4e7-b1bb-415c-aa7e-36bc02893e96
begin
	B_range = 0:1:600
	interp_linear43 = σr_ltp(4, 3, B_range, "S", T=T, P=P, γ=γₚ_mix)
	interp_linear22 = σr_ltp(2, 2, B_range, "S", T=T, P=P, γ=γₚ_mix)
end

# ╔═╡ 9cc131da-fae0-4307-81e7-3fcdd5eacbe4
let
	fig = Figure()
	ax1 = Axis(fig[1, 1], yticklabelcolor = :black, rightspinevisible = false)
    ax2 = Axis(fig[1, 1], yticklabelcolor = :orangered, yaxisposition = :right,
        rightspinecolor = :orangered, ytickcolor = :orangered)

	x = 0:600
	field = pulse.(x, waveform = waveform)
	
	lines!(ax1, x, field)
	lines!(ax2, x, interp_linear43.(field), color = :green)
	lines!(ax2, x, interp_linear22.(field), color = :red)

	ylims!(ax2, 0, 1)
	
	fig
end

# ╔═╡ Cell order:
# ╠═7bdcd8f0-3559-438e-8bbd-6f2f7b8fbc49
# ╠═28d0a2f5-e2b6-4ba8-9d9d-b95351154124
# ╠═f4d8a19a-ccd0-4fea-b80d-c7dc71db5bb6
# ╠═8674c900-228a-4166-a280-ba161b3a96e9
# ╠═dbb681e1-2961-48ba-8670-6a8223eb0c17
# ╠═a744b17d-d260-4f0b-92ca-ee19457722e6
# ╠═7a8a8dcc-0cc4-484c-9aa4-241b99e27407
# ╠═cfbe01e2-a8b5-4a34-bd56-3cd9fb51bf14
# ╠═273f49a4-8c75-444f-98d2-c3fd2d9ddeab
# ╠═1912ddd3-2837-4400-8b8e-e0768d0a0cfe
# ╠═59c363f1-6321-46df-a15b-487b6e1c23db
# ╠═a4a6e3d5-6dff-4967-b819-d59e5c11f8cb
# ╟─eb3c0bd5-4290-490e-badd-e7c6ebcdca1e
# ╟─f7efb4f2-9671-4765-abca-0be96343164a
# ╠═e1c9d6bd-f22d-48db-b0f1-87bd61cbaffb
# ╟─39c570bb-3fcf-4d97-a2d4-bdad2373dbe3
# ╟─de41ffa2-7bff-49f6-99fb-d4e7ff4e18cd
# ╠═b1a7f74e-cf69-4359-87ee-c4d451de96af
# ╟─2b572f72-a380-4839-ad35-6da00d81301f
# ╟─fb8b99bf-ffd6-4697-a901-7d68f120b073
# ╟─bc7b1fb7-460e-4f58-b8c7-4804ea2864f8
# ╟─a23a9fcb-8339-48fe-99fe-36b5e48558f4
# ╟─4f8c3e7b-aa82-456e-a2ca-277a1996c238
# ╠═3a5c1b51-1189-49d8-973c-a2bab8ae84b4
# ╟─12704438-0acb-43ed-a568-9df1bfd1ca67
# ╟─5b4a51fd-935c-4afc-be5a-20ed79e644b9
# ╟─cef550ba-4907-4a6d-9118-2f21b159418e
# ╟─21368d29-c163-42be-9c1c-3aa954a0f805
# ╟─adfbce97-db30-42e8-a7a0-a6e9a3a0d0a7
# ╠═0cf584b1-c693-4055-8ce6-7777fae814f1
# ╠═e3d495a5-3012-4bdf-8ab6-6da703e38809
# ╠═d031e2ec-8de0-4f7f-b363-2b1d02760d39
# ╠═4e0ed1ce-e503-4576-a384-31c320c3d874
# ╠═71cd8b11-fbc0-4f10-98cd-c516ecb4b172
# ╟─e518580e-77b1-4f53-8831-583f9e10af09
# ╠═f64aebb9-3b17-4145-9ac7-7ee98332bb86
# ╟─d0d5a47b-b2ce-47b6-a37a-7cc832fd1d38
# ╟─86e5bf8b-74c4-4dc0-8cce-9db5482a6542
# ╠═b91176e0-de63-45b1-bbbd-fd504e4602f7
# ╠═694bae9c-e7d8-4880-8541-26b5d6ee79af
# ╟─05130e2c-4336-4b9d-a2f5-968387e65c7c
# ╠═e26f8690-e69d-417e-a5b1-d23d8a1b7c6e
# ╠═5f1042b7-da3e-4d42-ac3a-f8f2946d1282
# ╠═6d5c4ec3-6cd6-4604-be8f-2e957a27a403
# ╠═60f02c14-43e2-4845-88cd-33648f8edd4d
# ╠═8fdff8a0-9655-4e11-979d-2df5ce982898
# ╠═09dacd3f-6362-4fda-a07a-3ccd658e4482
# ╠═e3efc9f2-0e5e-41b0-aa4a-8caf5b7ad3e1
# ╠═5c8d9c58-94db-4364-8a5c-b59774be361f
# ╠═811a6af3-a168-4e95-b406-a62bc90317d6
# ╠═df4ebe72-1dd3-4c39-a1f7-5d44b6099f4f
# ╟─7335a276-740c-4590-b499-9addaf55f399
# ╟─445abac3-99a7-418c-9b2a-fb372dd8cf68
# ╟─bcf1dece-43c5-4413-911d-31875efe7090
# ╟─adb5b89c-3b2c-4c3b-918b-df47c46d2998
# ╟─1f0d0e7d-9ec4-4307-bca4-0c6e29aefa23
# ╟─c4f7a7fc-6875-401e-92cb-db13f4d31281
# ╟─02897d5f-366c-45a0-a0bc-d7d8768c934e
# ╟─227508ba-3b87-446f-b089-d69a977d8874
# ╟─5ecd964b-7f3c-4a9a-b0c8-8fb3367ea242
# ╟─c9170980-2523-4c14-b995-fdf536ae9197
# ╟─3278d6e7-7f66-4f02-abeb-5743bd394c8a
# ╟─b3f23618-a305-4eb3-b345-00fe6dc633f1
# ╠═c8070009-b96a-49cb-a257-0c89ff1e5670
# ╟─83d724b2-a6b8-4781-8f18-15240207f09f
# ╠═e0ecb826-8749-414c-969e-efbb82064c20
# ╠═8081ce03-8307-4341-9d63-96c92e06d48a
# ╠═bddf165b-f0f0-4429-8170-720f47f3d456
# ╠═9155d4e7-b1bb-415c-aa7e-36bc02893e96
# ╠═9cc131da-fae0-4307-81e7-3fcdd5eacbe4
