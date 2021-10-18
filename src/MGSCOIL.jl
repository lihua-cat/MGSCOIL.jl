module MGSCOIL

import PhysicalConstants.CODATA2018: R as 𝑅, N_A as 𝑁, c_0 as 𝑐, h as ℎ, ε_0 as 𝜀
import Statistics: mean
import ZeemanSpectra: line_I127, A_I127, σ0_I127, ATOM_DATA, fwhm_doppler
using DataFrames, StructArrays, Parameters
using Unitful
using CUDA; CUDA.allowscalar(false)
using CUDA.CUFFT
using GLMakie
using ProgressLogging
using Printf

const PRECISION = Float32

##  unit operation
export UNIT, uustrip, @uustrip

include("unit_operations.jl")
##  initial flow

include("initial_flow.jl")
##  model setup
export model_setup

include("model_setup.jl")
##  angular spectrum

include("angular_spectrum.jl")
##  flow refresh
export flow_refresh!, flow_refresh_fast!

include("flow_refresh.jl")
##  optical extraction

include("optical_extraction.jl")
##  laser
export outpower, bounce!, propagate, angular_spectrum_paras

include("laser.jl")
##  visualization
export plot1, plot2, plot3

include("visualization.jl")

end
