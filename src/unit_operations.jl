#   derived_dimension (g/mol, cm^-3)
@derived_dimension MolarMass 𝐌 / 𝐍
@derived_dimension ParticleDensity 𝐋^-3

const UNIT = (;
    Length = u"cm",
    Area = u"cm^2",
    ParticleDensity = u"cm^-3",
    Time = u"s",
    Wavenumber = u"cm^-1",
    Velocity = u"cm/s",
    Energy = u"J",
    Power = u"W",
    Frequency = u"s^-1",
    ReactRate = u"cm^3/s"
)

#   uustrip
for d in propertynames(UNIT)
    eval(:(uustrip(q::$d) = ustrip(UNIT.$d, q)))
end
uustrip(n::Number) = n
uustrip(t::DataType, q) = map(t, uustrip(q))

uustrip(nt::NamedTuple) = begin
    k = keys(nt)
    v = [uustrip.(val) for val in nt]
    nt_ustrip = (; zip(k, v)...)
    return nt_ustrip
end

uustrip(t::Tuple) = uustrip.(t)
uustrip(a::AbstractArray) = uustrip.(a)

uustrip(dict::AbstractDict) = begin
    k = keys(dict)
    v = [uustrip.(val) for val in values(dict)]
    dict_ustrip = Dict(zip(k, v))
    return dict_ustrip
end

#   @uustrip
macro uustrip(args...)
    tmp = Expr(:block)
    for q in args
        # push!(tmp.args, esc(:($q = $q isa Union{Tuple, AbstractArray} ? uustrip.($q) : uustrip($q))))
        push!(tmp.args, esc(:($q = uustrip($q))))
    end
    return tmp
end
