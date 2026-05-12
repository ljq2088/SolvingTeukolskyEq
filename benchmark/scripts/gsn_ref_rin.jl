#!/usr/bin/env julia
# Evaluate R_in(r) on a grid of r points using GSN.jl.
# Usage:
#   julia gsn_ref_rin.jl --s=-2 --l=2 --m=2 --a=0.1 --omega=1.0 --r-list=2.0,5.0,10.0,100.0 --M=1.0
# Outputs JSON to stdout: {"a":..., "omega":..., "r": [...], "Rin_re": [...], "Rin_im": [...]}

using GeneralizedSasakiNakamura

function main()
    s = -2; l = 2; m = 2; a = 0.1; omega = 1.0; M_val = 1.0
    r_list = nothing; r_file = nothing; n_r = 500; r_min = nothing; r_max = 1000.0
    r_eps = 1.0e-4

    for arg in ARGS
        if startswith(arg, "--s="); s = parse(Int, arg[5:end])
        elseif startswith(arg, "--l="); l = parse(Int, arg[5:end])
        elseif startswith(arg, "--m="); m = parse(Int, arg[5:end])
        elseif startswith(arg, "--a="); a = parse(Float64, arg[5:end])
        elseif startswith(arg, "--omega="); omega = parse(Float64, arg[9:end])
        elseif startswith(arg, "--M="); M_val = parse(Float64, arg[5:end])
        elseif startswith(arg, "--r-list=")
            raw = split(arg[10:end], ",")
            r_list = [parse(Float64, strip(x)) for x in raw if !isempty(strip(x))]
        elseif startswith(arg, "--r-file="); r_file = arg[10:end]
        elseif startswith(arg, "--n-r="); n_r = parse(Int, arg[7:end])
        elseif startswith(arg, "--r-min="); r_min = parse(Float64, arg[9:end])
        elseif startswith(arg, "--r-max="); r_max = parse(Float64, arg[9:end])
        elseif startswith(arg, "--r-eps="); r_eps = parse(Float64, arg[9:end])
        end
    end

    a_scaled = a / M_val
    omega_scaled = omega * M_val

    Rin = Teukolsky_radial(s, l, m, a_scaled, omega_scaled, IN)

    if r_list === nothing
        if r_file !== nothing
            r_list = map(x -> parse(Float64, x), split(readchomp(r_file), ","))
        else
            rp = 1.0 + sqrt(1.0 - a_scaled^2)
            if r_min === nothing
                r_min = rp + r_eps
            end
            r_list = collect(range(Float64(r_min), stop=Float64(r_max), length=n_r))
        end
    end

    Rin_re = Float64[]
    Rin_im = Float64[]
    for r_val in r_list
        val = Rin(r_val)  # Returns [R(r), dR/dr], we want R(r)
        push!(Rin_re, real(val))
        push!(Rin_im, imag(val))
    end

    r_str = join(string.(r_list), ",")
    re_str = join(string.(Rin_re), ",")
    im_str = join(string.(Rin_im), ",")

    println("{\"s\": $s, \"l\": $l, \"m\": $m, \"a\": $a, \"omega\": $omega, \"M\": $M_val, " *
            "\"r\": [$r_str], \"Rin_re\": [$re_str], \"Rin_im\": [$im_str]}")
end

main()
