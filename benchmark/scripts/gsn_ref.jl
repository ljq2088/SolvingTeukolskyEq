#!/usr/bin/env julia
# Usage:
#   julia gsn_ref.jl --s=-2 --l=2 --m=2 --a=0.1 --omega=1.0 --M=1.0
#   julia gsn_ref.jl --s=-2 --l=2 --m=2 --a=0.1 --omega-list=1e-4,1e-3,1e-2 --M=1.0
# Outputs one JSON object per omega to stdout.

using GeneralizedSasakiNakamura

function main()
    s = -2
    l = 2
    m = 2
    a = 0.1
    omega = 1.0
    omega_list = nothing
    M_val = 1.0

    for arg in ARGS
        if startswith(arg, "--s=")
            s = parse(Int, arg[5:end])
        elseif startswith(arg, "--l=")
            l = parse(Int, arg[5:end])
        elseif startswith(arg, "--m=")
            m = parse(Int, arg[5:end])
        elseif startswith(arg, "--a=")
            a = parse(Float64, arg[5:end])
        elseif startswith(arg, "--omega=")
            omega = parse(Float64, arg[9:end])
        elseif startswith(arg, "--omega-list=")
            raw = split(arg[14:end], ",")
            omega_list = [parse(Float64, strip(x)) for x in raw if !isempty(strip(x))]
        elseif startswith(arg, "--M=")
            M_val = parse(Float64, arg[5:end])
        end
    end

    if omega_list === nothing
        omega_list = [omega]
    end

    a_scaled = a / M_val

    for omega_val in omega_list
        omega_scaled = omega_val * M_val

        Rin = Teukolsky_radial(s, l, m, a_scaled, omega_scaled, IN)
        lambda = Rin.mode.lambda

        B_inc = Rin.incidence_amplitude
        B_ref = Rin.reflection_amplitude
        B_trans = Rin.transmission_amplitude

        println("{\"s\": $s, \"l\": $l, \"m\": $m, \"a\": $a, \"omega\": $omega_val, \"M\": $M_val, " *
                "\"lambda_re\": $(real(lambda)), \"lambda_im\": $(imag(lambda)), " *
                "\"B_inc_re\": $(real(B_inc)), \"B_inc_im\": $(imag(B_inc)), " *
                "\"B_ref_re\": $(real(B_ref)), \"B_ref_im\": $(imag(B_ref)), " *
                "\"B_trans_re\": $(real(B_trans)), \"B_trans_im\": $(imag(B_trans))}")
    end
end

main()
