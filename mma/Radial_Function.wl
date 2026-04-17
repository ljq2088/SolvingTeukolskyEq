(* ::Package:: *)

Needs["SpinWeightedSpheroidalHarmonics`"];
Needs["Teukolsky`"];

ClearAll[
    ComputeAmplitudes, ComputeRinAtR, SampleRinOnGrid, SampleRinAtPoints, SampleRinComplex,
    ComputeAmplitudesMST, ComputeRinAtRMST, SampleRinOnGridMST, SampleRinAtPointsMST, SampleRinComplexMST
];

ComputeAmplitudes[s_, l_, m_, a_, \[Omega]_] := Module[
    {R, ampsIn, Binc, Bref, Btrans, ratio1, ratio2},
    
    R = TeukolskyRadial[s,l,m,a,\[Omega],Method->"NumericalIntegration"];
    ampsIn = R["In"]["Amplitudes"];
    
    Binc = ampsIn["Incidence"];
    Bref = ampsIn["Reflection"];
    Btrans = ampsIn["Transmission"];
    
    ratio1 = Bref/Binc;
    ratio2 = Abs[Bref/Binc];
    
    <|
        "Incidence" -> Binc,
        "Reflection" -> Bref,
        "Transmission" -> Btrans,
        "ReflectionOverIncidence" -> ratio1,
        "AbsReflectionOverIncidence" -> ratio2
    |>
];

ComputeRinAtR[s_, l_, m_, a_, \[Omega]_, r_] := Module[
    {Rin},
    
    Rin = TeukolskyRadial[s, l, m, a, \[Omega],Method->"NumericalIntegration"]["In"];
    
    <|
        "Rin" -> N[Rin[r], 30],
        "RinPrime" -> N[Rin'[r], 30],
        "RinDoublePrime" -> N[Rin''[r], 30]
    |>
];

SampleRinOnGrid[s_, l_, m_, a_, \[Omega]_, rmin_, rmax_, n_] := Module[
    {Rin, rgrid, vals, data},
    
    Rin = TeukolskyRadial[s, l, m, a, \[Omega],Method->"NumericalIntegration"]["In"];
    rgrid = N[Subdivide[rmin, rmax, n - 1], 30];
    vals = N[Rin /@ rgrid, 30];
    
    data = Transpose[{rgrid, Re[vals], Im[vals]}];
    
    (* \:5982\:679c\:4f60\:8fd8\:60f3\:5728 Mathematica kernel \:7a97\:53e3\:91cc\:9010\:884c\:770b\:5230 *)
    (*Scan[Print, data];*)
    
    data
];
SampleRinOnX[s_, l_, m_, a_, \[Omega]_, xmin_,xmax_,n_] := Module[
    {Rin, xgrid, rgrid, vals, data},
    
    Rin = TeukolskyRadial[s, l, m, a, \[Omega],Method->"NumericalIntegration"]["In"];
    xgrid = N[Subdivide[xmin, xmax, n - 1], 30];
    rgrid=(1.0+Sqrt[1-a^2])*N[1/xgrid,30];
    vals = N[Rin /@ rgrid, 30];
    
    data = Transpose[{rgrid, Re[vals], Im[vals]}];
    
    (* \:5982\:679c\:4f60\:8fd8\:60f3\:5728 Mathematica kernel \:7a97\:53e3\:91cc\:9010\:884c\:770b\:5230 *)
    (*Scan[Print, data];*)
    
    data
];

SampleRinAtPoints[s_, l_, m_, a_, \[Omega]_, rlist_List] := Module[
    {Rin, rgrid, vals, data},

    Rin = TeukolskyRadial[s, l, m, a, \[Omega],Method->"NumericalIntegration"]["In"];
    rgrid = N[rlist, 30];
    vals = N[Rin /@ rgrid, 30];

    data = Transpose[{rgrid, Re[vals], Im[vals]}];

    data
];


ClearAll[SampleRinComplex];

SampleRinComplex[s_, l_, m_, a_, \[Omega]_, rmin_, rmax_, n_] := Module[
    {Rin, rgrid, vals},
    
    Rin = TeukolskyRadial[s, l, m, a, \[Omega],Method->"NumericalIntegration"]["In"];
    rgrid = N[Subdivide[rmin, rmax, n - 1], 30];
    vals = N[Rin /@ rgrid, 30];
    
    Transpose[{rgrid, vals}]
];


ComputeAmplitudesMST[s_, l_, m_, a_, \[Omega]_] := Module[
    {R, ampsIn, Binc, Bref, Btrans, ratio1, ratio2},

    R = TeukolskyRadial[s, l, m, a, \[Omega], Method -> "MST"];
    ampsIn = R["In"]["Amplitudes"];

    Binc = ampsIn["Incidence"];
    Bref = ampsIn["Reflection"];
    Btrans = ampsIn["Transmission"];

    ratio1 = Bref/Binc;
    ratio2 = Abs[Bref/Binc];

    <|
        "Incidence" -> Binc,
        "Reflection" -> Bref,
        "Transmission" -> Btrans,
        "ReflectionOverIncidence" -> ratio1,
        "AbsReflectionOverIncidence" -> ratio2
    |>
];

ComputeRinAtRMST[s_, l_, m_, a_, \[Omega]_, r_] := Module[
    {Rin},

    Rin = TeukolskyRadial[s, l, m, a, \[Omega], Method -> "MST"]["In"];

    <|
        "Rin" -> N[Rin[r], 30],
        "RinPrime" -> N[Rin'[r], 30],
        "RinDoublePrime" -> N[Rin''[r], 30]
    |>
];

SampleRinOnGridMST[s_, l_, m_, a_, \[Omega]_, rmin_, rmax_, n_] := Module[
    {Rin, rgrid, vals, data},

    Rin = TeukolskyRadial[s, l, m, a, \[Omega], Method -> "MST"]["In"];
    rgrid = N[Subdivide[rmin, rmax, n - 1], 30];
    vals = N[Rin /@ rgrid, 30];

    data = Transpose[{rgrid, Re[vals], Im[vals]}];

    data
];

SampleRinAtPointsMST[s_, l_, m_, a_, \[Omega]_, rlist_List] := Module[
    {Rin, rgrid, vals, data},

    Rin = TeukolskyRadial[s, l, m, a, \[Omega], Method -> "MST"]["In"];
    rgrid = N[rlist, 30];
    vals = N[Rin /@ rgrid, 30];

    data = Transpose[{rgrid, Re[vals], Im[vals]}];

    data
];

SampleRinComplexMST[s_, l_, m_, a_, \[Omega]_, rmin_, rmax_, n_] := Module[
    {Rin, rgrid, vals},

    Rin = TeukolskyRadial[s, l, m, a, \[Omega], Method -> "MST"]["In"];
    rgrid = N[Subdivide[rmin, rmax, n - 1], 30];
    vals = N[Rin /@ rgrid, 30];

    Transpose[{rgrid, vals}]
];
