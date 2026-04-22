from __future__ import annotations
import sys
sys.path.append("/home/ljq/code/PINN/SolvingTeukolsky")
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from domain.safe_eval import evaluate_param_point


def _lambda_cache_key(a: float, omega: float, l: int, m: int, s: int) -> str:
    return (
        f"a={round(float(a), 12):.12f}|"
        f"omega={round(float(omega), 12):.12f}|"
        f"l={int(l)}|m={int(m)}|s={int(s)}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=2)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=-2)

    parser.add_argument("--a-min", type=float, default=0.01)
    parser.add_argument("--a-max", type=float, default=0.99)
    parser.add_argument("--omega-min", type=float, default=1.0e-4)
    parser.add_argument("--omega-max", type=float, default=1.0)

    parser.add_argument("--n-a", type=int, default=25)
    parser.add_argument("--n-w", type=int, default=40)

    parser.add_argument("--k-margin", type=float, default=1.0e-2)
    parser.add_argument("--r-match", type=float, default=8.0)
    parser.add_argument("--n-cheb", type=int, default=32)

    parser.add_argument("--require-ramp", action="store_true")
    parser.add_argument("--save", type=str, default="outputs/domain/probe_l2_m2.json")

    args = parser.parse_args()

    a_grid = np.linspace(args.a_min, args.a_max, args.n_a)
    w_grid = np.linspace(args.omega_min, args.omega_max, args.n_w)

    records = []
    counter = Counter()
    lambda_cache = {}

    for a in a_grid:
        for w in w_grid:
            st = evaluate_param_point(
                a=float(a),
                omega=float(w),
                l=args.l,
                m=args.m,
                s=args.s,
                k_horizon_margin=args.k_margin,
                require_ramp=args.require_ramp,
                r_match=args.r_match,
                n_cheb=args.n_cheb,
            )
            counter[st.code] += 1
            lam_value = None if st.lambda_status is None else st.lambda_status.value
            lam_re = None if lam_value is None else complex(lam_value).real
            lam_im = None if lam_value is None else complex(lam_value).imag

            rec = {
                "a": st.a,
                "omega": st.omega,
                "valid": st.valid,
                "code": st.code,
                "message": st.message,
                "k_h": st.k_h,
                "lambda_re": lam_re,
                "lambda_im": lam_im,
            }
            records.append(rec)

            if lam_value is not None:
                lambda_cache[_lambda_cache_key(st.a, st.omega, st.l, st.m, st.s)] = {
                    "a": st.a,
                    "omega": st.omega,
                    "l": st.l,
                    "m": st.m,
                    "s": st.s,
                    "lambda_re": lam_re,
                    "lambda_im": lam_im,
                }

    total = len(records)
    valid = sum(1 for r in records if r["valid"])

    print("=" * 80)
    print(f"(l,m,s)=({args.l},{args.m},{args.s})")
    print(f"grid = {args.n_a} x {args.n_w} = {total}")
    print(f"valid = {valid}, invalid = {total - valid}")
    print("-" * 80)
    for k, v in counter.most_common():
        print(f"{k:24s} : {v}")
    print("=" * 80)

    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "l": args.l,
                    "m": args.m,
                    "s": args.s,
                    "a_min": args.a_min,
                    "a_max": args.a_max,
                    "omega_min": args.omega_min,
                    "omega_max": args.omega_max,
                    "n_a": args.n_a,
                    "n_w": args.n_w,
                    "k_margin": args.k_margin,
                    "r_match": args.r_match,
                    "n_cheb": args.n_cheb,
                    "require_ramp": args.require_ramp,
                },
                "counts": dict(counter),
                "lambda_cache": lambda_cache,
                "records": records,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"saved -> {save_path}")


if __name__ == "__main__":
    main()
