from math import isclose
from random import random

import polars as pl

from pl_simd import col as simd_col


def do_with_preinit(df: pl.DataFrame, tgt_vec):
    return df.with_columns(simd_col("a").spatial.cos(cmp=tgt_vec).alias("sim"))


def do_test(sim_vec, tgt_vec):
    df = pl.DataFrame({"a": [sim_vec]}).with_columns(
        simd_col("a").cast(pl.Array(pl.Float32, len(sim_vec)))
    )
    return do_with_preinit(df, tgt_vec)


def test_sim():
    s1 = [0.0, 1.0, 2.0]

    cossim = do_test(s1, [1.0, 1.0, 2.0]).get_column("sim").to_list()

    assert isclose(0.08712911517429056, cossim[0])


def rand_arr(sz: int):
    return [random() for _ in range(sz)]


def simple_bench(a1, a2):
    do_test(a1, a2)


if __name__ == "__main__":
    import timeit

    times = []
    for shp in [1, 10, 100, 10000, 100_000]:
        print(f"batch len: {shp}")
        for n in [4, 8, 16, 24, 32, 64, 128, 256, 384]:
            print(f"testing: {shp}x{n}")

            a1 = rand_arr(n)
            df = pl.DataFrame({"a": [rand_arr(n) for _ in range(shp)]}).with_columns(
                pl.col("a").cast(pl.Array(pl.Float32, n))
            )
            tt = timeit.timeit(
                "simple_bench(df, a1)",
                globals=dict(simple_bench=do_with_preinit, a1=a1, df=df),
                number=10000,
            )
            times.append(
                {
                    "name": f"{shp}x{n}",
                    "n_rows": shp,
                    "vector_size": n,
                    "n_tests": 10_000,
                    "total_time": tt,
                    "time_ms": (tt / 10),  # 10_000 / 1000
                }
            )

    times = pl.DataFrame(times)
    times.write_csv("benches_f32.csv")
    print(times)
