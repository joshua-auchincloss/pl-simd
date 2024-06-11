# pl-simd

`pl-simd` is a vector targeted library for usage between polars and simd optimized operations. The primary use case is to enable functionality for common vector-search queries against dataframes targeting datasets which may not be large enough to justify usage of a targeted database (where your data fits in memory). The API is written in rust, bridging the gap between [`polars`](https://github.com/pola-rs/polars) and [SimSIMD](https://github.com/ashvardanian/SimSIMD).

## Usage

### col

#### Spatial Similarity

```py
from pl_simd import col as simd_col


tgt_vec = [2., 3., 4.]

df = pl.DataFrame({"a": [[1., 2., 3.]]}).with_columns(
    simd_col("a").cast(pl.Array(pl.Float32, 3))
)

df.with_columns(
    simd_col("a").spatial.cos(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |

df.with_columns(
    simd_col("a").spatial.sqeuclidean(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |
```

#### Probability Similarity

```py
from pl_simd import col as simd_col


tgt_vec = [2., 3., 4.]

df = pl.DataFrame({"a": [[1., 2., 3.]]}).with_columns(
    simd_col("a").cast(pl.Array(pl.Float32, 3))
)

df.with_columns(
    simd_col("a").prob.jensenshannon(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |

df.with_columns(
    simd_col("a").prob.kullbackleibler(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |
```

#### Binary Similarity

```py
from pl_simd import col as simd_col


tgt_vec = [b"11110000", b"00001111"]

df = pl.DataFrame({"a": [[b"11110000", b"01010101"]]}).with_columns(
    simd_col("a").cast(pl.Array(pl.Uint8, 2))
)

df.with_columns(
    simd_col("a").binary.hamming(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |

df.with_columns(
    simd_col("a").binary.jaccard(cmp=tgt_vec).alias("sim"))
# |   a   | sim  |
# | [...] | ...  |
```

## Performance

Computing the cosine distance between 1x384 for 100,000 rows takes ~80ms.
