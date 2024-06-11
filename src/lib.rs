use polars::prelude::*;
use polars::series::Series;
use polars_core::utils::arrow::array::{Array, PrimitiveArray};
use pyo3::prelude::*;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use simsimd::{BinarySimilarity, ProbabilitySimilarity, SpatialSimilarity};

#[derive(Deserialize, Clone)]
pub struct FloatKwargs {
    cmp: Vec<f32>,

    #[serde(default = "default_0")]
    null_value: f64,
}

fn default_0() -> f64 {
    0.
}

#[derive(Deserialize, Clone)]
pub struct BinaryKwargs {
    cmp: Vec<u8>,

    #[serde(default = "default_0")]
    null_value: f64,
}

macro_rules! impl_similarity {
    (
        $super: ident:
        $cls: ident & $t: ty =>
        $($($fn: ident) + $(,)?)*,
    ) => {
        impl $super {
            paste::paste!{
                $(
                    $(
                        fn $fn(&self, arr: Vec<$t>) -> f64 {
                            <$t as $cls>::$fn(&self.cmp, &arr).unwrap_or(self.null_value)
                        }
                    )*
                )*
            }
        }
    };
}

impl_similarity! {
    FloatKwargs: SpatialSimilarity & f32 => cos, sqeuclidean,
}

impl_similarity! {
    FloatKwargs: ProbabilitySimilarity & f32 => jensenshannon, kullbackleibler,
}

impl_similarity! {
    BinaryKwargs: BinarySimilarity & u8 => hamming, jaccard,
}

macro_rules! as_arr_ty {
    (
        $ty: expr
    ) => {
        paste::paste! {
            fn [<as_ $ty _arr>](arr: Box<dyn Array>) -> Vec<$ty> {
                arr.as_any()
                    .downcast_ref::<PrimitiveArray<$ty>>()
                    .unwrap()
                    .values_iter()
                    .copied()
                    .collect::<Vec<_>>()
            }
        }
    };
}

as_arr_ty!(f32);
as_arr_ty!(u8);

macro_rules! f32_fn {
    (
        $fn: expr
    ) => {
        paste::paste! {
            #[polars_expr(output_type=Float64)]
            fn [<$fn>](inputs: &[Series], kwargs: FloatKwargs) -> PolarsResult<Series> {
                let ca = inputs[0].array()?;
                if ca.width() != kwargs.cmp.len() {
                    polars_bail!{
                        InvalidOperation: "expected fixed size of {}, found {}", kwargs.cmp.len(), ca.width()
                    }
                }
                let ca: ChunkedArray<Float64Type> = ca.apply_generic(|it| match it {
                    Some(arr) => Some(kwargs.$fn(as_f32_arr(arr))),
                    None => None,
                });
                Ok(ca.into_series())
            }
        }
    };
}

macro_rules! f32_fns {
    ($($($fn: expr) + $(,)?)*) => {
        $(
            $(f32_fn!{$fn})*
        )*
    };
}

f32_fns! {
    cos, sqeuclidean, jensenshannon, kullbackleibler
}

macro_rules! u8_fn {
    (
        $fn: expr
    ) => {
        paste::paste! {
            #[polars_expr(output_type=Float64)]
            fn [<$fn>](inputs: &[Series], kwargs: BinaryKwargs) -> PolarsResult<Series> {
                let ca = inputs[0].array()?;
                if ca.width() != kwargs.cmp.len() {
                    polars_bail!{
                        InvalidOperation: "expected fixed size of {}, found {}", kwargs.cmp.len(), ca.width()
                    }
                }
                let ca: ChunkedArray<Float64Type> = ca.apply_generic(|it| match it {
                    Some(arr) => Some(kwargs.$fn(as_u8_arr(arr))),
                    None => None,
                });
                Ok(ca.into_series())
            }
        }
    };
}

macro_rules! u8_fns {
    ($($($fn: expr) + $(,)?)*) => {
        $(
            $(u8_fn!{$fn})*
        )*
    };
}

u8_fns! {
    hamming, jaccard
}

#[pymodule]
fn _internal(_: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
