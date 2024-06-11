from pathlib import Path
from typing import List, cast, Iterable

import polars as pl
from polars.functions.col import Column
from polars.plugins import register_plugin_function
from polars.type_aliases import PolarsDataType

__resource__ = Path(__file__).parent


@pl.api.register_expr_namespace("spatial")
class Spatial:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def cos(self, cmp: List[float]) -> pl.Expr:
        """Computes the cosine similarity between two slices. The cosine similarity is a measure of similarity between two non-zero vectors of an dot product space that measures the cosine of the angle between them."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="cos",
            is_elementwise=True,
        )

    def sqeuclidean(self, cmp: List[float]) -> pl.Expr:
        """Computes the squared Euclidean distance between two slices. The squared Euclidean distance is the sum of the squared differences between corresponding elements of the two slices."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="sqeuclidean",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("prob")
class Probability:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def kullbackleibler(self, cmp: List[float]) -> pl.Expr:
        """Computes the Kullback-Leibler divergence between two probability distributions. The Kullback-Leibler divergence is a measure of how one probability distribution diverges from a second, expected probability distribution."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="kullbackleibler",
            is_elementwise=True,
        )

    def jensenshannon(self, cmp: List[float]) -> pl.Expr:
        """Computes the Jensen-Shannon divergence between two probability distributions. The Jensen-Shannon divergence is a method of measuring the similarity between two probability distributions. It is based on the Kullback-Leibler divergence, but is symmetric and always has a finite value."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="jensenshannon",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("binary")
class Binary:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming(self, cmp: bytes) -> pl.Expr:
        """Computes the Hamming distance between two binary data slices. The Hamming distance between two strings of equal length is the number of bits at which the corresponding values are different."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="hamming",
            is_elementwise=True,
        )

    def jaccard(self, cmp: bytes) -> pl.Expr:
        """Computes the Jaccard index between two bitsets represented by binary data slices. The Jaccard index, also known as the Jaccard similarity coefficient, is a statistic used for gauging the similarity and diversity of sample sets."""
        return register_plugin_function(
            plugin_path=__resource__,
            args=[self._expr],
            kwargs={"cmp": cmp},
            function_name="jaccard",
            is_elementwise=True,
        )


class Expr(pl.Expr):
    @property
    def spatial(self) -> Spatial:
        return Spatial(self)

    @property
    def prob(self) -> Probability:
        return Probability(self)

    @property
    def binary(self) -> Binary:
        return Binary(self)


class SimSIMD(Column):
    def __call__(
        self,
        name: str | PolarsDataType | Iterable[str] | Iterable[PolarsDataType],
        *more_names: str | PolarsDataType,
    ) -> Expr: ...

    def __getattr__(self, name: str) -> pl.Expr: ...

    def spatial(self) -> Spatial: ...
    def prob(self) -> Probability: ...
    def binary(self) -> Binary: ...


col = cast(SimSIMD, pl.col)


__all__ = ["col", "SimSIMD", "Spatial", "Probability", "Binary"]
