from doufo import dataclass
import typing


@dataclass
class EfficiencyMap:
    data: typing.Any
    center: typing.List[int]
    size: typing.List[int]


@dataclass
class InverseEfficiencyMap:
    data: typing.Any
    center: typing.List[int]
    size: typing.List[int]
