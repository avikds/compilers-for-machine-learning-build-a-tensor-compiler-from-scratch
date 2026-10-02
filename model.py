"""
Compilers for Machine Learning: Build a Tensor Compiler from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - UOp
import sys
import math
from enum import Enum, auto
import numpy as np

sys.setrecursionlimit(100000)
INF = float("inf")


class DType:
    def __init__(self, name, c_name, np_type):
        self.name, self.c_name, self.np = name, c_name, np_type

    def __repr__(self):
        return "dtypes." + self.name


class dtypes:
    bool = DType("bool", "int", np.int32)
    int32 = DType("int32", "int", np.int32)
    float32 = DType("float32", "float", np.float32)


class Ops(Enum):
    CONST = auto()
    PARAM = auto()
    BUFFER = auto()

    ADD = auto()
    MUL = auto()
    MAX = auto()
    CMPLT = auto()
    AND = auto()
    IDIV = auto()
    MOD = auto()

    RECIP = auto()
    EXP2 = auto()
    LOG2 = auto()
    SQRT = auto()
    CAST = auto()
    WHERE = auto()

    RESHAPE = auto()
    EXPAND = auto()
    PERMUTE = auto()
    FLIP = auto()
    PAD = auto()
    SHRINK = auto()
    REDUCE_AXIS = auto()

    RANGE = auto()
    SPECIAL = auto()
    INDEX = auto()
    LOAD = auto()
    STORE = auto()
    DEFINE_ACC = auto()
    SINK = auto()


BINARY = {
    Ops.ADD,
    Ops.MUL,
    Ops.MAX,
    Ops.CMPLT,
    Ops.AND,
    Ops.IDIV,
    Ops.MOD,
}

UNARY = {
    Ops.RECIP,
    Ops.EXP2,
    Ops.LOG2,
    Ops.SQRT,
    Ops.CAST,
}

ALU = BINARY | UNARY | {Ops.WHERE}

MOVEMENT = {
    Ops.RESHAPE,
    Ops.EXPAND,
    Ops.PERMUTE,
    Ops.FLIP,
    Ops.PAD,
    Ops.SHRINK,
}


class UOp:
    __slots__ = ("op", "dtype", "src", "arg")
    _cache = {}

    def __new__(cls, op, dtype=None, src=(), arg=None):
        src = tuple(src)
        key = (op, dtype, src, arg)

        cached = cls._cache.get(key)
        if cached is not None:
            return cached

        self = super().__new__(cls)
        self.op = op
        self.dtype = dtype
        self.src = src
        self.arg = arg

        cls._cache[key] = self
        return self

    def __init__(self, op, dtype=None, src=(), arg=None):
        # Fields are initialized in __new__ because UOps are hash-consed.
        pass

    def __repr__(self):
        src = ", ".join(repr(x) for x in self.src)
        return f"UOp({self.op.name}, {self.dtype!r}, ({src}), arg={self.arg!r})"

    @staticmethod
    def const(dtype, v):
        if dtype is dtypes.bool:
            v = bool(v)
        elif dtype is dtypes.int32:
            v = int(v)
        elif dtype is dtypes.float32:
            v = float(v)
        else:
            raise TypeError(f"Unsupported dtype: {dtype!r}")

        return UOp(Ops.CONST, dtype, (), v)

    @staticmethod
    def range(n, i):
        return UOp(
            Ops.RANGE,
            dtypes.int32,
            (UOp.const(dtypes.int32, n),),
            i,
        )

    def alu(self, op, *src):
        src = tuple(
            value if isinstance(value, UOp)
            else UOp.const(self.dtype, value)
            for value in src
        )

        out_dtype = (
            dtypes.bool
            if op in {Ops.CMPLT, Ops.AND}
            else self.dtype
        )

        return UOp(op, out_dtype, (self,) + src)

    def __add__(self, other):
        return self.alu(Ops.ADD, other)

    def __radd__(self, other):
        if not isinstance(other, UOp):
            other = UOp.const(self.dtype, other)
        return UOp(Ops.ADD, self.dtype, (other, self))

    def __mul__(self, other):
        return self.alu(Ops.MUL, other)

    def __rmul__(self, other):
        if not isinstance(other, UOp):
            other = UOp.const(self.dtype, other)
        return UOp(Ops.MUL, self.dtype, (other, self))

    def __floordiv__(self, other):
        return self.alu(Ops.IDIV, other)

    def __mod__(self, other):
        return self.alu(Ops.MOD, other)

    def __lt__(self, other):
        return self.alu(Ops.CMPLT, other)

    def __and__(self, other):
        return self.alu(Ops.AND, other)

    def __neg__(self):
        # Constants are folded directly.
        if self.op is Ops.CONST:
            return UOp.const(self.dtype, -self.arg)

        return self * UOp.const(self.dtype, -1)

    def __sub__(self, other):
        return self + (-other)

    def maximum(self, other):
        return self.alu(Ops.MAX, other)

    def recip(self):
        return UOp(Ops.RECIP, self.dtype, (self,))

    def exp2(self):
        return UOp(Ops.EXP2, self.dtype, (self,))

    def log2(self):
        return UOp(Ops.LOG2, self.dtype, (self,))

    def sqrt(self):
        return UOp(Ops.SQRT, self.dtype, (self,))

    def cast(self, dtype):
        return UOp(Ops.CAST, dtype, (self,))

    def where(self, a, b):
        # Both branches are numeric: use float32.
        if not isinstance(a, UOp) and not isinstance(b, UOp):
            a = UOp.const(dtypes.float32, a)
            b = UOp.const(dtypes.float32, b)

        # One branch is numeric: use the dtype of the other branch.
        elif not isinstance(a, UOp):
            a = UOp.const(b.dtype, a)

        elif not isinstance(b, UOp):
            b = UOp.const(a.dtype, b)

        return UOp(Ops.WHERE, a.dtype, (self, a, b))

    def toposort(self):
        """
        Return all nodes reachable from self in topological order.

        Every node appears after all of its source nodes, with self last.
        Uses an explicit stack instead of recursion.
        """
        visited = set()
        result = []
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()

            if node in visited:
                continue

            if expanded:
                visited.add(node)
                result.append(node)
                continue

            stack.append((node, True))

            for src in reversed(node.src):
                if src not in visited:
                    stack.append((src, False))

        return result

