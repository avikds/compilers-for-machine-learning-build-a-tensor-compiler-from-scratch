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

# Step 2 - bounds
import functools

@functools.lru_cache(maxsize=None)
def bounds(u):
    """
    Return (vmin, vmax) for integer and boolean expressions.

    Boolean values are represented numerically as 0 or 1.
    Floating-point expressions and unknown operations are unbounded.
    """
    # Floating-point values and unknown dtypes have no useful bounds.
    if u.dtype not in (dtypes.int32, dtypes.bool):
        return -INF, INF

    # Constants.
    if u.op is Ops.CONST:
        if u.dtype is dtypes.bool:
            v = int(bool(u.arg))
            return v, v
        return u.arg, u.arg

    # Range-like indices are always [0, n - 1].
    if u.op in (Ops.RANGE, Ops.SPECIAL):
        n = u.src[0].arg
        return 0, n - 1

    # Addition.
    if u.op is Ops.ADD:
        a0, a1 = bounds(u.src[0])
        b0, b1 = bounds(u.src[1])
        return a0 + b0, a1 + b1

    # Multiplication.
    if u.op is Ops.MUL:
        a0, a1 = bounds(u.src[0])
        b0, b1 = bounds(u.src[1])

        products = []
        for a in (a0, a1):
            for b in (b0, b1):
                # Skip 0 * inf, which would otherwise become NaN.
                if (a == 0 and math.isinf(b)) or (b == 0 and math.isinf(a)):
                    continue
                products.append(a * b)

        if not products:
            return 0, 0

        return min(products), max(products)

    # Maximum.
    if u.op is Ops.MAX:
        a0, a1 = bounds(u.src[0])
        b0, b1 = bounds(u.src[1])
        return max(a0, b0), max(a1, b1)

    # Integer floor division by a positive constant.
    if u.op is Ops.IDIV:
        a0, a1 = bounds(u.src[0])
        c = u.src[1].arg

        if not isinstance(c, (int, np.integer)) or c <= 0:
            return -INF, INF

        vmin = -INF if a0 == -INF else math.floor(a0 / c)
        vmax = INF if a1 == INF else math.floor(a1 / c)

        return vmin, vmax

    # Modulo by a positive constant.
    if u.op is Ops.MOD:
        a0, a1 = bounds(u.src[0])
        c = u.src[1].arg

        if not isinstance(c, (int, np.integer)) or c <= 0:
            return -INF, INF

        # Already inside [0, c - 1], so modulo does not change the interval.
        if 0 <= a0 and a1 < c:
            return a0, a1

        return 0, c - 1

    # Less-than comparison.
    if u.op is Ops.CMPLT:
        a0, a1 = bounds(u.src[0])
        b0, b1 = bounds(u.src[1])

        if a1 < b0:
            return 1, 1

        if a0 >= b1:
            return 0, 0

        return 0, 1

    # Logical AND.
    if u.op is Ops.AND:
        a0, a1 = bounds(u.src[0])
        b0, b1 = bounds(u.src[1])

        if a0 == 1 and b0 == 1:
            return 1, 1

        if a1 == 0 or b1 == 0:
            return 0, 0

        return 0, 1

    # Conditional expression.
    if u.op is Ops.WHERE:
        c0, c1 = bounds(u.src[0])

        # Condition is definitely true.
        if c0 == 1 and c1 == 1:
            return bounds(u.src[1])

        # Condition is definitely false.
        if c0 == 0 and c1 == 0:
            return bounds(u.src[2])

        # Unknown condition: union the two branch intervals.
        a0, a1 = bounds(u.src[1])
        b0, b1 = bounds(u.src[2])
        return min(a0, b0), max(a1, b1)

    # Casting preserves bounds when casting an integer/bool expression.
    if u.op is Ops.CAST:
        src = u.src[0]

        if src.dtype in (dtypes.int32, dtypes.bool):
            return bounds(src)

        return -INF, INF

    # All other operations are currently unknown to the bounds analysis.
    return -INF, INF

# Step 3 - UPat
_MISSING = object()

class UPat:
    def __init__(self, op=None, dtype=None, src=None, arg=_MISSING, name=None):
        self.op = op
        self.dtype = dtype
        self.src = src
        self.arg = arg
        self.name = name

    @staticmethod
    def var(name=None, dtype=None):
        return UPat(dtype=dtype, name=name)

    @staticmethod
    def cvar(name=None, dtype=None):
        return UPat(Ops.CONST, name=name, dtype=dtype)

    def match(self, u, store):
        """
        Match this pattern against u.

        On success, bindings are written into store.
        Repeated names must refer to the exact same UOp object.
        """
        original = dict(store)

        # Operation constraint.
        if self.op is not None:
            if isinstance(self.op, set):
                if u.op not in self.op:
                    store.clear()
                    store.update(original)
                    return False
            elif u.op is not self.op:
                store.clear()
                store.update(original)
                return False

        # DType constraint.
        if self.dtype is not None and u.dtype is not self.dtype:
            store.clear()
            store.update(original)
            return False

        # Argument constraint.
        if self.arg is not _MISSING and u.arg != self.arg:
            store.clear()
            store.update(original)
            return False

        # Bind the node by name.
        if self.name is not None:
            if self.name in store:
                if store[self.name] is not u:
                    store.clear()
                    store.update(original)
                    return False
            else:
                store[self.name] = u

        # Source constraint.
        if self.src is not None:
            if len(u.src) != len(self.src):
                store.clear()
                store.update(original)
                return False

            for pat, src in zip(self.src, u.src):
                if not pat.match(src, store):
                    store.clear()
                    store.update(original)
                    return False

        return True


class PatternMatcher:
    def __init__(self, patterns):
        self.patterns = list(patterns)

    def __add__(self, other):
        return PatternMatcher(self.patterns + other.patterns)

    def rewrite(self, u):
        """
        Try each rewrite rule in order.

        A rule applies only when its pattern matches and its function returns
        a replacement that is neither None nor the original UOp.
        """
        for pattern, fn in self.patterns:
            store = {}

            if not pattern.match(u, store):
                continue

            result = fn(**store)

            if result is not None and result is not u:
                return result

        return None

# Step 4 - graph_rewrite
def graph_rewrite(root, pm):
    """
    Rewrite a UOp graph bottom-up to a fixed point.

    Each original node is memoized so shared subgraphs are rewritten only once.
    Newly created replacement nodes are also recursively rewritten when they
    have sources.
    """
    memo = {}

    def rw(u):
        if u in memo:
            return memo[u]

        # Rewrite all sources first.
        new_src = tuple(rw(src) for src in u.src)

        # Rebuild only when at least one source changed.
        current = u if new_src == u.src else UOp(u.op, u.dtype, new_src, u.arg)

        # Repeatedly apply rewrite rules until reaching a fixed point.
        for _ in range(1000):
            r = pm.rewrite(current)

            # No rule applies.
            if r is None:
                break

            # A rewrite that maps the node back to itself is already a fixed
            # point, so stop.
            if r is current:
                break

            # A replacement may contain sources that have not yet been
            # rewritten. Run the same bottom-up rewrite on them.
            if r.src:
                r_rewritten = rw(r)
            else:
                r_rewritten = r

            # Stop if processing the replacement leads back to the current
            # node.
            if r_rewritten is current:
                break

            current = r_rewritten

        # Memoize the original node, not just the rewritten node.
        memo[u] = current
        return current

    return rw(root)

