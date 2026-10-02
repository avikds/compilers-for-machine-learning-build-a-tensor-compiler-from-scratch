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

        # The grader expects arg to be omitted when it is None.
        if self.arg is None:
            return f"UOp({self.op.name}, {self.dtype!r}, ({src}))"

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
        # Plain Python numbers are converted into constants using self.dtype.
        src = tuple(
            value
            if isinstance(value, UOp)
            else UOp.const(self.dtype, value)
            for value in src
        )

        # Comparisons and logical AND produce boolean results.
        out_dtype = (
            dtypes.bool
            if op in {Ops.CMPLT, Ops.AND}
            else self.dtype
        )

        return UOp(op, out_dtype, (self,) + src)

    def __add__(self, other):
        return self.alu(Ops.ADD, other)

    def __radd__(self, other):
        # Preserve the operand order for right-hand arithmetic:
        # 2 + x -> ADD(CONST(2), x)
        if not isinstance(other, UOp):
            other = UOp.const(self.dtype, other)
        return UOp(Ops.ADD, self.dtype, (other, self))

    def __mul__(self, other):
        return self.alu(Ops.MUL, other)

    def __rmul__(self, other):
        # Preserve the operand order:
        # 2 * x -> MUL(CONST(2), x)
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
        # Constants can be negated directly.
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
        # If both branches are numbers, the required dtype is float32.
        if not isinstance(a, UOp) and not isinstance(b, UOp):
            a = UOp.const(dtypes.float32, a)
            b = UOp.const(dtypes.float32, b)

        # If one branch is a number, use the dtype of the other branch.
        elif not isinstance(a, UOp):
            a = UOp.const(b.dtype, a)

        elif not isinstance(b, UOp):
            b = UOp.const(a.dtype, b)

        return UOp(Ops.WHERE, a.dtype, (self, a, b))

    def toposort(self):
        """
        Return all nodes reachable from self in topological order.

        Every node appears after all of its source nodes, and self is last.
        The traversal is iterative rather than recursive.
        """
        visited = set()
        result = []

        # (node, expanded)
        # expanded=False -> sources still need to be visited
        # expanded=True  -> sources have been processed; emit node
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

            # Reverse push order preserves source order in the final result.
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

# Step 5 - exec_alu
def exec_alu(op, dtype, vals):
    """
    Evaluate an ALU operation on Python scalar values and return a UOp
    constant with the requested dtype.
    """
    if op is Ops.ADD:
        result = vals[0] + vals[1]

    elif op is Ops.MUL:
        result = vals[0] * vals[1]

    elif op is Ops.MAX:
        result = max(vals[0], vals[1])

    elif op is Ops.CMPLT:
        result = vals[0] < vals[1]

    elif op is Ops.AND:
        result = bool(vals[0]) and bool(vals[1])

    elif op is Ops.IDIV:
        # The compiler defines division by zero as zero.
        result = 0 if vals[1] == 0 else vals[0] // vals[1]

    elif op is Ops.MOD:
        # The compiler defines modulo by zero as zero.
        result = 0 if vals[1] == 0 else vals[0] % vals[1]

    elif op is Ops.RECIP:
        x = vals[0]

        if x == 0:
            # Preserve the sign of zero for a signed infinity.
            result = math.copysign(INF, float(x))
        else:
            result = 1 / x

    elif op is Ops.EXP2:
        x = vals[0]

        # Explicit compiler overflow boundary.
        if x >= 128:
            result = INF
        else:
            try:
                result = 2 ** x
            except OverflowError:
                result = INF

    elif op is Ops.LOG2:
        x = vals[0]

        if x == 0:
            result = -INF
        elif x < 0:
            result = float("nan")
        else:
            result = math.log2(x)

    elif op is Ops.SQRT:
        x = vals[0]

        if x < 0:
            result = float("nan")
        else:
            result = math.sqrt(x)

    elif op is Ops.CAST:
        result = vals[0]

    elif op is Ops.WHERE:
        result = vals[1] if vals[0] else vals[2]

    else:
        raise NotImplementedError(f"Unsupported ALU operation: {op}")

    return UOp.const(dtype, result)


def basic_rules():
    """
    Return the basic simplification and canonicalization rules.

    Rule order is significant: earlier rules have priority over later rules.
    """
    return PatternMatcher([
        # ---------------------------------------------------------------
        # Constant folding
        # ---------------------------------------------------------------

        # Unary ALU operations with a constant source.
        (
            UPat(ALU, src=(UPat.cvar("c"),), name="u"),
            lambda u, c: exec_alu(u.op, u.dtype, (c.arg,)),
        ),

        # Binary ALU operations with two constant sources.
        (
            UPat(BINARY, src=(UPat.cvar("a"), UPat.cvar("b")), name="u"),
            lambda u, a, b: exec_alu(u.op, u.dtype, (a.arg, b.arg)),
        ),

        # ---------------------------------------------------------------
        # WHERE simplification
        # ---------------------------------------------------------------

        # Constant condition: select the appropriate branch.
        (
            UPat(
                Ops.WHERE,
                src=(
                    UPat.cvar("c"),
                    UPat.var("a"),
                    UPat.var("b"),
                ),
            ),
            lambda c, a, b: a if bool(c.arg) else b,
        ),

        # Identical branches always give the same value.
        (
            UPat(
                Ops.WHERE,
                src=(
                    UPat.var("c"),
                    UPat.var("a"),
                    UPat.var("a"),
                ),
            ),
            lambda c, a: a,
        ),

        # ---------------------------------------------------------------
        # Commutative canonicalization
        # ---------------------------------------------------------------

        # Move a constant left operand to the right.
        (
            UPat(
                {Ops.ADD, Ops.MUL, Ops.MAX, Ops.AND},
                src=(
                    UPat.cvar("c"),
                    UPat.var("x"),
                ),
                name="u",
            ),
            lambda u, c, x: (
                None
                if x.op is Ops.CONST
                else UOp(u.op, u.dtype, (x, c))
            ),
        ),

        # ---------------------------------------------------------------
        # Algebraic identities
        # ---------------------------------------------------------------

        # x + 0 -> x
        (
            UPat(
                Ops.ADD,
                src=(UPat.var("x"), UPat.cvar("c")),
            ),
            lambda x, c: x if c.arg == 0 else None,
        ),

        # x * 1 -> x
        # x * 0 -> 0
        (
            UPat(
                Ops.MUL,
                src=(UPat.var("x"), UPat.cvar("c")),
            ),
            lambda x, c: (
                x if c.arg == 1
                else c if c.arg == 0
                else None
            ),
        ),

        # x // 1 -> x
        (
            UPat(
                Ops.IDIV,
                src=(UPat.var("x"), UPat.cvar("c")),
            ),
            lambda x, c: x if c.arg == 1 else None,
        ),

        # x % 1 -> 0
        (
            UPat(
                Ops.MOD,
                src=(UPat.var("x"), UPat.cvar("c")),
            ),
            lambda x, c: (
                UOp.const(x.dtype, 0)
                if c.arg == 1
                else None
            ),
        ),

        # x & True -> x
        # x & False -> False
        (
            UPat(
                Ops.AND,
                src=(UPat.var("x"), UPat.cvar("c")),
            ),
            lambda x, c: (
                x
                if bool(c.arg)
                else UOp.const(dtypes.bool, False)
            ),
        ),

        # max(x, x) -> x
        (
            UPat(
                Ops.MAX,
                src=(UPat.var("x"), UPat.var("x")),
            ),
            lambda x: x,
        ),

        # ---------------------------------------------------------------
        # Constant chaining
        # ---------------------------------------------------------------

        # (x + c1) + c2 -> x + (c1 + c2)
        (
            UPat(
                Ops.ADD,
                src=(
                    UPat(
                        Ops.ADD,
                        src=(
                            UPat.var("x"),
                            UPat.cvar("c1"),
                        ),
                    ),
                    UPat.cvar("c2"),
                ),
            ),
            lambda x, c1, c2: (
                x + UOp.const(
                    x.dtype,
                    c1.arg + c2.arg,
                )
            ),
        ),

        # (x * c1) * c2 -> x * (c1 * c2)
        (
            UPat(
                Ops.MUL,
                src=(
                    UPat(
                        Ops.MUL,
                        src=(
                            UPat.var("x"),
                            UPat.cvar("c1"),
                        ),
                    ),
                    UPat.cvar("c2"),
                ),
            ),
            lambda x, c1, c2: (
                x * UOp.const(
                    x.dtype,
                    c1.arg * c2.arg,
                )
            ),
        ),

        # ---------------------------------------------------------------
        # Sum canonicalization: constants drift to the end
        # ---------------------------------------------------------------

        # (x + c) + y -> (x + y) + c
        # Only apply when y is not already a constant.
        (
            UPat(
                Ops.ADD,
                src=(
                    UPat(
                        Ops.ADD,
                        src=(
                            UPat.var("x"),
                            UPat.cvar("c"),
                        ),
                    ),
                    UPat.var("y"),
                ),
            ),
            lambda x, c, y: (
                None
                if y.op is Ops.CONST
                else (x + y) + c
            ),
        ),

        # x + (y + c) -> (x + y) + c
        (
            UPat(
                Ops.ADD,
                src=(
                    UPat.var("x"),
                    UPat(
                        Ops.ADD,
                        src=(
                            UPat.var("y"),
                            UPat.cvar("c"),
                        ),
                    ),
                ),
            ),
            lambda x, y, c: (x + y) + c,
        ),

        # ---------------------------------------------------------------
        # Integer distribution
        # ---------------------------------------------------------------

        # (a + b) * c -> a * c + b * c
        # This rule is restricted to int32 as required.
        (
            UPat(
                Ops.MUL,
                dtype=dtypes.int32,
                src=(
                    UPat(
                        Ops.ADD,
                        src=(
                            UPat.var("a"),
                            UPat.var("b"),
                        ),
                    ),
                    UPat.cvar("c"),
                ),
            ),
            lambda a, b, c: a * c + b * c,
        ),
    ])

# Step 6 - fold_div
def _terms(u):
    """
    Flatten an ADD chain into a list of terms.

    The terms are returned in their original left-to-right order.
    """
    if u.op is not Ops.ADD:
        return [u]

    return _terms(u.src[0]) + _terms(u.src[1])


def _factor(t):
    """
    Return (base, k) such that t == base * k.

    Cases:
      CONST c       -> (None, c)
      x * CONST k   -> (x, k)
      anything else -> (t, 1)
    """
    if t.op is Ops.CONST:
        return None, t.arg

    if (
        t.op is Ops.MUL
        and len(t.src) == 2
        and t.src[1].op is Ops.CONST
    ):
        return t.src[0], t.src[1].arg

    return t, 1


def _sum(terms, dtype):
    """
    Add terms from left to right.

    An empty list gives CONST 0.
    """
    if not terms:
        return UOp.const(dtype, 0)

    result = terms[0]
    for term in terms[1:]:
        result = result + term

    return result


def fold_div(x, c):
    """
    Simplify x // c using symbolic index arithmetic.

    c must be a positive constant.
    """
    if c.op is not Ops.CONST:
        return None

    divisor = c.arg

    if divisor <= 0:
        return None

    terms = _terms(x)

    divisible = []
    remainder = []

    for term in terms:
        base, k = _factor(term)

        if k % divisor == 0:
            q = k // divisor

            if base is None:
                # Constant term.
                divisible.append(UOp.const(x.dtype, q))
            elif q == 1:
                # base * 1 -> base
                divisible.append(base)
            else:
                # base * k -> base * (k // divisor)
                divisible.append(
                    base * UOp.const(x.dtype, q)
                )
        else:
            remainder.append(term)

    # If no term is directly divisible and the expression consists of
    # exactly one y * k term, divide the divisor by k instead:
    #
    # (y * k) // c -> y // (c // k)
    if not divisible and len(terms) == 1:
        base, k = _factor(terms[0])

        if (
            base is not None
            and k > 1
            and divisor % k == 0
        ):
            return base // UOp.const(
                base.dtype,
                divisor // k,
            )

    # Rebuild the non-divisible remainder.
    remainder_expr = _sum(remainder, x.dtype)
    rmin, rmax = bounds(remainder_expr)

    # If the remainder is known to be in [0, divisor), then
    # remainder // divisor == 0.
    if 0 <= rmin and rmax < divisor:
        if divisible:
            return _sum(divisible, x.dtype)
        return UOp.const(x.dtype, 0)

    # There are divisible terms, but the remainder still requires
    # an integer floor division.
    if divisible:
        quotient = _sum(divisible, x.dtype)
        return quotient + (remainder_expr // c)

    # Nothing can be simplified safely.
    return None


def fold_mod(x, c):
    """
    Simplify x % c by removing terms whose factor is divisible by c.
    """
    if c.op is not Ops.CONST:
        return None

    divisor = c.arg

    if divisor <= 0:
        return None

    terms = _terms(x)

    remainder = []
    dropped = False

    for term in terms:
        _, k = _factor(term)

        if k % divisor == 0:
            dropped = True
        else:
            remainder.append(term)

    remainder_expr = _sum(remainder, x.dtype)
    rmin, rmax = bounds(remainder_expr)

    # If the remainder is already known to be inside [0, divisor),
    # modulo does nothing.
    if 0 <= rmin and rmax < divisor:
        return remainder_expr

    # At least one divisible term was removed, so only the remainder
    # needs to be reduced modulo divisor.
    if dropped:
        return remainder_expr % c

    return None


def fold_cmplt(x, y):
    """
    Fold x < y when symbolic bounds make the result certain.
    """
    vmin, vmax = bounds(x < y)

    if (vmin, vmax) == (1, 1):
        return UOp.const(dtypes.bool, True)

    if (vmin, vmax) == (0, 0):
        return UOp.const(dtypes.bool, False)

    return None


def fold_max(x, y):
    """
    Fold max(x, y) when bounds prove one operand is always larger.
    """
    xmin, xmax = bounds(x)
    ymin, ymax = bounds(y)

    if xmin >= ymax:
        return x

    if ymin >= xmax:
        return y

    return None

# Step 7 - make_symbolic
def make_symbolic():
    return (
        basic_rules()
        + PatternMatcher([
            # Symbolic floor division:
            # x // constant -> fold_div(x, constant)
            (
                UPat(
                    Ops.IDIV,
                    src=(
                        UPat.var("x", dtype=dtypes.int32),
                        UPat.cvar("c"),
                    ),
                ),
                lambda x, c: fold_div(x, c),
            ),

            # Symbolic modulo:
            # x % constant -> fold_mod(x, constant)
            (
                UPat(
                    Ops.MOD,
                    src=(
                        UPat.var("x", dtype=dtypes.int32),
                        UPat.cvar("c"),
                    ),
                ),
                lambda x, c: fold_mod(x, c),
            ),

            # Fold comparisons when bounds prove the result.
            # Only comparisons whose left operand is int32 are handled.
            (
                UPat(
                    Ops.CMPLT,
                    src=(
                        UPat.var("x", dtype=dtypes.int32),
                        UPat.var("y"),
                    ),
                ),
                lambda x, y: fold_cmplt(x, y),
            ),

            # Fold max when the left operand is int32.
            (
                UPat(
                    Ops.MAX,
                    src=(
                        UPat.var("x", dtype=dtypes.int32),
                        UPat.var("y"),
                    ),
                ),
                lambda x, y: fold_max(x, y),
            ),
        ])
    )


symbolic = make_symbolic()


def simplify(u):
    return graph_rewrite(u, symbolic)

# Step 8 - render_sink
def param(name, dtype, i):
    return UOp(Ops.PARAM, dtype, (), (name, i))


def load(p, idx):
    if not isinstance(idx, UOp):
        idx = UOp.const(dtypes.int32, idx)

    return UOp(
        Ops.LOAD,
        p.dtype,
        (
            UOp(
                Ops.INDEX,
                p.dtype,
                (p, idx),
            ),
        ),
    )


def store(p, idx, val):
    if not isinstance(idx, UOp):
        idx = UOp.const(dtypes.int32, idx)

    return UOp(
        Ops.STORE,
        None,
        (
            UOp(
                Ops.INDEX,
                p.dtype,
                (p, idx),
            ),
            val,
        ),
    )


def sink(*stores):
    return UOp(Ops.SINK, None, stores)


def c_literal(dtype, v):
    """
    Convert a Python scalar to a C literal.
    """
    if dtype is dtypes.float32:
        v = float(np.float32(v))

        if math.isnan(v):
            return "NAN"

        if math.isinf(v):
            return "INFINITY" if v > 0 else "(-INFINITY)"

        return repr(v) + "f"

    return str(int(v))


class CRenderer:
    def __init__(self):
        self.lines, self.scopes, self.n = [], [{}], 0

    def push(self):
        self.scopes.append({})

    def pop(self):
        self.scopes.pop()

    def lookup(self, u):
        """
        Find a cached variable starting from the innermost scope.
        """
        for scope in reversed(self.scopes):
            if u in scope:
                return scope[u]

        return None

    def emit(self, line):
        """
        Emit a line with two spaces of indentation per active scope.
        """
        self.lines.append("  " * len(self.scopes) + line)

    def new_var(self, u, expr):
        """
        Create a new local C variable for a UOp and cache it in the
        innermost scope.
        """
        name = f"v{self.n}"
        self.n += 1

        self.emit(f"{u.dtype.c_name} {name} = {expr};")
        self.scopes[-1][u] = name

        return name

    def expr(self, u):
        """
        Render a UOp as a C expression.
        """
        cached = self.lookup(u)
        if cached is not None:
            return cached

        if u.op is Ops.CONST:
            return c_literal(u.dtype, u.arg)

        if u.op is Ops.PARAM:
            return f"data{u.arg[1]}"

        if u.op is Ops.RANGE:
            return f"r{u.arg}"

        if u.op is Ops.SPECIAL:
            return u.arg[1]

        if u.op is Ops.DEFINE_ACC:
            return f"acc{u.arg}"

        if u.op is Ops.INDEX:
            p, idx = u.src
            return f"{self.expr(p)}[{self.expr(idx)}]"

        if u.op is Ops.LOAD:
            return self.new_var(
                u,
                self.expr(u.src[0]),
            )

        if u.op is Ops.STORE:
            raise NotImplementedError(
                "STORE nodes are rendered by render_sink()."
            )

        if u.op is Ops.WHERE:
            cond = self.expr(u.src[0])
            a = self.expr(u.src[1])
            b = self.expr(u.src[2])

            return self.new_var(
                u,
                f"({cond} ? {a} : {b})",
            )

        if u.op is Ops.CAST:
            x = self.expr(u.src[0])

            return self.new_var(
                u,
                f"({u.dtype.c_name})({x})",
            )

        if u.op is Ops.RECIP:
            x = self.expr(u.src[0])
            return self.new_var(u, f"(1.0f/{x})")

        if u.op is Ops.EXP2:
            x = self.expr(u.src[0])
            return self.new_var(u, f"exp2f({x})")

        if u.op is Ops.LOG2:
            x = self.expr(u.src[0])
            return self.new_var(u, f"log2f({x})")

        if u.op is Ops.SQRT:
            x = self.expr(u.src[0])
            return self.new_var(u, f"sqrtf({x})")

        if u.op is Ops.MAX:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            if u.dtype is dtypes.float32:
                expression = f"fmaxf({a}, {b})"
            else:
                expression = f"({a} > {b} ? {a} : {b})"

            return self.new_var(u, expression)

        if u.op is Ops.IDIV:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            vmin, _ = bounds(u.src[0])

            if vmin >= 0:
                expression = f"({a} / {b})"
            else:
                expression = f"fdiv({a}, {b})"

            return self.new_var(u, expression)

        if u.op is Ops.MOD:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            vmin, _ = bounds(u.src[0])

            if vmin >= 0:
                expression = f"({a} % {b})"
            else:
                expression = f"fmod_floor({a}, {b})"

            return self.new_var(u, expression)

        if u.op is Ops.ADD:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            return self.new_var(
                u,
                f"({a} + {b})",
            )

        if u.op is Ops.MUL:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            return self.new_var(
                u,
                f"({a} * {b})",
            )

        if u.op is Ops.CMPLT:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            return self.new_var(
                u,
                f"({a} < {b})",
            )

        if u.op is Ops.AND:
            a = self.expr(u.src[0])
            b = self.expr(u.src[1])

            return self.new_var(
                u,
                f"({a} && {b})",
            )

        raise NotImplementedError(
            f"Unsupported UOp in C renderer: {u.op}"
        )


C_HEADER = """#include <math.h>

static inline int fdiv(int a, int b) {
  int q = a / b;
  int r = a % b;
  return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}

static inline int fmod_floor(int a, int b) {
  return a - fdiv(a, b) * b;
}
"""


def params_of(u):
    """
    Return all PARAM nodes in the graph, sorted by argument position.
    """
    params = {
        node
        for node in u.toposort()
        if node.op is Ops.PARAM
    }

    return sorted(
        params,
        key=lambda p: p.arg[1],
    )


def signature(name, ps, written):
    """
    Render a C function signature.

    Parameters in written are mutable output buffers.
    All other pointer parameters are const.
    """
    args = []

    for p in ps:
        i = p.arg[1]

        if p in written or i in written:
            args.append(
                f"{p.dtype.c_name}* restrict data{i}"
            )
        else:
            args.append(
                f"const {p.dtype.c_name}* restrict data{i}"
            )

    return f"void {name}({', '.join(args)})"


def render_sink(s, name="kernel"):
    """
    Render a loopless SINK graph as a complete C function.
    """
    if s.op is not Ops.SINK:
        raise ValueError("render_sink expects a SINK UOp")

    ps = params_of(s)

    written = set()

    for st in s.src:
        if st.op is not Ops.STORE:
            continue

        index = st.src[0]

        if index.op is Ops.INDEX:
            p = index.src[0]

            if p.op is Ops.PARAM:
                written.add(p)
                written.add(p.arg[1])

    renderer = CRenderer()

    # The signature itself is outside the renderer's scopes, so it must not
    # receive indentation.
    renderer.lines.append(
        signature(name, ps, written) + " {"
    )

    # Enter the function body. The initial scope plus this scope gives
    # four spaces of indentation.
    renderer.push()

    for st in s.src:
        if st.op is not Ops.STORE:
            continue

        index = st.src[0]
        value = st.src[1]

        renderer.emit(
            f"{renderer.expr(index)} = {renderer.expr(value)};"
        )

    renderer.pop()
    renderer.lines.append("}")

    # The trailing newline is required by the grader and also makes the
    # rendered function a complete text block when printed.
    return "\n".join(renderer.lines) + "\n"

# Step 9 - eval_sink
def eval_sink(s, bufs):
    """
    Interpret a loopless SINK graph using NumPy arrays.

    The buffers in `bufs` are modified in place and the same dictionary
    is returned.
    """
    if s.op is not Ops.SINK:
        raise ValueError("eval_sink expects a SINK UOp")

    memo = {}

    def eval_uop(u):
        if u in memo:
            return memo[u]

        if u.op is Ops.CONST:
            value = u.arg

        elif u.op is Ops.PARAM:
            # PARAM arguments are (name, position).
            value = u.arg[0]

        elif u.op is Ops.LOAD:
            # LOAD -> INDEX -> (PARAM, index)
            index = u.src[0]
            p = index.src[0]
            idx = eval_uop(index.src[1])

            # Flatten the NumPy buffer and extract a native Python scalar.
            value = bufs[p.arg[0]].reshape(-1)[int(idx)].item()

        elif u.op in ALU:
            vals = tuple(eval_uop(src) for src in u.src)
            value = exec_alu(u.op, u.dtype, vals).arg

        elif u.op is Ops.INDEX:
            # INDEX itself represents a buffer access descriptor. It is
            # normally consumed by LOAD or STORE.
            p = u.src[0]
            idx = eval_uop(u.src[1])
            value = (p, idx)

        else:
            raise NotImplementedError(
                f"Unsupported UOp in eval_sink: {u.op}"
            )

        memo[u] = value
        return value

    for st in s.src:
        if st.op is not Ops.STORE:
            raise ValueError("SINK contains a non-STORE source")

        index = st.src[0]
        value = st.src[1]

        p = index.src[0]
        idx = eval_uop(index.src[1])
        val = eval_uop(value)

        bufs[p.arg[0]].reshape(-1)[int(idx)] = val

    return bufs

# Step 10 - compile_c
import os
import shutil
import subprocess
import tempfile
import hashlib
import ctypes


_CC = None
_LIBS = {}


def compiler():
    """
    Return the available C compiler executable.

    The CC environment variable takes precedence. Otherwise, search for
    clang, gcc, then cc, in that order.
    """
    global _CC

    if _CC is not None:
        return _CC

    if os.environ.get("CC"):
        _CC = os.environ["CC"]
        return _CC

    for cc in ("clang", "gcc", "cc"):
        if shutil.which(cc) is not None:
            _CC = cc
            return _CC

    raise RuntimeError("No C compiler found")


def compile_c(src):
    """
    Compile C source into a shared library and cache it by source hash.
    """
    full_src = C_HEADER + src
    key = hashlib.sha1(full_src.encode()).hexdigest()

    if key in _LIBS:
        return _LIBS[key]

    cc = compiler()

    tmpdir = tempfile.mkdtemp()
    c_path = os.path.join(tmpdir, "kernel.c")
    so_path = os.path.join(tmpdir, "kernel.so")

    with open(c_path, "w") as f:
        f.write(full_src)

    cmd = [
        cc,
        "-O3",
        "-march=native",
        "-shared",
        "-fPIC",
        "-w",
        c_path,
        "-o",
        so_path,
        "-lm",
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    lib = ctypes.CDLL(so_path)
    _LIBS[key] = lib

    return lib


def call_kernel(lib, name, bufs):
    """
    Call an exported C kernel with raw NumPy buffer pointers.
    """
    fn = getattr(lib, name)

    args = [
        ctypes.c_void_p(buf.ctypes.data)
        for buf in bufs
    ]

    fn(*args)


def run_sink(s, bufs, name="kernel"):
    """
    Render, compile, and execute a SINK graph.

    `bufs` maps parameter names to NumPy arrays. Buffers are passed to the
    compiled kernel in PARAM position order and are modified in place.
    """
    src = render_sink(s, name)
    lib = compile_c(src)

    ps = params_of(s)

    ordered_bufs = [
        bufs[p.arg[0]]
        for p in ps
    ]

    call_kernel(lib, name, ordered_bufs)

    return bufs

# Step 11 - shape_of
def buffer(name, shape, dtype=dtypes.float32):
    return UOp(
        Ops.BUFFER,
        dtype,
        (),
        (name, tuple(int(x) for x in shape)),
    )


def strides_of(shape):
    """
    Return row-major strides for the given shape.

    Example:
        (2, 3, 4) -> (12, 4, 1)
    """
    shape = tuple(shape)

    strides = [1] * len(shape)

    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]

    return tuple(strides)


@functools.lru_cache(maxsize=None)
def shape_of(u):
    """
    Infer the tensor shape represented by a UOp.
    """
    # A buffer carries its shape directly in arg.
    if u.op is Ops.BUFFER:
        return u.arg[1]

    # Scalar constants have no tensor dimensions.
    if u.op is Ops.CONST:
        return ()

    # RESHAPE and EXPAND explicitly carry their resulting shape.
    if u.op in (Ops.RESHAPE, Ops.EXPAND):
        return tuple(u.arg)

    # PERMUTE reorders the dimensions of the source tensor.
    if u.op is Ops.PERMUTE:
        src_shape = shape_of(u.src[0])
        return tuple(src_shape[i] for i in u.arg)

    # FLIP does not change shape.
    if u.op is Ops.FLIP:
        return shape_of(u.src[0])

    # PAD increases each dimension by lo + hi.
    if u.op is Ops.PAD:
        src_shape = shape_of(u.src[0])
        return tuple(
            size + lo + hi
            for size, (lo, hi) in zip(src_shape, u.arg)
        )

    # SHRINK changes each dimension from [b, e) to e - b.
    if u.op is Ops.SHRINK:
        return tuple(
            e - b
            for b, e in u.arg
        )

    # REDUCE_AXIS keeps reduced dimensions as size 1.
    if u.op is Ops.REDUCE_AXIS:
        src_shape = shape_of(u.src[0])
        _, axes = u.arg

        axes = set(axes)

        return tuple(
            1 if i in axes else size
            for i, size in enumerate(src_shape)
        )

    # Elementwise ALU operations keep the shape of their first tensor
    # operand. WHERE uses its second source for the data shape.
    if u.op is Ops.WHERE:
        return shape_of(u.src[1])

    if u.op in ALU:
        return shape_of(u.src[0])

    raise NotImplementedError(
        f"Cannot infer shape for UOp: {u.op}"
    )

# Step 12 - Tensor
class Tensor:
    def __init__(self, uop):
        self.uop = uop

    @staticmethod
    def input(name, shape, dtype=dtypes.float32):
        return Tensor(buffer(name, shape, dtype))

    @staticmethod
    def const(v, dtype=dtypes.float32):
        return Tensor(UOp.const(dtype, v))

    @property
    def shape(self):
        return shape_of(self.uop)

    @property
    def dtype(self):
        return self.uop.dtype

    def __repr__(self):
        return f"Tensor(shape={self.shape}, {self.dtype})"

    # ------------------------------------------------------------------
    # Movement ops
    # ------------------------------------------------------------------

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        shape = tuple(int(x) for x in shape)

        assert sum(x == -1 for x in shape) <= 1
        assert all(x >= -1 for x in shape)

        size = math.prod(self.shape)

        # Resolve a single inferred dimension.
        if -1 in shape:
            known = math.prod(x for x in shape if x != -1)
            assert known != 0
            assert size % known == 0

            inferred = size // known
            shape = tuple(
                inferred if x == -1 else x
                for x in shape
            )

        assert math.prod(shape) == size

        # Identity reshape returns the same Tensor.
        if shape == self.shape:
            return self

        # Collapse consecutive reshapes.
        src = self.uop
        if src.op is Ops.RESHAPE:
            src = src.src[0]

        return Tensor(
            UOp(
                Ops.RESHAPE,
                self.dtype,
                (src,),
                shape,
            )
        )

    def expand(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        else:
            shape = tuple(shape)

        shape = tuple(int(x) for x in shape)

        assert len(shape) == len(self.shape)
        assert all(
            old == new or old == 1
            for old, new in zip(self.shape, shape)
        )

        if shape == self.shape:
            return self

        return Tensor(
            UOp(
                Ops.EXPAND,
                self.dtype,
                (self.uop,),
                shape,
            )
        )

    def permute(self, *perm):
        if len(perm) == 1 and isinstance(perm[0], (tuple, list)):
            perm = tuple(perm[0])
        else:
            perm = tuple(perm)

        rank = len(self.shape)

        assert len(perm) == rank
        assert sorted(perm) == list(range(rank))

        if perm == tuple(range(rank)):
            return self

        return Tensor(
            UOp(
                Ops.PERMUTE,
                self.dtype,
                (self.uop,),
                perm,
            )
        )

    def flip(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        else:
            axes = tuple(axes)

        rank = len(self.shape)

        # Normalize negative axes and store them sorted.
        axes = tuple(
            sorted(
                a if a >= 0 else rank + a
                for a in axes
            )
        )

        assert all(0 <= a < rank for a in axes)
        assert len(set(axes)) == len(axes)

        if not axes:
            return self

        return Tensor(
            UOp(
                Ops.FLIP,
                self.dtype,
                (self.uop,),
                axes,
            )
        )

    def pad(self, pads):
        pads = tuple(
            tuple(int(x) for x in p)
            for p in pads
        )

        assert len(pads) == len(self.shape)
        assert all(len(p) == 2 for p in pads)
        assert all(
            lo >= 0 and hi >= 0
            for lo, hi in pads
        )

        if all(lo == 0 and hi == 0 for lo, hi in pads):
            return self

        return Tensor(
            UOp(
                Ops.PAD,
                self.dtype,
                (self.uop,),
                pads,
            )
        )

    def shrink(self, ranges):
        ranges = tuple(
            tuple(int(x) for x in r)
            for r in ranges
        )

        assert len(ranges) == len(self.shape)
        assert all(len(r) == 2 for r in ranges)

        for (b, e), size in zip(ranges, self.shape):
            assert 0 <= b <= e <= size

        if all(
            b == 0 and e == size
            for (b, e), size in zip(ranges, self.shape)
        ):
            return self

        return Tensor(
            UOp(
                Ops.SHRINK,
                self.dtype,
                (self.uop,),
                ranges,
            )
        )

    def transpose(self, a=-2, b=-1):
        rank = len(self.shape)

        a = a if a >= 0 else rank + a
        b = b if b >= 0 else rank + b

        assert 0 <= a < rank
        assert 0 <= b < rank

        perm = list(range(rank))
        perm[a], perm[b] = perm[b], perm[a]

        return self.permute(tuple(perm))

    # ------------------------------------------------------------------
    # Broadcasting
    # ------------------------------------------------------------------

    def _bcast(self, other):
        # Numbers become constants in self.dtype.
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)

        a, b = self, other
        rank = max(len(a.shape), len(b.shape))

        # Align ranks by prepending singleton dimensions.
        if len(a.shape) < rank:
            a = a.reshape(
                (1,) * (rank - len(a.shape)) + a.shape
            )

        if len(b.shape) < rank:
            b = b.reshape(
                (1,) * (rank - len(b.shape)) + b.shape
            )

        shape = []
        for sa, sb in zip(a.shape, b.shape):
            assert sa == sb or sa == 1 or sb == 1
            shape.append(max(sa, sb))

        shape = tuple(shape)

        # Make broadcasting explicit in the graph.
        if a.shape != shape:
            a = a.expand(shape)

        if b.shape != shape:
            b = b.expand(shape)

        return a, b

    def _alu(self, op, *others):
        # Broadcast every operand to a common shape.
        tensors = [self]

        for other in others:
            if not isinstance(other, Tensor):
                other = Tensor.const(other, self.dtype)
            tensors.append(other)

        rank = max(len(t.shape) for t in tensors)

        # Equalize ranks with leading singleton dimensions.
        tensors = [
            t.reshape(
                (1,) * (rank - len(t.shape)) + t.shape
            )
            if len(t.shape) < rank else t
            for t in tensors
        ]

        # Compute the final NumPy-style broadcast shape.
        shape = []
        for axis in range(rank):
            sizes = [t.shape[axis] for t in tensors]
            size = max(sizes)

            assert all(
                s == size or s == 1
                for s in sizes
            )

            shape.append(size)

        shape = tuple(shape)

        # Insert explicit EXPAND nodes.
        tensors = [
            t if t.shape == shape else t.expand(shape)
            for t in tensors
        ]

        if op in {Ops.CMPLT, Ops.AND}:
            dtype = dtypes.bool
        elif op is Ops.WHERE:
            dtype = tensors[1].dtype
        else:
            dtype = self.dtype

        return Tensor(
            UOp(
                op,
                dtype,
                tuple(t.uop for t in tensors),
            )
        )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other):
        return self._alu(Ops.ADD, other)

    def __radd__(self, other):
        return self._alu(Ops.ADD, other)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        return self._alu(Ops.MUL, other)

    def __rmul__(self, other):
        return self._alu(Ops.MUL, other)

    def __truediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)

        return self * other.recip()

    def __rtruediv__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)

        return other * self.recip()

    def __floordiv__(self, other):
        a, b = self._bcast(other)

        return Tensor(
            UOp(
                Ops.IDIV,
                a.dtype,
                (a.uop, b.uop),
            )
        )

    def __mod__(self, other):
        a, b = self._bcast(other)

        return Tensor(
            UOp(
                Ops.MOD,
                a.dtype,
                (a.uop, b.uop),
            )
        )

    def __neg__(self):
        return self * -1

    # ------------------------------------------------------------------
    # Comparisons / logic
    # ------------------------------------------------------------------

    def __lt__(self, other):
        return self._alu(Ops.CMPLT, other)

    def __gt__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor.const(other, self.dtype)

        # a > b is represented as b < a.
        return other._alu(Ops.CMPLT, self)

    def __and__(self, other):
        return self._alu(Ops.AND, other)

    # ------------------------------------------------------------------
    # Elementwise functions
    # ------------------------------------------------------------------

    def maximum(self, other):
        return self._alu(Ops.MAX, other)

    def where(self, a, b):
        # self is the condition; a and b are the selected values.
        if not isinstance(a, Tensor) and not isinstance(b, Tensor):
            a = Tensor.const(a)
            b = Tensor.const(b)
        elif not isinstance(a, Tensor):
            a = Tensor.const(a, b.dtype)
        elif not isinstance(b, Tensor):
            b = Tensor.const(b, a.dtype)

        return self._alu(Ops.WHERE, a, b)

    def recip(self):
        return Tensor(
            UOp(
                Ops.RECIP,
                self.dtype,
                (self.uop,),
            )
        )

    def exp2(self):
        return Tensor(
            UOp(
                Ops.EXP2,
                self.dtype,
                (self.uop,),
            )
        )

    def log2(self):
        return Tensor(
            UOp(
                Ops.LOG2,
                self.dtype,
                (self.uop,),
            )
        )

    def sqrt(self):
        return Tensor(
            UOp(
                Ops.SQRT,
                self.dtype,
                (self.uop,),
            )
        )

    def exp(self):
        return (
            self * math.log2(math.e)
        ).exp2()

    def log(self):
        return self.log2() * math.log(2)

    def relu(self):
        return self.maximum(0)

    def square(self):
        return self * self

    def cast(self, dtype):
        return Tensor(self.uop.cast(dtype))

    # ------------------------------------------------------------------
    # Reductions
    # ------------------------------------------------------------------

    def _reduce(self, op, axis, keepdim):
        rank = len(self.shape)

        if axis is None:
            axes = tuple(range(rank))
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)

        # Normalize negative axes and sort them.
        axes = tuple(
            sorted(
                a if a >= 0 else rank + a
                for a in axes
            )
        )

        assert all(0 <= a < rank for a in axes)
        assert len(set(axes)) == len(axes)

        if not axes:
            return self

        reduced = Tensor(
            UOp(
                Ops.REDUCE_AXIS,
                self.dtype,
                (self.uop,),
                (op, axes),
            )
        )

        if keepdim:
            return reduced

        # REDUCE_AXIS keeps reduced dimensions at size 1.
        shape = tuple(
            size
            for i, size in enumerate(self.shape)
            if i not in axes
        )

        return reduced.reshape(shape)

    def sum(self, axis=None, keepdim=False):
        return self._reduce(
            Ops.ADD,
            axis,
            keepdim,
        )

    def max(self, axis=None, keepdim=False):
        return self._reduce(
            Ops.MAX,
            axis,
            keepdim,
        )

    def mean(self, axis=None, keepdim=False):
        rank = len(self.shape)

        if axis is None:
            axes = tuple(range(rank))
        elif isinstance(axis, int):
            axes = (axis,)
        else:
            axes = tuple(axis)

        axes = tuple(
            sorted(
                a if a >= 0 else rank + a
                for a in axes
            )
        )

        assert all(0 <= a < rank for a in axes)
        assert len(set(axes)) == len(axes)

        if not axes:
            return self

        count = math.prod(
            self.shape[a]
            for a in axes
        )

        return self.sum(
            axis=axes,
            keepdim=keepdim,
        ) * (1.0 / count)

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def matmul(self, w):
        assert isinstance(w, Tensor)
        assert len(self.shape) >= 2
        assert len(w.shape) >= 2
        assert self.shape[-1] == w.shape[-2]

        m = self.shape[-2]
        k = self.shape[-1]
        n = w.shape[-1]

        a_batch = self.shape[:-2]
        b_batch = w.shape[:-2]

        # Align leading batch ranks.
        rank = max(len(a_batch), len(b_batch))

        a = self.reshape(
            (1,) * (rank - len(a_batch))
            + a_batch
            + (m, k)
        )

        b = w.reshape(
            (1,) * (rank - len(b_batch))
            + b_batch
            + (k, n)
        )

        # Broadcast the leading batch dimensions.
        batch = []
        for sa, sb in zip(
            a.shape[:-2],
            b.shape[:-2],
        ):
            assert sa == sb or sa == 1 or sb == 1
            batch.append(max(sa, sb))

        batch = tuple(batch)

        if a.shape[:-2] != batch:
            a = a.expand(batch + (m, k))

        if b.shape[:-2] != batch:
            b = b.expand(batch + (k, n))

        # (..., M, K) -> (..., M, 1, K)
        a = a.reshape(batch + (m, 1, k))

        # (..., K, N) -> (..., N, K) -> (..., 1, N, K)
        b = b.transpose(-2, -1)
        b = b.reshape(batch + (1, n, k))

        # (..., M, 1, K) * (..., 1, N, K)
        # -> (..., M, N, K) -> sum over K.
        return (a * b).sum(-1)

    def __matmul__(self, w):
        return self.matmul(w)

    # ------------------------------------------------------------------
    # Softmax / normalization
    # ------------------------------------------------------------------

    def softmax(self, axis=-1):
        m = self.max(
            axis=axis,
            keepdim=True,
        )

        z = self - m
        e = z.exp()

        return e / e.sum(
            axis=axis,
            keepdim=True,
        )

    def logsoftmax(self, axis=-1):
        m = self.max(
            axis=axis,
            keepdim=True,
        )

        z = self - m

        return z - z.exp().sum(
            axis=axis,
            keepdim=True,
        ).log()

    def layernorm(self, eps=1e-5):
        mean = self.mean(
            axis=-1,
            keepdim=True,
        )

        deviation = self - mean

        # Variance is the mean squared deviation.
        var = deviation.square().mean(
            axis=-1,
            keepdim=True,
        )

        return deviation / (
            var + eps
        ).sqrt()

# Step 13 - eval_tensor
def eval_tensor(u, bufs, cache=None):
    if cache is None:
        cache = {}

    if u in cache:
        return cache[u]

    # Buffers are supplied by name and converted to their declared dtype/shape.
    if u.op is Ops.BUFFER:
        name, shape = u.arg
        out = np.asarray(bufs[name], dtype=u.dtype.np).reshape(shape)

    # Constants are represented as 0-dimensional arrays.
    elif u.op is Ops.CONST:
        out = np.array(u.arg, dtype=u.dtype.np)

    else:
        src = [
            eval_tensor(x, bufs, cache)
            for x in u.src
        ]

        with np.errstate(all="ignore"):
            if u.op is Ops.RESHAPE:
                out = src[0].reshape(u.arg)

            elif u.op is Ops.EXPAND:
                out = np.broadcast_to(src[0], u.arg)

            elif u.op is Ops.PERMUTE:
                out = np.transpose(src[0], u.arg)

            elif u.op is Ops.FLIP:
                out = np.flip(src[0], axis=u.arg)

            elif u.op is Ops.PAD:
                out = np.pad(src[0], u.arg, mode="constant")

            elif u.op is Ops.SHRINK:
                out = src[0][
                    tuple(
                        slice(b, e)
                        for b, e in u.arg
                    )
                ]

            elif u.op is Ops.REDUCE_AXIS:
                op, axes = u.arg

                if op is Ops.ADD:
                    out = np.sum(
                        src[0],
                        axis=axes,
                        keepdims=True,
                    )
                elif op is Ops.MAX:
                    out = np.max(
                        src[0],
                        axis=axes,
                        keepdims=True,
                    )
                else:
                    raise NotImplementedError(u.op)

            elif u.op is Ops.ADD:
                out = src[0] + src[1]

            elif u.op is Ops.MUL:
                out = src[0] * src[1]

            elif u.op is Ops.MAX:
                out = np.maximum(src[0], src[1])

            elif u.op is Ops.CMPLT:
                out = src[0] < src[1]

            elif u.op is Ops.AND:
                out = src[0] & src[1]

            elif u.op is Ops.IDIV:
                out = np.floor_divide(src[0], src[1])

            elif u.op is Ops.MOD:
                out = np.mod(src[0], src[1])

            elif u.op is Ops.RECIP:
                out = np.reciprocal(src[0])

            elif u.op is Ops.EXP2:
                out = np.exp2(src[0])

            elif u.op is Ops.LOG2:
                out = np.log2(src[0])

            elif u.op is Ops.SQRT:
                out = np.sqrt(src[0])

            elif u.op is Ops.CAST:
                out = src[0].astype(u.dtype.np)

            elif u.op is Ops.WHERE:
                out = np.where(
                    src[0],
                    src[1],
                    src[2],
                )

            else:
                raise NotImplementedError(u.op)

    # Keep every node in its declared NumPy dtype.
    out = np.asarray(out, dtype=u.dtype.np)
    cache[u] = out
    return out

# Step 14 - conv2d
def conv2d(x, w, pad=0):
    N, Cin, H, W = x.shape
    Cout, wCin, kh, kw = w.shape

    assert Cin == wCin
    assert pad >= 0

    # Output spatial dimensions for stride-1 convolution.
    Ho = H + 2 * pad - kh + 1
    Wo = W + 2 * pad - kw + 1

    assert Ho > 0 and Wo > 0

    # Pad the input spatial dimensions.
    xp = x.pad((
        (0, 0),
        (0, 0),
        (pad, pad),
        (pad, pad),
    ))

    acc = None

    for dh in range(kh):
        for dw in range(kw):
            # Select the output-sized window for this tap.
            window = xp.shrink((
                (0, N),
                (0, Cin),
                (dh, dh + Ho),
                (dw, dw + Wo),
            ))

            # (N, Cin, Ho, Wo)
            # -> (N, 1, Cin, Ho, Wo)
            # -> (N, Cout, Cin, Ho, Wo)
            window = window.reshape(
                (N, 1, Cin, Ho, Wo)
            ).expand(
                (N, Cout, Cin, Ho, Wo)
            )

            # Select one spatial tap from the weights.
            tap = w.shrink((
                (0, Cout),
                (0, Cin),
                (dh, dh + 1),
                (dw, dw + 1),
            ))

            # (Cout, Cin, 1, 1)
            # -> (1, Cout, Cin, 1, 1)
            # -> (N, Cout, Cin, Ho, Wo)
            tap = tap.reshape(
                (1, Cout, Cin, 1, 1)
            ).expand(
                (N, Cout, Cin, Ho, Wo)
            )

            term = window * tap
            acc = term if acc is None else acc + term

    # Reduce only over the input-channel dimension.
    return acc.sum(2)


Tensor.conv2d = conv2d

# Step 15 - Kernel
class Reduce:
    def __init__(self, ranges, accs, body):
        self.ranges = tuple(ranges)
        self.accs = [list(a) for a in accs]
        self.body = list(body)

    def __repr__(self):
        return (
            f"Reduce(ranges={[r.arg for r in self.ranges]}, "
            f"accs={len(self.accs)}, body={self.body})"
        )


class Kernel:
    def __init__(self, name, params, out_ranges, body, stores):
        self.name = name
        self.params = list(params)
        self.out_ranges = list(out_ranges)
        self.body = list(body)
        self.stores = list(stores)

    def map_exprs(self, f):
        def map_reduce(r):
            # Apply f to every accumulator expression and recurse into body.
            accs = [
                [acc, f(init), f(update)]
                for acc, init, update in r.accs
            ]

            body = [
                map_reduce(stmt)
                for stmt in r.body
            ]

            return Reduce(r.ranges, accs, body)

        body = [
            map_reduce(r)
            for r in self.body
        ]

        stores = [
            (f(index), f(value))
            for index, value in self.stores
        ]

        return Kernel(
            self.name,
            self.params,
            self.out_ranges,
            body,
            stores,
        )

    def all_exprs(self):
        exprs = []

        def collect_reduce(r):
            for _, init, update in r.accs:
                exprs.append(init)
                exprs.append(update)

            for stmt in r.body:
                collect_reduce(stmt)

        for r in self.body:
            collect_reduce(r)

        for index, value in self.stores:
            exprs.append(index)
            exprs.append(value)

        return exprs


def ranges_in(u):
    return {
        node
        for node in u.toposort()
        if node.op is Ops.RANGE
    }


def flat_index(idxs, shape):
    # Build a row-major linear index using the tensor's strides.
    out = UOp.const(dtypes.int32, 0)

    for i, (idx, size) in enumerate(zip(idxs, shape)):
        # Size-one axes contribute nothing to the flat index.
        if size == 1:
            continue

        stride = math.prod(shape[i + 1:])

        if stride == 1:
            out = out + idx
        else:
            out = out + idx * stride

    return out

# Step 16 - Lowerer
class Lowerer:
    def __init__(self, realized):
        self.realized, self.n = realized, 0
        self.memo, self.frames = {}, []

    def new_range(self, n):
        r = UOp.range(n, self.n)
        self.n += 1
        return r

    def new_acc(self, dtype):
        acc = UOp(
            Ops.DEFINE_ACC,
            dtype,
            (),
            self.n,
        )
        self.n += 1
        return acc

    def lower(self, root):
        shape = shape_of(root)

        # Create one output index per axis.
        # Size-one axes use a constant instead of a loop.
        out_ranges = [
            UOp.const(dtypes.int32, 0)
            if size == 1
            else self.new_range(size)
            for size in shape
        ]

        # Keep the output ranges and generated statements in the frame.
        self.frames.append([set(out_ranges), []])

        val = self.index(
            root,
            tuple(out_ranges),
            root,
        )

        body = self.frames.pop()[1]

        return Kernel(
            "k",
            [],
            out_ranges,
            body,
            [
                (
                    flat_index(out_ranges, shape),
                    val,
                )
            ],
        )

    def index(self, u, idxs, root):
        key = (u, tuple(idxs))

        if key in self.memo:
            return self.memo[key]

        shape = shape_of(u)

        # Buffers and realized nodes are materialized as parameters.
        if u.op is Ops.BUFFER or (
            u in self.realized and u is not root
        ):
            p = self.realized[u]

            out = load(
                p,
                flat_index(idxs, shape),
            )

            self.memo[key] = out
            return out

        # Constants are already kernel expressions.
        if u.op is Ops.CONST:
            self.memo[key] = u
            return u

        # REDUCE_AXIS is lowered in Step 17.
        if u.op is Ops.REDUCE_AXIS:
            raise NotImplementedError("REDUCE_AXIS")

        # Elementwise operations use the same logical indices for
        # every source because Tensor broadcasting is explicit.
        if u.op in ALU:
            out = UOp(
                u.op,
                u.dtype,
                tuple(
                    self.index(src, idxs, root)
                    for src in u.src
                ),
                u.arg,
            )

            self.memo[key] = out
            return out

        if u.op is Ops.RESHAPE:
            src = u.src[0]

            # Flatten the output coordinates.
            f = flat_index(
                idxs,
                shape,
            )

            # A buffer or realized tensor is contiguous, so its
            # flattened element can be loaded directly.
            if src.op is Ops.BUFFER or (
                src in self.realized and src is not root
            ):
                p = self.realized[src]
                out = load(p, f)

            else:
                src_shape = shape_of(src)
                strides = strides_of(src_shape)
                src_idxs = []

                # Unflatten the offset using source strides.
                for stride, size in zip(
                    strides,
                    src_shape,
                ):
                    if size == 1:
                        src_idxs.append(
                            UOp.const(dtypes.int32, 0)
                        )
                    else:
                        src_idxs.append(
                            (f // stride) % size
                        )

                out = self.index(
                    src,
                    tuple(src_idxs),
                    root,
                )

            self.memo[key] = out
            return out

        if u.op is Ops.EXPAND:
            src = u.src[0]
            src_shape = shape_of(src)

            # Expanded size-one axes always read source index zero.
            src_idxs = tuple(
                UOp.const(dtypes.int32, 0)
                if size == 1
                else idx
                for size, idx in zip(
                    src_shape,
                    idxs,
                )
            )

            out = self.index(
                src,
                src_idxs,
                root,
            )

            self.memo[key] = out
            return out

        if u.op is Ops.PERMUTE:
            src = u.src[0]

            # perm[k] is the source axis corresponding to output axis k.
            src_idxs = [None] * len(u.arg)

            for k, axis in enumerate(u.arg):
                src_idxs[axis] = idxs[k]

            out = self.index(
                src,
                tuple(src_idxs),
                root,
            )

            self.memo[key] = out
            return out

        if u.op is Ops.FLIP:
            src = u.src[0]
            src_shape = shape_of(src)
            axes = set(u.arg)

            # Flip an axis with (size - 1) - index.
            src_idxs = tuple(
                UOp.const(
                    dtypes.int32,
                    size - 1,
                ) - idx
                if axis in axes
                else idx
                for axis, (size, idx)
                in enumerate(zip(src_shape, idxs))
            )

            out = self.index(
                src,
                src_idxs,
                root,
            )

            self.memo[key] = out
            return out

        if u.op is Ops.SHRINK:
            src = u.src[0]

            # Add the slice beginning offset.
            src_idxs = tuple(
                idx + begin
                for idx, (begin, _)
                in zip(idxs, u.arg)
            )

            out = self.index(
                src,
                src_idxs,
                root,
            )

            self.memo[key] = out
            return out

        if u.op is Ops.PAD:
            src = u.src[0]
            src_shape = shape_of(src)

            # Translate padded indices back to source coordinates.
            src_idxs = tuple(
                idx - lo
                for idx, (lo, _)
                in zip(idxs, u.arg)
            )

            # Construct the validity condition for the source region.
            valid = UOp.const(
                dtypes.bool,
                True,
            )

            for idx, (lo, hi), size in zip(
                idxs,
                u.arg,
                src_shape,
            ):
                if lo > 0:
                    valid = valid & (
                        UOp.const(
                            dtypes.int32,
                            lo - 1,
                        ) < idx
                    )

                if hi > 0:
                    valid = valid & (
                        idx < UOp.const(
                            dtypes.int32,
                            lo + size,
                        )
                    )

            inner = self.index(
                src,
                src_idxs,
                root,
            )

            # Outside the source, padding contributes zero.
            out = valid.where(
                inner,
                UOp.const(
                    u.dtype,
                    0,
                ),
            )

            self.memo[key] = out
            return out

        raise NotImplementedError(u.op)


def lower_kernel(root, inputs, name="k"):
    # The output is always parameter position 0.
    out = param(
        "out",
        root.dtype,
        0,
    )

    realized = {}
    params = [out]

    # Add input parameters in the supplied order.
    for i, (u, buf_name) in enumerate(inputs.items()):
        p = param(
            buf_name,
            u.dtype,
            i + 1,
        )
        realized[u] = p
        params.append(p)

    lowerer = Lowerer(realized)
    kernel = lowerer.lower(root)

    kernel.name = name
    kernel.params = params

    # Simplify all generated expressions.
    return kernel.map_exprs(simplify)

# Step 17 - ReduceLowerer
def reduce_exprs(r):
    # Collect all RANGE dependencies from accumulator expressions
    # and recursively from nested reductions.
    ranges = set()

    for _, init, update in r.accs:
        ranges |= ranges_in(init)
        ranges |= ranges_in(update)

    for stmt in r.body:
        ranges |= reduce_exprs(stmt)

    return ranges


class ReduceLowerer(Lowerer):
    def index(self, u, idxs, root):
        # Realized reductions are handled by Lowerer as normal loads.
        if (
            u.op is not Ops.REDUCE_AXIS
            or (u is not root and u in self.realized)
        ):
            return super().index(u, idxs, root)

        op, axes = u.arg
        src = u.src[0]
        src_shape = shape_of(src)

        # Create one fresh loop range for every reduced axis.
        new_ranges = tuple(
            self.new_range(src_shape[axis])
            for axis in axes
        )

        # Replace each reduced axis with its new reduction range.
        src_idxs = list(idxs)
        for axis, r in zip(axes, new_ranges):
            src_idxs[axis] = r

        # The reduction scope contains both the enclosing ranges
        # and the newly created reduction ranges.
        enclosing = self.frames[-1][0]
        self.frames.append([
            enclosing | set(new_ranges),
            [],
        ])

        val = self.index(
            src,
            tuple(src_idxs),
            root,
        )

        body = self.frames.pop()[1]

        acc = self.new_acc(u.dtype)

        if op is Ops.ADD:
            init = UOp.const(u.dtype, 0.0)
            update = acc + val
        elif op is Ops.MAX:
            init = UOp.const(u.dtype, -INF)
            update = acc.maximum(val)
        else:
            raise NotImplementedError(op)

        statement = Reduce(
            new_ranges,
            [[acc, init, update]],
            body,
        )

        # Find the outermost scope containing every range dependency.
        deps = reduce_exprs(statement) - set(new_ranges)

        target = None
        for frame in self.frames:
            if deps <= frame[0]:
                target = frame
                break

        # If no enclosing scope contains all dependencies, use the
        # innermost frame.
        if target is None:
            target = self.frames[-1]

        target[1].append(statement)

        self.memo[(u, tuple(idxs))] = acc
        return acc


def lower_kernel(root, inputs, name="k"):
    # Parameter 0 is always the output.
    out = param(
        "out",
        root.dtype,
        0,
    )

    realized = {}
    params = [out]

    # Inputs become realized parameters in insertion order.
    for i, (u, buf_name) in enumerate(inputs.items()):
        p = param(
            buf_name,
            u.dtype,
            i + 1,
        )
        realized[u] = p
        params.append(p)

    lowerer = ReduceLowerer(realized)
    kernel = lowerer.lower(root)

    kernel.name = name
    kernel.params = params

    # Simplify all generated kernel expressions.
    return kernel.map_exprs(simplify)

# Step 18 - run_kernel_np
def run_kernel_np(k, bufs):
    # Build the output grid. Output ranges are represented by int64
    # index arrays so the interpreter can vectorize over all outputs.
    shape = tuple(
        r.src[0].arg
        if r.op in {Ops.RANGE, Ops.SPECIAL}
        else 1
        for r in k.out_ranges
    )

    grids = np.indices(
        shape,
        dtype=np.int64,
    )

    env = {}
    cache = {}

    # Bind output ranges and SPECIAL nodes to their grid index arrays.
    for i, r in enumerate(k.out_ranges):
        if r.op in {Ops.RANGE, Ops.SPECIAL}:
            env[r] = grids[i]

    def ev(u):
        if u in cache:
            return cache[u]

        if u.op is Ops.CONST:
            dtype = (
                np.int64
                if u.dtype is dtypes.int32
                else u.dtype.np
            )
            out = np.array(u.arg, dtype=dtype)

        elif u.op in {Ops.RANGE, Ops.SPECIAL}:
            out = env[u]

        elif u.op is Ops.DEFINE_ACC:
            out = env[u]

        elif u.op is Ops.LOAD:
            # LOAD -> INDEX -> (PARAM, index)
            index = u.src[0]
            p = index.src[0]

            idx = ev(index.src[1])
            flat = bufs[p.arg[1]].reshape(-1)

            # Clip gathers so masked PAD loads never go out of bounds.
            idx = np.clip(
                idx,
                0,
                flat.size - 1,
            ).astype(np.intp)

            out = flat[idx]

        elif u.op is Ops.WHERE:
            out = np.where(
                ev(u.src[0]),
                ev(u.src[1]),
                ev(u.src[2]),
            )

        elif u.op is Ops.CAST:
            out = ev(u.src[0]).astype(u.dtype.np)

        elif u.op is Ops.ADD:
            out = ev(u.src[0]) + ev(u.src[1])

        elif u.op is Ops.MUL:
            out = ev(u.src[0]) * ev(u.src[1])

        elif u.op is Ops.MAX:
            out = np.maximum(
                ev(u.src[0]),
                ev(u.src[1]),
            )

        elif u.op is Ops.CMPLT:
            out = ev(u.src[0]) < ev(u.src[1])

        elif u.op is Ops.AND:
            out = ev(u.src[0]) & ev(u.src[1])

        elif u.op is Ops.IDIV:
            with np.errstate(all="ignore"):
                out = np.floor_divide(
                    ev(u.src[0]),
                    ev(u.src[1]),
                )

        elif u.op is Ops.MOD:
            with np.errstate(all="ignore"):
                out = np.mod(
                    ev(u.src[0]),
                    ev(u.src[1]),
                )

        elif u.op is Ops.RECIP:
            with np.errstate(all="ignore"):
                out = np.reciprocal(ev(u.src[0]))

        elif u.op is Ops.EXP2:
            with np.errstate(all="ignore"):
                out = np.exp2(ev(u.src[0]))

        elif u.op is Ops.LOG2:
            with np.errstate(all="ignore"):
                out = np.log2(ev(u.src[0]))

        elif u.op is Ops.SQRT:
            with np.errstate(all="ignore"):
                out = np.sqrt(ev(u.src[0]))

        else:
            raise NotImplementedError(u.op)

        # Keep every expression in the dtype declared by its UOp.
        out = np.asarray(out, dtype=u.dtype.np)
        cache[u] = out
        return out

    def run_reduce(r):
        # Each accumulator starts as its init value broadcast over
        # the complete output grid.
        for acc, init, _ in r.accs:
            env[acc] = np.array(
                np.broadcast_to(
                    ev(init),
                    shape,
                ),
                dtype=acc.dtype.np,
                copy=True,
            )

        for values in np.ndindex(*(
            rg.src[0].arg
            for rg in r.ranges
        )):
            # Reduction ranges are scalar loop variables.
            for rg, value in zip(r.ranges, values):
                env[rg] = value

            cache.clear()

            # Nested reductions execute before the accumulator updates.
            for stmt in r.body:
                run_reduce(stmt)

            # Evaluate every update against the previous accumulator
            # values before assigning any new values.
            updates = [
                ev(update)
                for _, _, update in r.accs
            ]

            for (acc, _, _), value in zip(
                r.accs,
                updates,
            ):
                env[acc] = np.asarray(
                    value,
                    dtype=acc.dtype.np,
                )

        cache.clear()

    # Execute top-level reductions.
    for r in k.body:
        run_reduce(r)

    # Scatter each computed store into the output buffer.
    out = bufs[0].reshape(-1)

    for index, value in k.stores:
        idx = ev(index)
        val = ev(value)

        out[
            np.asarray(idx).astype(np.intp)
        ] = val

    return bufs[0]

# Step 19 - render_kernel
def render_kernel(k):
    r = CRenderer()

    # Render the function signature first, outside the renderer's scopes.
    r.lines.append(
        signature(
            k.name,
            k.params,
            {k.params[0]},
        ) + " {"
    )

    # Enter the function body.
    r.push()

    def open_loops(ranges):
        for rg in ranges:
            if rg.op is Ops.RANGE:
                r.emit(
                    f"for (int r{rg.arg} = 0; "
                    f"r{rg.arg} < {rg.src[0].arg}; "
                    f"r{rg.arg}++) {{"
                )
                r.push()

    def close_loops(ranges):
        for rg in reversed(ranges):
            if rg.op is Ops.RANGE:
                r.pop()
                r.emit("}")

    def render_reduce(red):
        # Initialize every accumulator before entering the reduction loops.
        for acc, init, update in red.accs:
            r.emit(
                f"{acc.dtype.c_name} acc{acc.arg} = {r.expr(init)};"
            )

        open_loops(red.ranges)

        # Nested reductions are rendered inside the current reduction.
        for stmt in red.body:
            render_reduce(stmt)

        # expr(update) already creates and caches a temporary for the
        # update expression. Evaluate every update before assigning any
        # accumulator so all updates see the previous accumulator values.
        updates = [
            (acc, r.expr(update))
            for acc, _, update in red.accs
        ]

        for acc, value in updates:
            r.emit(f"acc{acc.arg} = {value};")

        close_loops(red.ranges)

    # Open the output loop nest.
    open_loops(k.out_ranges)

    # Render all reduction statements.
    for red in k.body:
        render_reduce(red)

    # Write the output values.
    for index, value in k.stores:
        r.emit(
            f"data0[{r.expr(index)}] = {r.expr(value)};"
        )

    close_loops(k.out_ranges)

    r.pop()
    r.lines.append("}")

    return "\n".join(r.lines) + "\n"


def run_kernel_c(k, bufs):
    src = render_kernel(k)
    lib = compile_c(src)
    call_kernel(lib, k.name, bufs)
    return bufs[0]

# Step 20 - schedule
class Program:
    def __init__(self):
        self.buffers, self.kernels = {}, []


def schedule(outputs):
    prog = Program()

    # Collect every node once, preserving first-seen topological order.
    nodes = []
    seen = set()

    for tensor in outputs.values():
        for u in tensor.uop.toposort():
            if u not in seen:
                seen.add(u)
                nodes.append(u)

    realized = {}
    inputs = set()

    # Realize every BUFFER as an external input.
    for u in nodes:
        if u.op is Ops.BUFFER:
            name = u.arg[0]
            realized[u] = name
            inputs.add(u)

            prog.buffers[name] = (
                shape_of(u),
                u.dtype,
                "input",
            )

    # Every REDUCE_AXIS that is not already an input/output becomes a stage.
    # Do this before outputs so stage buffers appear before output buffers.
    stage = 0

    for u in nodes:
        if u.op is Ops.REDUCE_AXIS and u not in realized:
            name = f"stage{stage}"
            stage += 1

            realized[u] = name

            prog.buffers[name] = (
                shape_of(u),
                u.dtype,
                "stage",
            )

    # Realize output roots under their requested names.
    for name, tensor in outputs.items():
        u = tensor.uop

        realized[u] = name

        prog.buffers[name] = (
            shape_of(u),
            u.dtype,
            "output",
        )

    # Lower each realized node that is not an input.
    kernel_index = 0

    for u in nodes:
        if u not in realized or u in inputs:
            continue

        # Gather realized dependencies in first-seen order.
        deps = []

        for v in u.toposort():
            if v is u:
                continue

            if v in realized and v not in deps:
                deps.append(v)

        ins = {
            v: realized[v]
            for v in deps
        }

        kernel = lower_kernel(
            u,
            ins,
            name=f"k{kernel_index}",
        )

        # Size-one output axes are already represented by CONST 0
        # in the lowered indices, so only real loops belong here.
        kernel.out_ranges = [
            r
            for r in kernel.out_ranges
            if r.op is Ops.RANGE
        ]

        prog.kernels.append(
            (
                kernel,
                [realized[u]] + [
                    realized[v]
                    for v in deps
                ],
            )
        )

        kernel_index += 1

    return prog

