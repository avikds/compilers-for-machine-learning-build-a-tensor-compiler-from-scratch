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

