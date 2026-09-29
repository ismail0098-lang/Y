import time
import ctypes
from typing import List, Dict, Any, Callable, Optional, Tuple, Union
from .compiler import generate_autotune_search_space

# Below this, a measurement cannot be about the kernel: a launch plus a
# synchronise costs tens of microseconds at minimum, so anything faster is the
# Python call overhead of a decorated function that never launched anything.
MEASUREMENT_FLOOR_MS = 0.05


def measurement_beats_model(
    winner_ms: float, model_ms: float, within: float
) -> bool:
    """Is a measured candidate's win over the model's pick real?

    A pure predicate, deliberately, so the two rules can be pinned with exact
    numbers instead of a clock. Timing them end to end only detected a dropped
    floor in about 4 runs out of 10 - microsecond-scale repeatability is not
    something a loaded machine offers, and a probabilistic guard on a rule this
    small is not worth keeping.

    `within` is the widest repeat-to-repeat dispersion seen in the run, as a
    fraction: 0.2 means some candidate's slowest sample was 20% above its own
    fastest.
    """
    # A GPU kernel launch plus a synchronise does not complete in
    # microseconds. Below this the harness is timing itself - the decorated
    # function never launched anything - and no tile ordering can be read out
    # of it, however consistent the difference looks.
    if winner_ms < MEASUREMENT_FLOOR_MS:
        return False
    # ...and above the floor, only a margin wider than the run's own noise
    # counts. Anything smaller is the scheduler's ordering, not the kernel's.
    return winner_ms * (1.0 + within) < model_ms


class AutotuneConfig:
    """Configuration container for autotuning tile parameters."""
    def __init__(
        self,
        cta_m: int = 128,
        cta_n: int = 128,
        cta_k: int = 32,
        warps_m: int = 4,
        warps_n: int = 2,
        num_stages: int = 3,
        num_warps: int = 8,
        kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.cta_m = cta_m
        self.cta_n = cta_n
        self.cta_k = cta_k
        self.warps_m = warps_m
        self.warps_n = warps_n
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.kwargs = kwargs or {}

    def tile(self) -> Tuple[int, int, int, int, int, int, int]:
        """Identity of the configuration, for deduplicating candidate lists."""
        return (
            self.cta_m, self.cta_n, self.cta_k,
            self.warps_m, self.warps_n, self.num_stages, self.num_warps,
        )

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "cta_m": self.cta_m,
            "cta_n": self.cta_n,
            "cta_k": self.cta_k,
            "warps_m": self.warps_m,
            "warps_n": self.warps_n,
            "num_stages": self.num_stages,
            "num_warps": self.num_warps,
        }
        res.update(self.kwargs)
        return res

    def __repr__(self) -> str:
        return f"AutotuneConfig(cta_m={self.cta_m}, cta_n={self.cta_n}, cta_k={self.cta_k}, num_warps={self.num_warps}, num_stages={self.num_stages}, kwargs={self.kwargs})"

# Alias Config for Triton API compatibility
Config = AutotuneConfig

class AutotunedKernel:
    """Kernel wrapper that dynamically autotunes tile configurations upon first execution."""
    def __init__(
        self,
        fn: Callable,
        configs: Optional[List[AutotuneConfig]] = None,
        key: Optional[List[Union[str, int]]] = None,
        warmup: int = 3,
        rep: int = 10,
    ):
        self.fn = fn
        self.configs = configs or []
        self.key = key or []
        self.warmup = warmup
        self.rep = rep
        self._best_config_cache: Dict[Tuple, AutotuneConfig] = {}

    def _extract_key(self, args: Tuple, kwargs: Dict[str, Any]) -> Tuple:
        if not self.key:
            key_items = []
            for arg in args:
                if hasattr(arg, "shape"):
                    key_items.extend(arg.shape)
                elif isinstance(arg, (int, float)):
                    key_items.append(arg)
            for v in kwargs.values():
                if isinstance(v, (int, float)):
                    key_items.append(v)
            return tuple(key_items)

        key_items = []
        for k in self.key:
            if isinstance(k, int) and k < len(args):
                arg = args[k]
                if hasattr(arg, "shape"):
                    key_items.extend(arg.shape)
                else:
                    key_items.append(arg)
            elif isinstance(k, str):
                if k in kwargs:
                    v = kwargs[k]
                    if hasattr(v, "shape"):
                        key_items.extend(v.shape)
                    else:
                        key_items.append(v)
        return tuple(key_items) if key_items else (0,)

    def benchmark_config(self, config: AutotuneConfig, args: Tuple, kwargs: Dict[str, Any]) -> float:
        """Measures execution latency of kernel under candidate configuration."""
        try:
            import torch
            has_torch = True
        except ImportError:
            has_torch = False

        call_kwargs = dict(kwargs)
        call_kwargs.update(config.kwargs)

        for _ in range(self.warmup):
            try:
                self.fn(config, *args, **call_kwargs)
            except Exception:
                return float("inf")

        if has_torch and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(self.rep):
                self.fn(config, *args, **call_kwargs)
            end_event.record()
            torch.cuda.synchronize()
            return start_event.elapsed_time(end_event) / self.rep
        else:
            t0 = time.perf_counter()
            for _ in range(self.rep):
                self.fn(config, *args, **call_kwargs)
            t1 = time.perf_counter()
            return (t1 - t0) * 1000.0 / self.rep

    def autotune(self, *args, **kwargs) -> AutotuneConfig:
        cache_key = self._extract_key(args, kwargs)
        if cache_key in self._best_config_cache:
            return self._best_config_cache[cache_key]

        if not self.configs:
            m, n, k = 1024, 1024, 1024
            if len(cache_key) >= 3:
                m, n, k = cache_key[0], cache_key[1], cache_key[2]
            raw_space = generate_autotune_search_space(m, n, k)
            self.configs = [
                AutotuneConfig(
                    c["cta_m"], c["cta_n"], c["cta_k"], c.get("warps_m", 4), c.get("warps_n", 2), c.get("num_stages", 3), c.get("num_warps", 8)
                )
                for c in raw_space
            ]
            if m <= 1024 or n <= 1024:
                small_configs = [
                    AutotuneConfig(16, 32, 32, warps_m=1, warps_n=1, num_stages=2, num_warps=1),
                    AutotuneConfig(32, 64, 32, warps_m=1, warps_n=2, num_stages=2, num_warps=2),
                    AutotuneConfig(64, 64, 32, warps_m=2, warps_n=2, num_stages=2, num_warps=4),
                ]
                self.configs = small_configs + self.configs

        # The analytic model's pick is the BASELINE, not one candidate among
        # many. That is what the compiler itself does: prefer an on-device
        # measurement, fall back to the model. Measurement here can be pure
        # noise - a decorated function that does not actually launch a kernel
        # is timed at sub-microsecond Python call overhead - and ranking 30
        # candidates by noise returned a different tile on every run. The
        # documented `python -m unittest` suite failed about one run in four
        # on exactly that.
        model_config = self._model_config(cache_key)

        candidates = list(self.configs)
        model_idx = None
        if model_config is not None:
            model_idx = next(
                (i for i, c in enumerate(candidates) if c.tile() == model_config.tile()),
                None,
            )
            if model_idx is None:
                candidates.append(model_config)
                model_idx = len(candidates) - 1
        if not candidates:
            return None

        # Interleave: one pass per round over ALL candidates, then take each
        # candidate's MINIMUM. A single sequential pass charges whichever
        # candidate ran during a scheduling hiccup, and the minimum is the
        # only order statistic that is not dragged around by one outlier.
        rounds = max(1, min(3, self.rep))
        samples: Dict[int, List[float]] = {i: [] for i in range(len(candidates))}
        for _ in range(rounds):
            for i, cfg in enumerate(candidates):
                samples[i].append(self.benchmark_config(cfg, args, kwargs))

        times = {i: min(v) for i, v in samples.items() if v}
        finite = {i: t for i, t in times.items() if t != float("inf")}
        if not finite:
            best = model_config or candidates[0]
            self._best_config_cache[cache_key] = best
            return best

        winner_idx = min(finite, key=lambda i: finite[i])
        best_config = candidates[winner_idx]
        model_time = finite.get(model_idx) if model_idx is not None else None

        if model_time is not None and winner_idx != model_idx:
            # WITHIN: how much one candidate's own repeats disagree. A
            # candidate only wins if it beats the model by MORE than that -
            # anything smaller is the scheduler's ordering, not the kernel's.
            #
            # A `between > within` clause was here too (candidate-to-candidate
            # spread must exceed repeat-to-repeat spread) and is deliberately
            # gone: it is implied. With `lo` the winner's time and `hi` the
            # slowest, `between >= (model_time - lo)/lo`, and the margin test
            # below is exactly `within < (model_time - lo)/lo`. Mutation could
            # not pin the clause because nothing depends on it; the answer to
            # an unpinnable clause is to delete it, not to leave it untested.
            within = 0.0
            for v in samples.values():
                lo = min(v)
                if 0.0 < lo < float("inf"):
                    within = max(within, (max(v) - lo) / lo)

            if not measurement_beats_model(finite[winner_idx], model_time, within):
                best_config = candidates[model_idx]

        self._best_config_cache[cache_key] = best_config
        return best_config

    def _model_config(self, cache_key: Tuple) -> Optional[AutotuneConfig]:
        """The compiler's analytic pick for this shape, or None if unavailable."""
        if len(cache_key) < 3:
            return None
        try:
            from .compiler import select_autotune_config
            c = select_autotune_config(cache_key[0], cache_key[1], cache_key[2])
        except Exception:
            return None
        if not c:
            return None
        return AutotuneConfig(
            c["cta_m"], c["cta_n"], c["cta_k"],
            c.get("warps_m", 4), c.get("warps_n", 2),
            c.get("num_stages", 3), c.get("num_warps", 8),
        )

    def __call__(self, *args, **kwargs):
        config = self.autotune(*args, **kwargs)
        call_kwargs = dict(kwargs)
        call_kwargs.update(config.kwargs)
        return self.fn(config, *args, **call_kwargs)

def autotune(
    configs: Optional[List[AutotuneConfig]] = None,
    key: Optional[List[Union[str, int]]] = None,
    warmup: int = 3,
    rep: int = 10,
):
    """Decorator to enable dynamic empirical autotuning for Y kernels."""
    def decorator(fn):
        return AutotunedKernel(fn, configs=configs, key=key, warmup=warmup, rep=rep)
    return decorator

def heuristics(values: Dict[str, Callable]):
    """Triton-compatible heuristics decorator to dynamically set kernel parameters based on arguments."""
    def decorator(fn):
        def wrapper(*args, **kwargs):
            for param, func in values.items():
                if param not in kwargs:
                    try:
                        kwargs[param] = func(*args)
                    except Exception:
                        try:
                            kwargs[param] = func(kwargs)
                        except Exception:
                            kwargs[param] = func(args)
            return fn(*args, **kwargs)
        return wrapper
    return decorator


