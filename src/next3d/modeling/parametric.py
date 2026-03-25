"""Full parametric engine — parameter binding, dependency graph, selective replay.

This makes next3d a true parametric CAD system:
1. Parameters are named values: wall_t = 3.0
2. Operations bind to parameters: add_shell(thickness=@wall_t)
3. Changing a parameter replays only affected operations
4. Design tables generate variants from parameter combinations

The AI agent or user defines parameters, then builds geometry using @param
references. When a parameter changes, the engine figures out which operations
are affected and replays from the earliest affected operation forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


PARAM_PREFIX = "@"


@dataclass
class ParameterDef:
    """A named design parameter."""

    name: str
    value: float
    description: str = ""
    unit: str = "mm"

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "description": self.description,
            "unit": self.unit,
        }


@dataclass
class BoundOperation:
    """An operation with parameter bindings recorded.

    Stores both the resolved params (actual numbers) and the bindings
    (which params map to which parameter names).
    """

    op_index: int
    op_type: str
    raw_params: dict[str, Any]  # params as passed, may contain @param refs
    bindings: dict[str, str]  # param_key → parameter_name (e.g. "thickness" → "wall_t")
    description: str = ""

    def depends_on(self, param_name: str) -> bool:
        """Check if this operation depends on a given parameter."""
        return param_name in self.bindings.values()

    def resolve(self, parameters: dict[str, ParameterDef]) -> dict[str, Any]:
        """Resolve @param references to actual values."""
        resolved = dict(self.raw_params)
        for key, param_name in self.bindings.items():
            if param_name in parameters:
                resolved[key] = parameters[param_name].value
        return resolved


class ParametricEngine:
    """Manages parameters, bindings, and selective replay.

    Usage:
        engine = ParametricEngine()
        engine.define("wall_t", 3.0, "Wall thickness")
        engine.define("bolt_d", 6.0, "Bolt diameter")

        # Record operations with bindings
        engine.record_operation(0, "add_shell", {"thickness": "@wall_t"})
        engine.record_operation(1, "add_hole", {"diameter": "@bolt_d"})

        # Change a parameter → get affected operations
        affected = engine.change_parameter("wall_t", 2.0)
        # → [0]  (only the shell operation needs replay)

        # Generate design table variants
        variants = engine.design_table({
            "wall_t": [2, 3, 4],
            "bolt_d": [4, 6, 8],
        })
        # → 9 combinations, each with resolved params for all operations
    """

    def __init__(self) -> None:
        self._parameters: dict[str, ParameterDef] = {}
        self._operations: list[BoundOperation] = []

    @property
    def parameters(self) -> dict[str, ParameterDef]:
        return self._parameters

    @property
    def operations(self) -> list[BoundOperation]:
        return self._operations

    # ------------------------------------------------------------------
    # Parameter management
    # ------------------------------------------------------------------

    def define(
        self,
        name: str,
        value: float,
        description: str = "",
        unit: str = "mm",
    ) -> ParameterDef:
        """Define or update a parameter."""
        p = ParameterDef(name=name, value=value, description=description, unit=unit)
        self._parameters[name] = p
        return p

    def get(self, name: str) -> float:
        """Get a parameter value."""
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not defined")
        return self._parameters[name].value

    def change_parameter(self, name: str, new_value: float) -> list[int]:
        """Change a parameter value and return indices of affected operations.

        Returns the operation indices that need to be replayed, starting
        from the earliest affected operation.
        """
        if name not in self._parameters:
            raise KeyError(f"Parameter '{name}' not defined")

        self._parameters[name].value = new_value

        # Find all operations that depend on this parameter
        affected = [
            op.op_index for op in self._operations
            if op is not None and op.depends_on(name)
        ]

        if not affected:
            return []

        # Everything from the earliest affected operation onward needs replay
        earliest = min(affected)
        return list(range(earliest, len(self._operations)))

    # ------------------------------------------------------------------
    # Operation recording
    # ------------------------------------------------------------------

    def record_operation(
        self,
        op_index: int,
        op_type: str,
        params: dict[str, Any],
        description: str = "",
    ) -> BoundOperation:
        """Record an operation, extracting @param bindings.

        Params containing "@param_name" strings are recorded as bindings.
        The actual values are resolved from current parameter values.
        """
        bindings: dict[str, str] = {}
        raw_params: dict[str, Any] = {}

        for key, value in params.items():
            if isinstance(value, str) and value.startswith(PARAM_PREFIX):
                param_name = value[len(PARAM_PREFIX):]
                bindings[key] = param_name
                # Store the reference, not the resolved value
                raw_params[key] = value
            else:
                raw_params[key] = value

        bound_op = BoundOperation(
            op_index=op_index,
            op_type=op_type,
            raw_params=raw_params,
            bindings=bindings,
            description=description,
        )

        # Extend or replace at index
        if op_index < len(self._operations):
            self._operations[op_index] = bound_op
        else:
            # Fill gaps if needed
            while len(self._operations) < op_index:
                self._operations.append(None)  # type: ignore
            self._operations.append(bound_op)

        return bound_op

    def clear_operations(self) -> None:
        """Clear all recorded operations (for full rebuild)."""
        self._operations.clear()

    # ------------------------------------------------------------------
    # Resolution and replay planning
    # ------------------------------------------------------------------

    def resolve_all(self) -> list[dict[str, Any]]:
        """Resolve all operations to concrete parameter values.

        Returns list of {op_index, op_type, params} with @refs replaced.
        """
        result = []
        for op in self._operations:
            if op is None:
                continue
            result.append({
                "op_index": op.op_index,
                "op_type": op.op_type,
                "params": op.resolve(self._parameters),
                "description": op.description,
                "bindings": op.bindings,
            })
        return result

    def get_replay_plan(self, changed_params: list[str]) -> list[dict[str, Any]]:
        """Get the operations that need replay after parameter changes.

        Returns resolved operations from the earliest affected onward.
        """
        earliest = len(self._operations)
        for param_name in changed_params:
            for op in self._operations:
                if op and op.depends_on(param_name):
                    earliest = min(earliest, op.op_index)

        if earliest >= len(self._operations):
            return []

        return [
            {
                "op_index": op.op_index,
                "op_type": op.op_type,
                "params": op.resolve(self._parameters),
                "description": op.description,
            }
            for op in self._operations[earliest:]
            if op is not None
        ]

    def dependency_graph(self) -> dict[str, list[int]]:
        """Return which operations depend on each parameter.

        Returns: {param_name: [op_index, ...]}
        """
        graph: dict[str, list[int]] = {name: [] for name in self._parameters}
        for op in self._operations:
            if op is None:
                continue
            for param_name in op.bindings.values():
                if param_name in graph:
                    graph[param_name].append(op.op_index)
        return graph

    # ------------------------------------------------------------------
    # Design table — variant generation
    # ------------------------------------------------------------------

    def design_table(
        self,
        param_ranges: dict[str, list[float]],
    ) -> list[dict[str, Any]]:
        """Generate design variants from parameter combinations.

        Args:
            param_ranges: {param_name: [value1, value2, ...]}

        Returns:
            List of variant dicts, each with parameter values and
            resolved operations ready for replay.
        """
        import itertools

        # Validate all params exist
        for name in param_ranges:
            if name not in self._parameters:
                raise KeyError(f"Parameter '{name}' not defined")

        # Generate all combinations
        names = list(param_ranges.keys())
        value_lists = [param_ranges[n] for n in names]
        combinations = list(itertools.product(*value_lists))

        variants = []
        for i, combo in enumerate(combinations):
            # Set parameter values for this variant
            variant_params = {}
            for name, val in zip(names, combo):
                variant_params[name] = val

            # Temporarily override parameters to resolve operations
            saved = {n: self._parameters[n].value for n in names}
            for name, val in variant_params.items():
                self._parameters[name].value = val

            resolved_ops = self.resolve_all()

            # Restore
            for name, val in saved.items():
                self._parameters[name].value = val

            variants.append({
                "variant_index": i,
                "parameters": variant_params,
                "operations": resolved_ops,
                "label": ", ".join(f"{n}={v}" for n, v in variant_params.items()),
            })

        return variants

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full parametric state."""
        return {
            "parameters": {
                name: p.to_dict() for name, p in self._parameters.items()
            },
            "operations": [
                {
                    "op_index": op.op_index,
                    "op_type": op.op_type,
                    "raw_params": op.raw_params,
                    "bindings": op.bindings,
                    "description": op.description,
                }
                for op in self._operations if op is not None
            ],
            "dependency_graph": self.dependency_graph(),
        }
