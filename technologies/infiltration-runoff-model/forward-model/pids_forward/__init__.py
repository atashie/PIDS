"""PIDS Pillar-2 forward hydrologic model (modular DOLFINx/FEniCSx FEM).

Built bottom-up, one module at a time, each gated by the three-tier sanity
routine (``governance/claude-sanity-check-routine.md``). The first module is the
subsurface: a dimension-agnostic, mass-conservative (mixed-form) Richards solver.

Design: ``docs/plans/2026-06-04-pids-forward-model-architecture-design.md``.
"""
