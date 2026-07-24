# Development Guidelines

These rules define how OpticalLoop should evolve. They keep the repository focused on a self-contained, Timeloop-backed optical computing simulator instead of a collection of paper-specific simulators.

## Planning Principles

- Use object-oriented structure for project-level components.
- Prefer clear classes with single responsibilities and simple composition.
- For simple features, design only for implemented requirements.

## Coding Principles

- Add comments for non-obvious logic, assumptions, and interface boundaries.
- Avoid broad or unnecessary `try/except`; prefer explicit checks and simple control flow.
- Follow KISS: use the simplest correct implementation and avoid unnecessary abstractions.

## Implementation Rules

- Prefer extending existing modules over creating parallel helpers.
- Keep raw external tool or library calls inside adapter layers.
- Update tests for behavior changes, or state why no test change is needed.

## OpticalLoop-Specific Boundary

Live simulation must stay on the single adapter path:

```text
OpticalLoop CLI/Python API -> TimeloopBackend -> workspace/scripts/utils.quick_run -> Timeloop mapper
```

Application code, notebooks, and docs can organize inputs and explain results, but they should not introduce a second energy, latency, cycle, area, or component-power simulator.

## Cleanup Standard

Keep the repository organized around the pieces needed to reproduce CIMLoop/Timeloop optical computing flows:

- core Timeloop-backed simulator objects;
- ROSA as the formal application workflow;
- DEAP-CNNs as a generic macro/workload notebook example;
- necessary vendored `workspace/` assets;
- focused tests and portable docs.

Avoid committing generated full result trees, local paper/reference context, or temporary experiments.
