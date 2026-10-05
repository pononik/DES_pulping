# DES Pulping (LA:ChCl) — Process Model and LCA vs Kraft

This repository contains a conceptual process model and life cycle assessment (LCA) for pulping with a deep eutectic solvent (DES) composed of lactic acid (LA) and choline chloride (ChCl), compared against the kraft benchmark.

## Summary
- Motivation: Kraft pulping is industrially dominant but has low material efficiency (<50% yield), complex causticizing recovery, and high energy intensity.
- Approach: Build a cradle-to-gate inventory for unbleached pulp using LA:ChCl based on lab-scale literature and process design; compare to kraft using consistent system boundaries.
- Process highlights:
  - Yield: 50–60% unbleached pulp
  - Lower process temperature: ~130 °C
  - Chemical recovery via evaporation
  - Lignin coproduct
- LCA findings:
  - DES (LA:ChCl) pulping shows lower climate change impacts than kraft.
  - Sensitivity drivers include solvent recovery efficiency, pulp yield, and energy supply.
- Bayesian optimization (BO):
  - A scalarized BO was performed to balance performance objectives and constraints.
  - The recommended operating window is 120–130 °C and 3.26–4.46 h at liquor-to-wood (L/W) ratios of 5–10.
- Implication: DES pulping shows potential for industrial implementation, contingent on solvent loss minimization and robust recovery.

## Repository contents
- `scripts_des/` — Process models, inventory construction, and LCA scripts (EF method ready)
- `LICENSE`, `CITATION.cff` — Licensing and citation info

## How to cite
Please cite the repository (see `CITATION.cff`) and the associated study when available.
