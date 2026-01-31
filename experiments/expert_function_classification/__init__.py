"""Expert Function Classification Experiment.

Classifies MoE experts by function via systematic ablation:
- Storage (externalizable): retrievable knowledge, recoverable via memory bank
- Computation (irreducible): transformations that produce structure errors when removed
- Routing (structural): experts whose removal disrupts downstream expert selection
- Redundant (removable): experts whose ablation causes no measurable change
"""
