# Documentation visual style

Use the same NVIDIA color palette across workflow SVGs and Mermaid diagrams.

| Element | Style |
|---|---|
| Primary accent, scientific compute, accepted outcomes | NVIDIA green `#76B900`, with black labels |
| Control headers and primary text | Black or near-black `#111111` |
| Data artifacts | White fill, dark text, gray border |
| Runtime handling and secondary surfaces | Neutral gray fill, dark text |
| Arrows, separators and secondary text | Neutral gray with clear contrast |

Use Arial/Helvetica with a sans-serif fallback, simple rectangular shapes,
consistent spacing and thin connector lines. Label scientific operations and
outcomes explicitly so color is never the only way to distinguish them. Dashed
outlines denote Slurm jobs; solid outlines denote runtime containers.

Prefer editable SVGs for detailed architecture diagrams. Include a `viewBox`,
accessible `title` and `desc`, and self-contained styles. Check legibility at
the displayed documentation width. Mermaid diagrams must specify node and
connector colors rather than inherit a renderer's default palette.

The pipeline SVGs are the displayed diagrams; the older PNGs are retained as
historical assets. Keep diagram content aligned with the implementation and
the [pipeline overview](../pipeline-overview.md).

For official logo assets and their use, follow the
[NVIDIA brand guidelines](https://www.nvidia.com/en-us/about-nvidia/legal-info/logo-brand-usage/).
