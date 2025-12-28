# AI usage policy and prompt design

## Allowed AI tasks (constrained)
- Flattening (planarizing the saree image)
- Texture warping to follow body curvature (constrained)
- Lighting/shadow adaptation to match model asset

## Forbidden AI behaviours
- Inventing motifs or borders
- Color hallucination or color replacement beyond ΔE tolerance
- Adding embellishments, logos, or duplicated patterns
- Making creative alterations to motifs or border geometry

## Prompt template (example)
System: "You are a deterministic image assistant. Preserve motifs, borders and colors. Do NOT change pattern layout or invent details."

User: "Flatten the saree in `S_clean.png`. Output requirements: no folds, neutral background, preserve border geometry and motif scale. Allowed: minimal smoothing, lighting normalization. Forbidden: motif scaling, border redesign. Log actions and provide minimal metadata."

## Retry injection
On failure, append targeted instruction:
"Previous output failed due to BORDER_DISTORTION. Preserve border geometry exactly; do not change border width or motif spacing. If warping, use only the provided placement map."

## Determinism
- Supply same prompt + seed and same assets → deterministic behavior from AI adapter wherever possible.
- Log prompt + response in `retry_log.json`.
- Track and persist the exact prompt used per attempt along with any model parameters.

## Implementation notes
- Abstract AI usage behind an adapter interface so model backends (local or API) can be swapped.
- Adapter must accept:
  - `seed` (if model supports)
  - `strict_instructions` (text)
  - `placement_map` (per-pose)
  - `forbidden_changes` (list)
- Log the adapter input and output for reproducibility and debugging.
