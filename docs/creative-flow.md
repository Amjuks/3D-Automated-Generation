# Creative scene workflow

## Problems confirmed

The old planner asks every category to be a museum. Its small enum of furnishing generators constrains all resulting scenes. The compiler selects museum geometry from layout alone. Asset selection uses three fixed queries and placement consumes only the first material and model. Renderer fallback can silently omit lighting and previews.

## Implementation plan

1. Add a structured scene brief and human-readable scene.md before geometry. Describe category, composition, environment, lighting, buildings, floors, zones, materials and population. Persist each step.
2. Design one zone at a time, using only the master concept, its allocation and a short neighboring-zone summary. Give each zone a bounded object palette and semantic asset requests. Never ask the model to write executable code.
3. Use one category-independent compiler: arbitrary semantic object names resolve to models or individual data-only geometry recipes. No category-name dispatch is used in the active flow. Deterministic allocation owns placement, paths and floor connections. Keep the previous compiler for legacy checkpoints.
4. Resolve multiple texture/model requests by material and object role, vary equally relevant search results by seed, reuse downloads and preserve provenance. Report unavailable requests and every procedural fallback. Apply all selected assets, not just index zero.
5. Render with Blender when requested, including exterior cameras for outdoor scenes, and report renderer limitations. Generate a coverage/quality report rather than equating geometry validity with photorealism.
6. Test category differentiation, material/model assignment, bounded calls, resume, constraints and geometry validity; inspect representative rendered outputs.

## Scope and limits

This remains a bounded scene compiler, not an unrestricted modeling agent. Its bounded primitive vocabulary, object-local recipes and open asset catalog determine the attainable fidelity. People can use licensed local model assets; an LLM-designed assembly is an explicitly reported fallback. Realism needs PBR assets, lighting and visual review. Existing runs retain their saved workflow; create a new run to adopt the new design process.


## Online assets and rendering

The resolver searches [Poly Haven's public API](https://github.com/Poly-Haven/Public-API) and preserves its required credit. Asset records retain the source URL and [CC0 license](https://polyhaven.com/license), checksums and PBR texture dependencies. An optional local catalog can supplement the online catalog with any user-supplied models and their licenses; semantic tags map them to requested object roles. Downloads, texture resolution and asset count remain configurable budgets.

`examples/creative.yaml` requests 2K textures, up to 20 assets, a 350 MB selection budget, Cycles previews, and an editable Blender project. `require_blender: true` prevents a silent unlit GLB-only fallback. The installed portable Blender lives under `.cache/tools`; normal Blender installations and BLENDER_PATH take precedence.

## Outputs and resume

Each scene writes `brief.json`, `scene.md`, `zone-designs/`, `asset-requests.json`, `assets.json`, `object-recipes/`, `asset-coverage.json`, `quality.json`, and geometry/export checkpoints. Geometry recipes describe one object in a normalized allocation, never executable code. Failed model searches produce recipe requests, not a generic vase. A new run uses `workflow: creative`; snapshots predating that setting keep `legacy` to preserve their contracts.


The tested local endpoint sometimes corrupts `json_object` responses into malformed envelope keys. The creative profile therefore omits the provider response-format parameter (`text`) while retaining typed JSON validation. JSON-object mode also falls back to this transport after malformed structure. Single-field answer envelopes and exact Markdown JSON fences are accepted only when the inner payload passes the full requested schema.
