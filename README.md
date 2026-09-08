# Scene Generator

A runnable Python pipeline that turns category counts into furnished, connected 3D scenes. It separates semantic design from deterministic spatial allocation and geometry, checkpoints every component, and exports one GLB per scene.

Offline mode exercises the pipeline with placeholder design data; use live mode for semantic creativity. Live design uses an OpenAI-compatible endpoint; geometry remains procedural. Blender provides headless export, PBR materials, lighting and rendered previews. Trimesh provides a portable GLB fallback.

**Current workflow:** category names are free-form. New runs expand each category into a written scene brief, individual zone designs, semantic asset requests and object-local geometry recipes. A shared spatial compiler assembles the result; it does not route every category into museum geometry. Existing checkpoints retain their legacy workflow.

## Creative scenes

```bash
python -m scene_generator generate --config examples/batch.yaml --settings examples/creative.yaml --name creative-batch
```

Keep any category/count mapping in `batch.yaml`. The creative profile uses your `.env` LLM connection, multiple online PBR textures and models, high-detail geometry and Blender Cycles rendering. It requires Blender (system install, `BLENDER_PATH`, or the project's portable installation). Change the budgets and render resolution in `examples/creative.yaml` as needed.

The flow works in small, checkpointed steps:

1. Expand the category into `brief.json` and `scene.md`: concept, environment, spatial composition, building floors, zones, lighting and people.
2. Detail one zone at a time, including finishes, arbitrary object names and asset searches.
3. Retrieve assets by role, using multiple materials/models and seeded tie-breaking among equally relevant results. Optionally supply `assets.local_catalog` with tagged models to supplement Poly Haven.
4. Design each missing object separately as a validated geometry recipe, then compile geometry while preserving circulation and floor connections.
5. Validate and export. Inspect `asset-coverage.json` and `quality.json` for missing assets, procedural approximations, scaled objects and rendering limitations.

`scene.md` is a generated view of the structured design, not executable code or a Markdown input parser. The creative profile produces `scene.glb`, overview `scene.png`, eye-level `detail.png`, and `scene.blend` when rendering completes. Re-running `resume` reuses individual design and geometry checkpoints. Start a new run to use this flow for batches that already generated legacy scenes.

**Fidelity limits:** accepting arbitrary category names does not guarantee accurate or photorealistic models for every idea. The available assets and geometric primitives limit the result. Building envelopes use a bounded spatial grammar (up to three floors); object recipes can compose arbitrary semantic objects, but characters and complex organic forms may remain approximations. Weather is recorded as art direction; it is not a fluid or volumetric simulation. Read the [implementation plan and limits](docs/creative-flow.md).

## Quick start

Python 3.10+ on Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python -m scene_generator generate --config examples/small.yaml
```

On Debian/Ubuntu, install the distribution's `python3-venv` package if `venv` reports that `ensurepip` is unavailable. Alternatively, use `python -m virtualenv .venv` if virtualenv is installed.

A generated [sample GLB](examples/artifacts/room.glb), [rendered preview](examples/artifacts/room.png) and [validation report](examples/artifacts/room-validation.json) are included.

The small example uses mock LLM and asset modes. It needs no API key, internet connection, or Blender installation. Progress goes to stderr; the machine-readable summary goes to stdout. The run ID and directory are printed immediately.

```bash
python -m scene_generator generate --config examples/museums.yaml
python -m scene_generator status --run-id <id>
python -m scene_generator resume --run-id <id>
python -m scene_generator validate --scene runs/<id>/scenes/<scene-id>
python -m scene_generator export --scene runs/<id>/scenes/<scene-id> --format glb
```

For a custom `generation.output.root`, pass `--root /path/to/runs` to `status` and `resume`, or set `SCENE_GENERATOR_HOME`. Resume uses the saved configuration and absolute paths, so changing the working directory does not change scene generation. Do not move a run and its asset cache without updating its path references.

Live LLM requests print `llm_request`, `llm_success`, `llm_error`, and `llm_retry` events to stderr and save them in the run's `events.jsonl`. Validation failures include field paths and error types; HTTP failures include status codes. A single-field answer envelope or a JSON Markdown fence is unwrapped; its inner object must still validate against the same schema. A request can remain pending until the configured timeout, and `retries: 3` allows four attempts. A successful short chat request verifies connectivity, but scene generation also requires a complete response that passes the scene schema. Resume reads LLM settings from the SQLite configuration snapshot; editing the original YAML or the informational `config.json` does not change an existing run. Endpoint, model, and credentials are read from the current environment (including `.env`).

Exit codes: `0` success, `1` runtime failure, `2` invalid input or failed standalone validation, `130` interrupted. A failed export leaves completed geometry available for resume.

## Generate a batch from category counts

Put your categories and desired counts in `examples/batch.yaml`:

```yaml
museum: 10
aquarium: 5
```

Run it with the high-quality live settings from your existing museum config:

```bash
python -m scene_generator generate --config examples/batch.yaml --settings examples/realistic-museum.yaml --name mixed-venues
```

This creates one resumable batch containing `museum-01` through `museum-10` and `aquarium-01` through `aquarium-05`, each with its own `scenes/<scene-id>/scene.glb`. Scenes run sequentially; room-design requests and geometry work use the configured concurrency. The terminal shows the current scene and total count. If a scene fails, the batch stops; fix the cause and run `python -m scene_generator resume --run-id <printed-id>` to reuse saved work.

`--settings` copies the source file's entire `generation` block, ignoring its scene counts and output name. `--name` names the new batch. Omit `--settings` to use defaults (live LLM, standard quality, procedural assets, GLB export); the original full YAML format still works. Credentials come from your existing `.env`.

Each scene gets a distinct, reproducible seed. Styles cycle first, then environments; combinations repeat when the count exceeds their product, while seeds and design context continue to vary. Counts request separate variations, but do not guarantee visually unique LLM designs. Legacy workflows cycle the listed styles; creative workflows invent their own category-appropriate composition and environment. The creative workflow expands category-specific objects through asset searches and individual geometry recipes; it does not use category-specific routing.

## Live LLM setup

```bash
cp .env.example .env
# Edit .env locally, then:
python -m scene_generator generate --config examples/live.yaml
```

```dotenv
OPENAI_BASE_URL=https://api.tensorstudio.ai/v1
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt
```

Credentials are read from the environment and never written to manifests, SQLite or logs. `.env` is gitignored. Blender child processes receive no `OPENAI_*` environment variables. LLM error records contain error classes/status codes, not provider response bodies or prompts.

The client uses `/chat/completions`. `json_object` is the default; the creative example uses `response_format: text` for providers whose JSON-mode decoding corrupts the output. Text mode omits the API response-format parameter but still requires a JSON answer. JSON-object requests retry in text mode after malformed structure; strict `json_schema` requests keep their explicit format. every response is still validated by Pydantic. Select `json_schema` only if your provider supports structured outputs and the supplied schemas. Provider/model availability and compatibility must be tested against your account. See the [official structured outputs documentation](https://developers.openai.com/api/docs/guides/structured-outputs).

Only the compact scene plan and individual room design decisions require LLM calls. A room request contains its contract, a small neighbor description, selected design tokens and its seed. It never contains the complete scene. Model-generated code is never executed.

Normalized request hashes include model, endpoint, mode, schema and input. Only valid typed responses are cached. Invalid schema responses retry the same component with compact field/type feedback. Transport retries use exponential backoff, `Retry-After`, timeouts, a request spacing limiter and a concurrency semaphore. Local cache hits incur zero provider tokens; provider cached tokens are reported separately. Mock calls report zero tokens rather than invented usage.

Costs remain `null` until input/output prices are configured. For example, set the following **to your provider's actual prices**, in currency units per million tokens:

```yaml
generation:
  llm:
    input_cost_per_million: 1.0
    output_cost_per_million: 2.0
    cached_input_cost_per_million: 0.5
```

Those numbers are configuration examples, not advertised model prices. Unreported usage from failed provider requests cannot be measured; reconcile with your provider's billing records.

## Configuration

A category mapping is enough. With no generation settings, live LLM mode is used:

```yaml
scenes:
  museum: 10
  mansion: 5

generation:
  seed: 42
  quality: high                  # draft | standard | high
  variations:
    architectural_styles: [neoclassical, modern, brutalist]
    environments: [urban, forest]
  frameworks: [blender, trimesh] # ordered backend preference
  workers: 4
  max_scene_triangles: 2000000
  max_texture_bytes: 256000000
  repair_attempts: 2
  llm:
    mode: mock                  # live | mock
    concurrency: 4
    requests_per_minute: 30
    timeout: 60
    retries: 3
  assets:
    mode: mock                  # mock | local | polyhaven
  output:
    format: glb                 # glb | obj | ply
    root: runs
    preview: false              # Blender only
    component_glbs: false
```

Unknown keys and unsupported formats/backends are rejected. Counts must be positive integers. All distances use meters, with Z up in contracts and Y up in GLB exports.

Optional site limits:

```yaml
generation:
  dimensions: [30, 24, 6]
  bounding_space:
    min: [100, 200, 0]
    max: [130, 224, 6]
```

The compiler fits room counts/proportions before placing furniture. It rejects spaces too small for the supported grammar; it does not shrink doorways or people-scale furniture to force a fit. The minimum one-room site is approximately 11 × 14 × 3.5 m including circulation and landscape margins.

Scene seeds are derived from the run seed, category and ordinal; component seeds are derived from stable hierarchical IDs. Mock/procedural geometry is reproducible with the same installed dependency versions. A live LLM is not guaranteed deterministic on a cache miss. To make a new variation, create a new run with a different seed/configuration.

Variations change room count, one- versus two-sided hall layout, room proportions, collections, furnishing density, environment, materials and atmosphere. The default museum batch spans sculpture, antiquities, science, art and natural-history concepts. Arbitrary style words affect semantic planning/material context; dedicated geometry details currently exist for the included architectural styles.

## Blender

Install Blender 4.x and put `blender` on PATH, or set:

```bash
export BLENDER_PATH=/absolute/path/to/blender
python -m scene_generator generate --config examples/small.yaml
```

The worker starts Blender with factory settings, disabled auto-execution, an explicit error exit code, and a configurable `blender_timeout`. It creates geometry from validated NPZ data, preserves object hierarchy, reuses mesh datablocks, applies PBR textures, adds exported point lights and optionally renders a camera preview using Cycles. A portable Blender 4.5.9 build was exercised during development.

If Blender is unavailable, the next configured adapter is selected and the actual backend is recorded. A Blender runtime failure is surfaced with `blender.log`, rather than silently downgrading the result. A Blender-only configuration fails cleanly at export with resumable geometry checkpoints.

GLB preserves hierarchy, mesh instances, materials and textures. OBJ/PLY are geometry-oriented alternatives; they do not preserve the same PBR/lighting/hierarchy semantics. Trimesh does not export the Blender lighting/preview setup. HDRIs illuminate Blender previews; the GLB does not embed a world-environment extension.

## Assets and Poly Haven

**Powered by Poly Haven** — live assets come from [Poly Haven](https://polyhaven.com), whose assets are CC0. This client uses the [official API](https://api.polyhaven.com) and its [documented attribution/User-Agent requirements](https://polyhaven.com/our-api).

```yaml
generation:
  assets:
    mode: polyhaven
    cache_dir: .cache/scene-generator/assets
    offline: false
    max_download_bytes: 30000000
    max_assets: 3
    required: false
```

Asset discovery ranks `/assets` metadata by semantic query, then examines `/files/{id}`. It selects floor textures, an environment HDRI and a collection model where available. It prefers small 1K variants within the download budget. Asset matching is heuristic, not a learned aesthetic ranking system.

Supported assets:

- PBR color, roughness and OpenGL normal maps, applied to floor/stone materials.
- HDR/EXR environments, applied by Blender.
- GLB models and glTF models with declared external buffers/textures. Bundles are checked for path traversal and undeclared references, then cached as portable GLB files. Models retain packed base-color appearance and UVs and are uniformly fit to their allocation. Imported multi-material/normal/metallic appearances are not fully preserved by atlas conversion.

Downloads are streamed with byte caps, MD5 checks when provided, and SHA-256 cache verification. Original metadata, source links, licenses and dependency hashes are saved. `offline: true` forbids API/download requests and uses populated metadata/file caches. A missing optional asset records an error and keeps procedural content; `required: true` stops at the asset stage. Mock mode uses procedural materials/objects and makes no asset requests.

Local mode uses a JSON catalog, with paths relative to the catalog:

```json
[
  {"id": "my-statue", "type": "model", "path": "statue.glb", "license": "CC0-1.0"},
  {"id": "my-floor", "type": "texture", "path": "wood.jpg", "license": "user-provided"},
  {"id": "my-sky", "type": "hdri", "path": "sky.hdr", "license": "user-provided"}
]
```

```yaml
generation:
  assets:
    mode: local
    local_catalog: /path/to/catalog.json
```

Texture files are embedded during GLB export. Manifests continue referencing the cache, which must remain available for regeneration. Imported meshes that fail the supported manifold/budget checks are replaced with a recorded procedural fallback during repair; downloaded assets are not automatically trusted as valid geometry.

## Architecture and checkpoints

```text
scene_generator/
  config.py, models.py           typed YAML, plans and component contracts
  llm.py, planning.py            compact typed design requests and caches
  decomposition.py, spatial.py  parent-owned allocations and spatial index
  generators.py                 reusable parametric geometry and UVs
  assets.py, polyhaven.py        asset selection, provenance and caching
  backends/                     Trimesh adapter and headless Blender worker
  validation.py, repair.py       deterministic checks and bounded repairs
  checkpoint.py, migrations/    SQLite state and ordered schema upgrades
  logging.py, statistics.py     JSONL events, token/cost and run reports
  exporters.py, pipeline.py      orchestration and complete-scene exports
  cli.py                        command-line entry points
```

A scene is decomposed through site → building → floor → room → structural elements/furniture → collection objects/books → small labels/reliefs. Each component has a stable ID, reciprocal parent/child links, allocation bounds, local/world transforms, sockets, reserved/clearance volumes, materials, parameters, dependencies, budgets, seed and validation state. Transform matrices are computed from the typed transforms. Empty fields explicitly mean no such constraint for that component.

```text
runs/<run-id>/
  config.json
  state.sqlite                  WAL mode; user_version migrations
  events.jsonl
  summary.json
  scenes/<scene-id>/
    plan.json
    room-designs/*.json          saved immediately after each typed design
    scene.json                  compact manifest of component paths
    design-tokens.json
    assets.json
    indexes/{spatial,dependencies,connections,assets}.json
    components/scene/site/building/floor-0/room-0-0/...
      contract.json             original compiler allocation
      spec.json                 current validated/repaired specification
      geometry.npz              pickle-free local geometry checkpoint
      geometry.glb              optional component export
    validation.json
    export.json
    report.json
    scene.glb
    scene.png                   optional Blender preview
    blender.log                 Blender diagnostics
```

SQLite stores run/scene/component stages and statuses, attempts and timing, dependencies, validation history, LLM calls/usage, asset events/provenance and file hashes/statistics. Reports distinguish cached input from local response-cache hits. Run summaries aggregate geometry, assets and generated files; scene reports include coverage, component counts, materials, textures, export backend and file size.

Atomic file replacement precedes marking a checkpoint complete in SQLite. An orphaned file after interruption is harmless: resume checks status, spec hash, dependencies and geometry hash before reuse. A corrupt leaf regenerates independently. Changed contracts invalidate only their dependency descendants. Valid geometry and exports are reused; export corruption triggers re-export. An advisory run lock prevents simultaneous writers and is released by the operating system on process exit. Current locking targets POSIX systems.

Repairs restore drifting allocations from their original contracts, regenerate invalid meshes, reduce procedural detail, or replace invalid imported assets with a supported primitive. Repair passes are bounded. Unsupported structural conflicts stop with a report rather than marking the scene valid. Repair is currently deterministic; live LLM-driven geometry repair and arbitrary bespoke mesh generation are extension points.

## Validation coverage and limits

| Check | Implemented coverage |
| --- | --- |
| Bounds/transforms | Finite typed values, positive extents, allocation containment, local/world composition, mesh containment |
| Hierarchy/dependencies | Unique IDs, reciprocal links, missing targets, cycle detection |
| Collisions | Spatial-hash broad phase with conservative AABB overlap; touching faces are allowed |
| Floating/attachments | Declared support contact, footprint overlap, support-socket alignment and opaque wall-mount contact |
| Doorways/paths | Reserved circulation/approach volumes and minimum doorway widths |
| Connectivity | Room graph, geometric portal positions and reachability from an exterior entrance |
| Mesh integrity | Face indices, finite vertices, degenerate faces, welded watertightness and winding consistency |
| Materials/UVs | Required materials, UV dimensions/finiteness and texture presence |
| Scale/budgets | Template room/door ergonomics, component/scene triangles and aggregate encoded texture bytes |
| Export | GLB reload and nonempty geometry; Blender process status and output existence |

AABB checks can reject valid concave interlocking shapes. They are not an exact triangle collision engine. Manifold checks do not establish absence of all self-intersections. Support checking is not rigid-body physics and does not prove stability of every furniture part or wall mount. Path checks assume the generated straight-hall layout; there is no general navmesh solver, stairs/elevators, multi-storey routing, building-code certification or arbitrary rotations. Texture budgets measure encoded file bytes, not GPU memory. Detailed per-material performance tuning and photorealistic art direction require additional authored generators/assets.

The adapter protocol supports future USD/engine exporters. Open3D is an optional installation extra for downstream processing; there is no Open3D backend or automatic Open3D pipeline stage in this version. Unsupported adapters/formats are rejected explicitly.

## Tests and development

```bash
pip install -e '.[test]'
pytest -q
ruff check scene_generator tests
ruff format --check scene_generator tests
# Include the real headless Blender integration test when available:
BLENDER_PATH=/absolute/path/to/blender pytest -q
```

Tests exercise invalid configs/contracts, contact semantics, LLM schema retries and redaction, response-cache accounting, asset integrity/offline reuse, bundle path safety, textured local assets, schema migrations, run locking, scene variation, deterministic exports, interrupted resumes, selective repair, constrained sites, validation failures and Blender fallback/headless execution. Network calls are mocked in the automated suite. Live Poly Haven and a portable Blender build were also exercised manually; a real LLM provider call requires your credentials.

To extend the system, add a typed design choice and a deterministic generator, allocate its children in the parent grammar, and test its contracts/geometry. Keep LLM outputs declarative. Add adapters through `backends.Backend` and `exporters.py`; do not put engine-specific code into the spatial compiler.
