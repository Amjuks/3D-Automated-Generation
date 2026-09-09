# Pipeline investigation and hardcoding audit

Investigated on 2026-09-09 against the source loaded from this checkout, the SQLite snapshot and JSONL events of `creative-sample-1-02`, saved scene/zone/object designs, a baseline test run, and live requests to the configured endpoint. Credentials and provider response/reasoning text are excluded from this report and diagnostic events.

## Root cause and evidence

The client buffered `/chat/completions` responses while this endpoint performed substantial reasoning and generation before sending a complete answer. HTTPX's read timeout expired during that silent interval. This was not evidence that the API was down. The old exception boundary then retried several unrelated conditions alike, replaced transport retry prompts with schema feedback, and provided no indication of headers, body progress, reasoning or JSON generation.

Observed evidence:

- The exact saved failing object request was approximately 3.8 KB, including a 2.1 KB schema. It contained one object, not the whole scene. The installed client was HTTPX 0.28.1; no HTTP proxy environment variables were present.
- Creative scene, zone and object requests are synchronous and sequential. Assets finish before object design; geometry workers start afterward. There was no concurrent geometry job, async event loop, local cancellation or exhausted connection pool explaining these timeouts. Historical `KeyboardInterrupt` entries occurred after users interrupted pending attempts.
- A fresh-connection buffered probe reproduced `ReadTimeout` at 180.16 seconds **before response headers**. The streaming version of the same payload received HTTP 200 and bytes at 5.43 seconds and continued receiving past 210 seconds, when the diagnostic stopped it. Fresh connections reproducing the problem argue against stale pooled connections as the primary cause.
- The revised client completed the saved inhabitant request in **109.96 seconds on one attempt**, with the idle read timeout still at 180 seconds. Headers arrived at 0.224 seconds; visible JSON began at 87.709 seconds. The result passed `ObjectRecipe` validation (five parts), and usage was recorded.
- Subsequent live scene planning explicitly reported thousands of `reasoning_chars` before the first content delta. Only counts are recorded. A successful brief took 90.15 seconds, with content beginning at 67.77 seconds.
- Historical logs include a 165.35-second successful object request and numerous 180.1-second timeouts. The local, pre-existing edit had also removed the configured `max_tokens` parameter while leaving it in configuration, making output budgeting ineffective. The temperature setting from that edit is preserved; token limits now work and can deliberately be omitted with `max_tokens: null`.

The historical `RemoteProtocolError` at 98.43 seconds means the remote HTTP exchange ended improperly. The old logs do not reveal whether the provider, a gateway, or another network hop closed it, nor whether a terminal answer existed before closure. That exact failure was not reproduced during the new live tests. Streaming reduces the silent interval and now distinguishes incomplete output from a valid terminal answer whose optional trailer is interrupted; it cannot guarantee that remote disconnections never happen.

HTTPX documents read timeouts as limits on waiting for network data, rather than total request duration: [HTTPX timeouts](https://www.python-httpx.org/advanced/timeouts/). OpenAI's streaming and schema contracts informed the compatible decoder and wire-schema conversion, but do not establish the third-party endpoint's implementation: [streaming](https://developers.openai.com/api/docs/guides/streaming-responses), [structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs).

## Execution path

`__main__ → cli.main → Pipeline(run_path)` loads configuration from SQLite, preserving the saved workflow and scene seeds. `Pipeline.run` holds the run lock, creates one shared synchronous LLM client and processes scene rows in order.

The creative path is:

1. `brief.design_scene`: input category, optional batch description, preferences, scene seed and recent concepts → typed `SceneBrief` → checkpointed `ZoneDesign` requests.
2. `creative.compile_creative` preflight: verify dimensions, allocations and packing before asset downloads and recipe calls.
3. `assets`: role-based environment/texture/model requests, local or Poly Haven selection, integrity-checked checkpoints.
4. `recipes.design_recipes`: reuse completed recipes; request one normalized assembly per distinct unresolved object design. Zero-count and acquired objects make no recipe calls.
5. `layout` and `creative`: seeded placement, optional paths routed around actual zone footprints, rectangular building envelopes and floor connections, arbitrary semantic objects assembled from geometry operations/assets.
6. Shared contracts → parallel procedural geometry → bounded validation/repair → Blender or Trimesh export → quality/coverage reports and checkpoints.

Legacy snapshots without a workflow retain `legacy`. Its old planners, schema enums, museum layouts and fixed furnishings are isolated under `scene_generator/legacy/`. The top-level `museum.py`, `planning.py` and `decomposition.py` are import compatibility shims. The creative path does not import these compilers; a subprocess regression checks that boundary. Explicit legacy mode remains available for compatibility and still has its historical content restrictions.

## Implementation changes

- Streaming SSE decoder handles chunk boundaries, comments, reasoning-only events, terminal reasons, usage events and buffered JSON responses. Only fully validated terminal answers enter the cache. Partial outputs are discarded on retry.
- Connection, write, pool and idle-read limits are separate; connection limits match configured concurrency. Streams close on success, failure and cancellation. Response bytes and elapsed receive time are bounded. The elapsed-time check runs on incoming blocks; an already blocked read retains its idle timeout.
- Transient network failures and HTTP 408/429/500/502/503/504 retry with jitter and `Retry-After`. Transport retries preserve the payload. Local protocol/pool errors, serialization/storage errors, refusals and output-limit failures are not disguised as repeated API requests.
- Logs distinguish queue time, headers, first bytes, reasoning/content counts, completion and schema validation. No provider bodies, credentials or reasoning text are logged.
- Strict schema conversion requires all declared fields, removes defaults and converts homogeneous tuple schemas without deleting fields named `title`. Local Pydantic validation remains authoritative.
- LLM cache keys include output settings and validation context. Recipe identity includes name, material, visual detail and dimensions; request context includes the scene concept, requested geometry quality and derived seed. Existing local recipe checkpoints remain reusable.
- Default values are validated/coerced consistently. This fixes an observed `2` versus `2.0` cache-key difference between fresh and restored objects that previously broke seeded replay.
- Trimesh receives floating-point RGBA values. Integer `(1,1,1,1)` material copies had been interpreted as byte colors, exporting white at 1/255 intensity/opacity. Export fingerprints are versioned so existing geometry can be re-exported with the fix.
- Asset repairs are reflected in coverage as placeholders, instead of incorrectly reporting repaired bounding boxes as successful model instances.

## Hardcoding audit

| Location / assumption | Classification and resolution |
| --- | --- |
| Museum/gallery category dispatch, collection themes, poetic titles, gallery counts/layouts, prescribed exhibits, residential furniture and style ornament | Example-specific legacy functionality. Isolated the planners, compilers, schemas and asset queries under `legacy/`; retained checkpoint compatibility. No category dispatch in the creative path. |
| Creative plan coerced into `ScenePlan(collection="art", ...)` | Semantic leakage. Replaced with generic export/report metadata (`SceneSummary`) and retained the complete brief as content authority. |
| Default neoclassical/modern/brutalist styles and urban/forest/coastal environments | Input configuration, not universal scene assumptions. New defaults are empty; explicit preferences reach creative planning. Legacy defaults stay inside its compatibility planner. |
| Human-sized, fabric-clothed population injected after zone validation, with a short human-name detection list | Inappropriate content logic. Population is now ordinary LLM-designed object data with arbitrary kind, dimensions and materials. Its assigned count is validated; no post-validation append or human default geometry. |
| Objects duplicated on every floor | Count semantics bug. Counts now apply across the zone. Explicit floor targeting/repetition is supported. Old zone checkpoints retain historical repetition when the new field is absent. |
| Mandatory furnished zones, positive-only object counts, slug-only material names | Unnecessary semantic restrictions. Empty zones, zero-count objects and free-form material names are accepted. Zero counts trigger neither assets nor recipes. |
| Fixed promenades drawn from a grid even for freeform layouts | Incorrect layout logic. Paths are optional; requested circulation routes around the actual zone footprints and reaches their entrances. |
| Clustered placement identical to scattered, perimeter using only two edges, open objects sized as if inside a two-sided room | Generalized placement. Arrangements have distinct seeded behavior; open zones use their area, interior aisles remain reserved, and a complete packing fallback avoids mixing conflicting random and grid placements. |
| Mandatory windows/roof | Zone choices now control both. Rectangular envelopes, bounded floors and minimum passage dimensions remain supported geometry infrastructure. |
| `COLLECTION / MUSEUM` footer on every printed label | Example-specific content in a shared utility. Removed the footer and replaced the fallback label text with the actual node name. |
| `sculpture` always producing a classical bust; `tree` producing an organic blob | Misleading geometry names. New recipes use explicit `bust`/`organic` names; old names decode compatibly. Added cone, capsule and profile-driven lathe operations to expand arbitrary assemblies. Named shape operations are not semantic object selection. |
| Fixed run seed and missing seed in zone/recipe context | New runs choose and persist a fresh seed unless an integer is supplied. All design stages derive seeds from it. Explicit seeded runs remain reproducible offline; live cache misses remain probabilistic. |
| Slug collisions between arbitrary category names | Unique deterministic scene prefixes now resolve repeated collisions, preserving every input category. |
| Box dimensions, Z-up contracts, resource/schema caps, geometric primitive tessellation, material defaults, supported render/export modes, standard lighting and camera fallbacks | Required bounded infrastructure, retained. These are documented capability limits, not an object/scene catalog. |
| `mock_design.py`, mock callbacks, test fixtures and `examples/*.yaml` | Offline test data and explicit example inputs. Retained outside live semantic decisions. The creative profile now defaults to a fresh saved seed. |
| `build/lib`, `.cache`, existing `runs` | Ignored generated copies/artifacts, not active source. Import inspection confirmed this checkout's `scene_generator` was loaded. Original run artifacts were not rewritten for the diagnostics. |

The review also covered assets/downloads, surface generation, geometry operations, spatial validation/repair, checkpoints, statistics, CLI, exporters and both backends. No additional scene-category routing was found in the active creative flow.

## Validation

- Baseline: **59 tests passed** before implementation.
- Final full suite: **98 tests passed in 173.96 seconds**, including the installed headless Blender integration. The final quality-context and manifest-accounting changes were followed by the focused creative/generic suite: **23 tests passed**. Ruff lint, Ruff formatting and `git diff --check` passed; every example YAML parsed successfully.
- Tests cover stream fragmentation and usage, incomplete versus terminal disconnects, real local HTTP streaming beyond an idle-time interval, retry classification, local failure propagation, deadlines/byte limits, schema conversion, cache identity, fresh/explicit seeds, meaningful placement variation, generic input preferences, arbitrary population, empty/zero-count zones, path clearance, floor count semantics, new shape operations, material colors and resume/export compatibility.
- Existing headless Blender integration remains part of the suite when the installed portable Blender is discovered.
- A copy of the interrupted scene completed with **1,249,910 triangles** and then resumed with **zero geometry regenerated**. Saved briefs, zone designs, existing object recipes and 19 downloaded assets were reused; the three missing recipes were supplied in mock mode for this assembly test. Nine invalid imported instances went through the existing bounded repair and are now honestly reported as placeholders. This was not a complete live regeneration of the original seven-scene batch.
- A separate live input, `resonant observatory`, requests one uninhabited open zone and two invented acoustic instruments without buildings or paths. Its events demonstrate actual reasoning traffic and successful schema-correction retries. **Completed and exported**, then resumed with zero new LLM calls and zero regenerated geometry. It produced one open zone, no paths/population, and two different instrument objects (228 triangles at draft quality). Across the entire diagnostic, four schema errors were visible; the original over-restrictive material/count rules were generalized, and later genuine length/geometry errors were corrected by bounded retries. Standalone validation and GLB export/reload also passed.

## Limits

The third-party endpoint can still take minutes to reason and can return invalid schema values, as the live tests demonstrated. These are observable and bounded, not hidden. Streaming does not make inference instantaneous or reveal which network hop caused a historical disconnect. Strict schema mode was tested at the wire/validation level, not against this provider's account.

This is a generic **bounded** scene compiler: axis-aligned allocations, rectangular building envelopes, limited floors, resource budgets, procedural approximations and heuristic asset matching remain. It is not an unrestricted topology generator, navigation/physics solver or guarantee of photorealism. Legacy saved designs retain their existing semantic content. Starting a new creative run uses the generalized design choices.


### Reproduction commands and artifacts

```bash
.venv/bin/pytest -q
.venv/bin/pytest -q tests/test_generic_generation.py tests/test_creative.py
.venv/bin/ruff check scene_generator tests
.venv/bin/ruff format --check scene_generator tests
git diff --check
.venv/bin/python -m scene_generator generate --config /tmp/scene-diagnostics/live-smoke.yaml
.venv/bin/python -m scene_generator resume --run-id generic-live --root /tmp/scene-diagnostics/live-runs
.venv/bin/python -m scene_generator validate --scene /tmp/scene-diagnostics/live-runs/generic-live/scenes/resonant_observatory-01
.venv/bin/python -m scene_generator export --scene /tmp/scene-diagnostics/live-runs/generic-live/scenes/resonant_observatory-01 --format glb
```

Local diagnostic artifacts are under `/tmp/scene-diagnostics/`: the isolated original-object request (`fixed-request/`), cloned interrupted scene (`resume-runs/resume-copy/`), and complete live scene (`live-runs/generic-live/`). They are outside the source tree. The live run was resumed while fixes were being validated, so its cumulative event history includes failures from intermediate revisions. The original `runs/creative-sample-1-02` was left available for the user's normal resume command.
