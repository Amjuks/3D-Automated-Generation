# Version 2 design graph

New configurations default to `generation.workflow: graph`. `creative` selects the earlier zone workflow and `legacy` selects the template workflow for compatibility. Saved snapshots keep their workflow; snapshots without a workflow retain the original legacy interpretation. Neither old plans nor zone briefs are automatically converted into graphs.

Run `python -m scene_generator generate --config examples/graph.yaml`. The example uses an explicitly labelled abstract mock fixture. Change `llm.mode` to `live` for natural-language design. Existing endpoint configuration and typed transport retries apply. A live provider run is necessary to assess semantic and visual fidelity; offline tests do not establish it.

## Contracts and ownership

- `models.py`: shared versioned component contracts, materials, bounds, and transforms. Version 1 retains translation-only behavior. Version 2 supports rotation, scale and affine local/world frames, including composed shear.
- `design.py`: hierarchical semantic graph, relationships, anchors, explicit envelopes, polygonal regions, reusable assemblies, asset requests, provenance and diagnostics. All architectural elements are optional. Repetition requires explicit instance frames; count zero produces no geometry.
- `policy.py`: user site bounds, geometry/memory/checkpoint limits, available operations, tolerance and solver limits, optional route width and reachability. Existing configuration continues to own download, response, worker, timeout and render limits.
- `graph_planning.py`: nine cumulative structured LLM stages. Inputs include the prior graph and policy. Outputs preserve stable IDs. Each stage has an input identity, typed output, checksum and atomic checkpoint. Changed inputs invalidate downstream stages; a damaged checkpoint is retained as one bounded `.invalid.json` copy and regenerated through the validated request cache.
- `graph_spatial.py`: composed local frames, relationship projections, derived envelopes, spatial checks and constraint diagnostics. Solver adjustments record original/new local matrices, the forcing constraint and intent impact.
- `capabilities.py`, `graph_compile.py`: explicit geometry operations compiled into reusable component geometry. Semantic names never select geometry. Unsupported operations fail with diagnostics. Missing assets follow the requested recipe, omission or error policy; no implicit placeholder is inserted.
- `graph_render.py`: explicit camera, light and world-atmosphere data with backend compatibility checks. Blender graph export does not add room lights, sun, floor, roof or camera. Trimesh supports one camera; unsupported lighting is reported or rejected when required.
- `graph_pipeline.py`: compile, generate, validate, export and resume. Manifest, graph, recipes, components, assets, completion checkpoint and export identity carry version information. Geometry and export checksums protect resume. Corrupt component specs are rebuilt from the authoritative graph on resume.

`scene.json` version 2 points to the graph and records derived bounds and policy. Components retain their source graph node IDs; render meshes are flattened into world affine frames without losing the semantic hierarchy stored in the design graph. `resolved.json` stores matrices, envelopes and diagnostics. `asset-manifest.json` records chosen candidates, licenses, source checksums, conversion policy and fallback reasons. `quality.json` distinguishes geometry validity from unassessed creative fidelity. The LLM never supplies executable Python or Blender code.

## Supported operations and limits

The initial graph geometry registry supports box, sphere, cylinder, cone, capsule, torus and explicit triangle meshes. Closed triangle meshes permit irregular terrain, architecture, openings and arbitrary forms without semantic dispatch. Unsupported booleans, sweeps and other operations fail explicitly; the planner must choose a supported representation. Material IDs and names are arbitrary.

Relative frames, anchors, attachment, vertical support, directed adjacency, directed separation, axis-aligned containment and orientation can project coordinates. Missing anchors, contradictory constraints and unresolved relationships identify the constraint and both nodes. Polygon containment and collision checks are conservative envelope tests; containment samples polygon corners and does not certify arbitrary concave boundaries. Visibility and route checks use straight segments against obstacle envelopes. Curved-route navigation, exact mesh collision, structural engineering and physical accessibility are not certified. Approximations are reported. Required failures block export; optional unsatisfied relationships are explicitly marked relaxed. No automatic scene-concept repair occurs.

Independent visual nodes need an explicit overlap/attachment/containment relationship to permit intersecting envelopes. Parts inside one recipe are treated as an intentional assembly. Recipe-local relationships name source and target part IDs and use the same resolver for reusable support and attachment assemblies. Negative-space nodes create exclusion regions without adding geometry. Accessibility width and reachability checks run only when requested by policy.

Geometry preflight estimates triangle and memory use before generation; actual mesh arrays, triangles, winding, watertightness, degeneracy, UVs and texture bytes are checked before export. These are application limits, not an operating-system memory quota. Native importer peak memory is not strictly bounded by input file size. Empty semantic scenes can export an empty GLB; OBJ/PLY require mesh content. Assets must have a usable license and explicit target envelope. Successful asset selection records uniform scaling policy; geometry does not distort source proportions.

Atmosphere is a single world background, not volumetric weather. Blender saves it in `.blend`; GLB does not embed a world environment. Render-feature compatibility is separate from mesh validity. Optional unsupported features appear in quality diagnostics. Required unsupported features block export. Blender timeouts preserve checkpoints and do not silently switch to a backend that might lose required features.

## Baseline audit and compatibility

At task start, Git reported tracked deletions of top-level `models.py`, `assets.py`, `planning.py`, `museum.py` and `decomposition.py`. The working-tree history did not identify the reason. The committed shared contracts and asset service were restored; the three thin import shims were restored for existing callers. No active graph imports point to legacy models or compilers. The existing 100-test suite passed on that restored baseline.

The import audit found shared `.models` use in builder/config/spatial/validation/recipes/brief/creative/repair and compatibility model lookups in legacy planning/decomposition/museum. Pipeline asset handling uses `.assets`; old planning/museum/decomposition callers retain thin shims to `legacy/`. A subprocess smoke test verifies that a complete new mock graph run loads no `scene_generator.legacy` modules.

`contract_migration.migrate_component` offers an explicit version 1 to 2 metadata upgrade preserving geometry, transforms and identity. Unknown versions and downgrades fail. No graph migration is inferred from an old scene category. Legacy resume, standalone export and corruption-repair tests continue to exercise the existing compatibility path. Generated artifacts and saved runs were not rewritten.
