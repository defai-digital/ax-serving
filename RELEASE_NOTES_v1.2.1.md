## ax-serving v1.2.1

This release finalizes the trusted offline baseline for AX Fabric.

### Highlights

- clarifies AX Serving as the execution and serving control plane for AX Fabric
- refines the public roadmap around offline enterprise LLM and AI systems
- removes public reliance on local-only `automatosx/` paths
- keeps benchmark and CI artifacts under `target/` instead of local planning folders
- finishes config test serialization for process-global environment variable mutation

### Notes

- `automatosx/` remains local-only planning space and is not part of the public product surface
- this release stays on the `v1.2.x` line and does not include `v1.4` scheduler work
