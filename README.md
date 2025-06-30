# iODT Remesher

- Builds halfedge connectivity and intrinsic lengths from a 3D model

- Signpost class wraps halfedge connectivity and initializes corner/signpost angles

## Delaunay Flips

- Flip non‑Delaunay edges

## Quality Refinement

- Identifies triangles with min angle below a threshold or area above max area

- Inserts vertices at the circumcenter or splits edges as a fallback

## Optimal Positioning
- Moves inserted vertices toward their weighted circumcenters

## Call Example
```
bool success = remesher.optimalDelaunayTriangulation(1);

```
## TODO
- Inserted vertices are mapped extrinsically by barycentric coordinates, needs proper geodesic tracing
  
- Min angle refinement is limited, currently operates on area thresholds 
