# iODT Remesher

- Builds halfedge connectivity and intrinsic lengths from a 3D model
- Signpost class wraps halfedge connectivity and initializes corner/signpost angles

## Delaunay Flips

- Flip non‑Delaunay edges

## Quality Refinement

- Identifies triangles with min angle below a threshold or area above max area
- Inserts vertices at the circumcenter or splits edges as a fallback

## Optimal Positioning

- Splits edges at midpoint that are higher than defined edge length
- Moves inserted vertices toward an area weighted average direction of adjacent circumcenters

## Call Example
```cpp
#include "iODT.hpp"
bool success = remesher.optimalDelaunayTriangulation(1);
```
## TODO
- Intrinsic metrics are still represented extrinsically  
