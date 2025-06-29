#include <glm/gtc/constants.hpp>

#include <iostream>
#include <chrono>
#include <algorithm>
#include <queue>

#include "SignpostMesh.hpp"
#include "iODT.hpp"

iODT::iODT(Model& model) : model(model) {
}

bool iODT::optimalDelaunayTriangulation(SignpostMesh& mesh, int maxIterations)
{
    // 1) Build connectivity & geometry from your model
    mesh.buildFromModel(model);
    auto& conn = mesh.getConnectivity();
    conn.rebuildConnectivity();
    conn.rebuildEdges();

    std::cout << "Computing initial corner angles...\n";
    mesh.updateAllCornerAngles();
    mesh.updateAllSignposts();

    // 2) Make mesh Delaunay
    std::cout << "\n=== Delaunay Flipping Phase ===" << std::endl;
    for (int iter = 0; iter < maxIterations; ++iter) {
        std::cout << "Delaunay pass " << (iter + 1) << "/" << maxIterations << "...\n";
        int flips = conn.makeDelaunay(1);
        if (flips == 0) {
            std::cout << "  no more flips needed.\n";
            break;
        }
    }

    // 3) Delaunay refinement to improve triangle quality
    std::cout << "\n=== Delaunay Refinement Phase ===" << std::endl;
    if (!delaunayRefinement(mesh)) {
        std::cerr << "Warning: Delaunay refinement failed" << std::endl;
    }

    // 4) Global delaunay call
    std::cout << "\n=== Final Global Delaunay Cleanup ===" << std::endl;
    conn.rebuildEdges();
    conn.rebuildConnectivity();
    conn.rebuildOpposites();
    conn.initializeIntrinsicLengths();

    for (int iter = 0; iter < maxIterations; ++iter) {
        std::cout << "Global Delaunay pass " << (iter + 1) << "/" << maxIterations << "...\n";
        int flips = conn.makeDelaunay(1);
        if (flips == 0) {
            std::cout << "  no more flips needed.\n";
            break;
        }
    }


    // 6) Push the result back to the model & GPU
    mesh.applyToModel(model);
    model.recreateBuffers();

    return true;
}

void iODT::repositionInsertedVertices(SignpostMesh& mesh, int maxIters, double tol) {
    auto& conn = mesh.getConnectivity();
    auto& verts = conn.getVertices();

    std::cout << "Repositioning of " << insertedVertices.size() << " Inserted vertices" << std::endl;

    if (insertedVertices.empty()) {
        std::cout << "  No inserted vertices to reposition" << std::endl;
        return;
    }

    for (int iter = 0; iter < maxIters; ++iter) {
        double maxMove = 0;
        std::vector<glm::vec3> newPos(verts.size());
        int movedCount = 0;

        // Initialize all positions (only inserted vertices will be changed)
        for (uint32_t v = 0; v < verts.size(); ++v) {
            newPos[v] = verts[v].position;
        }

        // Only process inserted vertices
        for (uint32_t v : insertedVertices) {
            if (v >= verts.size()) continue;

            if (conn.isBoundaryVertex(v)) {
                continue; // Don't move boundary vertices
            }

            // Grab displacement vector
            glm::vec3 vector = computeWeightedCircumcenterVector(mesh, v);

            // Check if we got a valid vector
            if (!std::isfinite(vector.x) || !std::isfinite(vector.y) || !std::isfinite(vector.z)) {
                continue;
            }

            // The vector points from current position toward optimal position
            // Apply a damping factor for stability
            const float DAMPING = 0.5f; // Adjust as needed
            glm::vec3 displacement = vector * DAMPING;

            // Limit movement to prevent extreme displacements
            float displacementLength = glm::length(displacement);
            const float MAX_DISPLACEMENT = 0.05f; // Conservative limit

            if (displacementLength > MAX_DISPLACEMENT) {
                displacement = displacement * (MAX_DISPLACEMENT / displacementLength);
            }

            // Apply the displacement
            newPos[v] = verts[v].position + displacement;

            double move = glm::length(displacement);
            maxMove = std::max(maxMove, (double)move);
            if (move > tol) movedCount++;
        }

        // Commit new positions
        for (uint32_t v = 0; v < verts.size(); ++v) {
            verts[v].position = newPos[v];
        }

        std::cout << "  Iteration " << (iter + 1) << ": moved " << movedCount
            << " vertices, max displacement = " << maxMove << std::endl;

        // Update intrinsic lengths after repositioning
        conn.initializeIntrinsicLengths();

        if (maxMove < tol) {
            std::cout << "  Converged after " << (iter + 1) << " iterations" << std::endl;
            break;
        }
    }
}

bool iODT::delaunayRefinement(SignpostMesh& mesh) {
    const float MIN_ANGLE = 1.0f * glm::pi<float>() / 180.0f;
    const float MAX_AREA = 0.0025f;
    const float MIN_AREA = 1e-4f;
    auto& conn = mesh.getConnectivity();

    for (int iter = 0; iter < 100; ++iter) {
        std::cout << "Refinement iteration " << (iter + 1) << "\n";

        mesh.updateAllCornerAngles();
        auto cands = findRefinementCandidates(mesh, MIN_ANGLE, MAX_AREA);

        if (cands.empty()) {
            std::cout << "Done.\n";
            return true;
        }

        std::sort(cands.begin(), cands.end(),
            [&](auto const& a, auto const& b) {
                return a.priority > b.priority;
            });

        bool did = false;
        std::vector<uint32_t> newVertices;
        std::unordered_set<uint32_t> processedEdges; // Track processed edges
        std::unordered_set<uint32_t> processedFaces; // Track processed faces

        for (auto const& C : cands) {
            // Skip if face was already processed (invalidated)
            if (processedFaces.count(C.faceIdx)) continue;

            if (mesh.computeFaceArea(C.faceIdx) < MIN_AREA) continue;

            uint32_t newV = UINT32_MAX;
            bool success = false;

            if (C.type == RefinementType::CIRCUMCENTER_INSERTION) {
                success = insertCircumcenter(mesh, C.faceIdx, newV);
                if (success) {
                    processedFaces.insert(C.faceIdx);
                }
            }
            else {
                // Skip if edge was already processed
                if (processedEdges.count(C.edgeIdx)) continue;

                std::cout << "[REFINE] splitting edge " << C.edgeIdx << "\n";
                success = splitEdge(mesh, C.edgeIdx, newV);
                if (success) {
                    processedEdges.insert(C.edgeIdx);
                    // Mark faces that used this edge as processed
                    // (they will be invalidated by the split)
                    auto& edges = conn.getEdges();
                    if (C.edgeIdx < edges.size()) {
                        uint32_t he = edges[C.edgeIdx].halfEdgeIdx;
                        if (he != HalfEdgeMesh::INVALID_INDEX) {
                            processedFaces.insert(conn.getHalfEdges()[he].face);
                            uint32_t oppHe = conn.getHalfEdges()[he].opposite;
                            if (oppHe != HalfEdgeMesh::INVALID_INDEX) {
                                processedFaces.insert(conn.getHalfEdges()[oppHe].face);
                            }
                        }
                    }
                }
            }

            if (success) {
                did = true;
                newVertices.push_back(newV);
            }
        }

        if (!did) {
            std::cout << "No refinement possible.\n";
            return true;
        }

        // Rebuild after split
        conn.rebuildEdges();
        conn.rebuildConnectivity();
        conn.initializeIntrinsicLengths();

        // 5) local Delaunay flips around every new vertex
        std::queue<uint32_t> Q;
        for (uint32_t v : newVertices) {
            for (auto he : conn.getVertexHalfEdges(v)) {
                if (he != HalfEdgeMesh::INVALID_INDEX)
                    Q.push(he);
            }
        }
        while (!Q.empty()) {
            uint32_t he = Q.front(); Q.pop();
            if (!conn.isDelaunayEdge(he) && conn.flipEdge(he)) {
                auto& H = conn.getHalfEdges();
                Q.push(H[he].next);
                Q.push(H[H[he].opposite].next);
            }
        }

        // 6) Update corner angles after all flipping is done
        std::cout << "Updating corner angles and signposts after flips...\n";
        mesh.updateAllCornerAngles();
        mesh.updateAllSignposts();

        std::cout << "Processed " << newVertices.size()
            << " refinements this iteration.\n";
        std::cout << "Local flipping completed.\n";
    }

    std::cout << "Reached max iterations\n";
    return true;
}

bool iODT::insertCircumcenter(SignpostMesh& mesh, uint32_t faceIdx, uint32_t& outNewVertex) {
    auto& conn = mesh.getConnectivity();
    const auto& faces = conn.getFaces();

    std::cout << "[insertCircumcenter] called on faceIdx = " << faceIdx << std::endl;

    // 1) Validate face index
    if (faceIdx >= faces.size()) {
        std::cout << "  -> faceIdx out of range. Returning false.\n";
        return false;
    }
    if (faces[faceIdx].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) {
        std::cout << "  -> face " << faceIdx << " already invalidated. Returning false.\n";
        return false;
    }

    // 2) Reject near zero area
    float area = mesh.computeFaceArea(faceIdx);
    std::cout << "  -> face area = " << area << std::endl;
    if (area < 1e-8f) {
        std::cout << "  -> area < 1e-8, skipping insertion.\n";
        return false;
    }

    // 3) Get intrinsic edge lengths
    std::vector<uint32_t> faceHEs = conn.getFaceHalfEdges(faceIdx);
    if (faceHEs.size() != 3) {
        std::cout << "  -> not a triangle, skipping.\n";
        return false;
    }

    const auto& HEs = conn.getHalfEdges();
    // 1) Compute the three corner vertex indices:
    uint32_t he0 = faceHEs[0];
    uint32_t he1 = faceHEs[1];
    uint32_t he2 = faceHEs[2];

    // v0 is the origin of he0
    uint32_t v0 = HEs[he0].origin;
    // v1 is the origin of he1
    uint32_t v1 = HEs[he1].origin;
    // v2 is the origin of he2
    uint32_t v2 = HEs[he2].origin;

    // 2) The 3 intrinsic lengths (a,b,c) already come from exactly those halfedges:
    double a = HEs[he0].intrinsicLength; // length(v0->v1)
    double b = HEs[he1].intrinsicLength; // length(v1->v2)
    double c = HEs[he2].intrinsicLength; // length(v2->v0)

    std::cout << "  -> intrinsic edge lengths: a=" << a << ", b=" << b << ", c=" << c << std::endl;

    // 4) Layout triangle in 2D
    glm::dvec2 P0(0.0, 0.0);
    glm::dvec2 P1(a, 0.0);

    // Compute P2 using law of cosines
    double x = (a * a + c * c - b * b) / (2.0 * a);
    double y2 = c * c - x * x;
    if (y2 < 0) {
        std::cout << "  -> degenerate triangle in 2D layout, skipping\n";
        return false;
    }
    glm::dvec2 P2(x, std::sqrt(y2));

    std::cout << "  -> 2D layout: P0=(" << P0.x << "," << P0.y
        << "), P1=(" << P1.x << "," << P1.y
        << "), P2=(" << P2.x << "," << P2.y << ")" << std::endl;

    std::cout << "  -> 2D layout:\n"
        << "       P0 (vertex " << v0 << ") = (" << P0.x << "," << P0.y << ")\n"
        << "       P1 (vertex " << v1 << ") = (" << P1.x << "," << P1.y << ")\n"
        << "       P2 (vertex " << v2 << ") = (" << P2.x << "," << P2.y << ")\n";

    // 5) Compute 2D circumcenter
    glm::dvec2 cc2d = mesh.computeCircumcenter2D(P0, P1, P2);
    if (!std::isfinite(cc2d.x) || !std::isfinite(cc2d.y)) {
        std::cout << "  -> 2D circumcenter is NaN/infinite, skipping\n";
        return false;
    }

    std::cout << "  -> 2D circumcenter: (" << cc2d.x << "," << cc2d.y << ")" << std::endl;

    // 5) Use your existing barycentric function
    glm::dvec3 bary = mesh.computeBarycentric2D(cc2d, P0, P1, P2);
    double lambda0 = bary.x;  // Weight for vertex 0 (P0)
    double lambda1 = bary.y;  // Weight for vertex 1 (P1)
    double lambda2 = bary.z;  // Weight for vertex 2 (P2)

    std::cout << "  -> barycentric coords: lambda0=" << lambda0 << ", lambda1=" << lambda1 << ", lambda2=" << lambda2 << std::endl;

    double sum = bary.x + bary.y + bary.z;
    std::cout << "  -> bary sum = " << sum << std::endl;

    const double BARY_TOLERANCE = 1e-2;

    // Check if circumcenter is too close to any edge
    if (lambda0 < BARY_TOLERANCE) {
        std::cout << "  -> circumcenter too close to edge opposite vertex 0 (edge v1-v2), splitting edge instead\n";
        uint32_t edgeIdx = conn.getEdgeIndexFromHalfEdge(he1);
        if (edgeIdx != HalfEdgeMesh::INVALID_INDEX) {
            return splitEdge(mesh, edgeIdx, outNewVertex);
        }
        return false;
    }

    if (lambda1 < BARY_TOLERANCE) {
        std::cout << "  -> circumcenter too close to edge opposite vertex 1 (edge v2-v0), splitting edge instead\n";
        uint32_t edgeIdx = conn.getEdgeIndexFromHalfEdge(he2);
        if (edgeIdx != HalfEdgeMesh::INVALID_INDEX) {
            return splitEdge(mesh, edgeIdx, outNewVertex);
        }
        return false;
    }

    if (lambda2 < BARY_TOLERANCE) {
        std::cout << "  -> circumcenter too close to edge opposite vertex 2 (edge v0-v1), splitting edge instead\n";
        uint32_t edgeIdx = conn.getEdgeIndexFromHalfEdge(he0);
        if (edgeIdx != HalfEdgeMesh::INVALID_INDEX) {
            return splitEdge(mesh, edgeIdx, outNewVertex);
        }
        return false;
    }

    // 6) Compute new intrinsic edge lengths in 2D
    double r0 = glm::length(cc2d - P0);
    double r1 = glm::length(cc2d - P1);
    double r2 = glm::length(cc2d - P2);

    std::cout << "  -> new intrinsic lengths: r0=" << r0 << ", r1=" << r1 << ", r2=" << r2 << std::endl;

    // 7) Validate triangle inequalities BEFORE splitting
    const double EPS = 1e-12;

    // Triangle (v0, v1, newV): edges a, r0, r1
    if (!(a + r0 > r1 + EPS && a + r1 > r0 + EPS && r0 + r1 > a - EPS)) {
        std::cout << "  -> triangle inequality violation for triangle 1, skipping\n";
        return false;
    }

    // Triangle (v1, v2, newV): edges b, r1, r2
    if (!(b + r1 > r2 + EPS && b + r2 > r1 + EPS && r1 + r2 > b - EPS)) {
        std::cout << "  -> triangle inequality violation for triangle 2, skipping\n";
        return false;
    }

    // Triangle (v2, v0, newV): edges c, r2, r0
    if (!(c + r2 > r0 + EPS && c + r0 > r2 + EPS && r2 + r0 > c - EPS)) {
        std::cout << "  -> triangle inequality violation for triangle 3, skipping\n";
        return false;
    }

    std::cout << "  -> all triangle inequalities satisfied, proceeding with split\n";

    // 8) Perform intrinsic split
    uint32_t newV = conn.splitTriangleIntrinsic(faceIdx, r0, r1, r2);
    if (newV == HalfEdgeMesh::INVALID_INDEX) {
        std::cout << "  -> splitTriangleIntrinsic failed.\n";
        return false;
    }

    // Track this as an inserted vertex
    insertedVertices.insert(newV);
    std::cout << "  -> splitTriangleIntrinsic succeeded.\n";

    // 10) Set the 3D position of the new vertex
    glm::vec3 v0Pos = conn.getVertices()[v0].position;
    glm::vec3 v1Pos = conn.getVertices()[v1].position;
    glm::vec3 v2Pos = conn.getVertices()[v2].position;

    // Barycentric interpolation in 3D:
    glm::dvec3 newPos3D_d =
        lambda0 * glm::dvec3(v0Pos) +
        lambda1 * glm::dvec3(v1Pos) +
        lambda2 * glm::dvec3(v2Pos);

    // Set it
    conn.getVertices()[newV].position = glm::vec3(newPos3D_d);

    // 11) Rebuild connectivity structures
    conn.rebuildEdges();
    conn.rebuildConnectivity();

    outNewVertex = newV;
    std::cout << "Successfully inserted circumcenter at face " << faceIdx << std::endl;
    return true;
}

std::vector<iODT::RefinementCandidate>iODT::findRefinementCandidates(const SignpostMesh& mesh, float minAngleThreshold, float maxAreaThreshold) {
    constexpr float MIN_AREA = 1e-5f;
    constexpr float MIN_EDGE_LEN = 1e-3f;

    auto& conn = mesh.getConnectivity();
    auto const& F = conn.getFaces();
    std::vector<RefinementCandidate> out;
    out.reserve(F.size());

    std::cout
        << "  [DEBUG] minAngleThreshold (rad) = "
        << minAngleThreshold
        << " (" << (minAngleThreshold * 180.0f / glm::pi<float>())
        << " deg), maxAreaThreshold = " << maxAreaThreshold
        << "\n";

    // Helper: lay out triangle in 2D, compute circumcenter, test barycentric coords
    auto tryCC = [&](uint32_t fIdx, glm::dvec2& cc) -> std::pair<bool, uint32_t> {
        uint32_t startHE = F[fIdx].halfEdgeIdx;
        uint32_t he[3] = {
            startHE,
            conn.getHalfEdges()[startHE].next,
            conn.getHalfEdges()[conn.getHalfEdges()[startHE].next].next
        };
        double L0 = conn.getHalfEdges()[he[0]].intrinsicLength;
        double L1 = conn.getHalfEdges()[he[1]].intrinsicLength;
        double L2 = conn.getHalfEdges()[he[2]].intrinsicLength;

        // 2D layout
        glm::dvec2 P0(0.0, 0.0), P1(L0, 0.0);
        double a = L0, b = L1, c = L2;
        double x = (a * a + c * c - b * b) / (2.0 * a);
        double y2 = c * c - x * x;
        double y = (y2 < 0 ? 0 : std::sqrt(y2));
        glm::dvec2 P2(x, y);

        // Calculate circumcenter
        double D = 2.0 * (P1.x * P2.y - P2.x * P1.y);
        if (std::fabs(D) < 1e-12) D = std::copysign(1e-12, D);
        cc = {
          (P2.y * (P1.x * P1.x + P1.y * P1.y)
          - P1.y * (P2.x * P2.x + P2.y * P2.y)) / D,
          (-P2.x * (P1.x * P1.x + P1.y * P1.y)
          + P1.x * (P2.x * P2.x + P2.y * P2.y)) / D
        };

        // Barycentric calculation
        double denom = (P1.y - P2.y) * (P0.x - P2.x) + (P2.x - P1.x) * (P0.y - P2.y);
        if (std::fabs(denom) < 1e-12)
            return { false, he[0] };  // degenerate

        double lambda0 = ((P1.y - P2.y) * (cc.x - P2.x) + (P2.x - P1.x) * (cc.y - P2.y)) / denom;
        double lambda1 = ((P2.y - P0.y) * (cc.x - P2.x) + (P0.x - P2.x) * (cc.y - P2.y)) / denom;
        double lambda2 = 1.0 - lambda0 - lambda1;

        const double eps = 1e-8;
        if (lambda0 < eps) return { false, he[1] }; // too close to edge v1-v2
        if (lambda1 < eps) return { false, he[2] }; // too close to edge v2-v0
        if (lambda2 < eps) return { false, he[0] }; // too close to edge v0-v1

        return { true, 0u };
        };

    // Scan all faces
    for (uint32_t f = 0; f < F.size(); ++f) {
        uint32_t startHE = F[f].halfEdgeIdx;
        if (startHE == HalfEdgeMesh::INVALID_INDEX) {
            std::cout << "  [DEBUG] face=" << f << " invalid, skip\n";
            continue;
        }

        float area = mesh.computeFaceArea(f);
        float minAng = computeMinAngle(mesh, f);

        std::cout
            << "  [DEBUG] face=" << f
            << "  area=" << area
            << "  minAng=" << minAng
            << " (" << (minAng * 180.0f / glm::pi<float>()) << " deg)\n";

        if (area < MIN_AREA) {
            std::cout << "    -> area<MIN_AREA skip\n";
            continue;
        }
        if (minAng <= 0.0f) {
            std::cout << "    -> minAng<=0 skip\n";
            continue;
        }

        bool angleBad = (minAng < minAngleThreshold);
        bool areaBad = (area > maxAreaThreshold);
        if (!angleBad && !areaBad) {
            std::cout << "    -> neither bad, skip\n";
            continue;
        }

        std::cout << "    -> candidate ("
            << (angleBad ? "bad angle" : "")
            << (angleBad && areaBad ? " + " : "")
            << (areaBad ? "bad area" : "")
            << ")\n";

        glm::dvec2 cc;
        auto [ok, badHE] = tryCC(f, cc);

        RefinementCandidate rc;
        rc.faceIdx = f;
        rc.minAngle = minAng;
        rc.area = area;
        rc.priority = ok
            ? (minAngleThreshold - minAng) / minAngleThreshold + area / maxAreaThreshold
            : 1.0f + area / maxAreaThreshold;

        if (ok) {
            rc.type = RefinementType::CIRCUMCENTER_INSERTION;
            rc.edgeIdx = HalfEdgeMesh::INVALID_INDEX;
        }
        else {
            uint32_t e = conn.getEdgeIndexFromHalfEdge(badHE);
            double L = conn.getIntrinsicLengthFromHalfEdge(e);
            if (e == HalfEdgeMesh::INVALID_INDEX) {
                std::cout << "      -> badHE->edgeIdx invalid, skip\n";
                continue;
            }
            if (L < 2 * MIN_EDGE_LEN) {
                std::cout << "      -> L=" << L << "<2*MIN_EDGE_LEN, skip\n";
                continue;
            }
            if (recentlySplit.count(std::minmax(
                conn.getHalfEdges()[conn.getEdges()[e].halfEdgeIdx].origin,
                conn.getHalfEdges()[conn.getEdges()[e].halfEdgeIdx].next
            )))
            {
                std::cout << "      -> edge blocked, skip\n";
                continue;
            }
            rc.type = RefinementType::EDGE_SPLIT;
            rc.edgeIdx = e;
        }

        std::cout << "    -> ADDED face=" << f
            << " type=" << (rc.type == RefinementType::CIRCUMCENTER_INSERTION ? "CC" : "SPLIT")
            << (rc.type == RefinementType::EDGE_SPLIT ? (" edge=" + std::to_string(rc.edgeIdx)) : "")
            << "\n";

        out.push_back(rc);
    }

    return out;
}

bool iODT::splitEdge(SignpostMesh& mesh, uint32_t edgeIdx, uint32_t& outNewVertex) {
    auto& conn = mesh.getConnectivity();
    constexpr uint32_t INV = HalfEdgeMesh::INVALID_INDEX;

    // Get original endpoints for 3D position interpolation
    uint32_t parentHE = conn.getEdges()[edgeIdx].halfEdgeIdx;
    uint32_t oppHE = conn.getHalfEdges()[parentHE].opposite;
    uint32_t VA = conn.getHalfEdges()[parentHE].origin;
    uint32_t VB = conn.getHalfEdges()[oppHE].origin;

    // Call splitEdgeTopo
    auto R = conn.splitEdgeTopo(edgeIdx, 0.5);
    if (R.newV == INV) {
        std::cout << "[splitEdge] splitEdgeTopo failed\n";
        return false;
    }

    // Track this as an inserted vertex
    insertedVertices.insert(R.newV);

    // Set the 3D position of the new vertex
    auto& V = conn.getVertices();
    V[R.newV].position = glm::mix(V[VA].position, V[VB].position, 0.5f);

    outNewVertex = R.newV;
    return true;
}

bool iODT::isBlockedEdge(const SignpostMesh& mesh, uint32_t edgeIdx) {
    const auto& conn = mesh.getConnectivity();
    uint32_t he0 = conn.getEdges()[edgeIdx].halfEdgeIdx;
    uint32_t v0 = conn.getHalfEdges()[he0].origin;
    uint32_t v1 = conn.getHalfEdges()[conn.getHalfEdges()[he0].next].origin;
    auto     p = std::minmax(v0, v1);
    return (recentlySplit.count(p) > 0);
}

float iODT::computeMinAngle(const SignpostMesh& mesh, uint32_t faceIdx) {
    const auto& conn = mesh.getConnectivity();
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();

    if (faceIdx >= faces.size()) {
        return 0.0f;
    }

    // Get the halfedges of this face
    std::vector<uint32_t> faceEdges = conn.getFaceHalfEdges(faceIdx);
    if (faceEdges.size() != 3) {
        return 0.0f;
    }

    // Get the edge lengths directly from the halfedges
    float a = halfEdges[faceEdges[0]].intrinsicLength;
    float b = halfEdges[faceEdges[1]].intrinsicLength;
    float c = halfEdges[faceEdges[2]].intrinsicLength;

    // Clamp minimum positive length 
    a = std::max(a, 1e-5f);
    b = std::max(b, 1e-5f);
    c = std::max(c, 1e-5f);

    // Compute angles using the law of cosines
    float cosA = (b * b + c * c - a * a) / (2.0f * b * c);
    float cosB = (a * a + c * c - b * b) / (2.0f * a * c);
    float cosC = (a * a + b * b - c * c) / (2.0f * a * b);

    // Clamp to valid range to avoid numerical issues
    cosA = glm::clamp(cosA, -1.0f, 1.0f);
    cosB = glm::clamp(cosB, -1.0f, 1.0f);
    cosC = glm::clamp(cosC, -1.0f, 1.0f);

    // Convert to angles
    float angleA = std::acos(cosA);
    float angleB = std::acos(cosB);
    float angleC = std::acos(cosC);

    // Return the minimum angle
    return std::min(std::min(angleA, angleB), angleC);
}

glm::vec3 iODT::computeWeightedCircumcenterVector(const SignpostMesh& mesh, uint32_t vertIdx) {
    const auto& conn = mesh.getConnectivity();
    const auto& halfEdges = conn.getHalfEdges();
    const auto& vertices = conn.getVertices();

    if (vertIdx >= vertices.size()) {
        return glm::vec3(0.0f);
    }

    // Get the first halfedge from this vertex
    uint32_t firstHalfEdge = vertices[vertIdx].halfEdgeIdx;
    if (firstHalfEdge == HalfEdgeMesh::INVALID_INDEX) {
        return glm::vec3(0.0f);
    }

    // Skip boundary vertices
    if (conn.isBoundaryVertex(vertIdx)) {
        return glm::vec3(0.0f);
    }

    // Find incident triangles
    std::unordered_set<uint32_t> incidentFaces;
    std::vector<uint32_t> vertexHalfEdges = conn.getVertexHalfEdges(vertIdx);

    for (uint32_t heIdx : vertexHalfEdges) {
        uint32_t faceIdx = halfEdges[heIdx].face;
        if (faceIdx != HalfEdgeMesh::INVALID_INDEX) {
            incidentFaces.insert(faceIdx);
        }
    }

    // Calculate intrinsic circumcenters and weights
    std::vector<glm::vec3> circumcenterVectors;
    std::vector<float> weights;

    // Get the 3D position of our target vertex
    glm::vec3 targetPos = vertices[vertIdx].position;

    for (uint32_t faceIdx : incidentFaces) {
        // Get the three halfedges of the face
        std::vector<uint32_t> faceEdges = conn.getFaceHalfEdges(faceIdx);
        if (faceEdges.size() != 3) continue;

        // Get the three vertices
        uint32_t v1 = halfEdges[faceEdges[0]].origin;
        uint32_t v2 = halfEdges[faceEdges[1]].origin;
        uint32_t v3 = halfEdges[faceEdges[2]].origin;

        // Skip if this vertex is not part of the face
        if (v1 != vertIdx && v2 != vertIdx && v3 != vertIdx) {
            continue;
        }

        // Compute triangle area using face area method (more robust)
        float area = mesh.computeFaceArea(faceIdx);
        if (area < 1e-6f) continue; // Skip degenerate triangles

        // Get intrinsic circumcenter of the triangle
        glm::vec3 circumcenter = mesh.computeIntrinsicCircumcenter(faceIdx);

        // Check for valid circumcenter
        if (!std::isfinite(circumcenter.x) || !std::isfinite(circumcenter.y) || !std::isfinite(circumcenter.z)) {
            continue;
        }

        // Get vector pointing from this vertex to the circumcenter
        glm::vec3 ccVector = circumcenter - targetPos;

        // Use triangle area as weight
        circumcenterVectors.push_back(ccVector);
        weights.push_back(area);
    }

    // Compute weighted average direction
    glm::vec3 avgDirection(0.0f);
    float totalWeight = 0.0f;

    for (size_t i = 0; i < circumcenterVectors.size(); i++) {
        avgDirection += circumcenterVectors[i] * weights[i];
        totalWeight += weights[i];
    }

    if (totalWeight > 0.0f) {
        avgDirection /= totalWeight;
    }

    return avgDirection;
}

bool iODT::validateMeshConnectivity(const SignpostMesh& mesh) {
    const auto& conn = mesh.getConnectivity();
    const auto& vertices = conn.getVertices();
    const auto& halfEdges = conn.getHalfEdges();

    std::cout << "[iODT] Validating mesh connectivity..." << std::endl;

    // Check for disconnected components using BFS
    std::vector<bool> visited(vertices.size(), false);
    int componentCount = 0;

    for (uint32_t startVertex = 0; startVertex < vertices.size(); ++startVertex) {
        if (visited[startVertex] || vertices[startVertex].halfEdgeIdx == HalfEdgeMesh::INVALID_INDEX) {
            continue;
        }

        // BFS from this vertex
        std::queue<uint32_t> queue;
        queue.push(startVertex);
        visited[startVertex] = true;
        int componentSize = 0;

        while (!queue.empty()) {
            uint32_t currentVertex = queue.front();
            queue.pop();
            componentSize++;

            // Get all halfedges from this vertex
            std::vector<uint32_t> vertexHalfEdges = conn.getVertexHalfEdges(currentVertex);

            for (uint32_t heIdx : vertexHalfEdges) {
                if (heIdx >= halfEdges.size()) continue;

                // Get the destination vertex of this halfedge
                uint32_t nextHE = halfEdges[heIdx].next;
                if (nextHE >= halfEdges.size()) continue;

                uint32_t destVertex = halfEdges[nextHE].origin;
                if (destVertex < vertices.size() && !visited[destVertex]) {
                    visited[destVertex] = true;
                    queue.push(destVertex);
                }
            }
        }

        std::cout << "  Component " << componentCount << ": " << componentSize << " vertices" << std::endl;
        componentCount++;
    }

    // Count unvisited vertices
    int unvisitedCount = 0;
    for (size_t i = 0; i < visited.size(); ++i) {
        if (!visited[i]) {
            std::cout << "  Unvisited vertex: " << i << std::endl;
            unvisitedCount++;
        }
    }

    std::cout << "  Total components: " << componentCount << std::endl;
    std::cout << "  Unvisited vertices: " << unvisitedCount << std::endl;

    return componentCount == 1 && unvisitedCount == 0;
}