#include <glm/gtc/constants.hpp>

#include <iostream>
#include <algorithm>
#include <queue>

#include "Model.hpp"
#include "SignPostMesh.hpp"

void SignpostMesh::buildFromModel(const Model& src) {
    // 1) build connectivity + vertex positions
    conn.buildFromModel(src);
    auto& HEs = conn.getHalfEdges();
    auto& V = conn.getVertices();

    // 2) Build faceNormals 
    const auto& connFaces = conn.getFaces();
    faceNormals.resize(connFaces.size());
    for (uint32_t fid = 0; fid < connFaces.size(); ++fid) {
        uint32_t startHe = connFaces[fid].halfEdgeIdx;
        if (startHe == INVALID_INDEX) {
            faceNormals[fid] = glm::vec3(0.0f);
            continue;
        }

        // walk the three halfedges of this face
        uint32_t he1 = startHe;
        uint32_t he2 = HEs[he1].next;
        uint32_t he3 = HEs[he2].next;

        // get their origin vertex indices
        uint32_t v0 = HEs[he1].origin;
        uint32_t v1 = HEs[he2].origin;
        uint32_t v2 = HEs[he3].origin;

        // sanity check 
        if (v0 >= V.size() ||
            v1 >= V.size() ||
            v2 >= V.size()) {
            faceNormals[fid] = glm::vec3(0.0f);
            continue;
        }

        // compute triangle normal
        glm::vec3 A = V[v0].position;
        glm::vec3 B = V[v1].position;
        glm::vec3 C = V[v2].position;
        faceNormals[fid] = glm::normalize(glm::cross(B - A, C - A));
    }

    // 3) set Euclidean lengths directly in HalfEdge struct
    for (uint32_t he = 0; he < HEs.size(); ++he) {
        uint32_t v1 = HEs[he].origin;
        uint32_t heN = HEs[he].next;

        // Check if next is valid
        if (heN == INVALID_INDEX || v1 == INVALID_INDEX) continue;

        uint32_t v2 = HEs[heN].origin;

        // Check if indices are in range
        if (v1 >= V.size() || v2 >= V.size()) continue;

        // Set intrinsic length
        glm::dvec3 dv1(V[v1].position.x, V[v1].position.y, V[v1].position.z);
        glm::dvec3 dv2(V[v2].position.x, V[v2].position.y, V[v2].position.z);
        HEs[he].intrinsicLength = glm::length(dv2 - dv1);
    }

    // 4) walk each vertex to initialize signpostAngle
    for (uint32_t vid = 0; vid < V.size(); ++vid) {
        uint32_t startHe = V[vid].halfEdgeIdx;
        if (startHe == INVALID_INDEX) continue;

        uint32_t curr = startHe;
        double running = 0.0;

        do {
            HEs[curr].signpostAngle = running;
            // cornerAngle is already stored per halfedge
            running += HEs[curr].cornerAngle;

            // Check if opposite exists before accessing it
            if (HEs[curr].opposite == INVALID_INDEX) break;

            // next around vertex = opposite.next
            curr = HEs[curr].opposite;

            // Check if next exists
            if (HEs[curr].next == INVALID_INDEX) break;

            curr = HEs[curr].next;
        } while (curr != startHe && curr != INVALID_INDEX);
    }
}

void SignpostMesh::applyToModel(Model& dstModel) const {
    // 1) Rebuild vertex array
    std::vector<::Vertex> newVtx;
    const auto& HEM_V = conn.getVertices();
    newVtx.reserve(HEM_V.size());

    for (uint32_t i = 0; i < HEM_V.size(); ++i) {
        ::Vertex v;
        // Use HEM_V[i].position instead of vertexPositions[i]
        v.pos = HEM_V[i].position;

        if (HEM_V[i].originalIndex < dstModel.getVertexCount()) {
            const auto& o = dstModel.getVertices()[HEM_V[i].originalIndex];
            v.color = o.color;
            v.normal = o.normal;
            v.texCoord = o.texCoord;
        }
        else {
            v.color = glm::vec3(0.0f);
            v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
            v.texCoord = glm::vec2(0.0f);
        }

        newVtx.push_back(v);
    }

    // 2) Build index buffer from faces (still needed for triangle rendering)
    std::vector<uint32_t> newIdx;
    const auto& HEs = conn.getHalfEdges();
    const auto& fdata = conn.getFaces();

    int orphanedFaces = 0;
    for (uint32_t fid = 0; fid < fdata.size(); ++fid) {
        uint32_t h0 = fdata[fid].halfEdgeIdx;
        if (h0 == INVALID_INDEX) {
            ++orphanedFaces;
            continue;
        }

        uint32_t h1 = HEs[h0].next;
        uint32_t h2 = HEs[h1].next;
        if (HEs[h2].next != h0) {
            std::cerr << "Warning: face " << fid << " is not a triangle!\n";
            continue;
        }

        uint32_t v0 = HEs[h0].origin;
        uint32_t v1 = HEs[h1].origin;
        uint32_t v2 = HEs[h2].origin;

        newIdx.push_back(v0);
        newIdx.push_back(v1);
        newIdx.push_back(v2);
    }

    std::cout << "Exporting mesh with " << fdata.size() << " faces\n";
    std::cout << "Orphaned faces: " << orphanedFaces << "\n";

    // 3) Edge diagnostics using ONLY conn.getEdges()
    const auto& edgeList = conn.getEdges();
    std::unordered_set<std::pair<uint32_t, uint32_t>, pair_hash> edgeSet;
    int duplicateEdgeCount = 0;

    for (const auto& edge : edgeList) {
        uint32_t he = edge.halfEdgeIdx;
        if (he == INVALID_INDEX) continue;

        uint32_t v1 = HEs[he].origin;
        uint32_t v2 = INVALID_INDEX;
        if (HEs[he].next != INVALID_INDEX)
            v2 = HEs[HEs[he].next].origin;

        if (v1 == INVALID_INDEX || v2 == INVALID_INDEX) continue;

        auto e = std::minmax(v1, v2);
        if (!edgeSet.insert(e).second) {
            std::cerr << "WARNING: Duplicate edge " << v1 << "-" << v2 << " in edge list\n";
            ++duplicateEdgeCount;
        }
    }

    std::cout << "Unique edges in original edge list: " << edgeSet.size()
        << " (from " << edgeList.size() << " edge entries)\n";
    std::cout << "Detected " << duplicateEdgeCount << " duplicate edges in original edge list\n";

    // 4) Final upload
    dstModel.setVertices(newVtx);
    dstModel.setIndices(newIdx);
    dstModel.recalculateNormals();

    std::cout << "Final model: " << newVtx.size() << " vertices, "
        << newIdx.size() / 3 << " faces\n";
}

void SignpostMesh::initializeIntrinsicGeometry() {
    auto& HEs = conn.getHalfEdges();
    const auto& edges = conn.getEdges();
    const auto& vertices = conn.getVertices();

    // 1) Calculate Euclidean lengths for all halfedges
    for (uint32_t edgeIdx = 0; edgeIdx < edges.size(); edgeIdx++) {
        uint32_t he = edges[edgeIdx].halfEdgeIdx;
        if (he == INVALID_INDEX) continue;

        // Get the vertices of this edge
        uint32_t v1 = HEs[he].origin;
        uint32_t v2;
        uint32_t opp = HEs[he].opposite;

        if (opp != INVALID_INDEX) {
            // For interior edges, get the origin of the opposite halfedge
            v2 = HEs[opp].origin;
        }
        else {
            // For boundary edges, get the next vertex in the face
            uint32_t nextHe = HEs[he].next;
            if (nextHe == INVALID_INDEX) continue;
            v2 = HEs[nextHe].origin;
        }

        if (v1 >= vertices.size() || v2 >= vertices.size()) continue;

        // Calculate the Euclidean length
        glm::dvec3 dv1(vertices[v1].position.x, vertices[v1].position.y, vertices[v1].position.z);
        glm::dvec3 dv2(vertices[v2].position.x, vertices[v2].position.y, vertices[v2].position.z);
        double length = glm::length(dv2 - dv1);

        // Ensure minimum positive length
        length = std::max(length, 1e-12);

        // Store the length directly in the HalfEdge struct
        HEs[he].intrinsicLength = length;

        // Also set for the opposite halfedge if it exists
        if (opp != INVALID_INDEX) {
            HEs[opp].intrinsicLength = length;
        }
    }

    // 2) Calculate corner angles for all faces
    const auto& faces = conn.getFaces();
    for (uint32_t faceIdx = 0; faceIdx < faces.size(); faceIdx++) {
        updateCornerAnglesForFace(faceIdx);
    }

    // 3) Initialize signpost angles by walking around each vertex
    const auto& V = conn.getVertices();
    for (uint32_t vid = 0; vid < V.size(); vid++) {
        uint32_t startHe = V[vid].halfEdgeIdx;
        if (startHe == INVALID_INDEX) continue;

        uint32_t curr = startHe;
        double running = 0.0;

        do {
            HEs[curr].signpostAngle = running;
            running += HEs[curr].cornerAngle;

            // Move to the next halfedge around this vertex
            uint32_t nextHe = conn.getNextAroundVertex(curr);
            if (nextHe == INVALID_INDEX) break;

            curr = nextHe;
        } while (curr != startHe && curr != INVALID_INDEX);
    }

    // Print statistics to verify
    printMeshStatistics();
}

void SignpostMesh::updateSignpostAngles(uint32_t he) {
    auto& HEs = conn.getHalfEdges();

    // prev halfedge in this face is HEs[he].prev
    uint32_t prevHe = HEs[he].prev;

    // Get the base angle from the previous halfedge's signpost angle
    double base = HEs[prevHe].signpostAngle;
    double corner = HEs[prevHe].cornerAngle;
    double ang = base + corner;

    // wrap into [0,2pi)
    ang = std::fmod(ang, 2 * glm::pi<double>());
    if (ang < 0) ang += 2 * glm::pi<double>();

    // Store the signpost angle directly in the halfedge
    HEs[he].signpostAngle = ang;
}

void SignpostMesh::updateAllSignposts() {
    const auto& V = conn.getVertices();
    auto& HEs = conn.getHalfEdges();

    for (uint32_t vid = 0; vid < V.size(); ++vid) {
        uint32_t startHe = V[vid].halfEdgeIdx;
        if (startHe == INVALID_INDEX) continue;

        // Get all halfedges around this vertex 
        std::vector<uint32_t> vertexHEs = conn.getVertexHalfEdges(vid);

        if (vertexHEs.empty()) continue;

        // Update signpost angles for all halfedges around this vertex
        double acc = 0.0;
        for (uint32_t he : vertexHEs) {
            HEs[he].signpostAngle = acc;
            acc += HEs[he].cornerAngle;
        }
    }
}

glm::vec3 SignpostMesh::computeIntrinsicCircumcenter(uint32_t faceIdx) const {
    const auto& halfEdges = conn.getHalfEdges();

    // Get the three halfedges of this face
    std::vector<uint32_t> faceEdges = conn.getFaceHalfEdges(faceIdx);
    if (faceEdges.size() != 3) {
        return glm::vec3(std::numeric_limits<float>::quiet_NaN());
    }

    // Read intrinsic lengths a, b, c
    double a = halfEdges[faceEdges[0]].intrinsicLength;
    double b = halfEdges[faceEdges[1]].intrinsicLength;
    double c = halfEdges[faceEdges[2]].intrinsicLength;

    // Lay out the triangle in the plane: p0 = (0, 0), p1 = (a, 0)
    glm::dvec2 p0(0.0, 0.0);
    glm::dvec2 p1(a, 0.0);

    // Compute the third vertex p2 = (x, y) by the law of cosines
    double x = (a * a + c * c - b * b) / (2.0 * a);
    double y2 = (c * c - x * x);
    if (y2 < 0.0) {
        // Triangle is degenerate in the intrinsic layout
        std::cout << "Triangle is degenerate in 2d layout, computing midpoint of longest edge instead" << std::endl;
        return computeLongestEdgeMidpoint(faceIdx);
    }
    glm::dvec2 p2(x, std::sqrt(y2));

    // Compute the 2D circumcenter of (p0, p1, p2)
    glm::vec2 cc2d = computeCircumcenter2D(
        glm::vec2((float)p0.x, (float)p0.y),
        glm::vec2((float)p1.x, (float)p1.y),
        glm::vec2((float)p2.x, (float)p2.y)
    );

    // Large circumcenter fallback
    if (!std::isfinite(cc2d.x) || !std::isfinite(cc2d.y)) {
        std::cout << "Circumcenter is very large, computing midpoint of longest edge instead" << std::endl;
        return computeLongestEdgeMidpoint(faceIdx);
    }

    // Compute barycentric coordinates of cc2d in (p0, p1, p2)
    glm::dvec3 bary = computeBarycentric2D(
        glm::dvec2(cc2d.x, cc2d.y),
        p0, p1, p2
    );

    // If any barycentric coordinate is negative, fallback to longest edge midpoint
    if (bary.x < 0.0 || bary.y < 0.0 || bary.z < 0.0) {
        std::cout << "Circumcenter lies outside triangle 2d layout, computing midpoint of longest edge" << std::endl;
        return computeLongestEdgeMidpoint(faceIdx);
    }

    // Otherwise map the valid circumcenter back to 3D
    return mapIntrinsic2DTo3D(faceIdx, glm::dvec2(cc2d.x, cc2d.y), p0, p1, p2);
}

glm::dvec2 SignpostMesh::computeCircumcenter2D(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const {
    double d = 2.0 * (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y));

    if (std::abs(d) < 1e-12) {
        return glm::dvec2(std::numeric_limits<double>::quiet_NaN());
    }

    double aSq = a.x * a.x + a.y * a.y;
    double bSq = b.x * b.x + b.y * b.y;
    double cSq = c.x * c.x + c.y * c.y;

    double x = (aSq * (b.y - c.y) + bSq * (c.y - a.y) + cSq * (a.y - b.y)) / d;
    double y = (aSq * (c.x - b.x) + bSq * (a.x - c.x) + cSq * (b.x - a.x)) / d;

    return glm::dvec2(x, y);
}

glm::dvec3 SignpostMesh::computeBarycentric2D(const glm::dvec2& p, const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const {
    glm::dvec2 v0 = c - a;
    glm::dvec2 v1 = b - a;
    glm::dvec2 v2 = p - a;

    double dot00 = glm::dot(v0, v0);
    double dot01 = glm::dot(v0, v1);
    double dot02 = glm::dot(v0, v2);
    double dot11 = glm::dot(v1, v1);
    double dot12 = glm::dot(v1, v2);

    double invDenom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * invDenom;
    double v = (dot00 * dot12 - dot01 * dot02) * invDenom;
    double w = 1.0 - u - v;
    return glm::dvec3(w, v, u);
}

glm::vec3 SignpostMesh::mapIntrinsic2DTo3D(uint32_t faceIdx, const glm::dvec2& targetPoint, const glm::dvec2& p0, const glm::dvec2& p1, const glm::dvec2& p2) const {
    // Get the 3D vertices of the face
    std::vector<uint32_t> faceVerts = conn.getFaceVertices(faceIdx);
    const auto& vertices = conn.getVertices();

    glm::vec3 v0 = vertices[faceVerts[0]].position;
    glm::vec3 v1 = vertices[faceVerts[1]].position;
    glm::vec3 v2 = vertices[faceVerts[2]].position;

    // Compute barycentric coordinates of targetPoint in the 2D triangle (p0,p1,p2)
    glm::dvec3 bary = computeBarycentric2D(targetPoint, p0, p1, p2);

    // Apply same barycentric coordinates to 3D triangle
    glm::vec3 result = (float)bary.x * v0 + (float)bary.y * v1 + (float)bary.z * v2;
    return result;
}

glm::vec3 SignpostMesh::computeLongestEdgeMidpoint(uint32_t faceIdx) const {
    std::vector<uint32_t> faceVerts = conn.getFaceVertices(faceIdx);
    if (faceVerts.size() != 3) {
        return glm::vec3(std::numeric_limits<float>::quiet_NaN());
    }

    const auto& vertices = conn.getVertices();
    glm::vec3 v0 = vertices[faceVerts[0]].position;
    glm::vec3 v1 = vertices[faceVerts[1]].position;
    glm::vec3 v2 = vertices[faceVerts[2]].position;

    float L01 = glm::length(v1 - v0);
    float L12 = glm::length(v2 - v1);
    float L20 = glm::length(v0 - v2);

    if (L01 >= L12 && L01 >= L20) {
        return 0.5f * (v0 + v1);
    }
    else if (L12 >= L01 && L12 >= L20) {
        return 0.5f * (v1 + v2);
    }
    else {
        return 0.5f * (v2 + v0);
    }
}

float SignpostMesh::computeCornerAngleBetweenSignposts(uint32_t edge1Idx, uint32_t edge2Idx, uint32_t vertexIdx) const {
    // Validate input
    if (edge1Idx >= conn.getEdges().size() || edge2Idx >= conn.getEdges().size() ||
        vertexIdx >= conn.getVertices().size()) {
        std::cerr << "Error: Invalid indices in computeCornerAngleBetweenSignposts" << std::endl;
        return 0.0f;
    }

    const auto& edges = conn.getEdges();
    const auto& HEs = conn.getHalfEdges();

    // Get the half-edges
    uint32_t he1 = edges[edge1Idx].halfEdgeIdx;
    uint32_t he2 = edges[edge2Idx].halfEdgeIdx;

    // Make sure the half-edges originate from the specified vertex
    if (HEs[he1].origin != vertexIdx) {
        // Try the opposite half-edge
        if (HEs[he1].opposite != INVALID_INDEX) {
            he1 = HEs[he1].opposite;
        }
    }

    if (HEs[he2].origin != vertexIdx) {
        // Try the opposite half-edge
        if (HEs[he2].opposite != INVALID_INDEX) {
            he2 = HEs[he2].opposite;
        }
    }

    // Verify that both half-edges now originate from the vertex
    if (HEs[he1].origin != vertexIdx || HEs[he2].origin != vertexIdx) {
        std::cerr << "Error: Edges do not connect to the specified vertex" << std::endl;
        return 0.0f;
    }

    // Get the target vertices
    uint32_t v1 = HEs[HEs[he1].next].origin;
    uint32_t v2 = HEs[HEs[he2].next].origin;

    // Get the edge lengths from the half-edge intrinsic lengths
    double b = HEs[he1].intrinsicLength;  // Edge from vertex to v1
    double c = HEs[he2].intrinsicLength;  // Edge from vertex to v2

    // Find the edge between v1 and v2
    uint32_t he3 = conn.findEdge(v1, v2);

    if (he3 == INVALID_INDEX) {
        std::cerr << "Error: No edge between vertices " << v1 << " and " << v2
            << " when computing corner angle. Mesh structure is invalid." << std::endl;
        return 0.0f;
    }

    // Get the length of the third edge
    double a = HEs[he3].intrinsicLength;

    return computeAngleFromLengths(a, b, c);
}

double SignpostMesh::computeAngleFromLengths(double a, double b, double c) const {
    // Ensure minimum positive edge length for numerical stability
    const float MIN_LENGTH = 1e-6f;

    // Check if any edge is too small or triangle inequality is violated
    if (a < MIN_LENGTH || b < MIN_LENGTH || c < MIN_LENGTH ||
        a + b < c || a + c < b || b + c < a) {

        std::cout << "[WARNING] Degenerate triangle detected (a=" << a
            << " b=" << b << " c=" << c << ") - Skipping angle computation!" << std::endl;

        // Return a special value to indicate this is a degenerate triangle
        return -1.0f;
    }

    // Compute the angle using the law of cosines
    double cosAngle = (b * b + c * c - a * a) / (2.0 * b * c);

    // Clamp to valid range to avoid numerical issues
    cosAngle = std::max(-1.0, std::min(1.0, cosAngle));

    return std::acos(cosAngle);
}

void SignpostMesh::updateCornerAnglesForFace(uint32_t faceIdx) {
    const auto& faces = conn.getFaces();
    auto& HEs = conn.getHalfEdges();

    if (faceIdx >= faces.size() || faces[faceIdx].halfEdgeIdx == INVALID_INDEX)
        return;

    // Get the three halfedges of this face
    uint32_t he0 = faces[faceIdx].halfEdgeIdx;
    uint32_t he1 = HEs[he0].next;
    uint32_t he2 = HEs[he1].next;

    if (he0 == INVALID_INDEX || he1 == INVALID_INDEX || he2 == INVALID_INDEX)
        return;

    // Get the three edge lengths
    double a = HEs[he0].intrinsicLength;
    double b = HEs[he1].intrinsicLength;
    double c = HEs[he2].intrinsicLength;

    // Calculate using the law of cosines
    double angleAtV0 = computeAngleFromLengths(b, c, a);  // Angle at origin of he0
    double angleAtV1 = computeAngleFromLengths(c, a, b);  // Angle at origin of he1
    double angleAtV2 = computeAngleFromLengths(a, b, c);  // Angle at origin of he2

    // Each halfedge gets the angle at its origin vertex
    HEs[he0].cornerAngle = angleAtV0;  // Angle at origin of he0
    HEs[he1].cornerAngle = angleAtV1;  // Angle at origin of he1
    HEs[he2].cornerAngle = angleAtV2;  // Angle at origin of he2
}

void SignpostMesh::updateAllCornerAngles() {
    const auto& faces = conn.getFaces();
    for (uint32_t f = 0; f < faces.size(); ++f) {
        updateCornerAnglesForFace(f);
    }
}

float SignpostMesh::computeFaceArea(uint32_t faceIdx) const {
    const auto& faces = conn.getFaces();
    const auto& halfEdges = conn.getHalfEdges();

    if (faceIdx >= faces.size()) {
        return 0.0f;
    }

    // Get the half-edges of this face
    std::vector<uint32_t> faceEdges = conn.getFaceHalfEdges(faceIdx);
    if (faceEdges.size() != 3) {
        return 0.0f;
    }

    // Get the edge lengths directly from the half-edges
    double a = halfEdges[faceEdges[0]].intrinsicLength;
    double b = halfEdges[faceEdges[1]].intrinsicLength;
    double c = halfEdges[faceEdges[2]].intrinsicLength;

    // Ensure minimum positive length for numerical stability
    a = std::max(a, 1e-12);
    b = std::max(b, 1e-12);
    c = std::max(c, 1e-12);

    // Compute semi-perimeter
    double s = (a + b + c) / 2.0f;

    // Compute area using Heron's formula
    double area = std::sqrt(std::max(0.0, s * (s - a) * (s - b) * (s - c)));

    return static_cast<float>(area);
}

std::vector<float> SignpostMesh::getAllFaceAreas() const {
    std::vector<float> areas;
    areas.reserve(conn.getFaces().size());

    for (uint32_t i = 0; i < conn.getFaces().size(); ++i) {
        areas.push_back(computeFaceArea(i));
    }

    return areas;
}

bool SignpostMesh::isEdgeOnBoundary(uint32_t heIdx) const {
    const auto& HEs = conn.getHalfEdges();
    return (heIdx < HEs.size() && HEs[heIdx].opposite == INVALID_INDEX);
}

bool SignpostMesh::isBoundaryVertex(uint32_t vertexIdx) const {
    return conn.isBoundaryVertex(vertexIdx);
}

std::vector<uint32_t> SignpostMesh::getBoundaryVertices() const {
    std::vector<uint32_t> boundaryVerts;

    for (uint32_t i = 0; i < conn.getVertices().size(); ++i) {
        if (conn.isBoundaryVertex(i)) {
            boundaryVerts.push_back(i);
        }
    }

    return boundaryVerts;
}

uint32_t SignpostMesh::getVertexDegree(uint32_t vertexIdx) const {
    if (vertexIdx >= conn.getVertices().size()) {
        return 0;
    }

    std::vector<uint32_t> vertexHEs = conn.getVertexHalfEdges(vertexIdx);
    return static_cast<uint32_t>(vertexHEs.size());
}

void SignpostMesh::printMeshStatistics() const {
    const auto& edges = conn.getEdges();
    const auto& faces = conn.getFaces();
    const auto& vertices = conn.getVertices();
    const auto& halfEdges = conn.getHalfEdges();

    double minLength = FLT_MAX;
    double maxLength = 0.0f;
    double totalLength = 0.0f;
    int validEdgeCount = 0;

    // Compute edge length statistics from halfEdge class
    for (uint32_t i = 0; i < halfEdges.size(); i++) {
        double length = halfEdges[i].intrinsicLength;
        if (length > 0.0f) {
            minLength = std::min(minLength, length);
            maxLength = std::max(maxLength, length);
            totalLength += length;
            validEdgeCount++;
        }
    }

    double avgLength = validEdgeCount > 0 ? totalLength / validEdgeCount : 0.0f;

    std::cout << "Mesh Statistics:" << std::endl;
    std::cout << "  Vertices: " << vertices.size() << std::endl;
    std::cout << "  Faces: " << faces.size() << std::endl;
    std::cout << "  Edges: " << edges.size() << std::endl;
    std::cout << "  Half-edges with valid length: " << validEdgeCount << std::endl;
    std::cout << "  Edge lengths - Min: " << minLength << ", Max: " << maxLength
        << ", Avg: " << avgLength << std::endl;
}