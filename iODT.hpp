#pragma once
#include <glm/glm.hpp>
#include <set>

#include "Model.hpp"
#include "Structs.hpp"

class SignpostMesh;

class iODT {
public:
    iODT(Model& model);

    static const uint32_t INVALID_INDEX = static_cast<uint32_t>(-1);

    enum class RefinementType {
        CIRCUMCENTER_INSERTION,
        EDGE_SPLIT
    };

    struct RefinementCandidate {
        RefinementType type;
        uint32_t faceIdx        = 0;
        uint32_t edgeIdx        = 0;
        float priority          = 0.0f;
        float minAngle          = 0.0f;
        float area              = 0.0f;
        float quality           = 0.0f;
    };

    // High level ODT functions
    bool optimalDelaunayTriangulation(SignpostMesh& mesh, int iterations);
    void repositionInsertedVertices(SignpostMesh& mesh, int iterations, double tol);

    // Refinement operations
    bool delaunayRefinement(SignpostMesh& mesh);
    std::vector<RefinementCandidate> findRefinementCandidates(const SignpostMesh& mesh, float minAngleThreshold, float maxAreaThreshold);
    bool insertCircumcenter(SignpostMesh& mesh, uint32_t faceIdx, uint32_t& outNewVertex);
    bool splitEdge(SignpostMesh& mesh, uint32_t edgeIdx, uint32_t& outNewVertex);
    bool isBlockedEdge(const SignpostMesh& mesh, uint32_t edgeIdx);

    // Quality metrics
    float computeMinAngle(const SignpostMesh& mesh, uint32_t faceIdx);

    // Helpers
    glm::vec3 computeWeightedCircumcenterVector(const SignpostMesh& mesh, uint32_t vertIdx);

    // Validation
    bool validateMeshConnectivity(const SignpostMesh& mesh);

private:
    Model& model;
    std::set<std::pair<uint32_t, uint32_t>> recentlySplit;

    // Track inserted vertices for ODT repositioning
    std::unordered_set<uint32_t> insertedVertices;

    // Feature preservation
    bool isFeatureVertex(const SignpostMesh& mesh, uint32_t vertIdx);
    bool isFeatureEdge(const SignpostMesh& mesh, uint32_t signpostIdx);
    void preserveFeatures(SignpostMesh& mesh);
};
