#pragma once
#include <glm/glm.hpp>

#include <vector>

#include "HalfEdgeMesh.hpp"

class Model;

class SignpostMesh {
public:
    static const uint32_t INVALID_INDEX = static_cast<uint32_t>(-1);

    struct pair_hash {
        template <class T1, class T2>
        std::size_t operator()(const std::pair<T1, T2>& p) const {
            auto h1 = std::hash<T1>{}(p.first);
            auto h2 = std::hash<T2>{}(p.second);
            return h1 ^ (h2 << 1);
        }
    };

    // Construction
    void buildFromModel(const Model& srcModel);
    void applyToModel(Model& dstModel) const;

    // Intrinsic operations
    void initializeIntrinsicGeometry();
    void updateSignpostAngles(uint32_t vertexIdx);
    void updateAllSignposts();

    // Intrinsic helpers
    glm::dvec2 computeCircumcenter2D(const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const;
    glm::dvec3 computeBarycentric2D(const glm::dvec2& p, const glm::dvec2& a, const glm::dvec2& b, const glm::dvec2& c) const;
    glm::vec3 mapIntrinsic2DTo3D(uint32_t faceIdx, const glm::dvec2& targetPoint, const glm::dvec2& p0, const glm::dvec2& p1, const glm::dvec2& p2) const;
    glm::vec3 computeIntrinsicCircumcenter(uint32_t faceIdx) const;
    glm::vec3 computeLongestEdgeMidpoint(uint32_t faceIdx) const;

    // Angle operations
    float computeCornerAngleBetweenSignposts(uint32_t sp1Idx, uint32_t sp2Idx, uint32_t vertexIdx) const;
    double computeAngleFromLengths(double a, double b, double c) const;
    void updateCornerAnglesForFace(uint32_t faceIdx);
    void updateAllCornerAngles();

    // Face operations
    float computeFaceArea(uint32_t faceIdx) const;
    std::vector<float> getAllFaceAreas() const;

    // Edge operations  
    bool isEdgeOnBoundary(uint32_t heIdx) const;

    // Vertex operations
    bool isBoundaryVertex(uint32_t vertexIdx) const;
    std::vector<uint32_t> getBoundaryVertices() const;
    uint32_t getVertexDegree(uint32_t vertexIdx) const;

    // Debug
    void printMeshStatistics() const;

    // Get halfedge structure
    const HalfEdgeMesh& getConnectivity() const {
        return conn;
    }
    HalfEdgeMesh& getConnectivity() {
        return conn;
    }

private:
    HalfEdgeMesh conn;
    std::vector<glm::vec3> faceNormals;
    std::vector<double> vertexAngleSums;
};