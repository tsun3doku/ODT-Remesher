#pragma once
#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_LEFT_HANDED
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h> 

#include <glm/glm.hpp>
#include <glm/gtx/hash.hpp>

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <array>

#include "File_utils.h" 
#include "Structs.hpp"

class AABBTree;
class Camera;
class VulkanDevice;
class MemoryAllocator;

struct Vertex {
    glm::vec3 pos;      // Vertex position
    glm::vec3 color;    // Vertex color
    glm::vec3 normal;   // Vertex normal
    glm::vec2 texCoord; // Texture coordinates

    static std::array<VkVertexInputBindingDescription, 2> getBindingDescriptions() {
        std::array<VkVertexInputBindingDescription, 2> bindingDescriptions{};

        // Main vertex binding (positions, normals, texcoords)
        VkVertexInputBindingDescription mainBinding{};
        mainBinding.binding = 0;
        mainBinding.stride = sizeof(Vertex); 
        mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindingDescriptions[0] = mainBinding;

        // Surface binding (dynamic color from compute shader)
        VkVertexInputBindingDescription surfaceBinding{};
        surfaceBinding.binding = 1;
        surfaceBinding.stride = sizeof(SurfaceVertex); 
        surfaceBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;
        bindingDescriptions[1] = surfaceBinding;

        return bindingDescriptions;
    }

    // Separate attribute description functions
    static std::array<VkVertexInputAttributeDescription, 4> getVertexAttributes() {
        std::array<VkVertexInputAttributeDescription, 4> vertexAttributes{};

        vertexAttributes[0].binding = 0;
        vertexAttributes[0].location = 0;
        vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributes[0].offset = offsetof(Vertex, pos);

        vertexAttributes[1].binding = 0;
        vertexAttributes[1].location = 1;
        vertexAttributes[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributes[1].offset = offsetof(Vertex, color);

        vertexAttributes[2].binding = 0;
        vertexAttributes[2].location = 2;
        vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
        vertexAttributes[2].offset = offsetof(Vertex, normal);

        vertexAttributes[3].binding = 0;
        vertexAttributes[3].location = 3;
        vertexAttributes[3].format = VK_FORMAT_R32G32_SFLOAT;
        vertexAttributes[3].offset = offsetof(Vertex, texCoord);

        return vertexAttributes;
    }

    static std::array<VkVertexInputAttributeDescription, 2> getSurfaceVertexAttributes() {
        std::array<VkVertexInputAttributeDescription, 2> surfaceVertexAttributes{};

        surfaceVertexAttributes[0].binding = 1;
        surfaceVertexAttributes[0].location = 4;
        surfaceVertexAttributes[0].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        surfaceVertexAttributes[0].offset = offsetof(SurfaceVertex, position); 

        surfaceVertexAttributes[1].binding = 1;
        surfaceVertexAttributes[1].location = 5;
        surfaceVertexAttributes[1].format = VK_FORMAT_R32G32B32A32_SFLOAT;
        surfaceVertexAttributes[1].offset = offsetof(SurfaceVertex, color);

        return surfaceVertexAttributes;
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos &&
            color == other.color &&
            normal == other.normal &&
            texCoord == other.texCoord;
    }
};

namespace std {
    template<> struct hash<Vertex> {
        size_t operator()(Vertex const& vertex) const {
            size_t h1 = std::hash<float>{}(vertex.pos.x) ^ std::hash<float>{}(vertex.pos.y) ^ std::hash<float>{}(vertex.pos.z);
            size_t h2 = std::hash<float>{}(vertex.normal.x) ^ std::hash<float>{}(vertex.normal.y) ^ std::hash<float>{}(vertex.normal.z);
            size_t h3 = std::hash<float>{}(vertex.texCoord.x) ^ std::hash<float>{}(vertex.texCoord.y);
            return h1 ^ (h2 << 1) ^ (h3 << 2);
        }
    };
}

class Model {
public:
    Model(VulkanDevice& device, MemoryAllocator& allocator, Camera& camera);
    ~Model();
    void init(const std::string modelPath);

    void loadModel(const std::string& modelPath);
    void buildAABBTree();

    void createVertexBuffer();
    void createIndexBuffer();
    void createSurfaceBuffer();

    void equalizeFaceAreas();
    void weldVertices(float epsilon);
    void recalculateNormals();
    void updateGeometry(const std::vector<Vertex>& newVertices, const std::vector<uint32_t>& newIndices);
    void updateVertexBuffer();
    void updateIndexBuffer();
    void saveOBJ(const std::string& path) const;

    void recreateBuffers();
    void cleanup();

    glm::vec3 getBoundingBoxCenter();
    std::array<glm::vec3, 8> calculateBoundingBox(const std::vector<Vertex>& vertices, glm::vec3& mindBound, glm::vec3& maxBound);
   
    HitResult rayIntersect(const glm::vec3& rayOrigin, const glm::vec3& rayDir);
    void markEdge(uint32_t triIndex, int edgeNum);

    // Getters
    const std::vector<Vertex>& getVertices() const {
        return vertices;
    }
    const size_t& getVertexCount() const {
        return vertices.size();
    }
    const std::vector<uint32_t>& getIndices() const {
        return indices;
    }

    glm::vec3 getFaceNormal(uint32_t faceIndex) const;

    VkBuffer getVertexBuffer() {
        return vertexBuffer;
    }
    VkDeviceSize getVertexBufferOffset() {
        return vertexBufferOffset_;
    }

    VkBuffer getIndexBuffer() {
        return indexBuffer;
    }
    VkDeviceSize getIndexBufferOffset() {
        return indexBufferOffset_;
    }

    VkBuffer getSurfaceBuffer() {
        return surfaceBuffer;
    }
    VkDeviceSize getSurfaceBufferOffset() {
        return surfaceBufferOffset_;
    }

    VkBuffer getSurfaceVertexBuffer() {
        return surfaceVertexBuffer;
    }
    VkDeviceSize getSurfaceVertexBufferOffset() {
        return surfaceVertexBufferOffset_;
    }

    glm::vec3 getModelPosition() {
        return modelPosition;
    }

    glm::mat4 getModelMatrix() {
        return modelMatrix;
    }

    std::unique_ptr<AABBTree>& getAABBTree() {
        return aabbTree;
    }

    // Setters
    void setModelPosition(const glm::vec3& position) { 
        modelPosition = position; 
    }
    void setModelMatrix(const glm::mat4& matrix) {
        modelMatrix = matrix;
    }
    void setIndices(const std::vector<uint32_t>& newIndices) {
        indices = newIndices;
    }
    void setVertices(const std::vector<Vertex>& newVertices) { 
        vertices = newVertices; 
    }

private:
    VulkanDevice& vulkanDevice;
    MemoryAllocator& memoryAllocator;
    Camera& camera;

    std::unique_ptr<AABBTree> aabbTree;

	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	VkBuffer vertexBuffer;
    VkDeviceSize vertexBufferOffset_;

	VkBuffer indexBuffer;
    VkDeviceSize indexBufferOffset_;

    VkBuffer surfaceBuffer;
    VkDeviceSize surfaceBufferOffset_;
    SurfaceVertex* mappedSurfaceVertices = nullptr;

    VkBuffer surfaceVertexBuffer;
    VkDeviceSize surfaceVertexBufferOffset_;

    glm::vec3 modelPosition{};
    glm::mat4 modelMatrix = glm::mat4(1.0f);

    std::vector<glm::vec3> faceNormals;
    std::unordered_set<Edge, EdgeHash> featuredEdges;
}; 