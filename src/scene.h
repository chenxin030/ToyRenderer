#pragma once
#include "vk/vk_device.h"
#include <glm/glm.hpp>
#include <string>
struct Buffer {
  VkBuffer handle{};
  VkDeviceMemory memory{};
  void *mapped{};
};
struct Image {
  VkImage handle{};
  VkDeviceMemory memory{};
  VkImageView view{};
};
struct Vertex {
  glm::vec3 p, n;
  glm::vec4 tangent;
  glm::vec2 uv;
};
struct Primitive {
  uint32_t first, count;
  glm::mat4 model;
  glm::vec4 color, params;
  glm::vec4 emission{0,0,0,0};
  glm::vec4 alpha{0,0,0,0}; // cutoff, MASK enabled
  size_t material;
  bool transparent = false;
};
struct Material {
  Image base, normal, mr, emission;
};
struct Scene {
  bool street = false;
  size_t animatedDraw = size_t(-1);
  vk::Device *d{};
  Buffer vertices;
  std::vector<Primitive> draws;
  std::vector<Material> materials;
  Image irradiance, prefilter, lut;
  VkSampler sampler{}, materialSampler{};
  void init(vk::Device &device);
  void destroy();
};
Buffer makeBuffer(vk::Device &d, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void *data = nullptr);
Image makeImage(vk::Device &d, uint32_t w, uint32_t h, VkFormat format,
                VkImageUsageFlags usage, const void *data = nullptr,
                size_t bytes = 0);
void freeBuffer(vk::Device &d, Buffer &b);
void freeImage(vk::Device &d, Image &i);
