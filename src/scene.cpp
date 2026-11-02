#include "scene.h"
#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/gtc/constants.hpp>
#include <stb/stb_image.h>
#include <stdexcept>
static void ck(VkResult r) {
  if (r != VK_SUCCESS)
    throw std::runtime_error("scene Vulkan " + std::to_string(r));
}
static uint32_t memory(vk::Device &d, uint32_t bits,
                       VkMemoryPropertyFlags flags) {
  VkPhysicalDeviceMemoryProperties p;
  vkGetPhysicalDeviceMemoryProperties(d.physicalDevice, &p);
  for (uint32_t i = 0; i < p.memoryTypeCount; ++i)
    if ((bits & (1u << i)) && (p.memoryTypes[i].propertyFlags & flags) == flags)
      return i;
  throw std::runtime_error("memory type");
}
Buffer makeBuffer(vk::Device &d, VkDeviceSize size, VkBufferUsageFlags usage,
                  const void *data) {
  Buffer b;
  VkBufferCreateInfo c{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  c.size = size;
  const bool staticVertex = (usage & VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) && data;
  c.usage = usage | (staticVertex ? VK_BUFFER_USAGE_TRANSFER_DST_BIT : 0);
  ck(vkCreateBuffer(d.device, &c, nullptr, &b.handle));
  VkMemoryRequirements r;
  vkGetBufferMemoryRequirements(d.device, b.handle, &r);
  VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  a.allocationSize = r.size;
  a.memoryTypeIndex = memory(d, r.memoryTypeBits, staticVertex ? VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT : (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT));
  ck(vkAllocateMemory(d.device, &a, nullptr, &b.memory));
  ck(vkBindBufferMemory(d.device, b.handle, b.memory, 0));
  if(staticVertex) {
    Buffer staging = makeBuffer(d,size,VK_BUFFER_USAGE_TRANSFER_SRC_BIT,data);
    VkCommandPool pool{}; VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pi.queueFamilyIndex=d.graphicsFamily;
    ck(vkCreateCommandPool(d.device,&pi,nullptr,&pool));
    VkCommandBuffer cmd{}; VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO}; ca.commandPool=pool;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
    ck(vkAllocateCommandBuffers(d.device,&ca,&cmd));VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};ck(vkBeginCommandBuffer(cmd,&begin));
    VkBufferCopy copy{0,0,size};vkCmdCopyBuffer(cmd,staging.handle,b.handle,1,&copy);
    VkBufferMemoryBarrier barrier{VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT;barrier.srcQueueFamilyIndex=barrier.dstQueueFamilyIndex=VK_QUEUE_FAMILY_IGNORED;barrier.buffer=b.handle;barrier.size=VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,0,0,nullptr,1,&barrier,0,nullptr);
    ck(vkEndCommandBuffer(cmd));VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&cmd;ck(vkQueueSubmit(d.queue,1,&si,{}));ck(vkQueueWaitIdle(d.queue));
    vkDestroyCommandPool(d.device,pool,nullptr);freeBuffer(d,staging);return b;
  }
  ck(vkMapMemory(d.device, b.memory, 0, size, 0, &b.mapped));
  if (data)
    memcpy(b.mapped, data, size);
  return b;
}
void freeBuffer(vk::Device &d, Buffer &b) {
  if (b.mapped)
    vkUnmapMemory(d.device, b.memory);
  vkDestroyBuffer(d.device, b.handle, nullptr);
  vkFreeMemory(d.device, b.memory, nullptr);
  b = {};
}
void freeImage(vk::Device &d, Image &i) {
  vkDestroyImageView(d.device, i.view, nullptr);
  vkDestroyImage(d.device, i.handle, nullptr);
  vkFreeMemory(d.device, i.memory, nullptr);
  i = {};
}
Image makeImage(vk::Device &d, uint32_t w, uint32_t h, VkFormat format,
                VkImageUsageFlags usage, const void *data, size_t bytes) {
  Image i;
  VkFormatProperties fp;
  vkGetPhysicalDeviceFormatProperties(d.physicalDevice, format, &fp);
  VkFormatFeatureFlags need = 0;
  if (usage & VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT)
    need |= VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT;
  if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    need |= VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT;
  if (usage & VK_IMAGE_USAGE_SAMPLED_BIT)
    need |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            (format == VK_FORMAT_D32_SFLOAT ? 0 : VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT);
  if ((fp.optimalTilingFeatures & need) != need)
    throw std::runtime_error("image format unsupported");
  VkImageCreateInfo c{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
  c.imageType = VK_IMAGE_TYPE_2D;
  c.extent = {w, h, 1};
  c.format = format;
  c.mipLevels = c.arrayLayers = 1;
  c.samples = VK_SAMPLE_COUNT_1_BIT;
  c.tiling = VK_IMAGE_TILING_OPTIMAL;
  c.usage = usage | (data ? VK_IMAGE_USAGE_TRANSFER_DST_BIT : 0);
  ck(vkCreateImage(d.device, &c, nullptr, &i.handle));
  VkMemoryRequirements r;
  vkGetImageMemoryRequirements(d.device, i.handle, &r);
  VkMemoryAllocateInfo a{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
  a.allocationSize = r.size;
  if (usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
    printf("color image format=%d %ux%u allocation=%llu bytes\n", int(format),
           w, h, (unsigned long long)r.size);
  a.memoryTypeIndex =
      memory(d, r.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  ck(vkAllocateMemory(d.device, &a, nullptr, &i.memory));
  ck(vkBindImageMemory(d.device, i.handle, i.memory, 0));
  VkImageViewCreateInfo v{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  v.image = i.handle;
  v.viewType = VK_IMAGE_VIEW_TYPE_2D;
  v.format = format;
  v.subresourceRange = {VkImageAspectFlags(format == VK_FORMAT_D32_SFLOAT ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT), 0, 1, 0, 1};
  ck(vkCreateImageView(d.device, &v, nullptr, &i.view));
  if (data) {
    auto staging = makeBuffer(d, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data);
    VkCommandPool pool;
    VkCommandPoolCreateInfo pc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pc.queueFamilyIndex = d.graphicsFamily;
    ck(vkCreateCommandPool(d.device, &pc, nullptr, &pool));
    VkCommandBuffer cmd;
    VkCommandBufferAllocateInfo ca{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ca.commandPool = pool;
    ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ca.commandBufferCount = 1;
    ck(vkAllocateCommandBuffers(d.device, &ca, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    ck(vkBeginCommandBuffer(cmd, &bi));
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = i.handle;
    b.subresourceRange = v.subresourceRange;
    b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0,
                         nullptr, 1, &b);
    VkBufferImageCopy copy{};
    copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    copy.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cmd, staging.handle, i.handle,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
    b.oldLayout = b.newLayout;
    b.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    b.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr,
                         0, nullptr, 1, &b);
    ck(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    ck(vkQueueSubmit(d.queue, 1, &si, {}));
    ck(vkQueueWaitIdle(d.queue));
    vkDestroyCommandPool(d.device, pool, nullptr);
    freeBuffer(d, staging);
  }
  return i;
}
static float radical(uint32_t v) {
  v = (v << 16) | (v >> 16);
  v = ((v & 0x55555555) << 1) | ((v & 0xaaaaaaaa) >> 1);
  v = ((v & 0x33333333) << 2) | ((v & 0xcccccccc) >> 2);
  v = ((v & 0x0f0f0f0f) << 4) | ((v & 0xf0f0f0f0) >> 4);
  v = ((v & 0x00ff00ff) << 8) | ((v & 0xff00ff00) >> 8);
  return float(v) * 2.3283064365386963e-10f;
}
static glm::vec3 basis(glm::vec3 h, glm::vec3 n) {
  auto t = glm::normalize(
      glm::cross(abs(n.y) < .99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0), n));
  return t * h.x + glm::cross(n, t) * h.y + n * h.z;
}
static glm::vec3 sampleGGX(float x, float y, float rough, glm::vec3 n) {
  float a = rough * rough, phi = glm::two_pi<float>() * x;
  float z = sqrt((1 - y) / (1 + (a * a - 1) * y));
  return basis({cos(phi) * sqrt(1 - z * z), sin(phi) * sqrt(1 - z * z), z}, n);
}
static glm::vec3 env(glm::vec3 n) {
  return glm::vec3(.12, .15, .2) + glm::vec3(.65, .75, 1) * std::max(n.y, 0.f) +
         glm::vec3(4, 3, 2) *
             pow(std::max(glm::dot(n, glm::normalize(glm::vec3(-1, 2, 1))),
                          0.f),
                 64.f);
}
static glm::vec3 direction(int x, int y, int w, int h) {
  float phi = (float(x) + .5f) / w * glm::two_pi<float>(),
        theta = (float(y) + .5f) / h * glm::pi<float>();
  return {cos(phi) * sin(theta), cos(theta), sin(phi) * sin(theta)};
}
void Scene::init(vk::Device &device) {
  d = &device;
  auto environment = [&](glm::vec3 n) {
    return env(n);
  };
  VkSamplerCreateInfo s{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  s.magFilter = s.minFilter = VK_FILTER_LINEAR;
  s.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  s.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  s.addressModeV = s.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  ck(vkCreateSampler(d->device, &s, nullptr, &sampler));
  s.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
  ck(vkCreateSampler(d->device, &s, nullptr, &materialSampler));
  // Linear analytic environment; bake irradiance/pi, six GGX prefilter bands,
  // and split-sum DFG.
  std::vector<glm::vec4> pixels(32 * 16);
  for (int y = 0; y < 16; ++y)
    for (int x = 0; x < 32; ++x) {
      auto n = direction(x, y, 32, 16);
      glm::vec3 sum(0);
      for (int j = 0; j < 1024; ++j) {
        float u = (j + .5f) / 1024, z = sqrt(1 - u),
              phi = radical(j) * glm::two_pi<float>();
        sum +=
            environment(basis({sqrt(u) * cos(phi), sqrt(u) * sin(phi), z}, n));
      }
      pixels[y * 32 + x] = glm::vec4(sum / 1024.f, 1);
    }
  irradiance =
      makeImage(*d, 32, 16, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT, pixels.data(), pixels.size() * 16);
  pixels.resize(128 * 64 * 6);
  for (int k = 0; k < 6; ++k)
    for (int y = 0; y < 64; ++y)
      for (int x = 0; x < 128; ++x) {
        auto n = direction(x, y, 128, 64);
        glm::vec3 sum(0);
        float weight = 0;
        for (int j = 0; j < 256; ++j) {
          auto h = sampleGGX((j + .5f) / 256, radical(j),
                             std::max(.04f, k / 5.f), n);
          auto l = glm::reflect(-n, h);
          float nl = std::max(glm::dot(n, l), 0.f);
          sum += environment(l) * nl;
          weight += nl;
        }
        pixels[(k * 64 + y) * 128 + x] =
            glm::vec4(sum / std::max(weight, .001f), 1);
      }
  prefilter =
      makeImage(*d, 128, 384, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT, pixels.data(), pixels.size() * 16);
  pixels.resize(64 * 64);
  for (int y = 0; y < 64; ++y)
    for (int x = 0; x < 64; ++x) {
      float nv = (x + .5f) / 64, r = (y + .5f) / 64, k = r * r / 2;
      glm::vec3 v(sqrt(1 - nv * nv), 0, nv);
      float A = 0, B = 0;
      for (int j = 0; j < 1024; ++j) {
        auto h = sampleGGX((j + .5f) / 1024, radical(j), r, {0, 0, 1});
        auto l = glm::reflect(-v, h);
        float nl = std::max(l.z, 0.f), vh = std::max(glm::dot(v, h), 0.f);
        if (nl > 0) {
          float g = (nv / (nv * (1 - k) + k)) * (nl / (nl * (1 - k) + k));
          float vis = g * vh / std::max(h.z * nv, 1e-6f), f = pow(1 - vh, 5.f);
          A += (1 - f) * vis;
          B += f * vis;
        }
      }
      pixels[y * 64 + x] = {A / 1024, B / 1024, 0, 1};
    }
  lut =
      makeImage(*d, 64, 64, VK_FORMAT_R32G32B32A32_SFLOAT,
                VK_IMAGE_USAGE_SAMPLED_BIT, pixels.data(), pixels.size() * 16);
  auto path = std::filesystem::u8path(
      std::string(ASSET_DIR) + "bistro/bistro.gltf");
  street = path.filename()=="bistro.gltf";
  animatedDraw = size_t(-1);
  std::ifstream file(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(file)), {});
  cgltf_options options{};
  cgltf_data *raw = nullptr;
  if (cgltf_parse(&options, text.data(), text.size(), &raw) !=
      cgltf_result_success)
    throw std::runtime_error("glTF parse");
  struct Guard {
    cgltf_data *p;
    ~Guard() { cgltf_free(p); }
  } guard{raw};
  // The shipped asset uses an embedded buffer so Unicode filesystem paths never
  // reach C stdio.
  std::vector<std::vector<char>> external(raw->buffers_count);
  for (size_t bi = 0; bi < raw->buffers_count; ++bi) {
    auto &buffer = raw->buffers[bi];
    if (buffer.uri && strncmp(buffer.uri, "data:", 5) != 0) {
      std::ifstream f(path.parent_path() / std::filesystem::u8path(buffer.uri),
                      std::ios::binary);
      external[bi].assign(std::istreambuf_iterator<char>(f), {});
      if (external[bi].size() < buffer.size)
        throw std::runtime_error("glTF external buffer truncated");
      buffer.data = external[bi].data();
    }
  }
  if (cgltf_load_buffers(&options, raw, nullptr) != cgltf_result_success ||
      cgltf_validate(raw) != cgltf_result_success)
    throw std::runtime_error("glTF buffers/validation");
  auto texture = [&](cgltf_texture_view tv, bool srgb,
                     std::array<unsigned char, 4> fallback) {
    int w = 1, h = 1;
    unsigned char *decoded = nullptr;
    auto t = tv.texture;
    if (t) {
      if (tv.has_transform)
        throw std::runtime_error("texture transform unsupported");
      if (t->sampler && (t->sampler->wrap_s != cgltf_wrap_mode_repeat ||
                         t->sampler->wrap_t != cgltf_wrap_mode_repeat))
        throw std::runtime_error("material requires repeat wrap");
      if (tv.texcoord != 0 || !t->image || !t->image->uri)
        throw std::runtime_error("only external PNG/JPEG TEXCOORD_0 supported");
      auto p = path.parent_path() / std::filesystem::u8path(t->image->uri);
      std::ifstream f(p, std::ios::binary);
      std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)), {});
      decoded =
          stbi_load_from_memory(b.data(), int(b.size()), &w, &h, nullptr, 4);
      if (!decoded)
        throw std::runtime_error("texture decode");
    }
    auto im = makeImage(
        *d, w, h, srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_SAMPLED_BIT, decoded ? decoded : fallback.data(),
        size_t(w) * h * 4);
    stbi_image_free(decoded);
    return im;
  };
  for (size_t i = 0; i < raw->materials_count; ++i) {
    auto &m = raw->materials[i];
    materials.emplace_back();
    auto &mat = materials.back();
    mat.base = texture(m.pbr_metallic_roughness.base_color_texture, true,
                       {255, 255, 255, 255});
    mat.normal = texture(m.normal_texture, false, {128, 128, 255, 255});
    mat.emission = texture(m.emissive_texture, true, {255,255,255,255});
    mat.mr = texture(m.pbr_metallic_roughness.metallic_roughness_texture, false,
                     {255, 255, 255, 255});
  }
  std::vector<Vertex> verts;
  auto active = [&](cgltf_node *node) {
    while (node->parent)
      node = node->parent;
    auto sc = raw->scene ? raw->scene
                         : (raw->scenes_count ? &raw->scenes[0] : nullptr);
    if (!sc)
      return false;
    for (size_t i = 0; i < sc->nodes_count; ++i)
      if (sc->nodes[i] == node)
        return true;
    return false;
  };
  for (size_t ni = 0; ni < raw->nodes_count; ++ni) {
    auto &node = raw->nodes[ni];
    if (!node.mesh || !active(&node))
      continue;
    if (node.skin)
      throw std::runtime_error("skinning unsupported");
    glm::mat4 model;
    cgltf_node_transform_world(&node, &model[0][0]);
    if (abs(glm::determinant(model)) < 1e-8f)
      throw std::runtime_error("singular node");
    for (size_t pi = 0; pi < node.mesh->primitives_count; ++pi) {
      auto &p = node.mesh->primitives[pi];
      if (p.type != cgltf_primitive_type_triangles || !p.material ||
          p.targets_count)
        throw std::runtime_error("unsupported primitive");
      auto pos = cgltf_find_accessor(&p, cgltf_attribute_type_position, 0),
           norm = cgltf_find_accessor(&p, cgltf_attribute_type_normal, 0),
           tan = cgltf_find_accessor(&p, cgltf_attribute_type_tangent, 0),
           uv = cgltf_find_accessor(&p, cgltf_attribute_type_texcoord, 0);
      if (!pos || !norm || !tan || !uv)
        throw std::runtime_error("POSITION NORMAL TANGENT TEXCOORD_0 required");
      Primitive draw;
      draw.first = uint32_t(verts.size());
      draw.count = uint32_t(p.indices ? p.indices->count : pos->count);
      draw.model = model;
      draw.transparent = p.material->alpha_mode == cgltf_alpha_mode_blend;
      draw.emission = {p.material->emissive_factor[0],p.material->emissive_factor[1],p.material->emissive_factor[2],0};
      draw.alpha = {p.material->alpha_cutoff, p.material->alpha_mode == cgltf_alpha_mode_mask ? 1.f : 0.f,0,0};
      draw.material = size_t(p.material - raw->materials);
      auto &m = p.material->pbr_metallic_roughness;
      memcpy(&draw.color, m.base_color_factor, 16);
      draw.params = {m.metallic_factor, m.roughness_factor,
                     p.material->normal_texture.texture
                         ? p.material->normal_texture.scale
                         : 0,
                     0};
      for (size_t j = 0; j < draw.count; ++j) {
        size_t ix = p.indices ? cgltf_accessor_read_index(p.indices, j) : j;
        Vertex v;
        cgltf_accessor_read_float(pos, ix, &v.p.x, 3);
        cgltf_accessor_read_float(norm, ix, &v.n.x, 3);
        cgltf_accessor_read_float(tan, ix, &v.tangent.x, 4);
        cgltf_accessor_read_float(uv, ix, &v.uv.x, 2);
        verts.push_back(v);
      }
      if(node.name && std::string(node.name)=="ToyRenderer_MovingCrate") animatedDraw=draws.size();
      draws.push_back(draw);
    }
  }
  if (verts.empty())
    throw std::runtime_error("empty glTF");
  vertices = makeBuffer(*d, verts.size() * sizeof(Vertex),
                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, verts.data());
  printf("glTF primitives=%zu vertices=%zu materials=%zu; IBL baked\n",
         draws.size(), verts.size(), materials.size());
}
void Scene::destroy() {
  if (!d)
    return;
  freeBuffer(*d, vertices);
  for (auto &m : materials) {
    freeImage(*d, m.base);
    freeImage(*d, m.normal);
    freeImage(*d, m.mr);
    freeImage(*d, m.emission);
  }
  freeImage(*d, irradiance);
  freeImage(*d, prefilter);
  freeImage(*d, lut);
  vkDestroySampler(d->device, sampler, nullptr);
  vkDestroySampler(d->device, materialSampler, nullptr);
  d = nullptr;
}
