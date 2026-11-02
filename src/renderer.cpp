#include "renderer.h"
#include "scene.h"
#include "shadows.h"
#include "vk/vk_device.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <stdexcept>
#include <thread>
static void check(VkResult r) {
  if (r != VK_SUCCESS)
    throw std::runtime_error("Vulkan result " + std::to_string(r));
}
using Clock = std::chrono::steady_clock;
struct RenderSettings {
  bool replay = false;
  bool animate = false;
  bool dusk = false;
  bool shadows = true;
  bool normalMapping = true;
  bool deferred = true;
  bool taa=true, jitter=true, reprojection=true, depthReject=true, clipHistory=true;
  int taaDebug=0;
  bool ao=true, aoHalf=true, aoFilter=true;
  float aoRadius=.7f;
  int aoSamples=16;
  bool pointLights = true;
  int width=1280,height=720;
  bool tiled=true, largeLights=false;
  int lightCount=128, tileSize=16;
  int component = 0;
  int shadowResolution = 1024;
  bool shadowSnapping = true;
  int shadowPCF = 1;
  float shadowLambda = .65f;
};

struct Frame {
  VkCommandPool pool{};
  VkCommandBuffer cmd{};
  VkFence fence{};
  VkSemaphore acquired{};
  VkQueryPool queries{};
  bool submitted = false;
};
struct Depth {
  VkImage image{};
  VkDeviceMemory memory{};
  VkImageView view{};
};
struct App {
  vk::Device d;
  Scene scene;
  RenderSettings options;
  bool normalMapping = true;
  std::array<Image,2> shadowMaps;
  VkPipeline shadowPipeline{};
  VkPipelineLayout shadowLayout{};
  int shadowResolution = 1024;
  bool shadows = true, snapping = true, rotateSun = false;
  bool slopeBias = true;
  float splitLambda = .65f, constantBias = .0001f, normalBias = 1.f;
  int pcfRadius = 1;
  double shadowMs[4]{};
  std::vector<Image> hdr;
  std::array<Image,2> history;
  std::array<bool,2> historyLayout{};
  std::vector<Image> motion; std::array<Image,2> temporalDebug;
  std::vector<VkDescriptorSet> temporalSets;
  std::array<VkDescriptorSet,2> historyOutput{};
  VkPipeline motionPipeline{}; VkPipelineLayout motionLayout{};
  glm::mat4 previousVP{1}, previousView{1};
  glm::vec2 previousJitter{0};
  std::vector<glm::mat4> previousModels;
  bool historyValid=false;
  int previousSettings=-1;
  std::array<float,20> previousQuality{};
  glm::mat4 animatedBase{1};

  std::vector<std::array<Image,3>> aoImages;
  VkPipeline aoPipeline[4]{};
  VkPipelineLayout aoLayout[4]{};
  double aoMs[3]{}, aoChainMs=0, taaMs=0, cullMs=0, lightMs=0;
  std::vector<std::array<Image, 3>> gbuffer;
  std::vector<VkDescriptorSet> gSets;
  VkDescriptorSetLayout gLayout{};
  VkPipeline gPipeline{}, lightPipeline{}, transparentPipeline{};
  VkPipelineLayout gPipeLayout{}, lightLayout{}, transparentLayout{};
  VkSampler nearest{};
  VkFormat gFormats[3]{};
  VkDescriptorSetLayout materialLayout{}, outputLayout{};
  VkDescriptorPool descriptorPool{};
  std::vector<VkDescriptorSet> materialSets, outputSets;
  std::array<Buffer, 2> cameraBuffers, lightBuffers, tileBuffers;
  std::array<uint32_t,2> tileWritten{};uint32_t emptyTiles=0,overflowTiles=0;
  VkPipeline tiledPipeline{}; VkPipelineLayout tiledLayout{};

  VkPipeline outputPipeline{};
  VkPipelineLayout outputPipelineLayout{};
  void descriptors();
  GLFWwindow *window{};
  std::array<Frame, 2> frames{};
  std::vector<Depth> depths;
  std::vector<VkSemaphore> ready;
  VkPipelineLayout layout{};
  VkPipeline pipeline{};
  bool gui = false, glfwGui = false, context = false;
  bool recreate = false;
  int pendingPreset = -1;
  bool requestedAOHalf = true;
  uint32_t bits = 0;
  float period = 0;
  double gpuMs = 0, cubeMs = 0, uiMs = 0, cpuMs = 0;
  uint64_t tick = 0, rendered = 0;
  PFN_vkCmdBeginDebugUtilsLabelEXT beginLabel{};
  PFN_vkCmdEndDebugUtilsLabelEXT endLabel{};
  void init();
  void targets();
  void dropTargets();
  void createPipeline();
  void draw(bool replay);
  ~App();
};
void App::init() {
  if(options.tileSize!=8 && options.tileSize!=16 && options.tileSize!=32) throw std::runtime_error("tile size must be 8/16/32");
  shadows=options.shadows;
  shadowResolution = options.shadowResolution;
  snapping=options.shadowSnapping; pcfRadius=options.shadowPCF; splitLambda=options.shadowLambda;
  if (!glfwInit())
    throw std::runtime_error("GLFW init");
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
  if(options.height>=1440) glfwWindowHint(GLFW_DECORATED,GLFW_FALSE);
  window =
      glfwCreateWindow(options.width, options.height, "ToyRenderer - Bistro", nullptr, nullptr);
  if (!window)
    throw std::runtime_error("window");
  if (!vk::CreateInstance(d) || !vk::CreateSurface(d, window) ||
      !vk::PickPhysicalDevice(d) || !vk::CreateDevice(d) ||
      !vk::CreateSwapchain(d, window))
    throw std::runtime_error("device init");
  VkPhysicalDeviceProperties prop{};
  vkGetPhysicalDeviceProperties(d.physicalDevice, &prop);
  period = prop.limits.timestampPeriod;
  uint32_t n = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(d.physicalDevice, &n, nullptr);
  std::vector<VkQueueFamilyProperties> q(n);
  vkGetPhysicalDeviceQueueFamilyProperties(d.physicalDevice, &n, q.data());
  bits = q[d.graphicsFamily].timestampValidBits;
  std::printf("GPU=%s driver=%u API=%u timestampPeriod=%g validBits=%u "
              "VSync=FIFO seed=42 dt=1/60 build=%s\n",
              prop.deviceName, prop.driverVersion, prop.apiVersion, period,
              bits,
#ifdef NDEBUG
              "Release"
#else
              "Debug"
#endif
  );
  if (!bits)
    throw std::runtime_error("timestamp unsupported");
  beginLabel = (PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetDeviceProcAddr(
      d.device, "vkCmdBeginDebugUtilsLabelEXT");
  endLabel = (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetDeviceProcAddr(
      d.device, "vkCmdEndDebugUtilsLabelEXT");
  VkFormatProperties fp{};
  vkGetPhysicalDeviceFormatProperties(d.physicalDevice, VK_FORMAT_D32_SFLOAT,
                                      &fp);
  if (!(fp.optimalTilingFeatures &
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT))
    throw std::runtime_error("D32 unsupported");
  for (auto &f : frames) {
    VkCommandPoolCreateInfo p{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    p.queueFamilyIndex = d.graphicsFamily;
    check(vkCreateCommandPool(d.device, &p, nullptr, &f.pool));
    VkCommandBufferAllocateInfo a{
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = f.pool;
    a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    a.commandBufferCount = 1;
    check(vkAllocateCommandBuffers(d.device, &a, &f.cmd));
    VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fc.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    check(vkCreateFence(d.device, &fc, nullptr, &f.fence));
    VkSemaphoreCreateInfo sc{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(d.device, &sc, nullptr, &f.acquired));
    VkQueryPoolCreateInfo qc{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qc.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qc.queryCount = 24;
    check(vkCreateQueryPool(d.device, &qc, nullptr, &f.queries));
  }
  scene.init(d);
  if(!scene.draws.empty()) animatedBase=scene.animatedDraw<scene.draws.size()?scene.draws[scene.animatedDraw].model:glm::mat4(1);
  for (auto &b : cameraBuffers)
    b = makeBuffer(d, 656, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
  VkSamplerCreateInfo ss{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  ss.magFilter = ss.minFilter = VK_FILTER_NEAREST;
  ss.addressModeU = ss.addressModeV = ss.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  check(vkCreateSampler(d.device, &ss, nullptr, &nearest));
  for(auto &im:shadowMaps) im=makeImage(d,shadowResolution*2,shadowResolution*2,VK_FORMAT_D32_SFLOAT,
      VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  descriptors();
  VkDescriptorSetLayout tl[]={materialLayout,gLayout};
  VkPipelineLayoutCreateInfo plc{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};plc.setLayoutCount=2;plc.pSetLayouts=tl;check(vkCreatePipelineLayout(d.device,&plc,nullptr,&tiledLayout));
  std::ifstream tf(std::filesystem::u8path(std::string(SHADER_DIR)+"tiled.comp.spv"),std::ios::binary|std::ios::ate);if(!tf) throw std::runtime_error("tiled shader open failed");size_t bytes=size_t(tf.tellg());std::vector<uint32_t> code(bytes/4);tf.seekg(0);tf.read(reinterpret_cast<char*>(code.data()),bytes);
  VkShaderModuleCreateInfo sm{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};sm.codeSize=bytes;sm.pCode=code.data();VkShaderModule module;check(vkCreateShaderModule(d.device,&sm,nullptr,&module));
  VkComputePipelineCreateInfo cp{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cp.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=module;cp.stage.pName="main";cp.layout=tiledLayout;check(vkCreateComputePipelines(d.device,VK_NULL_HANDLE,1,&cp,nullptr,&tiledPipeline));vkDestroyShaderModule(d.device,module,nullptr);

  gFormats[0] = gFormats[2] = VK_FORMAT_R32G32B32A32_SFLOAT;
  gFormats[1] = VK_FORMAT_R32G32B32A32_SFLOAT;
  VkSamplerCreateInfo ns{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
  ns.magFilter = ns.minFilter = VK_FILTER_NEAREST;
  ns.addressModeU = ns.addressModeV = ns.addressModeW =
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;

  targets();
  createPipeline();
  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  context = true;
  ImGui::GetIO().IniFilename = nullptr;
  glfwGui = ImGui_ImplGlfw_InitForVulkan(window, true);
  ImGui_ImplVulkan_InitInfo info{};
  info.ApiVersion = VK_API_VERSION_1_3;
  info.Instance = d.instance;
  info.PhysicalDevice = d.physicalDevice;
  info.Device = d.device;
  info.QueueFamily = d.graphicsFamily;
  info.Queue = d.queue;
  info.MinImageCount = 2;
  info.ImageCount = 2;
  info.DescriptorPoolSize = 16;
  info.UseDynamicRendering = true;
  info.PipelineRenderingCreateInfo = {
      VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
  info.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
  info.PipelineRenderingCreateInfo.pColorAttachmentFormats = &d.swapFormat;
  info.CheckVkResultFn = check;
  gui = ImGui_ImplVulkan_Init(&info);
  if (!gui)
    throw std::runtime_error("ImGui init");
}
void App::targets() {
  for(int i=0;i<2;++i) {
    tileBuffers[i]=makeBuffer(d, VkDeviceSize((d.extent.width+7)/8)*((d.extent.height+7)/8)*258*4,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    for(size_t m=0;m<scene.materials.size();++m) {
      VkDescriptorBufferInfo info{tileBuffers[i].handle,0,VK_WHOLE_SIZE};
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet=materialSets[i*scene.materials.size()+m];w.dstBinding=9;w.descriptorCount=1;w.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;w.pBufferInfo=&info;vkUpdateDescriptorSets(d.device,1,&w,0,nullptr);
    }
  }
  historyValid=false; historyLayout={};
  previousModels.clear();
  for(auto &im:temporalDebug) im=makeImage(d,d.extent.width,d.extent.height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
  for(auto &im:history) im=makeImage(d,d.extent.width,d.extent.height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);

  hdr.resize(d.swapImages.size());
  outputSets.resize(d.swapImages.size());
  std::vector<VkDescriptorSetLayout> outLayouts(hdr.size(), outputLayout);
  VkDescriptorSetAllocateInfo outAlloc{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  outAlloc.descriptorPool = descriptorPool;
  outAlloc.descriptorSetCount = uint32_t(hdr.size());
  outAlloc.pSetLayouts = outLayouts.data();
  check(vkAllocateDescriptorSets(d.device, &outAlloc, outputSets.data()));
  for (size_t i = 0; i < hdr.size(); ++i) {
    hdr[i] = makeImage(
        d, d.extent.width, d.extent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkDescriptorImageInfo info{scene.sampler, hdr[i].view,
                               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = outputSets[i];
    w.dstBinding = 0;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &info;
    vkUpdateDescriptorSets(d.device, 1, &w, 0, nullptr);
  }
  depths.resize(d.swapImages.size());
  ready.resize(d.swapImages.size());
  for (size_t i = 0; i < depths.size(); ++i) {
    auto &z = depths[i];
    VkImageCreateInfo ic{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ic.imageType = VK_IMAGE_TYPE_2D;
    ic.format = VK_FORMAT_D32_SFLOAT;
    ic.extent = {d.extent.width, d.extent.height, 1};
    ic.mipLevels = ic.arrayLayers = 1;
    ic.samples = VK_SAMPLE_COUNT_1_BIT;
    ic.tiling = VK_IMAGE_TILING_OPTIMAL;
    ic.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
               VK_IMAGE_USAGE_SAMPLED_BIT;
    check(vkCreateImage(d.device, &ic, nullptr, &z.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(d.device, z.image, &req);
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(d.physicalDevice, &mp);
    uint32_t type = UINT32_MAX;
    for (uint32_t j = 0; j < mp.memoryTypeCount; ++j)
      if ((req.memoryTypeBits & (1u << j)) &&
          (mp.memoryTypes[j].propertyFlags &
           VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        type = j;
        break;
      }
    if (type == UINT32_MAX)
      throw std::runtime_error("depth memory");
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = req.size;
    ma.memoryTypeIndex = type;
    check(vkAllocateMemory(d.device, &ma, nullptr, &z.memory));
    check(vkBindImageMemory(d.device, z.image, z.memory, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = z.image;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = VK_FORMAT_D32_SFLOAT;
    vi.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
    check(vkCreateImageView(d.device, &vi, nullptr, &z.view));
    VkSemaphoreCreateInfo sc{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    check(vkCreateSemaphore(d.device, &sc, nullptr, &ready[i]));
    std::printf("depth allocation[%zu]=%llu bytes\n", i,
                (unsigned long long)req.size);
  }
  aoImages.resize(hdr.size());
  gbuffer.resize(hdr.size());
  gSets.resize(hdr.size());
  std::vector<VkDescriptorSetLayout> gl(hdr.size(), gLayout);
  outAlloc.pSetLayouts = gl.data();
  check(vkAllocateDescriptorSets(d.device, &outAlloc, gSets.data()));
  for (size_t i = 0; i < hdr.size(); ++i) {
    for (int j = 0; j < 3; ++j)
      gbuffer[i][j] = makeImage(d, d.extent.width, d.extent.height, gFormats[j],
                                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                                    VK_IMAGE_USAGE_SAMPLED_BIT);
    for(int j=0;j<3;++j) aoImages[i][j]=makeImage(d,
      j<2 && options.aoHalf ? (d.extent.width+1)/2 : d.extent.width,
      j<2 && options.aoHalf ? (d.extent.height+1)/2 : d.extent.height,
      VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    for (int j = 0; j < 7; ++j) {
      VkDescriptorImageInfo info{nearest,
                                 j < 3 ? gbuffer[i][j].view : j==3 ? depths[i].view : aoImages[i][j-4].view,
                                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = gSets[i];
      w.dstBinding = j;
      w.descriptorCount = 1;
      w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
      w.pImageInfo = &info;
      vkUpdateDescriptorSets(d.device, 1, &w, 0, nullptr);
    }
  }
  printf("G-buffer logical bytes/pixel=%d, per image=%llu (depth included; "
         "excludes HDR)\n",
         52,
         (unsigned long long)d.extent.width * d.extent.height *
             (52));
  motion.resize(hdr.size()); temporalSets.resize(hdr.size()*2);
  std::vector<VkDescriptorSetLayout> tls(temporalSets.size(),gLayout);
  outAlloc.descriptorSetCount=uint32_t(tls.size()); outAlloc.pSetLayouts=tls.data();
  check(vkAllocateDescriptorSets(d.device,&outAlloc,temporalSets.data()));
  for(size_t i=0;i<hdr.size();++i) {
    motion[i]=makeImage(d,d.extent.width,d.extent.height,VK_FORMAT_R16G16B16A16_SFLOAT,VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT|VK_IMAGE_USAGE_SAMPLED_BIT);
    for(int parity=0;parity<2;++parity) for(int j=0;j<10;++j) {
      VkImageView v=j<3?gbuffer[i][j].view:j==3?depths[i].view:j<7?aoImages[i][j-4].view:j==7?hdr[i].view:j==8?motion[i].view:history[1-parity].view;
      VkDescriptorImageInfo info{j==9?scene.sampler:nearest,v,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet=temporalSets[i*2+parity]; w.dstBinding=j; w.descriptorCount=1; w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&info;
      vkUpdateDescriptorSets(d.device,1,&w,0,nullptr);
    }
  }
  VkDescriptorSetLayout hol[]={outputLayout,outputLayout}; outAlloc.descriptorSetCount=2; outAlloc.pSetLayouts=hol;
  check(vkAllocateDescriptorSets(d.device,&outAlloc,historyOutput.data()));
  for(int j=0;j<2;++j) {
    VkDescriptorImageInfo info{nearest,history[j].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}; w.dstSet=historyOutput[j]; w.descriptorCount=1; w.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w.pImageInfo=&info;
    vkUpdateDescriptorSets(d.device,1,&w,0,nullptr);
    info.imageView=temporalDebug[j].view; w.dstBinding=1; vkUpdateDescriptorSets(d.device,1,&w,0,nullptr);
  }

}
void App::dropTargets() {
  tileWritten={};
  for(auto &b:tileBuffers) {freeBuffer(d,b); b={};}
  for(auto &im:temporalDebug) freeImage(d,im);
  for(auto &im:history) freeImage(d,im);
  for(auto &im:motion) freeImage(d,im); motion.clear();
  if(!temporalSets.empty()) vkFreeDescriptorSets(d.device,descriptorPool,uint32_t(temporalSets.size()),temporalSets.data()); temporalSets.clear();
  if(historyOutput[0]) vkFreeDescriptorSets(d.device,descriptorPool,2,historyOutput.data()); historyOutput={};

  for(auto &a:aoImages) for(auto &im:a) freeImage(d,im);
  aoImages.clear();
  for (auto &g : gbuffer)
    for (auto &im : g)
      freeImage(d, im);
  gbuffer.clear();
  if (!gSets.empty())
    vkFreeDescriptorSets(d.device, descriptorPool, uint32_t(gSets.size()),
                         gSets.data());
  gSets.clear();
  for (auto &i : hdr)
    freeImage(d, i);
  hdr.clear();
  if (!outputSets.empty())
    vkFreeDescriptorSets(d.device, descriptorPool, uint32_t(outputSets.size()),
                         outputSets.data());
  outputSets.clear();
  for (auto &z : depths) {
    vkDestroyImageView(d.device, z.view, nullptr);
    vkDestroyImage(d.device, z.image, nullptr);
    vkFreeMemory(d.device, z.memory, nullptr);
  }
  depths.clear();
  for (auto s : ready)
    vkDestroySemaphore(d.device, s, nullptr);
  ready.clear();
}
void App::createPipeline() {
  bool output = false, geometry = false, lighting = false, transparent = false, shadowPass = false;
  int aoPass=-1; bool motionPass=false;
  auto build = [&]() {
    VkPushConstantRange range{
        VkShaderStageFlags(output ? VK_SHADER_STAGE_FRAGMENT_BIT
                                  : VK_SHADER_STAGE_VERTEX_BIT |
                                        VK_SHADER_STAGE_FRAGMENT_BIT),
        0, output ? 16u : 128u};
    VkPipelineLayoutCreateInfo lc{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    VkDescriptorSetLayout ll[] = {materialLayout, gLayout};
    lc.setLayoutCount = lighting ? 2 : 1;
    lc.pSetLayouts = lighting ? ll : (output ? &outputLayout : &materialLayout);
    lc.pushConstantRangeCount = 1;
    lc.pPushConstantRanges = &range;
    check(vkCreatePipelineLayout(
        d.device, &lc, nullptr,
        motionPass ? &motionLayout : aoPass>=0 ? &aoLayout[aoPass] : shadowPass ? &shadowLayout : transparent ? &transparentLayout
                    : (lighting ? &lightLayout
                                : (geometry ? &gPipeLayout
                                            : (output ? &outputPipelineLayout
                                                      : &layout)))));
    auto load = [&](const char *name) {
      std::ifstream f(std::filesystem::u8path(std::string(SHADER_DIR) + name),
                      std::ios::binary | std::ios::ate);
      if (!f)
        throw std::runtime_error("shader missing");
      size_t size = size_t(f.tellg());
      std::vector<uint32_t> data(size / 4);
      f.seekg(0);
      f.read((char *)data.data(), size);
      VkShaderModuleCreateInfo ci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
      ci.codeSize = size;
      ci.pCode = data.data();
      VkShaderModule m;
      check(vkCreateShaderModule(d.device, &ci, nullptr, &m));
      return m;
    };
    VkShaderModule vs = load(motionPass ? "motion.vert.spv" : shadowPass ? "shadow.vert.spv" : (output || lighting) ? "tonemap.vert.spv"
                                                  : "cube.vert.spv"),
                   fs =
                       load(shadowPass ? "shadow.frag.spv" : motionPass ? "motion.frag.spv" : aoPass==3 ? "taa.frag.spv" : aoPass>=0 ? (aoPass==0?"ssao.frag.spv":aoPass==1?"ao_filter.frag.spv":"ao_upsample.frag.spv") : lighting ? "deferred.frag.spv"
                                     : (geometry ? "gbuffer.frag.spv"
                                                 : (output ? "tonemap.frag.spv"
                                                           : "cube.frag.spv")));
    VkPipelineShaderStageCreateInfo stages[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO}};
    stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    stages[0].module = vs;
    stages[0].pName = "main";
    stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    stages[1].module = fs;
    stages[1].pName = "main";
    VkPipelineVertexInputStateCreateInfo vertex{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkVertexInputBindingDescription binding{0, sizeof(Vertex),
                                            VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription attrs[] = {
        {0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, p)},
        {1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, n)},
        {2, 0, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, tangent)},
        {3, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(Vertex, uv)}};
    if (!output && !lighting) {
      vertex.vertexBindingDescriptionCount = 1;
      vertex.pVertexBindingDescriptions = &binding;
      vertex.vertexAttributeDescriptionCount = motionPass ? 1 : shadowPass ? 2 : 4;
      if(shadowPass) attrs[1]=attrs[3];
      vertex.pVertexAttributeDescriptions = attrs;
    }
    VkPipelineInputAssemblyStateCreateInfo assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    viewport.viewportCount = viewport.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo raster{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    raster.polygonMode = VK_POLYGON_MODE_FILL;
    raster.cullMode = VK_CULL_MODE_NONE;
    raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    raster.lineWidth = 1;
    VkPipelineMultisampleStateCreateInfo ms{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = !output && !lighting;
    if (transparent || motionPass) ds.depthWriteEnable = false;
    if(motionPass) ds.depthCompareOp=VK_COMPARE_OP_GREATER_OR_EQUAL;
    ds.depthCompareOp = motionPass ? VK_COMPARE_OP_EQUAL : shadowPass ? VK_COMPARE_OP_LESS : VK_COMPARE_OP_GREATER;
    VkPipelineColorBlendAttachmentState blend{};
    blend.colorWriteMask = 15;
    if (transparent) {
      blend.blendEnable = VK_TRUE;
      blend.srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
      blend.dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.colorBlendOp = VK_BLEND_OP_ADD;
      blend.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
      blend.dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
      blend.alphaBlendOp = VK_BLEND_OP_ADD;
    }
    VkPipelineColorBlendStateCreateInfo cb{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState blends[] = {blend, blend, blend};
    cb.attachmentCount = shadowPass ? 0 : geometry ? 3 : aoPass==3 ? 2 : 1;
    cb.pAttachments = blends;
    VkDynamicState states[] = {VK_DYNAMIC_STATE_VIEWPORT,
                               VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dynamic.dynamicStateCount = 2;
    dynamic.pDynamicStates = states;
    VkPipelineRenderingCreateInfo rendering{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    rendering.colorAttachmentCount = shadowPass ? 0 : geometry ? 3 : aoPass==3 ? 2 : 1;
    VkFormat hdrFormat = VK_FORMAT_R16G16B16A16_SFLOAT; VkFormat temporalFormats[]={hdrFormat,hdrFormat};
    rendering.pColorAttachmentFormats =
        aoPass==3 ? temporalFormats : geometry ? gFormats : (output ? &d.swapFormat : &hdrFormat);
    rendering.depthAttachmentFormat =
        (output || lighting) ? VK_FORMAT_UNDEFINED : VK_FORMAT_D32_SFLOAT;
    VkGraphicsPipelineCreateInfo pc{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    pc.pNext = &rendering;
    pc.stageCount = 2;
    pc.pStages = stages;
    pc.pVertexInputState = &vertex;
    pc.pInputAssemblyState = &assembly;
    pc.pViewportState = &viewport;
    pc.pRasterizationState = &raster;
    pc.pMultisampleState = &ms;
    pc.pDepthStencilState = &ds;
    pc.pColorBlendState = &cb;
    pc.pDynamicState = &dynamic;
    pc.layout = motionPass ? motionLayout : aoPass>=0 ? aoLayout[aoPass] : shadowPass ? shadowLayout : transparent ? transparentLayout
                : lighting
                    ? lightLayout
                    : (geometry ? gPipeLayout
                                : (output ? outputPipelineLayout : layout));
    auto result = vkCreateGraphicsPipelines(
        d.device, {}, 1, &pc, nullptr,
        motionPass ? &motionPipeline : aoPass>=0 ? &aoPipeline[aoPass] : shadowPass ? &shadowPipeline : transparent
            ? &transparentPipeline
            : (lighting ? &lightPipeline
                        : (geometry ? &gPipeline
                                    : (output ? &outputPipeline : &pipeline))));
    vkDestroyShaderModule(d.device, vs, nullptr);
    vkDestroyShaderModule(d.device, fs, nullptr);
    check(result);
  };
  build();
  transparent = true;
  build();
  transparent = false;
  geometry = true;
  build();
  geometry = false;
  lighting = true;
  build();
  for(aoPass=0;aoPass<4;++aoPass) build();
  aoPass=-1;
  lighting = false;
  output = true;
  build();
  output = false;
  motionPass=true; build(); motionPass=false;
  shadowPass = true;
  build();
}
void App::draw(bool replay) {
  if(options.animate && scene.animatedDraw<scene.draws.size()) {
    scene.draws[scene.animatedDraw].model=glm::translate(glm::mat4(1),glm::vec3(2.f*sin(float(tick)/60.f),0,0))*animatedBase;
  }
  auto &f = frames[rendered % 2];
  check(vkWaitForFences(d.device, 1, &f.fence, VK_TRUE, UINT64_MAX));
  if(tileWritten[rendered%2]) {
    emptyTiles=overflowTiles=0;auto *data=static_cast<const uint32_t*>(tileBuffers[rendered%2].mapped);
    for(uint32_t i=0;i<tileWritten[rendered%2];++i) {emptyTiles+=data[i*258]==0;overflowTiles+=data[i*258+1]!=0;}
  }
  if (f.submitted) {
    uint64_t values[48]{};
    auto r = vkGetQueryPoolResults(
        d.device, f.queries, 0, 24, sizeof(values), values, 16,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    if (r == VK_SUCCESS && values[1] && values[3] && values[5] && values[7]) {
      uint64_t mask = bits == 64 ? UINT64_MAX : ((1ull << bits) - 1);
      for(int j=0;j<4;++j) shadowMs[j]=double((values[(5+2*j)*2]-values[(4+2*j)*2])&mask)*period/1e6;
      for(int j=0;j<3;++j) aoMs[j]=double((values[(13+2*j)*2]-values[(12+2*j)*2])&mask)*period/1e6;
      aoChainMs=double((values[34]-values[24])&mask)*period/1e6;
      cullMs=double((values[42]-values[40])&mask)*period/1e6;lightMs=double((values[46]-values[44])&mask)*period/1e6;
      taaMs=double((values[38]-values[36])&mask)*period/1e6;
      gpuMs = double((values[6] - values[0]) & mask) * period / 1e6;
      cubeMs = double((values[2] - values[0]) & mask) * period / 1e6;
      uiMs = double((values[6] - values[4]) & mask) * period / 1e6;
    } else if (r != VK_NOT_READY)
      check(r);
    f.submitted=false;
  }
  uint32_t image = 0;
  auto acquired = vkAcquireNextImageKHR(d.device, d.swapchain, UINT64_MAX,
                                        f.acquired, {}, &image);
  if (acquired == VK_ERROR_OUT_OF_DATE_KHR) {
    recreate = true;
    return;
  }
  if (acquired == VK_SUBOPTIMAL_KHR)
    recreate = true;
  else
    check(acquired);
  check(vkResetCommandPool(d.device, f.pool, 0));
  ImGui_ImplVulkan_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();
  static bool direct = true, ibl = true;
  int &mode = options.component;
  {
  ImGui::Begin("ToyRenderer - Integrated renderer");
  if(ImGui::Button("1280 x 720")) glfwSetWindowSize(window,1280,720);
  ImGui::SameLine();
  if(ImGui::Button("1920 x 1080")) glfwSetWindowSize(window,1920,1080);
  if(ImGui::Button("Balanced")) pendingPreset=0;
  ImGui::SameLine();
  if(ImGui::Button("Quality")) pendingPreset=1;
  ImGui::SameLine();
  if(ImGui::Button("Baseline")) pendingPreset=2;
  if(ImGui::Checkbox("Camera tour", &options.replay)) historyValid=false;
  if(ImGui::Checkbox("Animate designated object", &options.animate)) historyValid=false;
  ImGui::Text("Empty tiles %u / overflow tiles %u",emptyTiles,overflowTiles);
  ImGui::Text("Cull %.3f + light %.3f = %.3f ms",cullMs,lightMs,cullMs+lightMs);
  ImGui::Checkbox("Tiled lighting",&options.tiled);
  ImGui::SliderInt("Lights",&options.lightCount,0,512);
  ImGui::Checkbox("Full screen light spheres",&options.largeLights);
  if(ImGui::Button("Tile 8")) options.tileSize=8; ImGui::SameLine();if(ImGui::Button("Tile 16")) options.tileSize=16;ImGui::SameLine();if(ImGui::Button("Tile 32")) options.tileSize=32;
  if(ImGui::Button("Light heatmap (magenta overflow)")) mode=18;
  ImGui::SameLine();
  if(ImGui::Button("Emissive component")) mode=19;

  if(ImGui::Checkbox("TAA", &options.taa)) historyValid=false;
  if(ImGui::Checkbox("Jitter", &options.jitter)) historyValid=false;
  ImGui::Checkbox("Reprojection", &options.reprojection);
  ImGui::Checkbox("Depth rejection", &options.depthReject);
  ImGui::Checkbox("Neighborhood clipping", &options.clipHistory);
  ImGui::SliderInt("TAA debug (0 off,1 velocity,2 weight,3 reject,4 clip)",&options.taaDebug,0,4);
  if(ImGui::Button("Reset history")) historyValid=false;
  if(ImGui::Checkbox("Dusk lighting", &options.dusk)) historyValid=false;

  ImGui::Checkbox("Shadows", &shadows);
  ImGui::Checkbox("Texel snapping", &snapping);
  ImGui::Checkbox("Slope bias", &slopeBias);
  ImGui::Checkbox("Rotate sun", &rotateSun);
  ImGui::SliderFloat("Split lambda", &splitLambda, 0, 1);
  ImGui::SliderInt("PCF radius", &pcfRadius, 0, 2);
  ImGui::SliderFloat("Constant depth bias", &constantBias, 0, .003f, "%.5f");
  ImGui::SliderFloat("Normal bias (texels)", &normalBias, 0, 4);
  for(int j=0;j<4;++j) ImGui::Text("Cascade %d: %.3f ms",j,shadowMs[j]);
  ImGui::Checkbox("SSAO (deferred)", &options.ao);
  if(ImGui::Checkbox("Half resolution AO", &requestedAOHalf)) recreate=true;
  ImGui::Checkbox("AO bilateral filter", &options.aoFilter);
  ImGui::SliderFloat("AO radius (world units)", &options.aoRadius, .05f, 3.f);
  ImGui::SliderInt("AO samples", &options.aoSamples, 4, 128);
  ImGui::Text("AO %.3f / filter %.3f / upsample %.3f ms",aoMs[0],aoMs[1],aoMs[2]);
  ImGui::Text("AO complete chain %.3f ms",aoChainMs);
  ImGui::Checkbox("Deferred", &options.deferred);
  ImGui::Checkbox("Point lights (all lights)", &options.pointLights);
  ImGui::Checkbox("Directional", &direct);
  ImGui::Checkbox("IBL", &ibl);
  ImGui::Checkbox("Normal map", &normalMapping);
  ImGui::Combo("Component", &mode,
               "Combined\0Direct\0Diffuse IBL\0Specular IBL\0World "
               "normal\0Base color linear\0MR data\0Depth\0Normal "
               "length\0Material ID\0World position\0Shadow mask\0Cascades\0Texel world size\0Shadow atlas\0AO raw\0AO filtered\0AO final\0");
  ImGui::Text("GPU interval %.3f ms / CPU submit %.3f ms", gpuMs, cpuMs);
  ImGui::Text("tick %llu / fixed dt 1/60 / seed 42", (unsigned long long)tick);
  ImGui::Text("Reversed Z: near=1 far=0; D32; FIFO");
  ImGui::Text("Drag RMB to orbit; wheel to zoom");
  ImGui::End();
  }
  ImGui::Render();
  static float yaw = 0, distance = 16;
  if (!replay && !ImGui::GetIO().WantCaptureMouse) {
    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT))
      yaw += ImGui::GetIO().MouseDelta.x * .005f;
    distance = glm::clamp(distance - ImGui::GetIO().MouseWheel, 5.f, 40.f);
  }
  float angle = replay ? float(tick % 720) * glm::two_pi<float>() / 720.f : yaw;
  glm::vec3 pan(0);
  glm::vec3 eye(sin(angle)*distance,6,cos(angle)*distance), target(0);
  if(scene.street) {
    float t=replay?float(tick%1800)/1800.f:0.f;
    eye=glm::vec3(-4.f+1.5f*sin(t*glm::two_pi<float>()),1.8f,4.f-7.f*sin(t*glm::pi<float>())*sin(t*glm::pi<float>()));
    target=eye+glm::vec3(1.f,0,-10.f);
    if(!replay) {target=eye+glm::vec3(cos(yaw)-10.f*sin(yaw),0,-sin(yaw)-10.f*cos(yaw));eye+=glm::normalize(target-eye)*(16-distance);target+=glm::normalize(target-eye)*(16-distance);}
  }
  auto view=glm::lookAtRH(eye,target,glm::vec3(0,1,0));
  auto projection = glm::perspectiveRH_ZO(
      glm::radians(55.f), float(d.extent.width) / d.extent.height, 100.f, .1f);
  projection[1][1] *= -1;
  auto unjitteredVP=projection*view;
  auto halton=[](uint64_t n,int base) { float v=0,f=1; while(n) { f/=base; v+=f*(n%base); n/=base; } return v; };
  glm::vec2 jitter=options.taa && options.jitter ? glm::vec2(halton(rendered%16+1,2)-.5f,halton(rendered%16+1,3)-.5f):glm::vec2(0);
  projection[2][0]-=2*jitter.x/d.extent.width; projection[2][1]-=2*jitter.y/d.extent.height;
  int settings=mode+32*options.deferred+64*direct+128*ibl+256*normalMapping+512*options.ao;
  if(settings!=previousSettings || previousModels.size()!=scene.draws.size()) historyValid=false;
  previousSettings=settings;
  std::array<float,20> quality{float(options.taa),float(options.jitter),float(shadows),float(snapping),float(pcfRadius),splitLambda,constantBias,normalBias,float(slopeBias),float(options.aoFilter),float(options.aoHalf),options.aoRadius,float(options.aoSamples),float(options.pointLights),float(options.lightCount),float(options.largeLights),float(options.taaDebug),float(options.reprojection),float(options.depthReject),float(options.clipHistory)};
  if(quality!=previousQuality) historyValid=false;
  previousQuality=quality;
  if(historyValid && (glm::length(glm::vec3(glm::inverse(view)[3])-glm::vec3(glm::inverse(previousView)[3]))>2.f || glm::dot(glm::vec3(view[2]),glm::vec3(previousView[2]))<.8f)) historyValid=false;
  struct Camera {
    glm::mat4 vp;
    glm::vec4 eye, light, settings;
    glm::mat4 inverseVP;
    glm::vec4 points;
    glm::mat4 shadowVP[4], view;
    glm::vec4 splits, texels, shadow, sun;
    glm::mat4 previousVP; glm::vec4 jitter;
  };
  static_assert(sizeof(Camera) == 656);
  Camera camera{projection * view,
                glm::vec4(eye, 1),
                glm::vec4(options.dusk?.2f:3.f, options.dusk?.25f:3.f, options.dusk?.4f:3.f, scene.street?(options.dusk?2.f:1.f):0.f),
                glm::vec4(direct, ibl*(options.dusk?.22f:1.f), normalMapping, mode),
                glm::inverse(projection * view),
                glm::vec4(options.pointLights, d.extent.width, d.extent.height, options.aoHalf)};
  glm::vec3 sun = glm::normalize(glm::vec3(-1,2,1));
  if(rotateSun) sun=glm::normalize(glm::vec3(sin(float(tick)*.005f),2,cos(float(tick)*.005f)));
  auto cascades = buildCascades(view,float(d.extent.width)/d.extent.height,sun,shadowResolution,splitLambda,snapping);
  for(int j=0;j<4;++j) camera.shadowVP[j]=cascades.matrix[j];
  camera.eye += glm::vec4(pan,0);
  camera.view=view; camera.splits=cascades.splits; camera.texels=cascades.texels;
  camera.shadow=glm::vec4(shadows,pcfRadius,constantBias,normalBias); camera.sun=glm::vec4(sun,slopeBias);
  camera.previousVP=historyValid?previousVP:unjitteredVP;
  glm::vec2 normalizedJitter(jitter.x/d.extent.width,jitter.y/d.extent.height);
  camera.jitter=glm::vec4(normalizedJitter,historyValid?previousJitter:normalizedJitter);
  memcpy(cameraBuffers[rendered % 2].mapped, &camera, sizeof(camera));
  struct Light {glm::vec4 sphere,intensity;}; static_assert(sizeof(Light)==32);
  auto *cfg=static_cast<uint32_t*>(lightBuffers[rendered%2].mapped);
  cfg[0]=std::clamp(options.lightCount,0,512);cfg[1]=options.tileSize;cfg[2]=256;cfg[3]=options.tiled;
  auto *ls=reinterpret_cast<Light*>(cfg+4);uint32_t seed=42;
  auto random=[&]() {seed=1664525*seed+1013904223;return float(seed>>8)/16777216.f;};
  for(uint32_t i=0;i<cfg[0];++i) {float x=random()*24-12,y=random()*6,z=random()*24-12; if(scene.street) {x=random()*28-14;y=1.5f+random()*3;z=random()*28-14;}ls[i]={glm::vec4(x,y,z,options.largeLights?100.f:2.5f),glm::vec4(12,8,5,0)};}

  VkCommandBuffer c = f.cmd;
  VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
  check(vkBeginCommandBuffer(c, &bi));
  vkCmdResetQueryPool(c, f.queries, 0, 24);
  vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, f.queries, 0);
  auto transition = [&](VkImage img, VkImageAspectFlags aspect,
                        VkImageLayout old, VkImageLayout next,
                        VkPipelineStageFlags2 src, VkAccessFlags2 sa,
                        VkPipelineStageFlags2 dst, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2};
    b.srcStageMask = src;
    b.srcAccessMask = sa;
    b.dstStageMask = dst;
    b.dstAccessMask = da;
    b.oldLayout = old;
    b.newLayout = next;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {aspect, 0, 1, 0, 1};
    VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
    dep.imageMemoryBarrierCount = 1;
    dep.pImageMemoryBarriers = &b;
    vkCmdPipelineBarrier2(c, &dep);
  };
  auto &sm=shadowMaps[rendered%2];
  transition(sm.handle,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
  VkRenderingAttachmentInfo shadowDepth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  shadowDepth.imageView=sm.view; shadowDepth.imageLayout=VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  shadowDepth.loadOp=VK_ATTACHMENT_LOAD_OP_CLEAR; shadowDepth.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
  shadowDepth.clearValue.depthStencil={1,0};
  VkRenderingInfo shadowRI{VK_STRUCTURE_TYPE_RENDERING_INFO};
  shadowRI.layerCount=1; shadowRI.pDepthAttachment=&shadowDepth;
  VkDeviceSize shadowOffset=0;
  vkCmdBindVertexBuffers(c,0,1,&scene.vertices.handle,&shadowOffset);
  for(int j=0;j<4;++j) {
    VkDebugUtilsLabelEXT sl{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
    const char *names[]={"CSM 0","CSM 1","CSM 2","CSM 3"}; sl.pLabelName=names[j];
    if(beginLabel) beginLabel(c,&sl);
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,4+j*2);
    VkRect2D rect{{(j%2)*shadowResolution,(j/2)*shadowResolution},{uint32_t(shadowResolution),uint32_t(shadowResolution)}};
    shadowRI.renderArea=rect;
    vkCmdBeginRendering(c,&shadowRI);
    VkViewport sv{float(rect.offset.x),float(rect.offset.y),float(shadowResolution),float(shadowResolution),0,1};
    vkCmdSetViewport(c,0,1,&sv); vkCmdSetScissor(c,0,1,&rect);
    vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,shadowPipeline);
    if(shadows) for(auto &draw:scene.draws) {
      if(draw.transparent) continue;
      struct Push {glm::mat4 m; glm::vec4 b,p,a;} push{cascades.matrix[j]*draw.model,draw.color,draw.params,draw.alpha};
      auto set=materialSets[(rendered%2)*scene.materials.size()+draw.material];
      vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_GRAPHICS,shadowLayout,0,1,&set,0,nullptr);
      vkCmdPushConstants(c,shadowLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(push),&push);
      vkCmdDraw(c,draw.count,1,draw.first,0);
    }
    vkCmdEndRendering(c);
    transition(sm.handle,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,5+j*2);
    if(endLabel) endLabel(c);
  }
  transition(sm.handle,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  // Old contents are discarded, but previous depth writes must finish before
  // this frame writes.
  transition(d.swapImages[image], VK_IMAGE_ASPECT_COLOR_BIT,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, 0,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  transition(depths[image].image, VK_IMAGE_ASPECT_DEPTH_BIT,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                 VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
             VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                 VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
             VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT |
                 VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
  transition(hdr[image].handle, VK_IMAGE_ASPECT_COLOR_BIT,
             VK_IMAGE_LAYOUT_UNDEFINED,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  VkRenderingAttachmentInfo color{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  color.imageView = hdr[image].view;
  color.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  color.clearValue.color = {{.025f, .035f, .06f, 1}};
  VkRenderingAttachmentInfo depth{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
  depth.imageView = depths[image].view;
  depth.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
  depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  depth.clearValue.depthStencil = {0, 0};
  VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
  ri.renderArea = {{0, 0}, d.extent};
  ri.layerCount = 1;
  ri.colorAttachmentCount = 1;
  ri.pColorAttachments = &color;
  ri.pDepthAttachment = &depth;
  std::array<VkRenderingAttachmentInfo, 3> ga{};
  if (options.deferred) {
    for (int j = 0; j < 3; ++j) {
      transition(gbuffer[image][j].handle, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_UNDEFINED,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      ga[j] = color;
      ga[j].imageView = gbuffer[image][j].view;
      ga[j].clearValue.color = {{0, 0, 0, 0}};
    }
    ri.colorAttachmentCount = 3;
    ri.pColorAttachments = ga.data();
  }
  VkDebugUtilsLabelEXT label{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
  label.pLabelName = options.deferred ? "GBuffer" : "ForwardPBR";
  if (beginLabel)
    beginLabel(c, &label);
  vkCmdBeginRendering(c, &ri);
  VkViewport vp{0, 0, float(d.extent.width), float(d.extent.height), 0, 1};
  VkRect2D sc{{0, 0}, d.extent};
  vkCmdSetViewport(c, 0, 1, &vp);
  vkCmdSetScissor(c, 0, 1, &sc);
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS,
                    options.deferred ? gPipeline : pipeline);
  VkDeviceSize offset = 0;
  vkCmdBindVertexBuffers(c, 0, 1, &scene.vertices.handle, &offset);
  for (const auto &draw : scene.draws) {
    if (draw.transparent)
      continue;
    struct Push {
      glm::mat4 model;
      glm::vec4 base, params, alpha, emission;
    } push{draw.model, draw.color, draw.params, draw.alpha, draw.emission};
    push.params.w = float(draw.material);
    auto set =
        materialSets[(rendered % 2) * scene.materials.size() + draw.material];
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1,
                            &set, 0, nullptr);
    vkCmdPushConstants(
        c, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
        sizeof(push), &push);
    vkCmdDraw(c, draw.count, 1, draw.first, 0);
  }
  vkCmdEndRendering(c);
  if (endLabel)
    endLabel(c);
  if (options.deferred) {
    for (int j = 0; j < 3; ++j)
      transition(gbuffer[image][j].handle, VK_IMAGE_ASPECT_COLOR_BIT,
                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                 VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                 VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                 VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                 VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    transition(depths[image].image, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
               VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    tileWritten[rendered%2]=0;
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,20);
    if(options.tiled || mode==18) {
      label.pLabelName="Tiled light culling";if(beginLabel) beginLabel(c,&label);
      vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_COMPUTE,tiledPipeline);
      VkDescriptorSet sets[]={materialSets[(rendered%2)*scene.materials.size()],gSets[image]};
      vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_COMPUTE,tiledLayout,0,2,sets,0,nullptr);
      tileWritten[rendered%2]=((d.extent.width+options.tileSize-1)/options.tileSize)*((d.extent.height+options.tileSize-1)/options.tileSize);
      vkCmdDispatch(c,(d.extent.width+options.tileSize-1)/options.tileSize,(d.extent.height+options.tileSize-1)/options.tileSize,1);
      VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};mb.srcStageMask=VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;mb.srcAccessMask=VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;mb.dstStageMask=VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_2_HOST_BIT;mb.dstAccessMask=VK_ACCESS_2_SHADER_STORAGE_READ_BIT|VK_ACCESS_2_HOST_READ_BIT;
      VkDependencyInfo dep{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};dep.memoryBarrierCount=1;dep.pMemoryBarriers=&mb;vkCmdPipelineBarrier2(c,&dep);
      if(endLabel) endLabel(c);
    }
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,21);
    for(int pass=0;pass<3;++pass) {
      auto &im=aoImages[image][pass];
      vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,12+pass*2);
      VkExtent2D extent={pass<2 && options.aoHalf?(d.extent.width+1)/2:d.extent.width,
                        pass<2 && options.aoHalf?(d.extent.height+1)/2:d.extent.height};
      transition(im.handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
      VkRenderingAttachmentInfo at{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
      at.imageView=im.view; at.imageLayout=VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      at.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE; at.storeOp=VK_ATTACHMENT_STORE_OP_STORE;
      VkRenderingInfo ar{VK_STRUCTURE_TYPE_RENDERING_INFO}; ar.renderArea={{0,0},extent}; ar.layerCount=1;
      ar.colorAttachmentCount=1; ar.pColorAttachments=&at;
      label.pLabelName=pass==0?"SSAO":pass==1?"AO bilateral filter":"AO edge-aware upsample";
      if(beginLabel) beginLabel(c,&label);
      vkCmdBeginRendering(c,&ar);
      VkViewport av{0,0,float(extent.width),float(extent.height),0,1}; VkRect2D sc{{0,0},extent};
      vkCmdSetViewport(c,0,1,&av); vkCmdSetScissor(c,0,1,&sc);
      vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,aoPipeline[pass]);
      VkDescriptorSet aset[]={materialSets[(rendered%2)*scene.materials.size()],gSets[image]};
      vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_GRAPHICS,aoLayout[pass],0,2,aset,0,nullptr);
      glm::vec4 ap(options.aoRadius,options.aoSamples,options.ao,options.aoFilter);
      vkCmdPushConstants(c,aoLayout[pass],VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(ap),&ap);
      vkCmdDraw(c,3,1,0,0); vkCmdEndRendering(c);
      if(endLabel) endLabel(c);
      transition(im.handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
      vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,13+pass*2);
    }
    VkViewport restore{0,0,float(d.extent.width),float(d.extent.height),0,1}; VkRect2D rs{{0,0},d.extent};
    vkCmdSetViewport(c,0,1,&restore); vkCmdSetScissor(c,0,1,&rs);
    ri.colorAttachmentCount = 1;
    ri.pColorAttachments = &color;
    ri.pDepthAttachment = nullptr;
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,22);
    label.pLabelName = "DeferredLighting: G-buffer/depth R, HDR W";
    if (beginLabel)
      beginLabel(c, &label);
    vkCmdBeginRendering(c, &ri);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, lightPipeline);
    VkDescriptorSet sets[] = {
        materialSets[(rendered % 2) * scene.materials.size()], gSets[image]};
    vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS, lightLayout, 0,
                            2, sets, 0, nullptr);
    vkCmdDraw(c, 3, 1, 0, 0);
    vkCmdEndRendering(c);
    if (endLabel)
      endLabel(c);
  }
    if(options.deferred) vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,23);
  if(!options.deferred) for(uint32_t q=20;q<24;++q) vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,q);
  if(!options.deferred) for(uint32_t q=12;q<18;++q)
    vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,q);
  // Match the opaque depth exactly; velocity is unjittered currentUV-previousUV.
  if(options.deferred) transition(depths[image].image,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
  transition(motion[image].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  auto motionColor=color; motionColor.imageView=motion[image].view; motionColor.clearValue.color={{0,0,0,0}};
  auto motionDepth=depth; motionDepth.loadOp=VK_ATTACHMENT_LOAD_OP_LOAD;
  auto mr=ri; mr.colorAttachmentCount=1; mr.pColorAttachments=&motionColor; mr.pDepthAttachment=&motionDepth;
  label.pLabelName="Motion vectors"; if(beginLabel) beginLabel(c,&label);
  vkCmdBeginRendering(c,&mr); vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,motionPipeline);
  auto camSet=materialSets[(rendered%2)*scene.materials.size()];
  vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_GRAPHICS,motionLayout,0,1,&camSet,0,nullptr);
  for(size_t j=0;j<scene.draws.size();++j) { auto &draw=scene.draws[j]; if(draw.transparent) continue;
    struct MP {glm::mat4 current,previous;} mp{draw.model,historyValid?previousModels[j]:draw.model};
    vkCmdPushConstants(c,motionLayout,VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(mp),&mp); vkCmdDraw(c,draw.count,1,draw.first,0);
  }
  vkCmdEndRendering(c);
  if(endLabel) endLabel(c);
  transition(motion[image].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  std::vector<const Primitive *> transparentDraws;
  for (const auto &draw : scene.draws)
    if (draw.transparent)
      transparentDraws.push_back(&draw);
  std::stable_sort(transparentDraws.begin(), transparentDraws.end(),
                   [&](auto a, auto b) {
                     return (view * a->model * glm::vec4(0, 0, 0, 1)).z <
                            (view * b->model * glm::vec4(0, 0, 0, 1)).z;
                   });
  if (!transparentDraws.empty()) {
    transition(hdr[image].handle, VK_IMAGE_ASPECT_COLOR_BIT,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
               VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                   VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    transition(depths[image].image, VK_IMAGE_ASPECT_DEPTH_BIT,
               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
               VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT,
               VK_ACCESS_2_SHADER_SAMPLED_READ_BIT |
                   VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
               VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                   VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
               VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT);
    color.loadOp = depth.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    ri.pDepthAttachment = &depth;
    label.pLabelName = "TransparentForward: HDR blend, opaque depth read";
    if (beginLabel)
      beginLabel(c, &label);
    vkCmdBeginRendering(c, &ri);
    vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, transparentPipeline);
    for (auto draw : transparentDraws) {
      struct Push {
        glm::mat4 model;
        glm::vec4 base, params, alpha, emission;
      } push{draw->model, draw->color, draw->params, draw->alpha, draw->emission};
      auto set = materialSets[(rendered % 2) * scene.materials.size() +
                              draw->material];
      vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS,
                              transparentLayout, 0, 1, &set, 0, nullptr);
      vkCmdPushConstants(c, transparentLayout,
                         VK_SHADER_STAGE_VERTEX_BIT |
                             VK_SHADER_STAGE_FRAGMENT_BIT,
                         0, sizeof(push), &push);
      vkCmdDraw(c, draw->count, 1, draw->first, 0);
    }
    vkCmdEndRendering(c);
    if (endLabel)
      endLabel(c);
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
  }
  vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 1);
  transition(hdr[image].handle, VK_IMAGE_ASPECT_COLOR_BIT,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
             VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
             VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transition(depths[image].image,VK_IMAGE_ASPECT_DEPTH_BIT,VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT|VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT|VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  int parity=rendered%2;
  // Both histories need a defined layout even on the first (rejected) history read.
  for(int j=0;j<2;++j) if(!historyLayout[j]) {
    transition(history[j].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_NONE,0,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT); historyLayout[j]=true;
  }
  transition(history[parity].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  transition(temporalDebug[parity].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_UNDEFINED,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  auto hc=color; hc.imageView=history[parity].view; hc.loadOp=VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  VkRenderingAttachmentInfo ta[]={hc,hc}; ta[1].imageView=temporalDebug[parity].view;
  auto tr=ri; tr.colorAttachmentCount=2; tr.pColorAttachments=ta; tr.pDepthAttachment=nullptr;
  label.pLabelName="TAA resolve: HDR + motion + previous history"; if(beginLabel) beginLabel(c,&label);
  vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,18);
  vkCmdBeginRendering(c,&tr); vkCmdBindPipeline(c,VK_PIPELINE_BIND_POINT_GRAPHICS,aoPipeline[3]);
  VkDescriptorSet ts[]={camSet,temporalSets[image*2+parity]};
  vkCmdBindDescriptorSets(c,VK_PIPELINE_BIND_POINT_GRAPHICS,aoLayout[3],0,2,ts,0,nullptr);
  glm::vec4 tp(options.taa && historyValid ? (unjitteredVP==previousVP?2.f:1.f) : 0.f,options.reprojection,options.depthReject,options.clipHistory);
  vkCmdPushConstants(c,aoLayout[3],VK_SHADER_STAGE_VERTEX_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(tp),&tp); vkCmdDraw(c,3,1,0,0); vkCmdEndRendering(c); if(endLabel) endLabel(c);
  vkCmdWriteTimestamp2(c,VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,f.queries,19);
  transition(history[parity].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  transition(temporalDebug[parity].handle,VK_IMAGE_ASPECT_COLOR_BIT,VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
  color.imageView = d.swapViews[image];
  ri.pDepthAttachment = nullptr;
  vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 2);
  label.pLabelName = "ToneMap: linear HDR R, output W";
  if (beginLabel)
    beginLabel(c, &label);
  vkCmdBeginRendering(c, &ri);
  vkCmdBindPipeline(c, VK_PIPELINE_BIND_POINT_GRAPHICS, outputPipeline);
  vkCmdBindDescriptorSets(c, VK_PIPELINE_BIND_POINT_GRAPHICS,
                          outputPipelineLayout, 0, 1, &historyOutput[parity], 0,
                          nullptr);
  struct Output {
    float exposure;
    int encode;
    int debug; int pad;
  } out{1, d.swapFormat == VK_FORMAT_B8G8R8A8_SRGB ||
                   d.swapFormat == VK_FORMAT_R8G8B8A8_SRGB
               ? 0
               : 1, options.taaDebug,0};
  vkCmdPushConstants(c, outputPipelineLayout, VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                     16, &out);
  vkCmdDraw(c, 3, 1, 0, 0);
  vkCmdEndRendering(c);
  if (endLabel)
    endLabel(c);
  transition(d.swapImages[image], VK_IMAGE_ASPECT_COLOR_BIT,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
             VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
             VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT |
                 VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
  label.pLabelName = "ImGui: color RW, font sampled R";
  if (beginLabel)
    beginLabel(c, &label);
  color.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
  ri.pDepthAttachment = nullptr;
  vkCmdBeginRendering(c, &ri);
  ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), c);
  vkCmdEndRendering(c);
  if (endLabel)
    endLabel(c);
  transition(
      d.swapImages[image], VK_IMAGE_ASPECT_COLOR_BIT,
      VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
      VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_2_NONE, 0);
  vkCmdWriteTimestamp2(c, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, f.queries, 3);
  check(vkEndCommandBuffer(c));
  VkPipelineStageFlags wait = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
  VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};
  submit.waitSemaphoreCount = 1;
  submit.pWaitSemaphores = &f.acquired;
  submit.pWaitDstStageMask = &wait;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers = &c;
  submit.signalSemaphoreCount = 1;
  submit.pSignalSemaphores = &ready[image];
  check(vkResetFences(d.device, 1, &f.fence));
  auto t = Clock::now();
  check(vkQueueSubmit(d.queue, 1, &submit, f.fence));
  cpuMs = std::chrono::duration<double, std::milli>(Clock::now() - t).count();
  f.submitted = true;
  VkPresentInfoKHR present{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
  present.waitSemaphoreCount = 1;
  present.pWaitSemaphores = &ready[image];
  present.swapchainCount = 1;
  present.pSwapchains = &d.swapchain;
  present.pImageIndices = &image;
  auto r = vkQueuePresentKHR(d.queue, &present);
  if (r == VK_SUBOPTIMAL_KHR || r == VK_ERROR_OUT_OF_DATE_KHR)
    recreate = true;
  else
    check(r);
  previousView=view; previousVP=unjitteredVP; previousJitter=normalizedJitter; 
  previousModels.clear(); for(auto &draw:scene.draws) previousModels.push_back(draw.model);
  historyValid=true;
  ++rendered;
  if (rendered % 120 == 0)
    std::printf(
        "frame=%llu tick=%llu GPU=%.4fms CPU-submit=%.4fms targets=%zu\n",
        (unsigned long long)rendered, (unsigned long long)tick, gpuMs, cpuMs,
        depths.size());
  if (rendered % 120 == 0)
    std::printf("Pass Geometry+Lighting+Motion+Transparent=%.4fms ToneMap+UI=%.4fms\n",
                cubeMs, uiMs);
  if(rendered%120==0) printf("Tiled ms %.6f %.6f total %.6f\n",cullMs,lightMs,cullMs+lightMs);
  if(rendered%120==0) printf("TAA resolve ms %.5f\n",taaMs);
  if(rendered%120==0) printf("AO ms %.5f %.5f %.5f\n",aoMs[0],aoMs[1],aoMs[2]);
  if(rendered%120==0) printf("CSM ms %.5f %.5f %.5f %.5f\n",shadowMs[0],shadowMs[1],shadowMs[2],shadowMs[3]);
}
App::~App() {
  if (d.device)
    vkDeviceWaitIdle(d.device);
  if (gui)
    ImGui_ImplVulkan_Shutdown();
  if (glfwGui)
    ImGui_ImplGlfw_Shutdown();
  if (context)
    ImGui::DestroyContext();
  if (d.device) {
    vkDestroyPipeline(d.device,tiledPipeline,nullptr);vkDestroyPipelineLayout(d.device,tiledLayout,nullptr);for(auto &b:lightBuffers) freeBuffer(d,b);
    vkDestroyPipeline(d.device,motionPipeline,nullptr); vkDestroyPipelineLayout(d.device,motionLayout,nullptr);
    for(int j=0;j<4;++j) { vkDestroyPipeline(d.device,aoPipeline[j],nullptr); vkDestroyPipelineLayout(d.device,aoLayout[j],nullptr); }
    vkDestroyPipeline(d.device, shadowPipeline, nullptr);
    vkDestroyPipelineLayout(d.device, shadowLayout, nullptr);
    for(auto &im:shadowMaps) freeImage(d,im);
    vkDestroyPipeline(d.device, transparentPipeline, nullptr);
    vkDestroyPipelineLayout(d.device, transparentLayout, nullptr);
    vkDestroyPipeline(d.device, gPipeline, nullptr);
    vkDestroyPipeline(d.device, lightPipeline, nullptr);
    vkDestroyPipelineLayout(d.device, gPipeLayout, nullptr);
    vkDestroyPipelineLayout(d.device, lightLayout, nullptr);
    vkDestroySampler(d.device, nearest, nullptr);
    vkDestroyDescriptorSetLayout(d.device, gLayout, nullptr);
    vkDestroyPipeline(d.device, pipeline, nullptr);
    vkDestroyPipelineLayout(d.device, layout, nullptr);
    dropTargets();
    vkDestroyPipeline(d.device, outputPipeline, nullptr);
    vkDestroyPipelineLayout(d.device, outputPipelineLayout, nullptr);
    vkDestroyDescriptorPool(d.device, descriptorPool, nullptr);
    vkDestroyDescriptorSetLayout(d.device, materialLayout, nullptr);
    vkDestroyDescriptorSetLayout(d.device, outputLayout, nullptr);
    for (auto &b : cameraBuffers)
      freeBuffer(d, b);
    scene.destroy();
    for (auto &f : frames) {
      vkDestroyQueryPool(d.device, f.queries, nullptr);
      vkDestroySemaphore(d.device, f.acquired, nullptr);
      vkDestroyFence(d.device, f.fence, nullptr);
      vkDestroyCommandPool(d.device, f.pool, nullptr);
    }
  }
  vk::DestroyDevice(d);
  if (window)
    glfwDestroyWindow(window);
  glfwTerminate();
}
int Run() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  try {
    App app;
    app.init();
    auto last = Clock::now();
    double accumulator = 0;
    while (!glfwWindowShouldClose(app.window)) {
      glfwPollEvents();
      auto now = Clock::now();
      accumulator += std::chrono::duration<double>(now - last).count();
      last = now;
      if (app.options.replay)
        accumulator=0;
      else
        while (accumulator >= 1. / 60) {
          ++app.tick;
          accumulator -= 1. / 60;
        }
      int w, h;
      glfwGetFramebufferSize(app.window, &w, &h);
      if (w <= 0 || h <= 0 || glfwGetWindowAttrib(app.window, GLFW_ICONIFIED)) {
        glfwWaitEventsTimeout(.05);
        continue;
      }
      if(app.pendingPreset>=0) {
        bool baseline=app.pendingPreset==2;
        app.options.ao=!baseline; app.options.taa=!baseline; app.options.tiled=!baseline;
        app.requestedAOHalf=app.pendingPreset!=1;
        app.options.aoSamples=app.pendingPreset==1?32:16;
        app.pendingPreset=-1;
        app.recreate=true;
      }
      if (app.recreate || uint32_t(w) != app.d.extent.width ||
          uint32_t(h) != app.d.extent.height) {
        check(vkDeviceWaitIdle(app.d.device));
        app.dropTargets();
        if (!vk::RecreateSwapchain(app.d, app.window))
          throw std::runtime_error("recreate");
        app.options.aoHalf=app.requestedAOHalf;
        app.targets();
        app.recreate = false;
      }
      auto previousTick=app.tick;
      auto previousRendered=app.rendered;
      if(app.options.replay) ++app.tick;
      app.draw(app.options.replay);
      if(app.options.replay && app.rendered==previousRendered) app.tick=previousTick;
    }
    std::printf("DONE frames=%llu ticks=%llu\n",
                (unsigned long long)app.rendered, (unsigned long long)app.tick);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAILED: %s\n", e.what());
    return 1;
  }
}

void App::descriptors() {
  for(auto &b:lightBuffers) b=makeBuffer(d,16+512*32,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  if(!materialLayout) {
  std::array<VkDescriptorSetLayoutBinding, 11> bindings{};
  for (uint32_t i = 0; i < 8; ++i)
    bindings[i] = {i,
                   i ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                     : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                   1,
                   VkShaderStageFlags(i ? VK_SHADER_STAGE_FRAGMENT_BIT
                                        : VK_SHADER_STAGE_VERTEX_BIT |
                                              VK_SHADER_STAGE_FRAGMENT_BIT),
                   nullptr};
  VkDescriptorSetLayoutCreateInfo l{
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
  bindings[0].stageFlags |= VK_SHADER_STAGE_COMPUTE_BIT;
  for(uint32_t j=8;j<10;++j) bindings[j]={j,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
  bindings[10]={10,VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,1,VK_SHADER_STAGE_FRAGMENT_BIT,nullptr};
  l.bindingCount = 11;
  l.pBindings = bindings.data();
  check(vkCreateDescriptorSetLayout(d.device, &l, nullptr, &materialLayout));
  auto ob = bindings[1];
  ob.binding = 0;
  VkDescriptorSetLayoutBinding outs[]={ob,ob}; outs[1].binding=1;
  l.bindingCount = 2;
  l.pBindings = outs;
  check(vkCreateDescriptorSetLayout(d.device, &l, nullptr, &outputLayout));
  std::array<VkDescriptorSetLayoutBinding, 10> gb{};
  for (uint32_t j = 0; j < 10; ++j)
    gb[j] = {j, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
             VK_SHADER_STAGE_FRAGMENT_BIT|VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  l.bindingCount = 10;
  l.pBindings = gb.data();
  check(vkCreateDescriptorSetLayout(d.device, &l, nullptr, &gLayout));
  }
  uint32_t count = uint32_t(scene.materials.size() * 2);
  VkDescriptorPoolSize sizes[] = {
      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count*2},
      {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, count},
      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, count * 8 + 512}};
  VkDescriptorPoolCreateInfo p{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
  p.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
  p.maxSets = count + 64;
  p.poolSizeCount = 3;
  p.pPoolSizes = sizes;
  check(vkCreateDescriptorPool(d.device, &p, nullptr, &descriptorPool));
  materialSets.resize(count);
  std::vector<VkDescriptorSetLayout> layouts(count, materialLayout);
  VkDescriptorSetAllocateInfo a{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
  a.descriptorPool = descriptorPool;
  a.descriptorSetCount = count;
  a.pSetLayouts = layouts.data();
  check(vkAllocateDescriptorSets(d.device, &a, materialSets.data()));
  for (uint32_t i = 0; i < count; ++i) {
    auto &m = scene.materials[i % scene.materials.size()];
    VkDescriptorBufferInfo b{cameraBuffers[i / scene.materials.size()].handle,
                             0, 656};
    VkDescriptorImageInfo images[] = {
        {scene.materialSampler, m.base.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {scene.materialSampler, m.normal.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {scene.materialSampler, m.mr.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {scene.sampler, scene.irradiance.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {scene.sampler, scene.prefilter.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {scene.sampler, scene.lut.view,
         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        {nearest,shadowMaps[i / scene.materials.size()].view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL}};
    VkDescriptorImageInfo ei{scene.materialSampler,m.emission.view,VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet ew{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};ew.dstSet=materialSets[i];ew.dstBinding=10;ew.descriptorCount=1;ew.descriptorType=VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;ew.pImageInfo=&ei;vkUpdateDescriptorSets(d.device,1,&ew,0,nullptr);
    VkDescriptorBufferInfo li{lightBuffers[i/scene.materials.size()].handle,0,VK_WHOLE_SIZE};
    VkWriteDescriptorSet lw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};lw.dstSet=materialSets[i];lw.dstBinding=8;lw.descriptorCount=1;lw.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;lw.pBufferInfo=&li;vkUpdateDescriptorSets(d.device,1,&lw,0,nullptr);
    for (uint32_t j = 0; j < 8; ++j) {
      VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
      w.dstSet = materialSets[i];
      w.dstBinding = j;
      w.descriptorCount = 1;
      w.descriptorType = j ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                           : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
      w.pBufferInfo = j ? nullptr : &b;
      w.pImageInfo = j ? &images[j - 1] : nullptr;
      vkUpdateDescriptorSets(d.device, 1, &w, 0, nullptr);
    }
  }
}



