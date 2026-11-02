#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <array>
#include <cmath>
// Fixed-radius frustum spheres keep XY scale invariant under camera rotation.
struct Cascades {
  std::array<glm::mat4, 4> matrix;
  glm::vec4 splits{}, texels{};
};
inline Cascades buildCascades(const glm::mat4 &view, float aspect,
                             glm::vec3 light, int resolution, float lambda,
                             bool snap) {
  Cascades out;
  auto inv = glm::inverse(view);
  auto lv = glm::lookAtRH(light * 150.f, glm::vec3(0),
      std::abs(light.y) > .99f ? glm::vec3(1,0,0) : glm::vec3(0,1,0));
  float previous = .1f;
  for (int c = 0; c < 4; ++c) {
    float t = (c + 1) / 4.f;
    float end = glm::mix(.1f + 59.9f*t, .1f*std::pow(600.f,t), lambda);
    out.splits[c] = end;
    // Include previous cascade's final 10 percent for transition sampling.
    float start = c ? previous - (previous - (c>1 ? out.splits[c-2] : .1f))*.1f : .1f;
    float mid = (start + end)*.5f;
    float tanY = std::tan(glm::radians(55.f)*.5f);
    float r = std::ceil(std::sqrt(end*end*tanY*tanY*(1+aspect*aspect) + (end-mid)*(end-mid))*16.f)/16.f;
    r *= float(resolution)/float(resolution-4); // guard PCF and rounding
    float texel = 2*r/resolution;
    out.texels[c] = texel;
    auto center = lv * inv * glm::vec4(0,0,-mid,1);
    if (snap) { center.x = std::round(center.x/texel)*texel; center.y = std::round(center.y/texel)*texel; }
    // Fixed light-space Z: includes off-frustum casters within documented scene bounds.
    out.matrix[c] = glm::orthoRH_ZO(center.x-r,center.x+r,center.y-r,center.y+r,.1f,300.f)*lv;
    previous = end;
  }
  return out;
}
