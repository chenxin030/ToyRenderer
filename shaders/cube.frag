#version 450
#extension GL_GOOGLE_include_directive : require
#include "brdf.glsl"
layout(set = 0, binding = 0) uniform Camera {
  mat4 vp;
  vec4 eye;
  vec4 light;
  vec4 settings;
  mat4 inverseVP;
  vec4 points;
  mat4 shadowVP[4];
  mat4 view;
  vec4 splits;
  vec4 texels;
  vec4 shadow; // enabled, PCF radius, constant bias, normal bias in texels
  vec4 sun;
}
camera;
layout(push_constant) uniform Object {
  mat4 model;
  vec4 base;
  vec4 params;
  vec4 alpha;
  vec4 emission;
}
pc;
layout(set = 0, binding = 1) uniform sampler2D baseTex;
layout(set = 0, binding = 2) uniform sampler2D normalTex;
layout(set = 0, binding = 3) uniform sampler2D mrTex;
layout(set = 0, binding = 4) uniform sampler2D irradianceTex;
layout(set = 0, binding = 5) uniform sampler2D prefilterTex;
layout(set = 0, binding = 6) uniform sampler2D lutTex;
layout(location = 0) in vec3 world;
layout(location = 1) in vec3 N;
layout(location = 2) in vec4 T;
layout(location = 3) in vec2 texcoord;
layout(location = 0) out vec4 color;
#include "lighting.glsl"
layout(set=0,binding=10) uniform sampler2D emissiveTex;
#include "material.glsl"
void main() {
  vec3 n, base;
  float metal, rough;
  material(n, base, metal, rough);
  vec3 e=pc.emission.rgb*texture(emissiveTex,texcoord).rgb;
  vec3 radiance=shade(world,n,base,metal,rough);
  if(int(camera.settings.w)==0) radiance+=e;
  if(int(camera.settings.w)==19) radiance=e;
  color = vec4(radiance,
               pc.base.a * texture(baseTex, texcoord).a);
}
