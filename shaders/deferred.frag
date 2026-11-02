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
}
pc;
layout(set = 0, binding = 1) uniform sampler2D baseTex;
layout(set = 0, binding = 2) uniform sampler2D normalTex;
layout(set = 0, binding = 3) uniform sampler2D mrTex;
layout(set = 0, binding = 4) uniform sampler2D irradianceTex;
layout(set = 0, binding = 5) uniform sampler2D prefilterTex;
layout(set = 0, binding = 6) uniform sampler2D lutTex;
layout(set = 1, binding = 0) uniform sampler2D gBase;
layout(set = 1, binding = 1) uniform sampler2D gNormal;
layout(set = 1, binding = 2) uniform sampler2D gParams;
layout(set = 1, binding = 3) uniform sampler2D gDepth;
layout(location = 0) out vec4 color;
#define DEFERRED_AO
layout(set=1,binding=4) uniform sampler2D aoRaw;
layout(set=1,binding=5) uniform sampler2D aoFiltered;
layout(set=1,binding=6) uniform sampler2D aoFinal;
#include "lighting.glsl"
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy);
  float z = texelFetch(gDepth, p, 0).r;
  if (z == 0) {
    color = camera.light.w==0 ? vec4(.025,.035,.06,1) : (camera.light.w==1 ? vec4(.45,.65,.9,1) : vec4(.035,.05,.12,1));
    return;
  }
  vec2 uv = gl_FragCoord.xy / vec2(textureSize(gDepth, 0));
  vec4 h = camera.inverseVP * vec4(uv * 2 - 1, z, 1);
  vec3 world = h.xyz / h.w;
  vec3 n = texelFetch(gNormal, p, 0).xyz, base = texelFetch(gBase, p, 0).rgb;
  vec4 mr = texelFetch(gParams, p, 0);
  color = vec4(shade(world, n, base, mr.y, mr.x), 1);
  int mode = int(camera.settings.w);
  vec3 emission=vec3(texelFetch(gBase,p,0).a,texelFetch(gNormal,p,0).a,mr.a);
  if(mode==0) color.rgb+=emission;
  if(mode==19) color.rgb=emission;
  if(mode==18) {uint o=tileOffset(); float t=float(tileData[o])/max(float(lightConfig.x),1); color=vec4(t,1-t,.1,1); if(tileData[o]==0) color=vec4(0,0,0,1); if(tileData[o+1]!=0) color=vec4(1,0,1,1);}

  if(mode==15) color=vec4(vec3(texture(aoRaw,uv).r),1);
  if(mode==16) color=vec4(vec3(texture(aoFiltered,uv).r),1);
  if(mode==17) color=vec4(vec3(texture(aoFinal,uv).r),1);
  if (mode == 7)
    color = vec4(vec3(z), 1);
  if (mode == 8)
    color = vec4(vec3(length(n)), 1);
  if (mode == 9)
    color = vec4(fract(mr.z * vec3(.31, .57, .79)), 1);
  if (mode == 10)
    color = vec4(abs(world) * .1, 1);
}
