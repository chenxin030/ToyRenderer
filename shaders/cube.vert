#version 450
invariant gl_Position;
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
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;
layout(location = 2) in vec4 tangent;
layout(location = 3) in vec2 uv;
layout(location = 0) out vec3 world;
layout(location = 1) out vec3 N;
layout(location = 2) out vec4 T;
layout(location = 3) out vec2 texcoord;
void main() {
  world = (pc.model * vec4(position, 1)).xyz;
  N = normalize(transpose(inverse(mat3(pc.model))) * normal);
  T = vec4(mat3(pc.model) * tangent.xyz,
           tangent.w * sign(determinant(mat3(pc.model))));
  texcoord = uv;
  gl_Position = camera.vp * vec4(world, 1);
}
