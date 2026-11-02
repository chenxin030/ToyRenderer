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
  mat4 previousVP; vec4 jitter;
}
camera;
