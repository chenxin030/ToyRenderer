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

layout(push_constant) uniform AO { vec4 settings; } ao;
layout(set=1,binding=1) uniform sampler2D normals;
layout(set=1,binding=3) uniform sampler2D depthTex;
layout(location=0) out vec4 color;
vec3 position(vec2 uv) {
 uv=(floor(uv*vec2(textureSize(depthTex,0)))+.5)/vec2(textureSize(depthTex,0));
 vec4 h=camera.inverseVP*vec4(uv*2-1,texture(depthTex,uv).r,1);
 return (camera.view*vec4(h.xyz/h.w,1)).xyz;
}
vec3 normalAt(vec2 uv) { return normalize(mat3(camera.view)*texture(normals,uv).xyz); }
float guide(vec2 a,vec2 b) {
 if(texture(depthTex,b).r==0) return 0;
 return exp(-abs(position(a).z-position(b).z)/max(.02,ao.settings.x*.1))*pow(max(dot(normalAt(a),normalAt(b)),0),32);
}
