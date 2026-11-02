#version 450
layout(set = 0, binding = 0) uniform sampler2D hdr;
layout(set=0,binding=1) uniform sampler2D diagnostic;
layout(push_constant) uniform Output {
  float exposure;
  int encode;
  int debug; int pad;
}
pc;
layout(location = 0) in vec2 uv;
layout(location = 0) out vec4 color;
vec3 srgb(vec3 x) {
  return mix(12.92 * x, 1.055 * pow(x, vec3(1 / 2.4)) - .055,
             greaterThan(x, vec3(.0031308)));
}
void main() {
  vec3 x = texture(hdr, uv).rgb * pc.exposure;
  if(pc.debug>0) {vec4 d=texture(diagnostic,uv); x=vec3(pc.debug==1?d.x:pc.debug==2?d.y:pc.debug==3?d.z:d.w);}
  else x = x / (1 + x);
  color = vec4(pc.encode != 0 ? srgb(x) : x, 1);
}
