#version 450
layout(push_constant) uniform Object { mat4 mvp; vec4 base; vec4 params; vec4 alpha; } pc;
layout(set=0,binding=1) uniform sampler2D baseTex;
layout(location=0) in vec2 texcoord;
void main() { if(pc.alpha.y>.5 && pc.base.a*texture(baseTex,texcoord).a<pc.alpha.x) discard; }
