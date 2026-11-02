#version 450
layout(push_constant) uniform Object { mat4 mvp; vec4 base; vec4 params; vec4 alpha; } pc;
layout(location=0) in vec3 position;
layout(location=3) in vec2 uv;
layout(location=0) out vec2 texcoord;
void main() { gl_Position=pc.mvp*vec4(position,1); texcoord=uv; }
