#version 450
invariant gl_Position;
#extension GL_GOOGLE_include_directive : require
#include "temporal_camera.glsl"
layout(push_constant) uniform Motion {mat4 current; mat4 previous;} pc;
layout(location=0) in vec3 position;
layout(location=0) out vec4 nowClip;
layout(location=1) out vec4 oldClip;
void main() {
 vec3 world=(pc.current*vec4(position,1)).xyz;
 gl_Position=camera.vp*vec4(world,1);
 nowClip=gl_Position; nowClip.xy-=2*camera.jitter.xy*nowClip.w;
 vec3 oldWorld=(pc.previous*vec4(position,1)).xyz;
 oldClip=camera.previousVP*vec4(oldWorld,1);
}
