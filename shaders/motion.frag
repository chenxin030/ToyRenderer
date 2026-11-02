#version 450
layout(location=0) in vec4 nowClip;
layout(location=1) in vec4 oldClip;
layout(location=0) out vec4 velocity;
void main() { if(oldClip.w<=0) velocity=vec4(0,0,0,0); else velocity=vec4((nowClip.xy/nowClip.w-oldClip.xy/oldClip.w)*.5,oldClip.z/oldClip.w,1); }
