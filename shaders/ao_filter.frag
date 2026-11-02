#version 450
#extension GL_GOOGLE_include_directive : require
#include "ao_common.glsl"
layout(set=1,binding=4) uniform sampler2D sourceAO;
void main(){
 if(ao.settings.z==0) {color=vec4(1);return;}
ivec2 sz=textureSize(sourceAO,0); vec2 uv=gl_FragCoord.xy/vec2(sz);
 if(texture(depthTex,uv).r==0) {color=vec4(1);return;}
 if(ao.settings.w==0) {color=vec4(vec3(texture(sourceAO,uv).r),1);return;}
 float sum=0,wSum=0;
 for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
  vec2 v=(clamp(ivec2(gl_FragCoord.xy)+ivec2(x,y),ivec2(0),sz-1)+.5)/vec2(sz);
  float w=guide(uv,v)*exp(-float(x*x+y*y)/4); sum+=texture(sourceAO,v).r*w; wSum+=w;
 }
 color=vec4(vec3(wSum>1e-6?sum/wSum:1),1);
}