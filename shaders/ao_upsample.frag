#version 450
#extension GL_GOOGLE_include_directive : require
#include "ao_common.glsl"
layout(set=1,binding=5) uniform sampler2D sourceAO;
void main(){
 if(ao.settings.z==0) {color=vec4(1);return;}
vec2 uv=gl_FragCoord.xy/vec2(textureSize(depthTex,0));
 if(texture(depthTex,uv).r==0) {color=vec4(1);return;}
 ivec2 sz=textureSize(sourceAO,0); vec2 grid=uv*vec2(sz)-.5; ivec2 base=ivec2(floor(grid)); vec2 f=fract(grid);
 float sum=0,wSum=0;
 for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
  vec2 v=(clamp(base+ivec2(x,y),ivec2(0),sz-1)+.5)/vec2(sz);
  float w=guide(uv,v)*(x==0?1-f.x:f.x)*(y==0?1-f.y:f.y);
  sum+=texture(sourceAO,v).r*w; wSum+=w;
 }
 color=vec4(vec3(wSum>1e-6?sum/wSum:1),1);
}