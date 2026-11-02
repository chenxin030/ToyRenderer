#version 450
#extension GL_GOOGLE_include_directive : require
#include "ao_common.glsl"
void main() {
 vec2 size=vec2(textureSize(depthTex,0));
 // Match ceil(full/2) attachment size, including odd extents.
 vec2 outSize=ceil(size / (camera.points.w>0?2:1));
 vec2 uv=gl_FragCoord.xy/outSize;
 if(ao.settings.z==0 || texture(depthTex,uv).r==0) {color=vec4(1);return;}
 vec3 p=position(uv),n=normalAt(uv);
 vec3 t=normalize(cross(abs(n.z)<.9?vec3(0,0,1):vec3(0,1,0),n)),b=cross(n,t);
 float angle=fract(sin(dot(floor(uv*size),vec2(12.9898,78.233)))*43758.5453)*6.2831853;
 float sum=0; int count=clamp(int(ao.settings.y),4,128);
 for(int i=0;i<count;++i) {
  float f=(float(i)+.5)/float(count),z=f;
  float a=float(i)*2.39996323+angle;
  vec3 dir=t*cos(a)*sqrt(1-z*z)+b*sin(a)*sqrt(1-z*z)+n*z;
  vec3 q=p+dir*ao.settings.x*mix(.1,1,f*f);
  vec4 clip=camera.vp*inverse(camera.view)*vec4(q,1);
  if(clip.w<=0) continue;
  vec2 v=clip.xy/clip.w*.5+.5;
  if(any(lessThan(v,vec2(0)))||any(greaterThanEqual(v,vec2(1)))||texture(depthTex,v).r==0) continue;
  vec3 hit=position(v);
  float range=1-smoothstep(ao.settings.x*.5,ao.settings.x,length(hit-p));
  // A nearest depth texel can be closer in view Z even on a coplanar receiver.
  // Only geometry above the receiver's tangent plane can occlude its hemisphere.
  float bias=max(.005,ao.settings.x*.02);
  sum+=float(hit.z>=q.z+bias && dot(hit-p,n)>bias)*range;
 }
 color=vec4(vec3(clamp(1-sum/float(count),0,1)),1);
}
