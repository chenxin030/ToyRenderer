#version 450
#extension GL_GOOGLE_include_directive : require
#include "temporal_camera.glsl"
layout(set=1,binding=3) uniform sampler2D depthTex;
layout(set=1,binding=7) uniform sampler2D currentTex;
layout(set=1,binding=8) uniform sampler2D motionTex;
layout(set=1,binding=9) uniform sampler2D historyTex;
layout(push_constant) uniform Temporal {vec4 flags;} pc;
layout(location=0) in vec2 uv;
layout(location=0) out vec4 result;
layout(location=1) out vec4 diagnostic;
void main() {
 ivec2 size=textureSize(currentTex,0), p=clamp(ivec2(gl_FragCoord.xy),ivec2(0),size-1);
 vec3 current=texelFetch(currentTex,p,0).rgb;
 float depth=texelFetch(depthTex,p,0).r;
 bool cameraStill=pc.flags.x>1.5;
 bool pixelStill=cameraStill && length(texelFetch(motionTex,p,0).xy*vec2(size))<.01;
 vec4 motion=texelFetch(motionTex,p,0);
 // History RGB is assigned to the unjittered output grid. Jitter changes
 // the subpixel sample of that grid, not the location of accumulated RGB.
 if(depth==0) {
   vec4 farPoint=camera.inverseVP*vec4((uv+camera.jitter.xy)*2-1,0,1);
   vec3 ray=farPoint.xyz/farPoint.w-camera.eye.xyz;
   vec4 oldClip=camera.previousVP*vec4(ray,0);
   motion=vec4(uv-(oldClip.xy/oldClip.w*.5+.5),0,oldClip.w>0?1:0);
 }
 vec2 oldUV=uv-(pc.flags.y>0?motion.xy:vec2(0));
 bool valid=pc.flags.x>0 && motion.w>0 && all(greaterThanEqual(oldUV,vec2(0))) && all(lessThan(oldUV,vec2(1)));
 vec3 h=current; float weight=0;
 if(valid) {
   if(pc.flags.z==0) {h=texture(historyTex,oldUV).rgb;weight=.9;}
   else if(pixelStill) {
     // For a stationary camera, subpixel coverage changes on static surfaces
     // are AA samples, not disocclusions. Exclude history written by moving
     // surfaces (negative alpha) so a departing actor cannot leave its color.
     vec2 coord=oldUV*vec2(size)-.5;ivec2 base=ivec2(floor(coord));vec2 f=fract(coord);
     vec3 sum=vec3(0);float coverage=0;
     for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
       vec4 tap=texelFetch(historyTex,clamp(base+ivec2(x,y),ivec2(0),size-1),0);
       float w=(x==0?1-f.x:f.x)*(y==0?1-f.y:f.y);
       if(tap.a>=0) {sum+=tap.rgb*w;coverage+=w;}
     }
     valid=coverage>.00001;if(valid) {h=sum/coverage;weight=.9;}
   } else {
     // History alpha is depth on the previous raster's jittered grid.
     vec2 depthUV=oldUV-camera.jitter.xy+camera.jitter.zw;
     ivec2 base=ivec2(floor(depthUV*vec2(size)-.5));float error=1e30;
     for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
       float oldDepth=abs(texelFetch(historyTex,clamp(base+ivec2(x,y),ivec2(0),size-1),0).a);
       error=min(error,abs(oldDepth-motion.z));
     }
     valid=error<max(.00002,abs(motion.z)*.01);
     if(valid) {h=texture(historyTex,oldUV).rgb;weight=.9;}
   }
 }
 vec3 beforeClip=h;
 if(pc.flags.w>0) {
   vec3 lo=current,hi=current;
   for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {vec3 n=texelFetch(currentTex,clamp(p+ivec2(x,y),ivec2(0),size-1),0).rgb;lo=min(lo,n);hi=max(hi,n);}
   h=clamp(h,lo,hi);
 }
 diagnostic=vec4(length(motion.xy*vec2(size)),weight,valid?0:1,length(beforeClip-h));
 // Alpha magnitude is raster depth; negative marks potentially moving history.
 result=vec4(mix(current,h,weight),cameraStill && pixelStill ? depth : -depth);
}

