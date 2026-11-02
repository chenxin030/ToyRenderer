layout(set=0,binding=7) uniform sampler2D shadowAtlas;
int cascadeIndex(vec3 world) {
 float d = -(camera.view*vec4(world,1)).z;
 return d<camera.splits.x?0:d<camera.splits.y?1:d<camera.splits.z?2:3;
}
float cascadeShadow(vec3 world, vec3 n, int c) {
 float nl = clamp(dot(n,camera.sun.xyz),0,1);
 vec3 w = world + n * camera.shadow.w*camera.texels[c]*(1-nl);
 vec3 p = (camera.shadowVP[c]*vec4(w,1)).xyz;
 vec2 uv=p.xy*.5+.5;
 if(any(lessThan(uv,vec2(0)))||any(greaterThan(uv,vec2(1)))||p.z<0||p.z>1) return 1;
 int size=textureSize(shadowAtlas,0).x/2;
 ivec2 tile=ivec2(c%2,c/2)*size;
 ivec2 pixel=ivec2(uv*size);
 float sum=0; int r=int(camera.shadow.y);
 for(int y=-r;y<=r;y++) for(int x=-r;x<=r;x++) {
   float z=texelFetch(shadowAtlas,tile+clamp(pixel+ivec2(x,y),ivec2(0),ivec2(size-1)),0).r;
   sum += p.z-camera.shadow.z-camera.sun.w*camera.texels[c]/299.9*min(sqrt(1-nl*nl)/max(nl,.1),4) <= z ? 1:0;
 }
 return sum/float((2*r+1)*(2*r+1));
}
float visibility(vec3 w,vec3 n) {
 float d=-(camera.view*vec4(w,1)).z;
 if(camera.shadow.x==0||d>camera.splits.w) return 1;
 int c=cascadeIndex(w);
 float a=cascadeShadow(w,n,c);
 float start=c==0?.1:camera.splits[c-1];
 float blend=smoothstep(mix(start,camera.splits[c],.9),camera.splits[c],d);
 return mix(a,c<3?cascadeShadow(w,n,c+1):1,blend);
}
