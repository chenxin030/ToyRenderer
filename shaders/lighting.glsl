#include "shadows.glsl"
#include "point_lights.glsl"
vec3 pref(vec3 r, float rough) {
  vec2 uv = envUV(r);
  float level = rough * 5;
  float lo = floor(level), hi = min(lo + 1, 5);
  float y = clamp(uv.y, 0.5 / 64, 63.5 / 64);
  return mix(texture(prefilterTex, vec2(uv.x, (lo + y) / 6)).rgb,
             texture(prefilterTex, vec2(uv.x, (hi + y) / 6)).rgb, fract(level));
}
vec3 shade(vec3 world, vec3 n, vec3 base, float metal, float rough) {
  vec3 v = normalize(camera.eye.xyz - world);
  float nv = max(dot(n, v), .0001);
  vec3 f0 = mix(vec3(.04), base, metal);
  vec3 F = fresnel(f0, nv);
  vec3 diffuse =
      texture(irradianceTex, envUV(n)).rgb * base * (1 - metal) * (1 - F);
#ifdef DEFERRED_AO
  diffuse *= texelFetch(aoFinal,ivec2(gl_FragCoord.xy),0).r;
#endif
  vec2 ab = texture(lutTex, vec2(nv, rough)).rg;
  vec3 spec = pref(reflect(-v, n), rough) * (f0 * ab.x + ab.y);
  vec3 direct =
      directBRDF(n, v, camera.sun.xyz, base, metal, rough) *
      camera.light.rgb * visibility(world,n);
  uint count=lightConfig.x, offset=0; bool tiled=false;
#ifdef DEFERRED_AO
  offset=tileOffset(); tiled=lightConfig.w!=0 && tileData[offset+1]==0;
  if(tiled) count=tileData[offset];
#endif
  for(uint j=0;j<count;++j) {
    uint i=j;
#ifdef DEFERRED_AO
    if(tiled) i=tileData[offset+2+j];
#endif
    vec3 delta=lights[i].positionRadius.xyz-world;
    float dist=length(delta), radius=lights[i].positionRadius.w;
    if(dist>=radius) continue;
    float falloff=pow(max(1-pow(dist/radius,4),0),2)/max(dist*dist,.01);
    direct+=directBRDF(n,v,delta/max(dist,.0001),base,metal,rough)*lights[i].intensity.rgb*falloff*camera.points.x;
  }
  int mode = int(camera.settings.w);
  vec3 result =
      direct * camera.settings.x + (diffuse + spec) * camera.settings.y;
  if (mode == 1)
    result = direct;
  if (mode == 2)
    result = diffuse;
  if (mode == 3)
    result = spec;
  if (mode == 4)
    result = n * .5 + .5;
  if (mode == 5)
    result = base;
  if (mode == 6)
    result = vec3(1, rough, metal);

  if(mode==11) result = vec3(visibility(world,n));
  if(mode==12) { int c=cascadeIndex(world); result = c==0?vec3(1,.2,.2):c==1?vec3(.2,1,.2):c==2?vec3(.2,.2,1):vec3(1,1,.2); }
  if(mode==13) result=vec3(camera.texels[cascadeIndex(world)]);
  if(mode==14) result=vec3(texture(shadowAtlas,gl_FragCoord.xy/camera.points.yz).r);
  return result;
}
