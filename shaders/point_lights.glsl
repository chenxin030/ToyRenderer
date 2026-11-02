struct PointLight { vec4 positionRadius; vec4 intensity; };
layout(std430,set=0,binding=8) readonly buffer Lights { uvec4 lightConfig; PointLight lights[]; };
layout(std430,set=0,binding=9) readonly buffer TileLists { uint tileData[]; };
#ifdef DEFERRED_AO
uint tileOffset() {
 uint size=lightConfig.y, columns=(uint(camera.points.y)+size-1)/size;
 return (uint(gl_FragCoord.y)/size*columns+uint(gl_FragCoord.x)/size)*258;
}

#endif

