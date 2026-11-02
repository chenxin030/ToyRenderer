void material(out vec3 n, out vec3 base, out float metal, out float rough) {
  if(pc.alpha.y > .5 && pc.base.a * texture(baseTex,texcoord).a < pc.alpha.x) discard;
  n = normalize(N);
  vec3 t = normalize(T.xyz - n * dot(n, T.xyz)), b = cross(n, t) * T.w;
  vec3 mapped = texture(normalTex, texcoord).xyz * 2 - 1;
  mapped.xy *= pc.params.z * camera.settings.z;
  n = normalize(mat3(t, b, n) * normalize(mapped));
  base = pc.base.rgb * texture(baseTex, texcoord).rgb;
  vec4 mr = texture(mrTex, texcoord);
  metal = pc.params.x * mr.b;
  rough = clamp(pc.params.y * mr.g, .04, 1);
}
