const float PI = 3.14159265359;
vec3 fresnel(vec3 f0, float vh) { return f0 + (1 - f0) * pow(1 - vh, 5); }
float geometry(float nv, float nl, float k) {
  return nv / (nv * (1 - k) + k) * nl / (nl * (1 - k) + k);
}
vec3 directBRDF(vec3 n, vec3 v, vec3 l, vec3 base, float metal, float rough) {
  vec3 h = normalize(v + l);
  float nv = max(dot(n, v), 1e-5), nl = max(dot(n, l), 0),
        nh = max(dot(n, h), 0), vh = max(dot(v, h), 0);
  float a = rough * rough, a2 = a * a, d = nh * nh * (a2 - 1) + 1;
  vec3 f = fresnel(mix(vec3(.04), base, metal), vh);
  float g = geometry(nv, nl, pow(rough + 1, 2) / 8);
  return ((1 - f) * (1 - metal) * base / PI +
          f * (a2 / (PI * d * d)) * g / max(4 * nv * nl, 1e-5)) *
         nl;
}
vec2 envUV(vec3 n) {
  return vec2(fract(atan(n.z, n.x) / (2 * PI) + 1),
              acos(clamp(n.y, -1, 1)) / PI);
}
