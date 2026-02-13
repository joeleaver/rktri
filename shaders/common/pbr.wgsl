// PBR (Physically Based Rendering) Utilities
// Implements Cook-Torrance BRDF with GGX distribution

const PI: f32 = 3.14159265359;

// GGX/Trowbridge-Reitz Normal Distribution Function
// Models microfacet distribution - how many microfacets are aligned with the half vector
// Args:
//   NdotH: dot(normal, half_vector)
//   roughness: surface roughness [0=smooth, 1=rough]
fn D_GGX(NdotH: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let NdotH2 = NdotH * NdotH;

    let denom = NdotH2 * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith's Geometry Function with GGX
// Models self-shadowing of microfacets
// Uses Schlick-GGX approximation for efficiency
fn G_SchlickGGX(NdotV: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return NdotV / (NdotV * (1.0 - k) + k);
}

fn G_Smith(NdotV: f32, NdotL: f32, roughness: f32) -> f32 {
    let ggx_v = G_SchlickGGX(NdotV, roughness);
    let ggx_l = G_SchlickGGX(NdotL, roughness);
    return ggx_v * ggx_l;
}

// Fresnel-Schlick Approximation
// Models how light reflects at different angles
// Args:
//   cosTheta: dot(half_vector, view_direction)
//   F0: base reflectivity at normal incidence
fn F_Schlick(cosTheta: f32, F0: vec3<f32>) -> vec3<f32> {
    let power = pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
    return F0 + (vec3<f32>(1.0) - F0) * power;
}

// Cook-Torrance Specular BRDF
// Combines D, G, and F terms for physically accurate specular reflection
// Args:
//   N: surface normal
//   V: view direction (towards camera)
//   L: light direction (towards light)
//   albedo: base color
//   metallic: metalness [0=dielectric, 1=metal]
//   roughness: surface roughness [0=smooth, 1=rough]
// Returns: outgoing radiance in view direction
fn brdf_cook_torrance(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32
) -> vec3<f32> {
    let H = normalize(V + L);

    let NdotV = max(dot(N, V), 0.0001);
    let NdotL = max(dot(N, L), 0.0001);
    let NdotH = max(dot(N, H), 0.0);
    let HdotV = max(dot(H, V), 0.0);

    // Base reflectivity - metals use albedo, dielectrics use 0.04
    let F0 = mix(vec3<f32>(0.04), albedo, metallic);

    // Cook-Torrance specular term
    let D = D_GGX(NdotH, roughness);
    let G = G_Smith(NdotV, NdotL, roughness);
    let F = F_Schlick(HdotV, F0);

    let specular = (D * G * F) / (4.0 * NdotV * NdotL);

    // Diffuse term (Lambertian)
    // Metals have no diffuse component
    let kD = (vec3<f32>(1.0) - F) * (1.0 - metallic);
    let diffuse = kD * albedo / PI;

    return (diffuse + specular) * NdotL;
}
