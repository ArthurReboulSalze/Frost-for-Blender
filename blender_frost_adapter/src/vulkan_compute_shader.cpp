#include "vulkan_compute_shader.hpp"

#include "cuda/tables.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <glslang/Public/resource_limits_c.h>
#include <vulkan/vulkan.h>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

const char *kParticleComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint particleCount;
    float planningRadiusScale;
    float voxelLength;
    uint fieldMode;
    float anisotropyMaxScale;
    float kernelSupportRadius;
    uint useVelocityBuffer;
    uint _padding;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputParticles {
    vec4 inputParticles[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputParticles {
    vec4 outputParticles[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputMinVoxelBounds {
    ivec4 outputMinVoxelBounds[];
};

layout(std430, set = 0, binding = 3) writeonly buffer OutputMaxVoxelBounds {
    ivec4 outputMaxVoxelBounds[];
};

layout(std430, set = 0, binding = 4) readonly buffer InputVelocities {
    vec4 inputVelocities[];
};

vec3 anisotropic_half_extents(vec4 particle, vec4 velocityData) {
    float baseRadius = max(particle.w * pc.planningRadiusScale, 0.0);
    if (baseRadius <= 0.0) {
        return vec3(0.0);
    }

    if (pc.fieldMode == 3u && pc.kernelSupportRadius > 0.0) {
        return vec3(pc.kernelSupportRadius);
    }

    if (pc.fieldMode != 5u || pc.useVelocityBuffer == 0u) {
        return vec3(baseRadius);
    }

    vec3 velocity = velocityData.xyz;
    float speed = length(velocity);
    if (speed <= 1.0e-5) {
        return vec3(baseRadius);
    }

    float safeRadius = max(baseRadius, max(pc.voxelLength, 1.0e-4));
    float stretch = min(1.0 + 0.5 * (speed / safeRadius), max(pc.anisotropyMaxScale, 1.0));
    float majorRadius = baseRadius * stretch;
    float minorRadius = baseRadius / sqrt(max(stretch, 1.0));
    vec3 direction = velocity / speed;
    vec3 directionSquared = direction * direction;
    float majorSquared = majorRadius * majorRadius;
    float minorSquared = minorRadius * minorRadius;
    float deltaSquared = max(majorSquared - minorSquared, 0.0);

    return sqrt(deltaSquared * directionSquared + vec3(minorSquared));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.particleCount) {
        return;
    }

    vec4 particle = inputParticles[idx];
    outputParticles[idx] = particle;

    vec4 velocityData = (pc.useVelocityBuffer != 0u) ? inputVelocities[idx] : vec4(0.0);
    vec3 halfExtents = anisotropic_half_extents(particle, velocityData);
    vec3 minCorner = particle.xyz - halfExtents;
    vec3 maxCorner = particle.xyz + halfExtents;

    ivec3 minVoxel = ivec3(floor(minCorner / pc.voxelLength));
    ivec3 maxVoxelExclusive = ivec3(ceil(maxCorner / pc.voxelLength));

    outputMinVoxelBounds[idx] = ivec4(minVoxel, 0);
    outputMaxVoxelBounds[idx] = ivec4(maxVoxelExclusive, 0);
}
)";

const char *kVoxelCoverageComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform CoveragePushConstants {
    uint particleCount;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    int domainDimX;
    int domainDimY;
    int domainDimZ;
    int _padding;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputMinVoxelBounds {
    ivec4 inputMinVoxelBounds[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputMaxVoxelBounds {
    ivec4 inputMaxVoxelBounds[];
};

layout(std430, set = 0, binding = 2) buffer OutputVoxelCoverage {
    uint voxelCoverage[];
};

uint linear_index(ivec3 p) {
    return uint(p.x) +
           uint(pc.domainDimX) * (uint(p.y) + uint(pc.domainDimY) * uint(p.z));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.particleCount) {
        return;
    }

    ivec3 minBound = inputMinVoxelBounds[idx].xyz - ivec3(pc.domainMinX, pc.domainMinY, pc.domainMinZ);
    ivec3 maxBound = inputMaxVoxelBounds[idx].xyz - ivec3(pc.domainMinX, pc.domainMinY, pc.domainMinZ);

    minBound = clamp(minBound, ivec3(0), ivec3(pc.domainDimX, pc.domainDimY, pc.domainDimZ));
    maxBound = clamp(maxBound, ivec3(0), ivec3(pc.domainDimX, pc.domainDimY, pc.domainDimZ));

    for (int z = minBound.z; z < maxBound.z; ++z) {
        for (int y = minBound.y; y < maxBound.y; ++y) {
            for (int x = minBound.x; x < maxBound.x; ++x) {
                atomicAdd(voxelCoverage[linear_index(ivec3(x, y, z))], 1u);
            }
        }
    }
}
)";

const char *kActiveVoxelCompactComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform ActiveVoxelCompactPushConstants {
    uint voxelCount;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputVoxelCoverage {
    uint voxelCoverage[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputActiveVoxelIndices {
    uint activeVoxelIndices[];
};

layout(std430, set = 0, binding = 2) buffer OutputActiveVoxelStats {
    uint activeVoxelStats[];
};

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.voxelCount) {
        return;
    }

    uint count = voxelCoverage[idx];
    if (count == 0u) {
        return;
    }

    uint slot = atomicAdd(activeVoxelStats[0], 1u);
    activeVoxelIndices[slot] = idx;
    atomicMax(activeVoxelStats[1], count);
    uint previousPairs = atomicAdd(activeVoxelStats[2], count);
    if (previousPairs > 0xffffffffu - count) {
        atomicOr(activeVoxelStats[3], 1u);
    }
}
)";

const char *kActiveVoxelPairFillComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform ActiveVoxelPairFillPushConstants {
    uint particleCount;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    int domainDimX;
    int domainDimY;
    int domainDimZ;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputMinVoxelBounds {
    ivec4 inputMinVoxelBounds[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputMaxVoxelBounds {
    ivec4 inputMaxVoxelBounds[];
};

layout(std430, set = 0, binding = 2) readonly buffer InputActiveVoxelCompactLookup {
    int activeVoxelCompactLookup[];
};

layout(std430, set = 0, binding = 3) readonly buffer InputActiveVoxelParticleOffsets {
    uint activeVoxelParticleOffsets[];
};

layout(std430, set = 0, binding = 4) buffer InOutActiveVoxelParticleCursor {
    uint activeVoxelParticleCursor[];
};

layout(std430, set = 0, binding = 5) writeonly buffer OutputActiveVoxelParticleIndices {
    uint activeVoxelParticleIndices[];
};

void main() {
    uint particleIndex = gl_GlobalInvocationID.x;
    if (particleIndex >= pc.particleCount) {
        return;
    }

    ivec3 minBound = inputMinVoxelBounds[particleIndex].xyz -
                     ivec3(pc.domainMinX, pc.domainMinY, pc.domainMinZ);
    ivec3 maxBound = inputMaxVoxelBounds[particleIndex].xyz -
                     ivec3(pc.domainMinX, pc.domainMinY, pc.domainMinZ);
    minBound = clamp(minBound, ivec3(0), ivec3(pc.domainDimX, pc.domainDimY, pc.domainDimZ));
    maxBound = clamp(maxBound, ivec3(0), ivec3(pc.domainDimX, pc.domainDimY, pc.domainDimZ));
    if (any(greaterThanEqual(minBound, maxBound))) {
        return;
    }

    uint sliceStride = uint(pc.domainDimX) * uint(pc.domainDimY);
    for (int z = minBound.z; z < maxBound.z; ++z) {
        for (int y = minBound.y; y < maxBound.y; ++y) {
            uint rowBase = uint(z) * sliceStride + uint(y) * uint(pc.domainDimX);
            for (int x = minBound.x; x < maxBound.x; ++x) {
                uint packedVoxelIndex = rowBase + uint(x);
                int compactIndex = activeVoxelCompactLookup[packedVoxelIndex];
                if (compactIndex < 0) {
                    continue;
                }
                uint localIndex = atomicAdd(activeVoxelParticleCursor[compactIndex], 1u);
                uint writeIndex = activeVoxelParticleOffsets[compactIndex] + localIndex;
                activeVoxelParticleIndices[writeIndex] = particleIndex;
            }
        }
    }
}
)";

const char *kVoxelScalarFieldComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform ScalarFieldPushConstants {
    uint activeVoxelCount;
    uint particleCount;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    int domainDimX;
    int domainDimY;
    uint fieldMode;
    float voxelLength;
    float fieldRadiusScale;
    float inverseMaxCoverage;
    float fieldThreshold;
    float surfaceIsoValue;
    float anisotropyMaxScale;
    float kernelSupportRadius;
    uint useRequestedParticleField;
    uint useVelocityBuffer;
    uint useCompactedParticlePairs;
    uint _padding0;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputVoxelCoverage {
    uint voxelCoverage[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputActiveVoxelIndices {
    uint activeVoxelIndices[];
};

layout(std430, set = 0, binding = 2) readonly buffer InputParticles {
    vec4 inputParticles[];
};

layout(std430, set = 0, binding = 3) readonly buffer InputVelocities {
    vec4 inputVelocities[];
};

layout(std430, set = 0, binding = 4) readonly buffer InputMinVoxelBounds {
    ivec4 inputMinVoxelBounds[];
};

layout(std430, set = 0, binding = 5) readonly buffer InputMaxVoxelBounds {
    ivec4 inputMaxVoxelBounds[];
};

layout(std430, set = 0, binding = 6) readonly buffer InputActiveVoxelParticleOffsets {
    uint activeVoxelParticleOffsets[];
};

layout(std430, set = 0, binding = 7) readonly buffer InputActiveVoxelParticleIndices {
    uint activeVoxelParticleIndices[];
};

layout(std430, set = 0, binding = 8) writeonly buffer OutputVoxelScalarField {
    float voxelScalarField[];
};

float metaball_function(float distance, float particleEffectRadius) {
    if (particleEffectRadius <= 0.0) {
        return 0.0;
    }

    float normalizedDistance = distance / particleEffectRadius;
    if (distance < 0.33333333 * particleEffectRadius) {
        return 1.0 - 3.0 * normalizedDistance * normalizedDistance;
    } else if (distance < particleEffectRadius) {
        float blend = 1.0 - normalizedDistance;
        return 1.5 * blend * blend;
    }

    return 0.0;
}

float plain_marching_cubes_function(float distance, float particleEffectRadius) {
    if (particleEffectRadius <= 0.0 || distance >= particleEffectRadius) {
        return 0.0;
    }

    return 1.0 - (distance / particleEffectRadius);
}

float zhu_bridson_weight(float distanceSquared, float kernelSupportRadius) {
    if (kernelSupportRadius <= 0.0) {
        return 0.0;
    }
    float kernelSupportSquared = kernelSupportRadius * kernelSupportRadius;
    if (distanceSquared >= kernelSupportSquared) {
        return 0.0;
    }
    float x = 1.0 - distanceSquared / kernelSupportSquared;
    return x * x * x;
}

float anisotropic_signed_distance(vec3 samplePosition, vec4 particle, vec4 velocityData,
                                  float particleEffectRadius, float anisotropyMaxScale,
                                  float voxelLength, float fieldThreshold) {
    if (particleEffectRadius <= 0.0) {
        return max(fieldThreshold, max(voxelLength, 1.0));
    }

    vec3 velocity = velocityData.xyz;
    float speed = length(velocity);
    float safeRadius = max(particleEffectRadius, max(voxelLength, 1.0e-4));
    float stretch = 1.0;
    if (speed > 1.0e-5) {
        stretch = min(1.0 + 0.5 * (speed / safeRadius), max(anisotropyMaxScale, 1.0));
    }

    float majorRadius = particleEffectRadius * stretch;
    float minorRadius = particleEffectRadius / sqrt(max(stretch, 1.0));
    float isoBiasDistance = (fieldThreshold - 0.5) * particleEffectRadius * 0.5;
    vec3 delta = samplePosition - particle.xyz;

    float signedDistance = 0.0;
    if (speed > 1.0e-5 && majorRadius > 1.0e-6 && minorRadius > 1.0e-6) {
        vec3 direction = velocity / speed;
        float parallelDistance = dot(delta, direction);
        vec3 perpendicular = delta - direction * parallelDistance;
        float normalizedDistance = sqrt(
            (parallelDistance * parallelDistance) / (majorRadius * majorRadius) +
            dot(perpendicular, perpendicular) / (minorRadius * minorRadius)
        );
        signedDistance = (normalizedDistance - 1.0) * particleEffectRadius;
    } else {
        signedDistance = length(delta) - particleEffectRadius;
    }

    return signedDistance - isoBiasDistance;
}

bool voxel_inside_particle_bounds(ivec3 voxelCoord, uint particleIndex) {
    ivec3 minBound = inputMinVoxelBounds[particleIndex].xyz;
    ivec3 maxBound = inputMaxVoxelBounds[particleIndex].xyz;
    return all(greaterThanEqual(voxelCoord, minBound)) &&
           all(lessThan(voxelCoord, maxBound));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.activeVoxelCount) {
        return;
    }

    uint voxelIndex = activeVoxelIndices[idx];
    uint candidateBegin = 0u;
    uint candidateEnd = pc.particleCount;
    bool useCompactedParticlePairs = (pc.useCompactedParticlePairs != 0u);
    if (useCompactedParticlePairs) {
        candidateBegin = activeVoxelParticleOffsets[idx];
        candidateEnd = activeVoxelParticleOffsets[idx + 1u];
    }
    uint count = voxelCoverage[voxelIndex];
    float outsideField = (pc.fieldMode == 1u || pc.fieldMode == 2u)
        ? max(pc.fieldThreshold, pc.surfaceIsoValue + 1.0e-3)
        : max(pc.fieldThreshold, max(pc.voxelLength, 1.0));
    if (count == 0u) {
        return;
    }

    float normalized = clamp(float(count) * pc.inverseMaxCoverage, 0.0, 1.0);
    float fallbackField = (pc.fieldMode == 1u || pc.fieldMode == 2u)
        ? (pc.fieldThreshold - normalized)
        : -normalized;
    if (pc.useRequestedParticleField == 0u || pc.particleCount == 0u) {
        voxelScalarField[voxelIndex] = fallbackField;
        return;
    }

    uint dimX = uint(pc.domainDimX);
    uint dimY = uint(pc.domainDimY);
    uint slice = dimX * dimY;
    uint z = voxelIndex / slice;
    uint rem = voxelIndex - z * slice;
    uint y = rem / dimX;
    uint x = rem - y * dimX;

    vec3 samplePosition = vec3(
        float(pc.domainMinX) + float(x),
        float(pc.domainMinY) + float(y),
        float(pc.domainMinZ) + float(z)
    ) * pc.voxelLength;
    ivec3 sampleVoxelCoord = ivec3(int(x) + pc.domainMinX,
                                   int(y) + pc.domainMinY,
                                   int(z) + pc.domainMinZ);

    if (pc.fieldMode == 1u) {
        float metaballField = pc.fieldThreshold;
        bool foundInfluence = false;
        for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
            uint particleIndex = useCompactedParticlePairs
                ? activeVoxelParticleIndices[candidateIndex]
                : candidateIndex;
            if (!useCompactedParticlePairs && !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
                continue;
            }
            vec4 particle = inputParticles[particleIndex];
            float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
            if (radius <= 0.0) {
                continue;
            }

            float distanceToParticle = length(samplePosition - particle.xyz);
            float weight = metaball_function(distanceToParticle, radius);
            if (weight <= 0.0) {
                continue;
            }

            metaballField -= weight;
            foundInfluence = true;
        }

        voxelScalarField[voxelIndex] = foundInfluence ? metaballField : outsideField;
        return;
    }

    if (pc.fieldMode == 2u) {
        float densityField = pc.fieldThreshold;
        bool foundInfluence = false;
        for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
            uint particleIndex = useCompactedParticlePairs
                ? activeVoxelParticleIndices[candidateIndex]
                : candidateIndex;
            if (!useCompactedParticlePairs && !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
                continue;
            }
            vec4 particle = inputParticles[particleIndex];
            float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
            if (radius <= 0.0) {
                continue;
            }

            float distanceToParticle = length(samplePosition - particle.xyz);
            float density = plain_marching_cubes_function(distanceToParticle, radius);
            if (density <= 0.0) {
                continue;
            }

            densityField -= density;
            foundInfluence = true;
        }

        voxelScalarField[voxelIndex] = foundInfluence ? densityField : outsideField;
        return;
    }

    if (pc.fieldMode == 3u) {
        float totalWeight = 0.0;
        vec3 blendedCenter = vec3(0.0);
        float blendedRadius = 0.0;
        float kernelSupportRadius = max(pc.kernelSupportRadius, 0.0);
        for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
            uint particleIndex = useCompactedParticlePairs
                ? activeVoxelParticleIndices[candidateIndex]
                : candidateIndex;
            if (!useCompactedParticlePairs && !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
                continue;
            }
            vec4 particle = inputParticles[particleIndex];
            float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
            if (radius <= 0.0) {
                continue;
            }

            vec3 delta = samplePosition - particle.xyz;
            float distanceSquared = dot(delta, delta);
            float weight = zhu_bridson_weight(
                distanceSquared,
                kernelSupportRadius > 0.0 ? kernelSupportRadius : radius);
            if (weight <= 0.0) {
                continue;
            }

            totalWeight += weight;
            blendedCenter += weight * particle.xyz;
            blendedRadius += weight * particle.w;
        }

        if (totalWeight <= 1.0e-6) {
            voxelScalarField[voxelIndex] = outsideField;
            return;
        }

        blendedCenter /= totalWeight;
        blendedRadius /= totalWeight;
        voxelScalarField[voxelIndex] = length(samplePosition - blendedCenter) - blendedRadius;
        return;
    }

    if (pc.fieldMode == 5u) {
        float minSignedDistance = 3.402823466e38;
        for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
            uint particleIndex = useCompactedParticlePairs
                ? activeVoxelParticleIndices[candidateIndex]
                : candidateIndex;
            if (!useCompactedParticlePairs && !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
                continue;
            }
            vec4 particle = inputParticles[particleIndex];
            float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
            if (radius <= 0.0) {
                continue;
            }

            vec4 velocityData = (pc.useVelocityBuffer != 0u)
                ? inputVelocities[particleIndex]
                : vec4(0.0);
            float signedDistance = anisotropic_signed_distance(
                samplePosition, particle, velocityData, radius,
                pc.anisotropyMaxScale, pc.voxelLength, pc.fieldThreshold);
            minSignedDistance = min(minSignedDistance, signedDistance);
        }

        if (minSignedDistance > 1.0e30) {
            voxelScalarField[voxelIndex] = outsideField;
            return;
        }

        voxelScalarField[voxelIndex] = minSignedDistance;
        return;
    }

    float minSignedDistance = 3.402823466e38;
    for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
        uint particleIndex = useCompactedParticlePairs
            ? activeVoxelParticleIndices[candidateIndex]
            : candidateIndex;
        if (!useCompactedParticlePairs && !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
            continue;
        }
        vec4 particle = inputParticles[particleIndex];
        float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
        if (radius <= 0.0) {
            continue;
        }

        float signedDistance = length(samplePosition - particle.xyz) - radius;
        minSignedDistance = min(minSignedDistance, signedDistance);
    }

    if (minSignedDistance > 1.0e30) {
        voxelScalarField[voxelIndex] = max(pc.fieldThreshold, fallbackField);
        return;
    }

    voxelScalarField[voxelIndex] = (pc.fieldThreshold > 0.0)
        ? min(minSignedDistance, pc.fieldThreshold)
        : minSignedDistance;
}
)";

const char *kZhuBridsonScalarFieldComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform ScalarFieldPushConstants {
    uint activeVoxelCount;
    uint particleCount;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    int domainDimX;
    int domainDimY;
    uint fieldMode;
    float voxelLength;
    float fieldRadiusScale;
    float inverseMaxCoverage;
    float fieldThreshold;
    float surfaceIsoValue;
    float anisotropyMaxScale;
    float kernelSupportRadius;
    uint useRequestedParticleField;
    uint useVelocityBuffer;
    uint useCompactedParticlePairs;
    uint _padding0;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputVoxelCoverage {
    uint voxelCoverage[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputActiveVoxelIndices {
    uint activeVoxelIndices[];
};

layout(std430, set = 0, binding = 2) readonly buffer InputParticles {
    vec4 inputParticles[];
};

layout(std430, set = 0, binding = 3) readonly buffer InputVelocities {
    vec4 inputVelocities[];
};

layout(std430, set = 0, binding = 4) readonly buffer InputMinVoxelBounds {
    ivec4 inputMinVoxelBounds[];
};

layout(std430, set = 0, binding = 5) readonly buffer InputMaxVoxelBounds {
    ivec4 inputMaxVoxelBounds[];
};

layout(std430, set = 0, binding = 6) readonly buffer InputActiveVoxelParticleOffsets {
    uint activeVoxelParticleOffsets[];
};

layout(std430, set = 0, binding = 7) readonly buffer InputActiveVoxelParticleIndices {
    uint activeVoxelParticleIndices[];
};

layout(std430, set = 0, binding = 8) writeonly buffer OutputVoxelScalarField {
    float voxelScalarField[];
};

float zhu_bridson_weight(float distanceSquared, float kernelSupportRadius) {
    if (kernelSupportRadius <= 0.0) {
        return 0.0;
    }
    float kernelSupportSquared = kernelSupportRadius * kernelSupportRadius;
    if (distanceSquared >= kernelSupportSquared) {
        return 0.0;
    }
    float x = 1.0 - distanceSquared / kernelSupportSquared;
    return x * x * x;
}

bool voxel_inside_particle_bounds(ivec3 voxelCoord, uint particleIndex) {
    ivec3 minBound = inputMinVoxelBounds[particleIndex].xyz;
    ivec3 maxBound = inputMaxVoxelBounds[particleIndex].xyz;
    return all(greaterThanEqual(voxelCoord, minBound)) &&
           all(lessThan(voxelCoord, maxBound));
}

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx >= pc.activeVoxelCount) {
        return;
    }

    uint voxelIndex = activeVoxelIndices[idx];
    uint count = voxelCoverage[voxelIndex];
    if (count == 0u) {
        return;
    }

    float normalized = clamp(float(count) * pc.inverseMaxCoverage, 0.0, 1.0);
    if (pc.useRequestedParticleField == 0u || pc.particleCount == 0u) {
        voxelScalarField[voxelIndex] = -normalized;
        return;
    }

    uint candidateBegin = 0u;
    uint candidateEnd = pc.particleCount;
    bool useCompactedParticlePairs = (pc.useCompactedParticlePairs != 0u);
    if (useCompactedParticlePairs) {
        candidateBegin = activeVoxelParticleOffsets[idx];
        candidateEnd = activeVoxelParticleOffsets[idx + 1u];
    }

    uint dimX = uint(pc.domainDimX);
    uint dimY = uint(pc.domainDimY);
    uint slice = dimX * dimY;
    uint z = voxelIndex / slice;
    uint rem = voxelIndex - z * slice;
    uint y = rem / dimX;
    uint x = rem - y * dimX;

    vec3 samplePosition = vec3(
        float(pc.domainMinX) + float(x),
        float(pc.domainMinY) + float(y),
        float(pc.domainMinZ) + float(z)
    ) * pc.voxelLength;
    ivec3 sampleVoxelCoord = ivec3(int(x) + pc.domainMinX,
                                   int(y) + pc.domainMinY,
                                   int(z) + pc.domainMinZ);

    float totalWeight = 0.0;
    vec3 blendedCenter = vec3(0.0);
    float blendedRadius = 0.0;
    float kernelSupportRadius = max(pc.kernelSupportRadius, 0.0);
    for (uint candidateIndex = candidateBegin; candidateIndex < candidateEnd; ++candidateIndex) {
        uint particleIndex = useCompactedParticlePairs
            ? activeVoxelParticleIndices[candidateIndex]
            : candidateIndex;
        if (!useCompactedParticlePairs &&
            !voxel_inside_particle_bounds(sampleVoxelCoord, particleIndex)) {
            continue;
        }

        vec4 particle = inputParticles[particleIndex];
        float radius = max(particle.w * pc.fieldRadiusScale, 0.0);
        if (radius <= 0.0) {
            continue;
        }

        vec3 delta = samplePosition - particle.xyz;
        float distanceSquared = dot(delta, delta);
        float weight = zhu_bridson_weight(
            distanceSquared,
            kernelSupportRadius > 0.0 ? kernelSupportRadius : radius);
        if (weight <= 0.0) {
            continue;
        }

        totalWeight += weight;
        blendedCenter += weight * particle.xyz;
        blendedRadius += weight * particle.w;
    }

    if (totalWeight <= 1.0e-6) {
        voxelScalarField[voxelIndex] = max(pc.fieldThreshold, max(pc.voxelLength, 1.0));
        return;
    }

    blendedCenter /= totalWeight;
    blendedRadius /= totalWeight;
    voxelScalarField[voxelIndex] = length(samplePosition - blendedCenter) - blendedRadius;
}
)";

const char *kSurfaceCellClassifyComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceCellPushConstants {
    uint candidateCellCount;
    int cellDimX;
    int cellDimY;
    int cellDimZ;
    int domainDimX;
    int domainDimY;
    float surfaceIsoValue;
    uint _padding0;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputCandidateCellIndices {
    uint candidateCellIndices[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputVoxelScalarField {
    float voxelScalarField[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputCandidateCubeIndices {
    uint candidateCubeIndices[];
};

const uvec3 kSurfaceCellCornerOffsets[8] = uvec3[8](
    uvec3(0, 0, 0),
    uvec3(1, 0, 0),
    uvec3(1, 1, 0),
    uvec3(0, 1, 0),
    uvec3(0, 0, 1),
    uvec3(1, 0, 1),
    uvec3(1, 1, 1),
    uvec3(0, 1, 1)
);

uint scalar_index(uvec3 p) {
    return p.x + uint(pc.domainDimX) * (p.y + uint(pc.domainDimY) * p.z);
}

void main() {
    uint candidateIndex = gl_GlobalInvocationID.x;
    if (candidateIndex >= pc.candidateCellCount) {
        return;
    }

    uint packedCellIndex = candidateCellIndices[candidateIndex];
    uint cellSlice = uint(pc.cellDimX) * uint(pc.cellDimY);
    uint z = packedCellIndex / cellSlice;
    uint rem = packedCellIndex - z * cellSlice;
    uint y = rem / uint(pc.cellDimX);
    uint x = rem - y * uint(pc.cellDimX);

    uint cubeIndex = 0u;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float cornerValue =
            voxelScalarField[scalar_index(uvec3(x, y, z) +
                                          kSurfaceCellCornerOffsets[corner])];
        if (cornerValue < pc.surfaceIsoValue) {
            cubeIndex |= (1u << corner);
        }
    }

    candidateCubeIndices[candidateIndex] = cubeIndex;
}
)";

const char *kSparseSurfaceCellFromActiveVoxelComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceCellPushConstants {
    uint candidateCellCount;
    int cellDimX;
    int cellDimY;
    int cellDimZ;
    int domainDimX;
    int domainDimY;
    float surfaceIsoValue;
    uint _padding0;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputActiveVoxelIndices {
    uint activeVoxelIndices[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputVoxelScalarField {
    float voxelScalarField[];
};

layout(std430, set = 0, binding = 2) buffer OutputCellCubeIndices {
    uint cellCubeIndices[];
};

const uvec3 kSurfaceCellCornerOffsets[8] = uvec3[8](
    uvec3(0, 0, 0),
    uvec3(1, 0, 0),
    uvec3(1, 1, 0),
    uvec3(0, 1, 0),
    uvec3(0, 0, 1),
    uvec3(1, 0, 1),
    uvec3(1, 1, 1),
    uvec3(0, 1, 1)
);

uint scalar_index(uvec3 p) {
    return p.x + uint(pc.domainDimX) * (p.y + uint(pc.domainDimY) * p.z);
}

void classify_cell(int cellX, int cellY, int cellZ) {
    if (cellX < 0 || cellY < 0 || cellZ < 0 ||
        cellX >= pc.cellDimX || cellY >= pc.cellDimY || cellZ >= pc.cellDimZ) {
        return;
    }

    uint x = uint(cellX);
    uint y = uint(cellY);
    uint z = uint(cellZ);
    uint cubeIndex = 0u;
    for (uint corner = 0u; corner < 8u; ++corner) {
        float cornerValue =
            voxelScalarField[scalar_index(uvec3(x, y, z) +
                                          kSurfaceCellCornerOffsets[corner])];
        if (cornerValue < pc.surfaceIsoValue) {
            cubeIndex |= (1u << corner);
        }
    }

    uint packedCellIndex =
        x + uint(pc.cellDimX) * (y + uint(pc.cellDimY) * z);
    atomicExchange(cellCubeIndices[packedCellIndex], cubeIndex);
}

void main() {
    uint activeVoxelIndex = gl_GlobalInvocationID.x;
    if (activeVoxelIndex >= pc.candidateCellCount) {
        return;
    }

    uint packedVoxelIndex = activeVoxelIndices[activeVoxelIndex];
    uint voxelSlice = uint(pc.domainDimX) * uint(pc.domainDimY);
    int z = int(packedVoxelIndex / voxelSlice);
    uint rem = packedVoxelIndex - uint(z) * voxelSlice;
    int y = int(rem / uint(pc.domainDimX));
    int x = int(rem - uint(y) * uint(pc.domainDimX));

    classify_cell(x, y, z);
    classify_cell(x - 1, y, z);
    classify_cell(x, y - 1, z);
    classify_cell(x, y, z - 1);
    classify_cell(x - 1, y - 1, z);
    classify_cell(x - 1, y, z - 1);
    classify_cell(x, y - 1, z - 1);
    classify_cell(x - 1, y - 1, z - 1);
}
)";

const char *kSurfaceCellCompactComputeShader = R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceCellCompactPushConstants {
    uint cellCount;
    uint _padding0;
    uint _padding1;
    uint _padding2;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputDenseCubeIndices {
    uint denseCubeIndices[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputActiveCellIndices {
    uint activeCellIndices[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputActiveCellCubeIndices {
    uint activeCellCubeIndices[];
};

layout(std430, set = 0, binding = 3) buffer OutputSurfaceCellStats {
    uint stats[];
};

void main() {
    uint cellIndex = gl_GlobalInvocationID.x;
    if (cellIndex >= pc.cellCount) {
        return;
    }

    uint cubeIndex = denseCubeIndices[cellIndex];
    if (cubeIndex == 0u || cubeIndex == 255u) {
        return;
    }

    uint outputIndex = atomicAdd(stats[0], 1u);
    activeCellIndices[outputIndex] = cellIndex;
    activeCellCubeIndices[outputIndex] = cubeIndex;
}
)";

std::string build_surface_triangle_compact_compute_shader_source() {
  std::ostringstream shader;
  shader << R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceMeshPushConstants {
    uint candidateCellCount;
    int cellDimX;
    int cellDimY;
    int domainDimX;
    int domainDimY;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    float voxelLength;
    float surfaceIsoValue;
    uint _padding0;
    uint _padding1;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputActiveCellIndices {
    uint activeCellIndices[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputActiveCellCubeIndices {
    uint activeCellCubeIndices[];
};

layout(std430, set = 0, binding = 2) readonly buffer InputVoxelScalarField {
    float voxelScalarField[];
};

layout(std430, set = 0, binding = 3) readonly buffer InputTriangleOffsets {
    uint triangleOffsets[];
};

layout(std430, set = 0, binding = 4) writeonly buffer OutputCompactTriangleVertices {
    float compactTriangleVertices[];
};

const ivec3 kCornerOffsets[8] = ivec3[8](
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    ivec3(1, 1, 1),
    ivec3(0, 1, 1)
);

const ivec2 kEdgeCorners[12] = ivec2[12](
    ivec2(0, 1),
    ivec2(1, 2),
    ivec2(2, 3),
    ivec2(3, 0),
    ivec2(4, 5),
    ivec2(5, 6),
    ivec2(6, 7),
    ivec2(7, 4),
    ivec2(0, 4),
    ivec2(1, 5),
    ivec2(2, 6),
    ivec2(3, 7)
);

const int kEdgeTable[256] = int[256](
)";
  for (int i = 0; i < 256; ++i) {
    if (i > 0) {
      shader << ((i % 12 == 0) ? ",\n" : ", ");
    }
    shader << edgeTable[i];
  }
  shader << R"(
);

const int kTriTable[4096] = int[4096](
)";
  for (int cubeIndex = 0; cubeIndex < 256; ++cubeIndex) {
    for (int slot = 0; slot < 16; ++slot) {
      const int flatIndex = cubeIndex * 16 + slot;
      if (flatIndex > 0) {
        shader << ((flatIndex % 16 == 0) ? ",\n" : ", ");
      }
      shader << triTable[cubeIndex][slot];
    }
  }
  shader << R"(
);

uint scalar_index(uvec3 p) {
    return p.x + uint(pc.domainDimX) * (p.y + uint(pc.domainDimY) * p.z);
}

vec3 world_position(uvec3 p) {
    vec3 domainMin = vec3(float(pc.domainMinX), float(pc.domainMinY), float(pc.domainMinZ));
    return (domainMin + vec3(p)) * pc.voxelLength;
}

vec3 interpolate_isovalue(vec3 a, vec3 b, float fa, float fb, float isoValue) {
    float t = 0.5;
    float denom = fb - fa;
    if (abs(denom) > 1.0e-6) {
        t = clamp((isoValue - fa) / denom, 0.0, 1.0);
    }
    return mix(a, b, t);
}

int tri_table_value(uint cubeIndex, uint slot) {
    return kTriTable[int(cubeIndex) * 16 + int(slot)];
}

void main() {
    uint candidateIndex = gl_GlobalInvocationID.x;
    if (candidateIndex >= pc.candidateCellCount) {
        return;
    }

    uint cubeIndex = activeCellCubeIndices[candidateIndex];
    if (cubeIndex == 0u || cubeIndex == 255u) {
        return;
    }

    uint packedCellIndex = activeCellIndices[candidateIndex];
    uint cellSlice = uint(pc.cellDimX) * uint(pc.cellDimY);
    uint z = packedCellIndex / cellSlice;
    uint rem = packedCellIndex - z * cellSlice;
    uint y = rem / uint(pc.cellDimX);
    uint x = rem - y * uint(pc.cellDimX);

    float cornerValues[8];
    vec3 cornerPositions[8];
    for (uint corner = 0u; corner < 8u; ++corner) {
        uvec3 gridPosition = uvec3(x, y, z) + uvec3(kCornerOffsets[corner]);
        cornerValues[corner] = voxelScalarField[scalar_index(gridPosition)];
        cornerPositions[corner] = world_position(gridPosition);
    }

    int edgeMask = kEdgeTable[int(cubeIndex)];
    vec3 edgeVertices[12];
    for (int edge = 0; edge < 12; ++edge) {
        if ((edgeMask & (1 << edge)) == 0) {
            continue;
        }
        ivec2 edgeCorners = kEdgeCorners[edge];
        int c0 = edgeCorners.x;
        int c1 = edgeCorners.y;
        edgeVertices[edge] = interpolate_isovalue(
            cornerPositions[c0],
            cornerPositions[c1],
            cornerValues[c0],
            cornerValues[c1],
            pc.surfaceIsoValue
        );
    }

    uint outputBase = triangleOffsets[candidateIndex] * 9u;
    uint triangleIndex = 0u;
    for (uint slot = 0u; slot < 15u; slot += 3u) {
        int e0 = tri_table_value(cubeIndex, slot + 0u);
        if (e0 < 0) {
            break;
        }
        int e1 = tri_table_value(cubeIndex, slot + 1u);
        int e2 = tri_table_value(cubeIndex, slot + 2u);
        uint triangleBase = outputBase + triangleIndex * 9u;
        compactTriangleVertices[triangleBase + 0u] = edgeVertices[e0].x;
        compactTriangleVertices[triangleBase + 1u] = edgeVertices[e0].y;
        compactTriangleVertices[triangleBase + 2u] = edgeVertices[e0].z;
        compactTriangleVertices[triangleBase + 3u] = edgeVertices[e1].x;
        compactTriangleVertices[triangleBase + 4u] = edgeVertices[e1].y;
        compactTriangleVertices[triangleBase + 5u] = edgeVertices[e1].z;
        compactTriangleVertices[triangleBase + 6u] = edgeVertices[e2].x;
        compactTriangleVertices[triangleBase + 7u] = edgeVertices[e2].y;
        compactTriangleVertices[triangleBase + 8u] = edgeVertices[e2].z;
        triangleIndex += 1u;
    }
}
)";
  return shader.str();
}

struct PushConstants {
  uint32_t particleCount = 0;
  float planningRadiusScale = 1.0f;
  float voxelLength = 1.0f;
  uint32_t fieldMode = 0;
  float anisotropyMaxScale = 1.0f;
  float kernelSupportRadius = 0.0f;
  uint32_t useVelocityBuffer = 0;
  uint32_t padding = 0;
};

struct CoveragePushConstants {
  uint32_t particleCount = 0;
  int32_t domainMinX = 0;
  int32_t domainMinY = 0;
  int32_t domainMinZ = 0;
  int32_t domainDimX = 0;
  int32_t domainDimY = 0;
  int32_t domainDimZ = 0;
  int32_t padding = 0;
};

struct ActiveVoxelCompactPushConstants {
  uint32_t voxelCount = 0;
};

struct ActiveVoxelPairFillPushConstants {
  uint32_t particleCount = 0;
  int32_t domainMinX = 0;
  int32_t domainMinY = 0;
  int32_t domainMinZ = 0;
  int32_t domainDimX = 0;
  int32_t domainDimY = 0;
  int32_t domainDimZ = 0;
};

struct ScalarFieldPushConstants {
  uint32_t activeVoxelCount = 0;
  uint32_t particleCount = 0;
  int32_t domainMinX = 0;
  int32_t domainMinY = 0;
  int32_t domainMinZ = 0;
  int32_t domainDimX = 0;
  int32_t domainDimY = 0;
  uint32_t fieldMode = 0;
  float voxelLength = 1.0f;
  float fieldRadiusScale = 1.0f;
  float inverseMaxCoverage = 1.0f;
  float fieldThreshold = 0.0f;
  float surfaceIsoValue = 0.0f;
  float anisotropyMaxScale = 1.0f;
  float kernelSupportRadius = 0.0f;
  uint32_t useRequestedParticleField = 0;
  uint32_t useVelocityBuffer = 0;
  uint32_t useCompactedParticlePairs = 0;
  uint32_t padding0 = 0;
};

struct SurfaceCellPushConstants {
  uint32_t candidateCellCount = 0;
  int32_t cellDimX = 0;
  int32_t cellDimY = 0;
  int32_t cellDimZ = 0;
  int32_t domainDimX = 0;
  int32_t domainDimY = 0;
  float surfaceIsoValue = 0.0f;
  uint32_t padding0 = 0;
};

struct SurfaceCellCompactPushConstants {
  uint32_t cellCount = 0;
  uint32_t padding0 = 0;
  uint32_t padding1 = 0;
  uint32_t padding2 = 0;
};

struct SurfaceTriangleCompactPushConstants {
  uint32_t cellCount = 0;
  uint32_t padding0 = 0;
  uint32_t padding1 = 0;
  uint32_t padding2 = 0;
};

struct SurfaceMeshPushConstants {
  uint32_t candidateCellCount = 0;
  int32_t cellDimX = 0;
  int32_t cellDimY = 0;
  int32_t domainDimX = 0;
  int32_t domainDimY = 0;
  int32_t domainMinX = 0;
  int32_t domainMinY = 0;
  int32_t domainMinZ = 0;
  float voxelLength = 1.0f;
  float surfaceIsoValue = 0.0f;
  uint32_t padding0 = 0;
  uint32_t padding1 = 0;
};

std::string build_surface_mesh_compute_shader_source() {
  std::ostringstream shader;
  shader << R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceMeshPushConstants {
    uint candidateCellCount;
    int cellDimX;
    int cellDimY;
    int domainDimX;
    int domainDimY;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    float voxelLength;
    float surfaceIsoValue;
    uint _padding0;
    uint _padding1;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputActiveCellIndices {
    uint activeCellIndices[];
};

layout(std430, set = 0, binding = 1) readonly buffer InputActiveCellCubeIndices {
    uint activeCellCubeIndices[];
};

layout(std430, set = 0, binding = 2) readonly buffer InputVoxelScalarField {
    float voxelScalarField[];
};

layout(std430, set = 0, binding = 3) writeonly buffer OutputTriangleCounts {
    uint triangleCounts[];
};

layout(std430, set = 0, binding = 4) writeonly buffer OutputTriangleVertices {
    float triangleVertices[];
};

const ivec3 kCornerOffsets[8] = ivec3[8](
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    ivec3(1, 1, 1),
    ivec3(0, 1, 1)
);

const ivec2 kEdgeCorners[12] = ivec2[12](
    ivec2(0, 1),
    ivec2(1, 2),
    ivec2(2, 3),
    ivec2(3, 0),
    ivec2(4, 5),
    ivec2(5, 6),
    ivec2(6, 7),
    ivec2(7, 4),
    ivec2(0, 4),
    ivec2(1, 5),
    ivec2(2, 6),
    ivec2(3, 7)
);

const int kEdgeTable[256] = int[256](
)";
  for (int i = 0; i < 256; ++i) {
    if (i > 0) {
      shader << ((i % 12 == 0) ? ",\n" : ", ");
    }
    shader << edgeTable[i];
  }
  shader << R"(
);

const int kTriTable[4096] = int[4096](
)";
  for (int cubeIndex = 0; cubeIndex < 256; ++cubeIndex) {
    for (int slot = 0; slot < 16; ++slot) {
      const int flatIndex = cubeIndex * 16 + slot;
      if (flatIndex > 0) {
        shader << ((flatIndex % 16 == 0) ? ",\n" : ", ");
      }
      shader << triTable[cubeIndex][slot];
    }
  }
  shader << R"(
);

uint scalar_index(uvec3 p) {
    return p.x + uint(pc.domainDimX) * (p.y + uint(pc.domainDimY) * p.z);
}

vec3 world_position(uvec3 p) {
    vec3 domainMin = vec3(float(pc.domainMinX), float(pc.domainMinY), float(pc.domainMinZ));
    return (domainMin + vec3(p)) * pc.voxelLength;
}

vec3 interpolate_isovalue(vec3 a, vec3 b, float fa, float fb, float isoValue) {
    float t = 0.5;
    float denom = fb - fa;
    if (abs(denom) > 1.0e-6) {
        t = clamp((isoValue - fa) / denom, 0.0, 1.0);
    }
    return mix(a, b, t);
}

int tri_table_value(uint cubeIndex, uint slot) {
    return kTriTable[int(cubeIndex) * 16 + int(slot)];
}

void main() {
    uint candidateIndex = gl_GlobalInvocationID.x;
    if (candidateIndex >= pc.candidateCellCount) {
        return;
    }

    triangleCounts[candidateIndex] = 0u;

    uint cubeIndex = activeCellCubeIndices[candidateIndex];
    if (cubeIndex == 0u || cubeIndex == 255u) {
        return;
    }

    uint packedCellIndex = activeCellIndices[candidateIndex];
    uint cellSlice = uint(pc.cellDimX) * uint(pc.cellDimY);
    uint z = packedCellIndex / cellSlice;
    uint rem = packedCellIndex - z * cellSlice;
    uint y = rem / uint(pc.cellDimX);
    uint x = rem - y * uint(pc.cellDimX);

    float cornerValues[8];
    vec3 cornerPositions[8];
    for (uint corner = 0u; corner < 8u; ++corner) {
        uvec3 gridPosition = uvec3(x, y, z) + uvec3(kCornerOffsets[corner]);
        cornerValues[corner] = voxelScalarField[scalar_index(gridPosition)];
        cornerPositions[corner] = world_position(gridPosition);
    }

    int edgeMask = kEdgeTable[int(cubeIndex)];
    vec3 edgeVertices[12];
    for (int edge = 0; edge < 12; ++edge) {
        if ((edgeMask & (1 << edge)) == 0) {
            continue;
        }
        ivec2 edgeCorners = kEdgeCorners[edge];
        int c0 = edgeCorners.x;
        int c1 = edgeCorners.y;
        edgeVertices[edge] = interpolate_isovalue(
            cornerPositions[c0],
            cornerPositions[c1],
            cornerValues[c0],
            cornerValues[c1],
            pc.surfaceIsoValue
        );
    }

    uint triangleCount = 0u;
    uint baseVertexFloatIndex = candidateIndex * 45u;
    for (uint slot = 0u; slot < 15u; slot += 3u) {
        int e0 = tri_table_value(cubeIndex, slot + 0u);
        if (e0 < 0) {
            break;
        }
        int e1 = tri_table_value(cubeIndex, slot + 1u);
        int e2 = tri_table_value(cubeIndex, slot + 2u);
        uint outputBase = baseVertexFloatIndex + triangleCount * 9u;
        triangleVertices[outputBase + 0u] = edgeVertices[e0].x;
        triangleVertices[outputBase + 1u] = edgeVertices[e0].y;
        triangleVertices[outputBase + 2u] = edgeVertices[e0].z;
        triangleVertices[outputBase + 3u] = edgeVertices[e1].x;
        triangleVertices[outputBase + 4u] = edgeVertices[e1].y;
        triangleVertices[outputBase + 5u] = edgeVertices[e1].z;
        triangleVertices[outputBase + 6u] = edgeVertices[e2].x;
        triangleVertices[outputBase + 7u] = edgeVertices[e2].y;
        triangleVertices[outputBase + 8u] = edgeVertices[e2].z;
        triangleCount += 1u;
    }

    triangleCounts[candidateIndex] = triangleCount;
}
)";
  return shader.str();
}

std::string build_dense_surface_mesh_compute_shader_source() {
  std::ostringstream shader;
  shader << R"(
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform SurfaceMeshPushConstants {
    uint candidateCellCount;
    int cellDimX;
    int cellDimY;
    int domainDimX;
    int domainDimY;
    int domainMinX;
    int domainMinY;
    int domainMinZ;
    float voxelLength;
    float surfaceIsoValue;
    uint _padding0;
    uint _padding1;
} pc;

layout(std430, set = 0, binding = 0) readonly buffer InputVoxelScalarField {
    float voxelScalarField[];
};

layout(std430, set = 0, binding = 1) writeonly buffer OutputCellCubeIndices {
    uint cellCubeIndices[];
};

layout(std430, set = 0, binding = 2) writeonly buffer OutputTriangleCounts {
    uint triangleCounts[];
};

layout(std430, set = 0, binding = 3) writeonly buffer OutputTriangleVertices {
    float triangleVertices[];
};

const ivec3 kCornerOffsets[8] = ivec3[8](
    ivec3(0, 0, 0),
    ivec3(1, 0, 0),
    ivec3(1, 1, 0),
    ivec3(0, 1, 0),
    ivec3(0, 0, 1),
    ivec3(1, 0, 1),
    ivec3(1, 1, 1),
    ivec3(0, 1, 1)
);

const ivec2 kEdgeCorners[12] = ivec2[12](
    ivec2(0, 1),
    ivec2(1, 2),
    ivec2(2, 3),
    ivec2(3, 0),
    ivec2(4, 5),
    ivec2(5, 6),
    ivec2(6, 7),
    ivec2(7, 4),
    ivec2(0, 4),
    ivec2(1, 5),
    ivec2(2, 6),
    ivec2(3, 7)
);

const int kEdgeTable[256] = int[256](
)";
  for (int i = 0; i < 256; ++i) {
    if (i > 0) {
      shader << ((i % 12 == 0) ? ",\n" : ", ");
    }
    shader << edgeTable[i];
  }
  shader << R"(
);

const int kTriTable[4096] = int[4096](
)";
  for (int cubeIndex = 0; cubeIndex < 256; ++cubeIndex) {
    for (int slot = 0; slot < 16; ++slot) {
      const int flatIndex = cubeIndex * 16 + slot;
      if (flatIndex > 0) {
        shader << ((flatIndex % 16 == 0) ? ",\n" : ", ");
      }
      shader << triTable[cubeIndex][slot];
    }
  }
  shader << R"(
);

uint scalar_index(uvec3 p) {
    return p.x + uint(pc.domainDimX) * (p.y + uint(pc.domainDimY) * p.z);
}

vec3 world_position(uvec3 p) {
    vec3 domainMin = vec3(float(pc.domainMinX), float(pc.domainMinY), float(pc.domainMinZ));
    return (domainMin + vec3(p)) * pc.voxelLength;
}

vec3 interpolate_isovalue(vec3 a, vec3 b, float fa, float fb, float isoValue) {
    float t = 0.5;
    float denom = fb - fa;
    if (abs(denom) > 1.0e-6) {
        t = clamp((isoValue - fa) / denom, 0.0, 1.0);
    }
    return mix(a, b, t);
}

int tri_table_value(uint cubeIndex, uint slot) {
    return kTriTable[int(cubeIndex) * 16 + int(slot)];
}

void main() {
    uint packedCellIndex = gl_GlobalInvocationID.x;
    if (packedCellIndex >= pc.candidateCellCount) {
        return;
    }

    triangleCounts[packedCellIndex] = 0u;
    cellCubeIndices[packedCellIndex] = 0u;

    uint cellSlice = uint(pc.cellDimX) * uint(pc.cellDimY);
    uint z = packedCellIndex / cellSlice;
    uint rem = packedCellIndex - z * cellSlice;
    uint y = rem / uint(pc.cellDimX);
    uint x = rem - y * uint(pc.cellDimX);

    float cornerValues[8];
    vec3 cornerPositions[8];
    uint cubeIndex = 0u;
    for (uint corner = 0u; corner < 8u; ++corner) {
        uvec3 gridPosition = uvec3(x, y, z) + uvec3(kCornerOffsets[corner]);
        float cornerValue = voxelScalarField[scalar_index(gridPosition)];
        cornerValues[corner] = cornerValue;
        cornerPositions[corner] = world_position(gridPosition);
        if (cornerValue < pc.surfaceIsoValue) {
            cubeIndex |= (1u << corner);
        }
    }

    cellCubeIndices[packedCellIndex] = cubeIndex;
    if (cubeIndex == 0u || cubeIndex == 255u) {
        return;
    }

    int edgeMask = kEdgeTable[int(cubeIndex)];
    vec3 edgeVertices[12];
    for (int edge = 0; edge < 12; ++edge) {
        if ((edgeMask & (1 << edge)) == 0) {
            continue;
        }
        ivec2 edgeCorners = kEdgeCorners[edge];
        int c0 = edgeCorners.x;
        int c1 = edgeCorners.y;
        edgeVertices[edge] = interpolate_isovalue(
            cornerPositions[c0],
            cornerPositions[c1],
            cornerValues[c0],
            cornerValues[c1],
            pc.surfaceIsoValue
        );
    }

    uint triangleCount = 0u;
    uint baseVertexFloatIndex = packedCellIndex * 45u;
    for (uint slot = 0u; slot < 15u; slot += 3u) {
        int e0 = tri_table_value(cubeIndex, slot + 0u);
        if (e0 < 0) {
            break;
        }
        int e1 = tri_table_value(cubeIndex, slot + 1u);
        int e2 = tri_table_value(cubeIndex, slot + 2u);
        uint outputBase = baseVertexFloatIndex + triangleCount * 9u;
        triangleVertices[outputBase + 0u] = edgeVertices[e0].x;
        triangleVertices[outputBase + 1u] = edgeVertices[e0].y;
        triangleVertices[outputBase + 2u] = edgeVertices[e0].z;
        triangleVertices[outputBase + 3u] = edgeVertices[e1].x;
        triangleVertices[outputBase + 4u] = edgeVertices[e1].y;
        triangleVertices[outputBase + 5u] = edgeVertices[e1].z;
        triangleVertices[outputBase + 6u] = edgeVertices[e2].x;
        triangleVertices[outputBase + 7u] = edgeVertices[e2].y;
        triangleVertices[outputBase + 8u] = edgeVertices[e2].z;
        triangleCount += 1u;
    }

    triangleCounts[packedCellIndex] = triangleCount;
}
)";
  return shader.str();
}

#if defined(_WIN32)
using LibraryHandle = HMODULE;

LibraryHandle load_vulkan_library() { return LoadLibraryW(L"vulkan-1.dll"); }

void close_vulkan_library(LibraryHandle handle) {
  if (handle) {
    FreeLibrary(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return handle ? reinterpret_cast<void *>(GetProcAddress(handle, name))
                : nullptr;
}
#else
using LibraryHandle = void *;

LibraryHandle load_vulkan_library() {
  return dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
}

void close_vulkan_library(LibraryHandle handle) {
  if (handle) {
    dlclose(handle);
  }
}

void *load_symbol(LibraryHandle handle, const char *name) {
  return handle ? dlsym(handle, name) : nullptr;
}
#endif

bool compile_compute_shader_source_spirv(const char *sourceCode,
                                         std::vector<uint32_t> &outSpirv,
                                         std::string &outError) {
  if (!sourceCode || sourceCode[0] == '\0') {
    outError = "empty compute shader source";
    return false;
  }

  if (glslang_initialize_process() == 0) {
    outError = "glslang_initialize_process failed.";
    return false;
  }

  glslang_input_t input{};
  input.language = GLSLANG_SOURCE_GLSL;
  input.stage = GLSLANG_STAGE_COMPUTE;
  input.client = GLSLANG_CLIENT_VULKAN;
  input.client_version = GLSLANG_TARGET_VULKAN_1_0;
  input.target_language = GLSLANG_TARGET_SPV;
  input.target_language_version = GLSLANG_TARGET_SPV_1_0;
  input.code = sourceCode;
  input.default_version = 450;
  input.default_profile = GLSLANG_NO_PROFILE;
  input.force_default_version_and_profile = 0;
  input.forward_compatible = 0;
  input.messages = static_cast<glslang_messages_t>(
      GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
  input.resource = glslang_default_resource();

  glslang_shader_t *shader = glslang_shader_create(&input);
  if (!shader) {
    outError = "glslang_shader_create failed.";
    return false;
  }

  const auto cleanupShader = [&]() { glslang_shader_delete(shader); };

  if (!glslang_shader_preprocess(shader, &input)) {
    outError = glslang_shader_get_info_log(shader);
    cleanupShader();
    return false;
  }

  if (!glslang_shader_parse(shader, &input)) {
    outError = glslang_shader_get_info_log(shader);
    cleanupShader();
    return false;
  }

  glslang_program_t *program = glslang_program_create();
  if (!program) {
    outError = "glslang_program_create failed.";
    cleanupShader();
    return false;
  }

  glslang_program_add_shader(program, shader);
  if (!glslang_program_link(
          program, static_cast<glslang_messages_t>(
                       GLSLANG_MSG_SPV_RULES_BIT |
                       GLSLANG_MSG_VULKAN_RULES_BIT))) {
    outError = glslang_program_get_info_log(program);
    glslang_program_delete(program);
    cleanupShader();
    return false;
  }

  glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
  const char *spirvMessages = glslang_program_SPIRV_get_messages(program);
  if (spirvMessages && spirvMessages[0] != '\0') {
    outError = spirvMessages;
    glslang_program_delete(program);
    cleanupShader();
    return false;
  }

  const size_t wordCount = glslang_program_SPIRV_get_size(program);
  outSpirv.resize(wordCount);
  glslang_program_SPIRV_get(program, outSpirv.data());

  glslang_program_delete(program);
  cleanupShader();
  outError.clear();
  return true;
}

bool compile_compute_shader_spirv(std::vector<uint32_t> &outSpirv,
                                  std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = glslang_initialize_process() != 0;
    if (!initOk) {
      cachedError = "glslang_initialize_process failed.";
      return;
    }

    glslang_input_t input{};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_COMPUTE;
    input.client = GLSLANG_CLIENT_VULKAN;
    input.client_version = GLSLANG_TARGET_VULKAN_1_0;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_0;
    input.code = kParticleComputeShader;
    input.default_version = 450;
    input.default_profile = GLSLANG_NO_PROFILE;
    input.force_default_version_and_profile = 0;
    input.forward_compatible = 0;
    input.messages = static_cast<glslang_messages_t>(
        GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
    input.resource = glslang_default_resource();

    glslang_shader_t *shader = glslang_shader_create(&input);
    if (!shader) {
      cachedError = "glslang_shader_create failed.";
      return;
    }

    const auto cleanupShader = [&]() { glslang_shader_delete(shader); };

    if (!glslang_shader_preprocess(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    if (!glslang_shader_parse(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    glslang_program_t *program = glslang_program_create();
    if (!program) {
      cachedError = "glslang_program_create failed.";
      cleanupShader();
      return;
    }

    glslang_program_add_shader(program, shader);
    if (!glslang_program_link(
            program, static_cast<glslang_messages_t>(
                         GLSLANG_MSG_SPV_RULES_BIT |
                         GLSLANG_MSG_VULKAN_RULES_BIT))) {
      cachedError = glslang_program_get_info_log(program);
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
    const char *spirvMessages = glslang_program_SPIRV_get_messages(program);
    if (spirvMessages && spirvMessages[0] != '\0') {
      cachedError = spirvMessages;
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    const size_t wordCount = glslang_program_SPIRV_get_size(program);
    cachedSpirv.resize(wordCount);
    glslang_program_SPIRV_get(program, cachedSpirv.data());

    glslang_program_delete(program);
    cleanupShader();
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty() ? "unknown glslang shader compilation error"
                                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_voxel_coverage_shader_spirv(std::vector<uint32_t> &outSpirv,
                                         std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = glslang_initialize_process() != 0;
    if (!initOk) {
      cachedError = "glslang_initialize_process failed.";
      return;
    }

    glslang_input_t input{};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_COMPUTE;
    input.client = GLSLANG_CLIENT_VULKAN;
    input.client_version = GLSLANG_TARGET_VULKAN_1_0;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_0;
    input.code = kVoxelCoverageComputeShader;
    input.default_version = 450;
    input.default_profile = GLSLANG_NO_PROFILE;
    input.force_default_version_and_profile = 0;
    input.forward_compatible = 0;
    input.messages = static_cast<glslang_messages_t>(
        GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
    input.resource = glslang_default_resource();

    glslang_shader_t *shader = glslang_shader_create(&input);
    if (!shader) {
      cachedError = "glslang_shader_create failed.";
      return;
    }

    const auto cleanupShader = [&]() { glslang_shader_delete(shader); };

    if (!glslang_shader_preprocess(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    if (!glslang_shader_parse(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    glslang_program_t *program = glslang_program_create();
    if (!program) {
      cachedError = "glslang_program_create failed.";
      cleanupShader();
      return;
    }

    glslang_program_add_shader(program, shader);
    if (!glslang_program_link(
            program, static_cast<glslang_messages_t>(
                         GLSLANG_MSG_SPV_RULES_BIT |
                         GLSLANG_MSG_VULKAN_RULES_BIT))) {
      cachedError = glslang_program_get_info_log(program);
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
    const char *spirvMessages = glslang_program_SPIRV_get_messages(program);
    if (spirvMessages && spirvMessages[0] != '\0') {
      cachedError = spirvMessages;
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    const size_t wordCount = glslang_program_SPIRV_get_size(program);
    cachedSpirv.resize(wordCount);
    glslang_program_SPIRV_get(program, cachedSpirv.data());

    glslang_program_delete(program);
    cleanupShader();
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty() ? "unknown glslang shader compilation error"
                                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_voxel_scalar_field_shader_spirv(std::vector<uint32_t> &outSpirv,
                                             std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = glslang_initialize_process() != 0;
    if (!initOk) {
      cachedError = "glslang_initialize_process failed.";
      return;
    }

    glslang_input_t input{};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_COMPUTE;
    input.client = GLSLANG_CLIENT_VULKAN;
    input.client_version = GLSLANG_TARGET_VULKAN_1_0;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_0;
    input.code = kVoxelScalarFieldComputeShader;
    input.default_version = 450;
    input.default_profile = GLSLANG_NO_PROFILE;
    input.force_default_version_and_profile = 0;
    input.forward_compatible = 0;
    input.messages = static_cast<glslang_messages_t>(
        GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
    input.resource = glslang_default_resource();

    glslang_shader_t *shader = glslang_shader_create(&input);
    if (!shader) {
      cachedError = "glslang_shader_create failed.";
      return;
    }

    const auto cleanupShader = [&]() { glslang_shader_delete(shader); };

    if (!glslang_shader_preprocess(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    if (!glslang_shader_parse(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    glslang_program_t *program = glslang_program_create();
    if (!program) {
      cachedError = "glslang_program_create failed.";
      cleanupShader();
      return;
    }

    glslang_program_add_shader(program, shader);
    if (!glslang_program_link(
            program, static_cast<glslang_messages_t>(
                         GLSLANG_MSG_SPV_RULES_BIT |
                         GLSLANG_MSG_VULKAN_RULES_BIT))) {
      cachedError = glslang_program_get_info_log(program);
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
    const char *spirvMessages = glslang_program_SPIRV_get_messages(program);
    if (spirvMessages && spirvMessages[0] != '\0') {
      cachedError = spirvMessages;
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    const size_t wordCount = glslang_program_SPIRV_get_size(program);
    cachedSpirv.resize(wordCount);
    glslang_program_SPIRV_get(program, cachedSpirv.data());

    glslang_program_delete(program);
    cleanupShader();
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty() ? "unknown glslang shader compilation error"
                                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_zhu_bridson_scalar_field_shader_spirv(
    std::vector<uint32_t> &outSpirv, std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = compile_compute_shader_source_spirv(
        kZhuBridsonScalarFieldComputeShader, cachedSpirv, cachedError);
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty() ? "unknown Zhu-Bridson shader compilation error"
                                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_surface_cell_classify_shader_spirv(
    std::vector<uint32_t> &outSpirv, std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = glslang_initialize_process() != 0;
    if (!initOk) {
      cachedError = "glslang_initialize_process failed.";
      return;
    }

    glslang_input_t input{};
    input.language = GLSLANG_SOURCE_GLSL;
    input.stage = GLSLANG_STAGE_COMPUTE;
    input.client = GLSLANG_CLIENT_VULKAN;
    input.client_version = GLSLANG_TARGET_VULKAN_1_0;
    input.target_language = GLSLANG_TARGET_SPV;
    input.target_language_version = GLSLANG_TARGET_SPV_1_0;
    input.code = kSurfaceCellClassifyComputeShader;
    input.default_version = 450;
    input.default_profile = GLSLANG_NO_PROFILE;
    input.force_default_version_and_profile = 0;
    input.forward_compatible = 0;
    input.messages = static_cast<glslang_messages_t>(
        GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT);
    input.resource = glslang_default_resource();

    glslang_shader_t *shader = glslang_shader_create(&input);
    if (!shader) {
      cachedError = "glslang_shader_create failed.";
      return;
    }

    const auto cleanupShader = [&]() { glslang_shader_delete(shader); };

    if (!glslang_shader_preprocess(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    if (!glslang_shader_parse(shader, &input)) {
      cachedError = glslang_shader_get_info_log(shader);
      cleanupShader();
      return;
    }

    glslang_program_t *program = glslang_program_create();
    if (!program) {
      cachedError = "glslang_program_create failed.";
      cleanupShader();
      return;
    }

    glslang_program_add_shader(program, shader);
    if (!glslang_program_link(
            program, static_cast<glslang_messages_t>(
                         GLSLANG_MSG_SPV_RULES_BIT |
                         GLSLANG_MSG_VULKAN_RULES_BIT))) {
      cachedError = glslang_program_get_info_log(program);
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    glslang_program_SPIRV_generate(program, GLSLANG_STAGE_COMPUTE);
    const char *spirvMessages = glslang_program_SPIRV_get_messages(program);
    if (spirvMessages && spirvMessages[0] != '\0') {
      cachedError = spirvMessages;
      glslang_program_delete(program);
      cleanupShader();
      return;
    }

    const size_t wordCount = glslang_program_SPIRV_get_size(program);
    cachedSpirv.resize(wordCount);
    glslang_program_SPIRV_get(program, cachedSpirv.data());

    glslang_program_delete(program);
    cleanupShader();
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty() ? "unknown glslang shader compilation error"
                                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_surface_mesh_shader_spirv(std::vector<uint32_t> &outSpirv,
                                       std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    const std::string shaderSource = build_surface_mesh_compute_shader_source();
    initOk = compile_compute_shader_source_spirv(shaderSource.c_str(),
                                                 cachedSpirv, cachedError);
    if (!initOk && cachedError.empty()) {
      cachedError = "unknown Vulkan surface mesh shader compilation error";
    }
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty()
                   ? "unknown Vulkan surface mesh shader compilation error"
                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_active_voxel_compact_shader_spirv(
    std::vector<uint32_t> &outSpirv, std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = compile_compute_shader_source_spirv(
        kActiveVoxelCompactComputeShader, cachedSpirv, cachedError);
    if (!initOk && cachedError.empty()) {
      cachedError =
          "unknown Vulkan active-voxel compaction shader compilation error";
    }
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty()
                   ? "unknown Vulkan active-voxel compaction shader compilation error"
                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_surface_cell_compact_shader_spirv(
    std::vector<uint32_t> &outSpirv, std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    initOk = compile_compute_shader_source_spirv(
        kSurfaceCellCompactComputeShader, cachedSpirv, cachedError);
    if (!initOk && cachedError.empty()) {
      cachedError =
          "unknown Vulkan surface-cell compaction shader compilation error";
    }
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty()
                   ? "unknown Vulkan surface-cell compaction shader compilation error"
                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool compile_surface_triangle_compact_shader_spirv(
    std::vector<uint32_t> &outSpirv, std::string &outError) {
  static std::once_flag initOnce;
  static bool initOk = false;
  static std::vector<uint32_t> cachedSpirv;
  static std::string cachedError;

  std::call_once(initOnce, []() {
    const std::string shaderSource =
        build_surface_triangle_compact_compute_shader_source();
    initOk = compile_compute_shader_source_spirv(shaderSource.c_str(),
                                                 cachedSpirv, cachedError);
    if (!initOk && cachedError.empty()) {
      cachedError =
          "unknown Vulkan surface-triangle compaction shader compilation error";
    }
  });

  if (!initOk || cachedSpirv.empty()) {
    outError = cachedError.empty()
                   ? "unknown Vulkan surface-triangle compaction shader compilation error"
                   : cachedError;
    return false;
  }

  outSpirv = cachedSpirv;
  outError.clear();
  return true;
}

bool find_memory_type_index(const VkPhysicalDeviceMemoryProperties &properties,
                            uint32_t memoryTypeBits,
                            VkMemoryPropertyFlags requiredFlags,
                            uint32_t &outIndex) {
  for (uint32_t i = 0; i < properties.memoryTypeCount; ++i) {
    const bool supportedByBuffer = (memoryTypeBits & (1u << i)) != 0;
    const bool hasRequiredFlags =
        (properties.memoryTypes[i].propertyFlags & requiredFlags) ==
        requiredFlags;
    if (supportedByBuffer && hasRequiredFlags) {
      outIndex = i;
      return true;
    }
  }
  return false;
}

float compute_vulkan_outside_field(
    const VulkanParticleComputeSettings &settings) {
  return (settings.fieldMode == VulkanScalarFieldMode::metaball ||
          settings.fieldMode == VulkanScalarFieldMode::plain_marching_cubes)
             ? std::max(settings.fieldThreshold, settings.surfaceIsoValue + 1.0e-3f)
             : std::max(settings.fieldThreshold,
                        std::max(settings.voxelLength, 1.0f));
}

bool reduce_particle_voxel_bounds(const std::vector<int32_t> &minBounds,
                                  const std::vector<int32_t> &maxBounds,
                                  std::size_t particleCount,
                                  int32_t &outMinX, int32_t &outMinY,
                                  int32_t &outMinZ, int32_t &outMaxX,
                                  int32_t &outMaxY, int32_t &outMaxZ) {
  if (minBounds.size() != particleCount * 4 ||
      maxBounds.size() != particleCount * 4 || particleCount == 0) {
    return false;
  }

  outMinX = minBounds[0];
  outMinY = minBounds[1];
  outMinZ = minBounds[2];
  outMaxX = maxBounds[0];
  outMaxY = maxBounds[1];
  outMaxZ = maxBounds[2];

  for (std::size_t i = 1; i < particleCount; ++i) {
    const std::size_t offset = i * 4;
    outMinX = std::min(outMinX, minBounds[offset + 0]);
    outMinY = std::min(outMinY, minBounds[offset + 1]);
    outMinZ = std::min(outMinZ, minBounds[offset + 2]);
    outMaxX = std::max(outMaxX, maxBounds[offset + 0]);
    outMaxY = std::max(outMaxY, maxBounds[offset + 1]);
    outMaxZ = std::max(outMaxZ, maxBounds[offset + 2]);
  }

  return true;
}

constexpr uint64_t kMaxVoxelCoverageCells = 8ull * 1024ull * 1024ull;
constexpr uint64_t kMaxVoxelScalarFieldCells =
    static_cast<uint64_t>(65535ull) * 64ull;
constexpr uint64_t kMaxVoxelParticleDistanceEvaluations =
    256ull * 1024ull * 1024ull;
constexpr uint64_t kMaxCompactedVoxelParticlePairs =
    32ull * 1024ull * 1024ull;

bool build_active_voxel_particle_lists(
    std::size_t particleCount, const std::vector<int32_t> &minBounds,
    const std::vector<int32_t> &maxBoundsExclusive, int32_t domainMinX,
    int32_t domainMinY, int32_t domainMinZ, int32_t domainDimX,
    int32_t domainDimY, int32_t domainDimZ,
    const std::vector<uint32_t> &activeVoxelIndices, uint64_t maxPairCount,
    std::vector<uint32_t> &outOffsets, std::vector<uint32_t> &outIndices,
    uint64_t &outPairCount, std::vector<int32_t> *outActiveVoxelLookup) {
  struct ActiveVoxelRowRange {
    uint32_t begin = std::numeric_limits<uint32_t>::max();
    uint32_t end = std::numeric_limits<uint32_t>::max();
  };

  outOffsets.clear();
  outIndices.clear();
  outPairCount = 0;

  if (particleCount == 0 || activeVoxelIndices.empty() || domainDimX <= 0 ||
      domainDimY <= 0 || domainDimZ <= 0 ||
      minBounds.size() != particleCount * 4 ||
      maxBoundsExclusive.size() != particleCount * 4) {
    if (activeVoxelIndices.empty()) {
      outOffsets.assign(1, 0);
      if (outActiveVoxelLookup) {
        outActiveVoxelLookup->clear();
      }
      return true;
    }
    return false;
  }

  const std::size_t voxelCount = static_cast<std::size_t>(domainDimX) *
                                 static_cast<std::size_t>(domainDimY) *
                                 static_cast<std::size_t>(domainDimZ);
  const std::size_t rowCount =
      static_cast<std::size_t>(domainDimY) * static_cast<std::size_t>(domainDimZ);
  std::vector<ActiveVoxelRowRange> activeVoxelRowRanges(rowCount);
  if (outActiveVoxelLookup) {
    outActiveVoxelLookup->assign(voxelCount, -1);
  }
  for (std::size_t i = 0; i < activeVoxelIndices.size(); ++i) {
    const uint32_t voxelIndex = activeVoxelIndices[i];
    if (voxelIndex >= voxelCount) {
      continue;
    }
    if (outActiveVoxelLookup) {
      (*outActiveVoxelLookup)[voxelIndex] = static_cast<int32_t>(i);
    }
    const std::size_t rowIndex =
        static_cast<std::size_t>(voxelIndex / static_cast<uint32_t>(domainDimX));
    ActiveVoxelRowRange &rowRange = activeVoxelRowRanges[rowIndex];
    if (rowRange.begin == std::numeric_limits<uint32_t>::max()) {
      rowRange.begin = static_cast<uint32_t>(i);
    }
    rowRange.end = static_cast<uint32_t>(i + 1);
  }

  outOffsets.assign(activeVoxelIndices.size() + 1, 0u);
  const int32_t sliceStride = domainDimX * domainDimY;

  for (std::size_t particleIndex = 0; particleIndex < particleCount;
       ++particleIndex) {
    const std::size_t offset = particleIndex * 4;
    const int32_t minX =
        std::max<int32_t>(minBounds[offset + 0] - domainMinX, 0);
    const int32_t minY =
        std::max<int32_t>(minBounds[offset + 1] - domainMinY, 0);
    const int32_t minZ =
        std::max<int32_t>(minBounds[offset + 2] - domainMinZ, 0);
    const int32_t maxX =
        std::min<int32_t>(maxBoundsExclusive[offset + 0] - domainMinX, domainDimX);
    const int32_t maxY =
        std::min<int32_t>(maxBoundsExclusive[offset + 1] - domainMinY, domainDimY);
    const int32_t maxZ =
        std::min<int32_t>(maxBoundsExclusive[offset + 2] - domainMinZ, domainDimZ);
    if (minX >= maxX || minY >= maxY || minZ >= maxZ) {
      continue;
    }

    for (int32_t z = minZ; z < maxZ; ++z) {
      for (int32_t y = minY; y < maxY; ++y) {
        const std::size_t rowIndex =
            static_cast<std::size_t>(z) * static_cast<std::size_t>(domainDimY) +
            static_cast<std::size_t>(y);
        const ActiveVoxelRowRange &rowRange = activeVoxelRowRanges[rowIndex];
        if (rowRange.begin == std::numeric_limits<uint32_t>::max()) {
          continue;
        }

        const uint32_t packedRowMin = static_cast<uint32_t>(
            z * sliceStride + y * domainDimX + minX);
        const uint32_t packedRowMax =
            static_cast<uint32_t>(z * sliceStride + y * domainDimX + maxX);
        auto rowBeginIt =
            activeVoxelIndices.begin() + static_cast<std::ptrdiff_t>(rowRange.begin);
        auto rowEndIt =
            activeVoxelIndices.begin() + static_cast<std::ptrdiff_t>(rowRange.end);
        auto lowerIt = std::lower_bound(rowBeginIt, rowEndIt, packedRowMin);
        auto upperIt = std::lower_bound(lowerIt, rowEndIt, packedRowMax);
        std::size_t compactIndex =
            static_cast<std::size_t>(lowerIt - activeVoxelIndices.begin());
        for (auto it = lowerIt; it != upperIt; ++it, ++compactIndex) {
          ++outOffsets[compactIndex + 1];
          ++outPairCount;
          if (outPairCount > maxPairCount ||
              outPairCount > std::numeric_limits<uint32_t>::max()) {
            outOffsets.clear();
            outIndices.clear();
            outPairCount = 0;
            if (outActiveVoxelLookup) {
              outActiveVoxelLookup->clear();
            }
            return false;
          }
        }
      }
    }
  }

  for (std::size_t i = 1; i < outOffsets.size(); ++i) {
    outOffsets[i] += outOffsets[i - 1];
  }

  outIndices.assign(static_cast<std::size_t>(outPairCount), 0u);
  std::vector<uint32_t> writeOffsets = outOffsets;

  for (std::size_t particleIndex = 0; particleIndex < particleCount;
       ++particleIndex) {
    const std::size_t offset = particleIndex * 4;
    const int32_t minX =
        std::max<int32_t>(minBounds[offset + 0] - domainMinX, 0);
    const int32_t minY =
        std::max<int32_t>(minBounds[offset + 1] - domainMinY, 0);
    const int32_t minZ =
        std::max<int32_t>(minBounds[offset + 2] - domainMinZ, 0);
    const int32_t maxX =
        std::min<int32_t>(maxBoundsExclusive[offset + 0] - domainMinX, domainDimX);
    const int32_t maxY =
        std::min<int32_t>(maxBoundsExclusive[offset + 1] - domainMinY, domainDimY);
    const int32_t maxZ =
        std::min<int32_t>(maxBoundsExclusive[offset + 2] - domainMinZ, domainDimZ);
    if (minX >= maxX || minY >= maxY || minZ >= maxZ) {
      continue;
    }

    for (int32_t z = minZ; z < maxZ; ++z) {
      for (int32_t y = minY; y < maxY; ++y) {
        const std::size_t rowIndex =
            static_cast<std::size_t>(z) * static_cast<std::size_t>(domainDimY) +
            static_cast<std::size_t>(y);
        const ActiveVoxelRowRange &rowRange = activeVoxelRowRanges[rowIndex];
        if (rowRange.begin == std::numeric_limits<uint32_t>::max()) {
          continue;
        }

        const uint32_t packedRowMin = static_cast<uint32_t>(
            z * sliceStride + y * domainDimX + minX);
        const uint32_t packedRowMax =
            static_cast<uint32_t>(z * sliceStride + y * domainDimX + maxX);
        auto rowBeginIt =
            activeVoxelIndices.begin() + static_cast<std::ptrdiff_t>(rowRange.begin);
        auto rowEndIt =
            activeVoxelIndices.begin() + static_cast<std::ptrdiff_t>(rowRange.end);
        auto lowerIt = std::lower_bound(rowBeginIt, rowEndIt, packedRowMin);
        auto upperIt = std::lower_bound(lowerIt, rowEndIt, packedRowMax);
        std::size_t compactIndex =
            static_cast<std::size_t>(lowerIt - activeVoxelIndices.begin());
        for (auto it = lowerIt; it != upperIt; ++it, ++compactIndex) {
          const std::size_t writeIndex =
              static_cast<std::size_t>(writeOffsets[compactIndex]++);
          outIndices[writeIndex] = static_cast<uint32_t>(particleIndex);
        }
      }
    }
  }

  return true;
}

bool build_active_voxel_lookup_and_offsets_from_dense_counts(
    const std::vector<uint32_t> &activeVoxelIndices,
    const uint32_t *voxelCoverageCounts, std::size_t voxelCoverageCount,
    int32_t domainDimX, int32_t domainDimY, int32_t domainDimZ,
    uint64_t maxPairCount, std::vector<int32_t> &outActiveVoxelLookup,
    std::vector<uint32_t> &outOffsets, uint64_t &outPairCount) {
  outActiveVoxelLookup.clear();
  outOffsets.clear();
  outPairCount = 0;

  if (domainDimX <= 0 || domainDimY <= 0 || domainDimZ <= 0) {
    return false;
  }

  const std::size_t expectedVoxelCount =
      static_cast<std::size_t>(domainDimX) * static_cast<std::size_t>(domainDimY) *
      static_cast<std::size_t>(domainDimZ);
  if (voxelCoverageCounts == nullptr || voxelCoverageCount != expectedVoxelCount ||
      expectedVoxelCount == 0u) {
    return false;
  }

  outActiveVoxelLookup.assign(expectedVoxelCount, -1);
  outOffsets.assign(activeVoxelIndices.size() + 1u, 0u);
  uint64_t runningPairCount = 0ull;
  for (std::size_t i = 0; i < activeVoxelIndices.size(); ++i) {
    const uint32_t voxelIndex = activeVoxelIndices[i];
    if (voxelIndex >= expectedVoxelCount) {
      outActiveVoxelLookup.clear();
      outOffsets.clear();
      outPairCount = 0;
      return false;
    }

    outActiveVoxelLookup[voxelIndex] = static_cast<int32_t>(i);
    outOffsets[i] = static_cast<uint32_t>(runningPairCount);
    runningPairCount += static_cast<uint64_t>(voxelCoverageCounts[voxelIndex]);
    if (runningPairCount > maxPairCount ||
        runningPairCount > std::numeric_limits<uint32_t>::max()) {
      outActiveVoxelLookup.clear();
      outOffsets.clear();
      outPairCount = 0;
      return false;
    }
  }

  outOffsets[activeVoxelIndices.size()] = static_cast<uint32_t>(runningPairCount);
  outPairCount = runningPairCount;
  return true;
}

bool build_active_voxel_lookup_and_offsets(
    const std::vector<uint32_t> &activeVoxelIndices,
    const std::vector<uint32_t> &voxelCoverageCounts, int32_t domainDimX,
    int32_t domainDimY, int32_t domainDimZ, uint64_t maxPairCount,
    std::vector<int32_t> &outActiveVoxelLookup,
    std::vector<uint32_t> &outOffsets, uint64_t &outPairCount) {
  return build_active_voxel_lookup_and_offsets_from_dense_counts(
      activeVoxelIndices, voxelCoverageCounts.data(), voxelCoverageCounts.size(),
      domainDimX, domainDimY, domainDimZ, maxPairCount, outActiveVoxelLookup,
      outOffsets, outPairCount);
}

void filter_boundary_active_voxels(
    const std::vector<uint32_t> &inputActiveVoxelIndices,
    const std::vector<uint32_t> &voxelCoverageCounts, int32_t domainDimX,
    int32_t domainDimY, int32_t domainDimZ,
    std::vector<uint32_t> &outBoundaryVoxelIndices, uint32_t &outMaxCoverage,
    uint64_t &outCoveredParticleVoxelPairs) {
  outBoundaryVoxelIndices.clear();
  outMaxCoverage = 0u;
  outCoveredParticleVoxelPairs = 0ull;

  if (domainDimX <= 0 || domainDimY <= 0 || domainDimZ <= 0) {
    return;
  }

  const std::size_t voxelCount = static_cast<std::size_t>(domainDimX) *
                                 static_cast<std::size_t>(domainDimY) *
                                 static_cast<std::size_t>(domainDimZ);
  if (voxelCoverageCounts.size() != voxelCount || inputActiveVoxelIndices.empty()) {
    return;
  }

  const uint32_t sliceStride =
      static_cast<uint32_t>(domainDimX * domainDimY);
  outBoundaryVoxelIndices.reserve(inputActiveVoxelIndices.size());
  for (uint32_t packedVoxelIndex : inputActiveVoxelIndices) {
    if (packedVoxelIndex >= voxelCount ||
        voxelCoverageCounts[packedVoxelIndex] == 0u) {
      continue;
    }

    const int32_t z =
        static_cast<int32_t>(packedVoxelIndex / sliceStride);
    const uint32_t rem = packedVoxelIndex - static_cast<uint32_t>(z) * sliceStride;
    const int32_t y =
        static_cast<int32_t>(rem / static_cast<uint32_t>(domainDimX));
    const int32_t x =
        static_cast<int32_t>(rem -
                             static_cast<uint32_t>(y) *
                                 static_cast<uint32_t>(domainDimX));

    const auto is_uncovered = [&](int32_t nx, int32_t ny, int32_t nz) {
      if (nx < 0 || ny < 0 || nz < 0 || nx >= domainDimX || ny >= domainDimY ||
          nz >= domainDimZ) {
        return true;
      }
      const std::size_t neighborIndex =
          static_cast<std::size_t>(nx) +
          static_cast<std::size_t>(domainDimX) *
              (static_cast<std::size_t>(ny) +
               static_cast<std::size_t>(domainDimY) *
                   static_cast<std::size_t>(nz));
      return voxelCoverageCounts[neighborIndex] == 0u;
    };

    if (is_uncovered(x - 1, y, z) || is_uncovered(x + 1, y, z) ||
        is_uncovered(x, y - 1, z) || is_uncovered(x, y + 1, z) ||
        is_uncovered(x, y, z - 1) || is_uncovered(x, y, z + 1)) {
      const uint32_t count = voxelCoverageCounts[packedVoxelIndex];
      outBoundaryVoxelIndices.push_back(packedVoxelIndex);
      outMaxCoverage = std::max(outMaxCoverage, count);
      outCoveredParticleVoxelPairs += static_cast<uint64_t>(count);
    }
  }
}

struct VulkanSharedContext {
  struct HostBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
    void *mapped = nullptr;
  };

  struct DeviceBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
  };

  std::mutex mutex;
  bool initializationAttempted = false;
  bool initialized = false;
  std::string initError;
  LibraryHandle library = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
  VkPhysicalDeviceMemoryProperties memoryProperties{};
  uint32_t queueFamilyIndex = UINT32_MAX;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout particleDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout particlePipelineLayout = VK_NULL_HANDLE;
  VkPipeline particlePipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout coverageDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout coveragePipelineLayout = VK_NULL_HANDLE;
  VkPipeline coveragePipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout activeVoxelCompactDescriptorSetLayout =
      VK_NULL_HANDLE;
  VkPipelineLayout activeVoxelCompactPipelineLayout = VK_NULL_HANDLE;
  VkPipeline activeVoxelCompactPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout activeVoxelPairFillDescriptorSetLayout =
      VK_NULL_HANDLE;
  VkPipelineLayout activeVoxelPairFillPipelineLayout = VK_NULL_HANDLE;
  VkPipeline activeVoxelPairFillPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout scalarFieldDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout scalarFieldPipelineLayout = VK_NULL_HANDLE;
  VkPipeline scalarFieldPipeline = VK_NULL_HANDLE;
  VkPipeline zhuScalarFieldPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout surfaceCellDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout surfaceCellPipelineLayout = VK_NULL_HANDLE;
  VkPipeline surfaceCellPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout sparseSurfaceCellDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout sparseSurfaceCellPipelineLayout = VK_NULL_HANDLE;
  VkPipeline sparseSurfaceCellPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout surfaceCellCompactDescriptorSetLayout =
      VK_NULL_HANDLE;
  VkPipelineLayout surfaceCellCompactPipelineLayout = VK_NULL_HANDLE;
  VkPipeline surfaceCellCompactPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout surfaceTriangleCompactDescriptorSetLayout =
      VK_NULL_HANDLE;
  VkPipelineLayout surfaceTriangleCompactPipelineLayout = VK_NULL_HANDLE;
  VkPipeline surfaceTriangleCompactPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout surfaceMeshDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout surfaceMeshPipelineLayout = VK_NULL_HANDLE;
  VkPipeline surfaceMeshPipeline = VK_NULL_HANDLE;
  VkDescriptorSetLayout denseSurfaceMeshDescriptorSetLayout = VK_NULL_HANDLE;
  VkPipelineLayout denseSurfaceMeshPipelineLayout = VK_NULL_HANDLE;
  VkPipeline denseSurfaceMeshPipeline = VK_NULL_HANDLE;
  VkDescriptorPool computeWorkDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet particleDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet coverageDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet activeVoxelCompactDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet activeVoxelPairFillDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet scalarFieldDescriptorSet = VK_NULL_HANDLE;
  VkCommandBuffer particleCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer coverageCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer activeVoxelCompactCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer activeVoxelPairFillCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer scalarFieldCommandBuffer = VK_NULL_HANDLE;
  VkDescriptorPool surfaceWorkDescriptorPool = VK_NULL_HANDLE;
  VkDescriptorSet surfaceCellDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet sparseSurfaceCellDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet surfaceCellCompactDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet surfaceTriangleCompactDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet surfaceMeshDescriptorSet = VK_NULL_HANDLE;
  VkDescriptorSet denseSurfaceMeshDescriptorSet = VK_NULL_HANDLE;
  VkCommandBuffer surfaceCellCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer sparseSurfaceCellCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer surfaceCellCompactCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer surfaceTriangleCompactCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer surfaceMeshCommandBuffer = VK_NULL_HANDLE;
  VkCommandBuffer denseSurfaceMeshCommandBuffer = VK_NULL_HANDLE;
  HostBuffer particleInputBuffer;
  HostBuffer particleOutputBuffer;
  HostBuffer particleVelocityBuffer;
  HostBuffer particleMinBoundsBuffer;
  HostBuffer particleMaxBoundsBuffer;
  HostBuffer voxelCoverageBuffer;
  HostBuffer activeVoxelIndexBuffer;
  HostBuffer activeVoxelStatsBuffer;
  HostBuffer activeVoxelCompactLookupBuffer;
  HostBuffer activeVoxelParticleOffsetBuffer;
  HostBuffer activeVoxelParticleCursorBuffer;
  HostBuffer activeVoxelParticleIndexBuffer;
  HostBuffer voxelScalarFieldBuffer;
  HostBuffer candidateCellIndexBuffer;
  HostBuffer candidateCellCubeIndexBuffer;
  HostBuffer denseSurfaceCellCubeIndexBuffer;
  HostBuffer surfaceCellStatsBuffer;
  HostBuffer surfaceTriangleCountBuffer;
  HostBuffer surfaceTriangleVertexBuffer;
  bool residentSurfaceTriangleVerticesCompacted = false;
  DeviceBuffer deviceActiveVoxelCompactLookupBuffer;
  DeviceBuffer deviceActiveVoxelParticleOffsetBuffer;
  DeviceBuffer deviceActiveVoxelParticleCursorBuffer;
  DeviceBuffer deviceActiveVoxelParticleIndexBuffer;
  DeviceBuffer deviceCandidateCellIndexBuffer;
  DeviceBuffer deviceCandidateCellCubeIndexBuffer;
  DeviceBuffer deviceDenseSurfaceCellCubeIndexBuffer;
  DeviceBuffer deviceSurfaceCellStatsBuffer;
  DeviceBuffer deviceSurfaceTriangleCountBuffer;
  DeviceBuffer deviceSurfaceTriangleVertexBuffer;

  PFN_vkDestroyInstance vkDestroyInstance = nullptr;
  PFN_vkDestroyDevice vkDestroyDevice = nullptr;
  PFN_vkDestroyBuffer vkDestroyBuffer = nullptr;
  PFN_vkFreeMemory vkFreeMemory = nullptr;
  PFN_vkDestroyDescriptorSetLayout vkDestroyDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorPool vkDestroyDescriptorPool = nullptr;
  PFN_vkDestroyShaderModule vkDestroyShaderModule = nullptr;
  PFN_vkDestroyPipelineLayout vkDestroyPipelineLayout = nullptr;
  PFN_vkDestroyPipeline vkDestroyPipeline = nullptr;
  PFN_vkDestroyCommandPool vkDestroyCommandPool = nullptr;
  PFN_vkDeviceWaitIdle vkDeviceWaitIdle = nullptr;

  PFN_vkCreateBuffer vkCreateBuffer = nullptr;
  PFN_vkGetBufferMemoryRequirements vkGetBufferMemoryRequirements = nullptr;
  PFN_vkAllocateMemory vkAllocateMemory = nullptr;
  PFN_vkBindBufferMemory vkBindBufferMemory = nullptr;
  PFN_vkMapMemory vkMapMemory = nullptr;
  PFN_vkUnmapMemory vkUnmapMemory = nullptr;
  PFN_vkCreateDescriptorSetLayout vkCreateDescriptorSetLayout = nullptr;
  PFN_vkCreateDescriptorPool vkCreateDescriptorPool = nullptr;
  PFN_vkAllocateDescriptorSets vkAllocateDescriptorSets = nullptr;
  PFN_vkUpdateDescriptorSets vkUpdateDescriptorSets = nullptr;
  PFN_vkCreateShaderModule vkCreateShaderModule = nullptr;
  PFN_vkCreatePipelineLayout vkCreatePipelineLayout = nullptr;
  PFN_vkCreateComputePipelines vkCreateComputePipelines = nullptr;
  PFN_vkAllocateCommandBuffers vkAllocateCommandBuffers = nullptr;
  PFN_vkBeginCommandBuffer vkBeginCommandBuffer = nullptr;
  PFN_vkCmdBindPipeline vkCmdBindPipeline = nullptr;
  PFN_vkCmdBindDescriptorSets vkCmdBindDescriptorSets = nullptr;
  PFN_vkCmdPushConstants vkCmdPushConstants = nullptr;
  PFN_vkCmdDispatch vkCmdDispatch = nullptr;
  PFN_vkCmdPipelineBarrier vkCmdPipelineBarrier = nullptr;
  PFN_vkCmdCopyBuffer vkCmdCopyBuffer = nullptr;
  PFN_vkCmdFillBuffer vkCmdFillBuffer = nullptr;
  PFN_vkEndCommandBuffer vkEndCommandBuffer = nullptr;
  PFN_vkQueueSubmit vkQueueSubmit = nullptr;
  PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
  PFN_vkResetCommandPool vkResetCommandPool = nullptr;

  void release_host_buffer(HostBuffer &hostBuffer) {
    if (device != VK_NULL_HANDLE && hostBuffer.mapped != nullptr &&
        vkUnmapMemory && hostBuffer.memory != VK_NULL_HANDLE) {
      vkUnmapMemory(device, hostBuffer.memory);
    }
    if (device != VK_NULL_HANDLE && hostBuffer.buffer != VK_NULL_HANDLE &&
        vkDestroyBuffer) {
      vkDestroyBuffer(device, hostBuffer.buffer, nullptr);
    }
    if (device != VK_NULL_HANDLE && hostBuffer.memory != VK_NULL_HANDLE &&
        vkFreeMemory) {
      vkFreeMemory(device, hostBuffer.memory, nullptr);
    }
    hostBuffer.buffer = VK_NULL_HANDLE;
    hostBuffer.memory = VK_NULL_HANDLE;
    hostBuffer.capacity = 0;
    hostBuffer.mapped = nullptr;
  }

  void release_device_buffer(DeviceBuffer &deviceBuffer) {
    if (device != VK_NULL_HANDLE && deviceBuffer.buffer != VK_NULL_HANDLE &&
        vkDestroyBuffer) {
      vkDestroyBuffer(device, deviceBuffer.buffer, nullptr);
    }
    if (device != VK_NULL_HANDLE && deviceBuffer.memory != VK_NULL_HANDLE &&
        vkFreeMemory) {
      vkFreeMemory(device, deviceBuffer.memory, nullptr);
    }
    deviceBuffer.buffer = VK_NULL_HANDLE;
    deviceBuffer.memory = VK_NULL_HANDLE;
    deviceBuffer.capacity = 0;
  }

  void release() {
    if (device != VK_NULL_HANDLE) {
      if (vkDeviceWaitIdle) {
        vkDeviceWaitIdle(device);
      }
      release_host_buffer(voxelScalarFieldBuffer);
      release_host_buffer(activeVoxelParticleIndexBuffer);
      release_host_buffer(activeVoxelParticleCursorBuffer);
    release_host_buffer(activeVoxelParticleOffsetBuffer);
    release_host_buffer(activeVoxelCompactLookupBuffer);
    release_host_buffer(activeVoxelStatsBuffer);
    release_host_buffer(activeVoxelIndexBuffer);
      release_host_buffer(voxelCoverageBuffer);
      release_host_buffer(surfaceCellStatsBuffer);
      release_host_buffer(denseSurfaceCellCubeIndexBuffer);
      release_host_buffer(surfaceTriangleVertexBuffer);
      release_host_buffer(surfaceTriangleCountBuffer);
      release_device_buffer(deviceSurfaceTriangleVertexBuffer);
      release_device_buffer(deviceSurfaceTriangleCountBuffer);
      release_device_buffer(deviceSurfaceCellStatsBuffer);
      release_device_buffer(deviceDenseSurfaceCellCubeIndexBuffer);
      release_device_buffer(deviceCandidateCellCubeIndexBuffer);
      release_device_buffer(deviceCandidateCellIndexBuffer);
      release_device_buffer(deviceActiveVoxelParticleIndexBuffer);
      release_device_buffer(deviceActiveVoxelParticleCursorBuffer);
      release_device_buffer(deviceActiveVoxelParticleOffsetBuffer);
      release_device_buffer(deviceActiveVoxelCompactLookupBuffer);
      release_host_buffer(candidateCellCubeIndexBuffer);
      release_host_buffer(candidateCellIndexBuffer);
      release_host_buffer(particleMaxBoundsBuffer);
      release_host_buffer(particleMinBoundsBuffer);
      release_host_buffer(particleVelocityBuffer);
      release_host_buffer(particleOutputBuffer);
      release_host_buffer(particleInputBuffer);
      if (particlePipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, particlePipeline, nullptr);
      }
      if (coveragePipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, coveragePipeline, nullptr);
      }
      if (activeVoxelCompactPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, activeVoxelCompactPipeline, nullptr);
      }
      if (activeVoxelPairFillPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, activeVoxelPairFillPipeline, nullptr);
      }
      if (scalarFieldPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, scalarFieldPipeline, nullptr);
      }
      if (zhuScalarFieldPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, zhuScalarFieldPipeline, nullptr);
      }
      if (surfaceCellPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, surfaceCellPipeline, nullptr);
      }
      if (sparseSurfaceCellPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, sparseSurfaceCellPipeline, nullptr);
      }
      if (surfaceCellCompactPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, surfaceCellCompactPipeline, nullptr);
      }
      if (surfaceTriangleCompactPipeline != VK_NULL_HANDLE &&
          vkDestroyPipeline) {
        vkDestroyPipeline(device, surfaceTriangleCompactPipeline, nullptr);
      }
      if (surfaceMeshPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, surfaceMeshPipeline, nullptr);
      }
      if (denseSurfaceMeshPipeline != VK_NULL_HANDLE && vkDestroyPipeline) {
        vkDestroyPipeline(device, denseSurfaceMeshPipeline, nullptr);
      }
      if (particlePipelineLayout != VK_NULL_HANDLE && vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, particlePipelineLayout, nullptr);
      }
      if (coveragePipelineLayout != VK_NULL_HANDLE && vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, coveragePipelineLayout, nullptr);
      }
      if (activeVoxelCompactPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, activeVoxelCompactPipelineLayout,
                                nullptr);
      }
      if (activeVoxelPairFillPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, activeVoxelPairFillPipelineLayout,
                                nullptr);
      }
      if (scalarFieldPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, scalarFieldPipelineLayout, nullptr);
      }
      if (surfaceCellPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, surfaceCellPipelineLayout, nullptr);
      }
      if (sparseSurfaceCellPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, sparseSurfaceCellPipelineLayout,
                                nullptr);
      }
      if (surfaceCellCompactPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, surfaceCellCompactPipelineLayout,
                                nullptr);
      }
      if (surfaceTriangleCompactPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, surfaceTriangleCompactPipelineLayout,
                                nullptr);
      }
      if (surfaceMeshPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, surfaceMeshPipelineLayout, nullptr);
      }
      if (denseSurfaceMeshPipelineLayout != VK_NULL_HANDLE &&
          vkDestroyPipelineLayout) {
        vkDestroyPipelineLayout(device, denseSurfaceMeshPipelineLayout, nullptr);
      }
      if (particleDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, particleDescriptorSetLayout,
                                     nullptr);
      }
      if (coverageDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, coverageDescriptorSetLayout,
                                     nullptr);
      }
      if (activeVoxelCompactDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     activeVoxelCompactDescriptorSetLayout,
                                     nullptr);
      }
      if (activeVoxelPairFillDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     activeVoxelPairFillDescriptorSetLayout,
                                     nullptr);
      }
      if (scalarFieldDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, scalarFieldDescriptorSetLayout,
                                     nullptr);
      }
      if (surfaceCellDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, surfaceCellDescriptorSetLayout,
                                     nullptr);
      }
      if (sparseSurfaceCellDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     sparseSurfaceCellDescriptorSetLayout,
                                     nullptr);
      }
      if (surfaceCellCompactDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     surfaceCellCompactDescriptorSetLayout,
                                     nullptr);
      }
      if (surfaceTriangleCompactDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     surfaceTriangleCompactDescriptorSetLayout,
                                     nullptr);
      }
      if (surfaceMeshDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device, surfaceMeshDescriptorSetLayout,
                                     nullptr);
      }
      if (denseSurfaceMeshDescriptorSetLayout != VK_NULL_HANDLE &&
          vkDestroyDescriptorSetLayout) {
        vkDestroyDescriptorSetLayout(device,
                                     denseSurfaceMeshDescriptorSetLayout,
                                     nullptr);
      }
      if (surfaceWorkDescriptorPool != VK_NULL_HANDLE &&
          vkDestroyDescriptorPool) {
        vkDestroyDescriptorPool(device, surfaceWorkDescriptorPool, nullptr);
      }
      if (computeWorkDescriptorPool != VK_NULL_HANDLE &&
          vkDestroyDescriptorPool) {
        vkDestroyDescriptorPool(device, computeWorkDescriptorPool, nullptr);
      }
      if (commandPool != VK_NULL_HANDLE && vkDestroyCommandPool) {
        vkDestroyCommandPool(device, commandPool, nullptr);
      }
      if (vkDestroyDevice) {
        vkDestroyDevice(device, nullptr);
      }
    }

    if (instance != VK_NULL_HANDLE && vkDestroyInstance) {
      vkDestroyInstance(instance, nullptr);
    }

    if (library) {
      close_vulkan_library(library);
    }

    library = nullptr;
    instance = VK_NULL_HANDLE;
    physicalDevice = VK_NULL_HANDLE;
    queueFamilyIndex = UINT32_MAX;
    device = VK_NULL_HANDLE;
    queue = VK_NULL_HANDLE;
    commandPool = VK_NULL_HANDLE;
    particleDescriptorSetLayout = VK_NULL_HANDLE;
    particlePipelineLayout = VK_NULL_HANDLE;
    particlePipeline = VK_NULL_HANDLE;
    coverageDescriptorSetLayout = VK_NULL_HANDLE;
    coveragePipelineLayout = VK_NULL_HANDLE;
    coveragePipeline = VK_NULL_HANDLE;
    activeVoxelCompactDescriptorSetLayout = VK_NULL_HANDLE;
    activeVoxelCompactPipelineLayout = VK_NULL_HANDLE;
    activeVoxelCompactPipeline = VK_NULL_HANDLE;
    activeVoxelPairFillDescriptorSetLayout = VK_NULL_HANDLE;
    activeVoxelPairFillPipelineLayout = VK_NULL_HANDLE;
    activeVoxelPairFillPipeline = VK_NULL_HANDLE;
    scalarFieldDescriptorSetLayout = VK_NULL_HANDLE;
    scalarFieldPipelineLayout = VK_NULL_HANDLE;
    scalarFieldPipeline = VK_NULL_HANDLE;
    zhuScalarFieldPipeline = VK_NULL_HANDLE;
    surfaceCellDescriptorSetLayout = VK_NULL_HANDLE;
    surfaceCellPipelineLayout = VK_NULL_HANDLE;
    surfaceCellPipeline = VK_NULL_HANDLE;
    sparseSurfaceCellDescriptorSetLayout = VK_NULL_HANDLE;
    sparseSurfaceCellPipelineLayout = VK_NULL_HANDLE;
    sparseSurfaceCellPipeline = VK_NULL_HANDLE;
    surfaceCellCompactDescriptorSetLayout = VK_NULL_HANDLE;
    surfaceCellCompactPipelineLayout = VK_NULL_HANDLE;
    surfaceCellCompactPipeline = VK_NULL_HANDLE;
    surfaceTriangleCompactDescriptorSetLayout = VK_NULL_HANDLE;
    surfaceTriangleCompactPipelineLayout = VK_NULL_HANDLE;
    surfaceTriangleCompactPipeline = VK_NULL_HANDLE;
    surfaceMeshDescriptorSetLayout = VK_NULL_HANDLE;
    surfaceMeshPipelineLayout = VK_NULL_HANDLE;
    surfaceMeshPipeline = VK_NULL_HANDLE;
    denseSurfaceMeshDescriptorSetLayout = VK_NULL_HANDLE;
    denseSurfaceMeshPipelineLayout = VK_NULL_HANDLE;
    denseSurfaceMeshPipeline = VK_NULL_HANDLE;
    computeWorkDescriptorPool = VK_NULL_HANDLE;
    particleDescriptorSet = VK_NULL_HANDLE;
    coverageDescriptorSet = VK_NULL_HANDLE;
    activeVoxelCompactDescriptorSet = VK_NULL_HANDLE;
    activeVoxelPairFillDescriptorSet = VK_NULL_HANDLE;
    scalarFieldDescriptorSet = VK_NULL_HANDLE;
    particleCommandBuffer = VK_NULL_HANDLE;
    coverageCommandBuffer = VK_NULL_HANDLE;
    activeVoxelCompactCommandBuffer = VK_NULL_HANDLE;
    activeVoxelPairFillCommandBuffer = VK_NULL_HANDLE;
    scalarFieldCommandBuffer = VK_NULL_HANDLE;
    surfaceWorkDescriptorPool = VK_NULL_HANDLE;
    surfaceCellDescriptorSet = VK_NULL_HANDLE;
    sparseSurfaceCellDescriptorSet = VK_NULL_HANDLE;
    surfaceCellCompactDescriptorSet = VK_NULL_HANDLE;
    surfaceTriangleCompactDescriptorSet = VK_NULL_HANDLE;
    surfaceMeshDescriptorSet = VK_NULL_HANDLE;
    denseSurfaceMeshDescriptorSet = VK_NULL_HANDLE;
    surfaceCellCommandBuffer = VK_NULL_HANDLE;
    sparseSurfaceCellCommandBuffer = VK_NULL_HANDLE;
    surfaceCellCompactCommandBuffer = VK_NULL_HANDLE;
    surfaceTriangleCompactCommandBuffer = VK_NULL_HANDLE;
    surfaceMeshCommandBuffer = VK_NULL_HANDLE;
    denseSurfaceMeshCommandBuffer = VK_NULL_HANDLE;
    particleInputBuffer = HostBuffer{};
    particleOutputBuffer = HostBuffer{};
    particleVelocityBuffer = HostBuffer{};
    particleMinBoundsBuffer = HostBuffer{};
    particleMaxBoundsBuffer = HostBuffer{};
    voxelCoverageBuffer = HostBuffer{};
    activeVoxelIndexBuffer = HostBuffer{};
    activeVoxelStatsBuffer = HostBuffer{};
    activeVoxelCompactLookupBuffer = HostBuffer{};
    activeVoxelParticleOffsetBuffer = HostBuffer{};
    activeVoxelParticleCursorBuffer = HostBuffer{};
    activeVoxelParticleIndexBuffer = HostBuffer{};
    voxelScalarFieldBuffer = HostBuffer{};
    candidateCellIndexBuffer = HostBuffer{};
    candidateCellCubeIndexBuffer = HostBuffer{};
    denseSurfaceCellCubeIndexBuffer = HostBuffer{};
    surfaceCellStatsBuffer = HostBuffer{};
    surfaceTriangleCountBuffer = HostBuffer{};
    surfaceTriangleVertexBuffer = HostBuffer{};
    residentSurfaceTriangleVerticesCompacted = false;
    deviceActiveVoxelCompactLookupBuffer = DeviceBuffer{};
    deviceActiveVoxelParticleOffsetBuffer = DeviceBuffer{};
    deviceActiveVoxelParticleCursorBuffer = DeviceBuffer{};
    deviceActiveVoxelParticleIndexBuffer = DeviceBuffer{};
    deviceCandidateCellIndexBuffer = DeviceBuffer{};
    deviceCandidateCellCubeIndexBuffer = DeviceBuffer{};
    deviceDenseSurfaceCellCubeIndexBuffer = DeviceBuffer{};
    deviceSurfaceCellStatsBuffer = DeviceBuffer{};
    deviceSurfaceTriangleCountBuffer = DeviceBuffer{};
    deviceSurfaceTriangleVertexBuffer = DeviceBuffer{};
    memoryProperties = VkPhysicalDeviceMemoryProperties{};

    vkDestroyInstance = nullptr;
    vkDestroyDevice = nullptr;
    vkDestroyBuffer = nullptr;
    vkFreeMemory = nullptr;
    vkDestroyDescriptorSetLayout = nullptr;
    vkDestroyDescriptorPool = nullptr;
    vkDestroyShaderModule = nullptr;
    vkDestroyPipelineLayout = nullptr;
    vkDestroyPipeline = nullptr;
    vkDestroyCommandPool = nullptr;
    vkDeviceWaitIdle = nullptr;

    vkCreateBuffer = nullptr;
    vkGetBufferMemoryRequirements = nullptr;
    vkAllocateMemory = nullptr;
    vkBindBufferMemory = nullptr;
    vkMapMemory = nullptr;
    vkUnmapMemory = nullptr;
    vkCreateDescriptorSetLayout = nullptr;
    vkCreateDescriptorPool = nullptr;
    vkAllocateDescriptorSets = nullptr;
    vkUpdateDescriptorSets = nullptr;
    vkCreateShaderModule = nullptr;
    vkCreatePipelineLayout = nullptr;
    vkCreateComputePipelines = nullptr;
    vkAllocateCommandBuffers = nullptr;
    vkBeginCommandBuffer = nullptr;
    vkCmdBindPipeline = nullptr;
    vkCmdBindDescriptorSets = nullptr;
    vkCmdPushConstants = nullptr;
    vkCmdDispatch = nullptr;
    vkCmdPipelineBarrier = nullptr;
    vkCmdCopyBuffer = nullptr;
    vkCmdFillBuffer = nullptr;
    vkEndCommandBuffer = nullptr;
    vkQueueSubmit = nullptr;
    vkQueueWaitIdle = nullptr;
    vkResetCommandPool = nullptr;

    initialized = false;
  }

};

VulkanSharedContext &get_vulkan_shared_context() {
  static VulkanSharedContext *context = new VulkanSharedContext();
  return *context;
}

bool ensure_shared_host_buffer(VulkanSharedContext &context,
                               VulkanSharedContext::HostBuffer &hostBuffer,
                               VkDeviceSize requestedSize, const char *label,
                               std::string &outError) {
  if (requestedSize == 0) {
    outError = std::string(label) + " requested zero bytes.";
    return false;
  }

  if (hostBuffer.buffer != VK_NULL_HANDLE && hostBuffer.memory != VK_NULL_HANDLE &&
      hostBuffer.mapped != nullptr && hostBuffer.capacity >= requestedSize) {
    outError.clear();
    return true;
  }

  context.release_host_buffer(hostBuffer);

  VkBufferCreateInfo bufferCreateInfo{};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size = requestedSize;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result =
      context.vkCreateBuffer(context.device, &bufferCreateInfo, nullptr,
                             &hostBuffer.buffer);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkCreateBuffer failed (VkResult " << result << ").";
    outError = stream.str();
    context.release_host_buffer(hostBuffer);
    return false;
  }

  VkMemoryRequirements memoryRequirements{};
  context.vkGetBufferMemoryRequirements(context.device, hostBuffer.buffer,
                                        &memoryRequirements);

  uint32_t memoryTypeIndex = 0;
  const VkMemoryPropertyFlags preferredFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
  const VkMemoryPropertyFlags fallbackFlags =
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  if (!find_memory_type_index(context.memoryProperties,
                              memoryRequirements.memoryTypeBits,
                              preferredFlags, memoryTypeIndex) &&
      !find_memory_type_index(context.memoryProperties,
                              memoryRequirements.memoryTypeBits,
                              fallbackFlags, memoryTypeIndex)) {
    outError = std::string(label) +
               ": failed to find a host-visible coherent memory type.";
    context.release_host_buffer(hostBuffer);
    return false;
  }

  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = memoryRequirements.size;
  allocateInfo.memoryTypeIndex = memoryTypeIndex;

  result = context.vkAllocateMemory(context.device, &allocateInfo, nullptr,
                                    &hostBuffer.memory);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkAllocateMemory failed (VkResult " << result
           << ").";
    outError = stream.str();
    context.release_host_buffer(hostBuffer);
    return false;
  }

  result = context.vkBindBufferMemory(context.device, hostBuffer.buffer,
                                      hostBuffer.memory, 0);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkBindBufferMemory failed (VkResult " << result
           << ").";
    outError = stream.str();
    context.release_host_buffer(hostBuffer);
    return false;
  }

  result = context.vkMapMemory(context.device, hostBuffer.memory, 0,
                               memoryRequirements.size, 0, &hostBuffer.mapped);
  if (result != VK_SUCCESS || hostBuffer.mapped == nullptr) {
    std::ostringstream stream;
    stream << label << ": vkMapMemory failed (VkResult " << result << ").";
    outError = stream.str();
    context.release_host_buffer(hostBuffer);
    return false;
  }

  hostBuffer.capacity = requestedSize;
  outError.clear();
  return true;
}

bool ensure_shared_device_buffer(VulkanSharedContext &context,
                                 VulkanSharedContext::DeviceBuffer &deviceBuffer,
                                 VkDeviceSize requestedSize, const char *label,
                                 std::string &outError) {
  if (requestedSize == 0) {
    outError = std::string(label) + " requested zero bytes.";
    return false;
  }

  if (deviceBuffer.buffer != VK_NULL_HANDLE &&
      deviceBuffer.memory != VK_NULL_HANDLE &&
      deviceBuffer.capacity >= requestedSize) {
    outError.clear();
    return true;
  }

  context.release_device_buffer(deviceBuffer);

  VkBufferCreateInfo bufferCreateInfo{};
  bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  bufferCreateInfo.size = requestedSize;
  bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                           VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                           VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkResult result =
      context.vkCreateBuffer(context.device, &bufferCreateInfo, nullptr,
                             &deviceBuffer.buffer);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkCreateBuffer failed (VkResult " << result << ").";
    outError = stream.str();
    context.release_device_buffer(deviceBuffer);
    return false;
  }

  VkMemoryRequirements memoryRequirements{};
  context.vkGetBufferMemoryRequirements(context.device, deviceBuffer.buffer,
                                        &memoryRequirements);

  uint32_t memoryTypeIndex = 0;
  if (!find_memory_type_index(context.memoryProperties,
                              memoryRequirements.memoryTypeBits,
                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                              memoryTypeIndex)) {
    outError =
        std::string(label) + ": failed to find a device-local memory type.";
    context.release_device_buffer(deviceBuffer);
    return false;
  }

  VkMemoryAllocateInfo allocateInfo{};
  allocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  allocateInfo.allocationSize = memoryRequirements.size;
  allocateInfo.memoryTypeIndex = memoryTypeIndex;

  result = context.vkAllocateMemory(context.device, &allocateInfo, nullptr,
                                    &deviceBuffer.memory);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkAllocateMemory failed (VkResult " << result << ").";
    outError = stream.str();
    context.release_device_buffer(deviceBuffer);
    return false;
  }

  result = context.vkBindBufferMemory(context.device, deviceBuffer.buffer,
                                      deviceBuffer.memory, 0);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << label << ": vkBindBufferMemory failed (VkResult " << result
           << ").";
    outError = stream.str();
    context.release_device_buffer(deviceBuffer);
    return false;
  }

  deviceBuffer.capacity = requestedSize;
  outError.clear();
  return true;
}

bool has_vulkan_scalar_field_samples(
    const VulkanParticleComputeResult &computeResult,
    std::size_t expectedSampleCount) {
  return expectedSampleCount > 0 &&
         ((!computeResult.voxelScalarField.empty() &&
           computeResult.voxelScalarField.size() == expectedSampleCount) ||
          computeResult.scalarFieldResidentOnGpu);
}

bool ensure_shared_scalar_field_buffer_for_result(
    VulkanSharedContext &sharedContext,
    const VulkanParticleComputeResult &computeResult,
    VkDeviceSize scalarFieldBufferSize, const char *label,
    std::string &outError) {
  if (!ensure_shared_host_buffer(sharedContext, sharedContext.voxelScalarFieldBuffer,
                                 scalarFieldBufferSize, label, outError)) {
    return false;
  }

  if (!computeResult.voxelScalarField.empty()) {
    std::memcpy(sharedContext.voxelScalarFieldBuffer.mapped,
                computeResult.voxelScalarField.data(),
                static_cast<size_t>(scalarFieldBufferSize));
    outError.clear();
    return true;
  }

  if (computeResult.scalarFieldResidentOnGpu) {
    outError.clear();
    return true;
  }

  outError = "Vulkan scalar-field data is unavailable for this operation.";
  return false;
}

bool initialize_vulkan_shared_context(VulkanSharedContext &context,
                                      std::string &outError) {
  if (context.initialized) {
    outError.clear();
    return true;
  }

  if (context.initializationAttempted) {
    outError = context.initError.empty() ? "Vulkan shared context is unavailable."
                                         : context.initError;
    return false;
  }

  context.initializationAttempted = true;
  context.release();

  auto fail = [&](const std::string &message) {
    context.initError = message;
    context.release();
    outError = message;
    return false;
  };

  auto create_shared_compute_pipeline =
      [&](const VkDescriptorSetLayoutBinding *bindings, uint32_t bindingCount,
          uint32_t pushConstantSize, const std::vector<uint32_t> &spirv,
          VkDescriptorSetLayout &outDescriptorSetLayout,
          VkPipelineLayout &outPipelineLayout, VkPipeline &outPipeline,
          const char *label) -> bool {
    if (outDescriptorSetLayout != VK_NULL_HANDLE &&
        outPipelineLayout != VK_NULL_HANDLE && outPipeline != VK_NULL_HANDLE) {
      return true;
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;

    auto cleanupLocal = [&]() {
      if (shaderModule != VK_NULL_HANDLE && context.vkDestroyShaderModule) {
        context.vkDestroyShaderModule(context.device, shaderModule, nullptr);
      }
      if (outPipeline != VK_NULL_HANDLE && context.vkDestroyPipeline) {
        context.vkDestroyPipeline(context.device, outPipeline, nullptr);
        outPipeline = VK_NULL_HANDLE;
      }
      if (outPipelineLayout != VK_NULL_HANDLE &&
          context.vkDestroyPipelineLayout) {
        context.vkDestroyPipelineLayout(context.device, outPipelineLayout,
                                        nullptr);
        outPipelineLayout = VK_NULL_HANDLE;
      }
      if (outDescriptorSetLayout != VK_NULL_HANDLE &&
          context.vkDestroyDescriptorSetLayout) {
        context.vkDestroyDescriptorSetLayout(context.device,
                                             outDescriptorSetLayout, nullptr);
        outDescriptorSetLayout = VK_NULL_HANDLE;
      }
    };

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutInfo{};
    descriptorSetLayoutInfo.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutInfo.bindingCount = bindingCount;
    descriptorSetLayoutInfo.pBindings = bindings;

    VkResult localResult =
        context.vkCreateDescriptorSetLayout(context.device,
                                            &descriptorSetLayoutInfo, nullptr,
                                            &outDescriptorSetLayout);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreateDescriptorSetLayout failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      cleanupLocal();
      return false;
    }

    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = spirv.data();
    localResult = context.vkCreateShaderModule(context.device, &shaderModuleInfo,
                                               nullptr, &shaderModule);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreateShaderModule failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      cleanupLocal();
      return false;
    }

    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = pushConstantSize;

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &outDescriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pushConstantRange;
    localResult = context.vkCreatePipelineLayout(context.device,
                                                 &pipelineLayoutInfo, nullptr,
                                                 &outPipelineLayout);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreatePipelineLayout failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      cleanupLocal();
      return false;
    }

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = outPipelineLayout;
    localResult = context.vkCreateComputePipelines(
        context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
        &outPipeline);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreateComputePipelines failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      cleanupLocal();
      return false;
    }

    context.vkDestroyShaderModule(context.device, shaderModule, nullptr);
    outError.clear();
    return true;
  };

  auto create_shared_compute_pipeline_variant =
      [&](const std::vector<uint32_t> &spirv, VkPipelineLayout pipelineLayout,
          VkPipeline &outPipeline, const char *label) -> bool {
    if (outPipeline != VK_NULL_HANDLE) {
      return true;
    }
    if (pipelineLayout == VK_NULL_HANDLE) {
      outError = std::string(label) + ": pipeline layout is unavailable.";
      return false;
    }

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    VkShaderModuleCreateInfo shaderModuleInfo{};
    shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleInfo.codeSize = spirv.size() * sizeof(uint32_t);
    shaderModuleInfo.pCode = spirv.data();
    VkResult localResult = context.vkCreateShaderModule(context.device,
                                                        &shaderModuleInfo,
                                                        nullptr, &shaderModule);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreateShaderModule failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      return false;
    }

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shaderModule;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = pipelineLayout;
    localResult = context.vkCreateComputePipelines(
        context.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr,
        &outPipeline);
    context.vkDestroyShaderModule(context.device, shaderModule, nullptr);
    if (localResult != VK_SUCCESS) {
      std::ostringstream stream;
      stream << label << ": vkCreateComputePipelines failed (VkResult "
             << localResult << ").";
      outError = stream.str();
      outPipeline = VK_NULL_HANDLE;
      return false;
    }

    outError.clear();
    return true;
  };

  auto ensure_shared_surface_dispatch_resources =
      [&](std::string &resourceError) -> bool {
    if (context.surfaceWorkDescriptorPool == VK_NULL_HANDLE) {
      VkDescriptorPoolSize poolSize{};
      poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      poolSize.descriptorCount = 24;

      VkDescriptorPoolCreateInfo poolInfo{};
      poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      poolInfo.maxSets = 6;
      poolInfo.poolSizeCount = 1;
      poolInfo.pPoolSizes = &poolSize;
      if (context.vkCreateDescriptorPool(context.device, &poolInfo, nullptr,
                                         &context.surfaceWorkDescriptorPool) !=
          VK_SUCCESS) {
        resourceError =
            "Failed to create shared Vulkan surface descriptor pool.";
        return false;
      }
    }

    if (context.surfaceCellDescriptorSet == VK_NULL_HANDLE ||
        context.sparseSurfaceCellDescriptorSet == VK_NULL_HANDLE ||
        context.surfaceCellCompactDescriptorSet == VK_NULL_HANDLE ||
        context.surfaceTriangleCompactDescriptorSet == VK_NULL_HANDLE ||
        context.surfaceMeshDescriptorSet == VK_NULL_HANDLE ||
        context.denseSurfaceMeshDescriptorSet == VK_NULL_HANDLE) {
      VkDescriptorSetLayout layouts[6] = {
          context.surfaceCellDescriptorSetLayout,
          context.sparseSurfaceCellDescriptorSetLayout,
          context.surfaceCellCompactDescriptorSetLayout,
          context.surfaceTriangleCompactDescriptorSetLayout,
          context.surfaceMeshDescriptorSetLayout,
          context.denseSurfaceMeshDescriptorSetLayout,
      };
      VkDescriptorSet sets[6] = {};
      VkDescriptorSetAllocateInfo setAllocateInfo{};
      setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      setAllocateInfo.descriptorPool = context.surfaceWorkDescriptorPool;
      setAllocateInfo.descriptorSetCount = 6;
      setAllocateInfo.pSetLayouts = layouts;
      if (context.vkAllocateDescriptorSets(context.device, &setAllocateInfo,
                                           sets) != VK_SUCCESS) {
        resourceError =
            "Failed to allocate shared Vulkan surface descriptor sets.";
        return false;
      }
      context.surfaceCellDescriptorSet = sets[0];
      context.sparseSurfaceCellDescriptorSet = sets[1];
      context.surfaceCellCompactDescriptorSet = sets[2];
      context.surfaceTriangleCompactDescriptorSet = sets[3];
      context.surfaceMeshDescriptorSet = sets[4];
      context.denseSurfaceMeshDescriptorSet = sets[5];
    }

    if (context.surfaceCellCommandBuffer == VK_NULL_HANDLE ||
        context.sparseSurfaceCellCommandBuffer == VK_NULL_HANDLE ||
        context.surfaceCellCompactCommandBuffer == VK_NULL_HANDLE ||
        context.surfaceTriangleCompactCommandBuffer == VK_NULL_HANDLE ||
        context.surfaceMeshCommandBuffer == VK_NULL_HANDLE ||
        context.denseSurfaceMeshCommandBuffer == VK_NULL_HANDLE) {
      VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
      commandBufferAllocateInfo.sType =
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = context.commandPool;
      commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 6;

      VkCommandBuffer commandBuffers[6] = {};
      if (context.vkAllocateCommandBuffers(context.device,
                                           &commandBufferAllocateInfo,
                                           commandBuffers) != VK_SUCCESS) {
        resourceError =
            "Failed to allocate shared Vulkan surface command buffers.";
        return false;
      }
      context.surfaceCellCommandBuffer = commandBuffers[0];
      context.sparseSurfaceCellCommandBuffer = commandBuffers[1];
      context.surfaceCellCompactCommandBuffer = commandBuffers[2];
      context.surfaceTriangleCompactCommandBuffer = commandBuffers[3];
      context.surfaceMeshCommandBuffer = commandBuffers[4];
      context.denseSurfaceMeshCommandBuffer = commandBuffers[5];
    }

    resourceError.clear();
    return true;
  };

  auto ensure_shared_compute_dispatch_resources =
      [&](std::string &resourceError) -> bool {
    if (context.computeWorkDescriptorPool == VK_NULL_HANDLE) {
      VkDescriptorPoolSize poolSize{};
      poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      poolSize.descriptorCount = 26;

      VkDescriptorPoolCreateInfo poolInfo{};
      poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
      poolInfo.maxSets = 5;
      poolInfo.poolSizeCount = 1;
      poolInfo.pPoolSizes = &poolSize;
      if (context.vkCreateDescriptorPool(context.device, &poolInfo, nullptr,
                                         &context.computeWorkDescriptorPool) !=
          VK_SUCCESS) {
        resourceError =
            "Failed to create shared Vulkan compute descriptor pool.";
        return false;
      }
    }

    if (context.particleDescriptorSet == VK_NULL_HANDLE ||
        context.coverageDescriptorSet == VK_NULL_HANDLE ||
        context.activeVoxelCompactDescriptorSet == VK_NULL_HANDLE ||
        context.activeVoxelPairFillDescriptorSet == VK_NULL_HANDLE ||
        context.scalarFieldDescriptorSet == VK_NULL_HANDLE) {
      VkDescriptorSetLayout layouts[5] = {
          context.particleDescriptorSetLayout,
          context.coverageDescriptorSetLayout,
          context.activeVoxelCompactDescriptorSetLayout,
          context.activeVoxelPairFillDescriptorSetLayout,
          context.scalarFieldDescriptorSetLayout,
      };
      VkDescriptorSet sets[5] = {};
      VkDescriptorSetAllocateInfo setAllocateInfo{};
      setAllocateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      setAllocateInfo.descriptorPool = context.computeWorkDescriptorPool;
      setAllocateInfo.descriptorSetCount = 5;
      setAllocateInfo.pSetLayouts = layouts;
      if (context.vkAllocateDescriptorSets(context.device, &setAllocateInfo,
                                           sets) != VK_SUCCESS) {
        resourceError =
            "Failed to allocate shared Vulkan compute descriptor sets.";
        return false;
      }
      context.particleDescriptorSet = sets[0];
      context.coverageDescriptorSet = sets[1];
      context.activeVoxelCompactDescriptorSet = sets[2];
      context.activeVoxelPairFillDescriptorSet = sets[3];
      context.scalarFieldDescriptorSet = sets[4];
    }

    if (context.particleCommandBuffer == VK_NULL_HANDLE ||
        context.coverageCommandBuffer == VK_NULL_HANDLE ||
        context.activeVoxelCompactCommandBuffer == VK_NULL_HANDLE ||
        context.activeVoxelPairFillCommandBuffer == VK_NULL_HANDLE ||
        context.scalarFieldCommandBuffer == VK_NULL_HANDLE) {
      VkCommandBufferAllocateInfo commandBufferAllocateInfo{};
      commandBufferAllocateInfo.sType =
          VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      commandBufferAllocateInfo.commandPool = context.commandPool;
      commandBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      commandBufferAllocateInfo.commandBufferCount = 5;

      VkCommandBuffer commandBuffers[5] = {};
      if (context.vkAllocateCommandBuffers(context.device,
                                           &commandBufferAllocateInfo,
                                           commandBuffers) != VK_SUCCESS) {
        resourceError =
            "Failed to allocate shared Vulkan compute command buffers.";
        return false;
      }
      context.particleCommandBuffer = commandBuffers[0];
      context.coverageCommandBuffer = commandBuffers[1];
      context.activeVoxelCompactCommandBuffer = commandBuffers[2];
      context.activeVoxelPairFillCommandBuffer = commandBuffers[3];
      context.scalarFieldCommandBuffer = commandBuffers[4];
    }

    resourceError.clear();
    return true;
  };

  context.library = load_vulkan_library();
  if (!context.library) {
    return fail("Vulkan runtime not found on this system.");
  }

  auto vkGetInstanceProcAddr =
      reinterpret_cast<PFN_vkGetInstanceProcAddr>(
          load_symbol(context.library, "vkGetInstanceProcAddr"));
  if (!vkGetInstanceProcAddr) {
    return fail("vkGetInstanceProcAddr is unavailable in the Vulkan loader.");
  }

  auto vkCreateInstance = reinterpret_cast<PFN_vkCreateInstance>(
      vkGetInstanceProcAddr(nullptr, "vkCreateInstance"));
  auto vkEnumerateInstanceVersion =
      reinterpret_cast<PFN_vkEnumerateInstanceVersion>(
          vkGetInstanceProcAddr(nullptr, "vkEnumerateInstanceVersion"));
  if (!vkCreateInstance) {
    return fail("required Vulkan instance entry points are missing.");
  }

  VkApplicationInfo appInfo{};
  appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  appInfo.pApplicationName = "Frost for Blender";
  appInfo.applicationVersion = 1;
  appInfo.pEngineName = "Frost";
  appInfo.engineVersion = 1;
  uint32_t apiVersion = VK_API_VERSION_1_0;
  if (vkEnumerateInstanceVersion) {
    vkEnumerateInstanceVersion(&apiVersion);
  }
  appInfo.apiVersion = apiVersion;

  VkInstanceCreateInfo instanceCreateInfo{};
  instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instanceCreateInfo.pApplicationInfo = &appInfo;

  VkResult result =
      vkCreateInstance(&instanceCreateInfo, nullptr, &context.instance);
  if (result != VK_SUCCESS || context.instance == VK_NULL_HANDLE) {
    std::ostringstream stream;
    stream << "vkCreateInstance failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  auto vkEnumeratePhysicalDevices =
      reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
          vkGetInstanceProcAddr(context.instance, "vkEnumeratePhysicalDevices"));
  context.vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
      vkGetInstanceProcAddr(context.instance, "vkDestroyInstance"));
  auto vkGetPhysicalDeviceQueueFamilyProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
          vkGetInstanceProcAddr(context.instance,
                                "vkGetPhysicalDeviceQueueFamilyProperties"));
  auto vkGetPhysicalDeviceMemoryProperties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
          vkGetInstanceProcAddr(context.instance,
                                "vkGetPhysicalDeviceMemoryProperties"));
  auto vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
      vkGetInstanceProcAddr(context.instance, "vkCreateDevice"));
  auto vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
      vkGetInstanceProcAddr(context.instance, "vkGetDeviceProcAddr"));

  if (!vkEnumeratePhysicalDevices) {
    vkEnumeratePhysicalDevices =
        reinterpret_cast<PFN_vkEnumeratePhysicalDevices>(
            vkGetInstanceProcAddr(nullptr, "vkEnumeratePhysicalDevices"));
  }
  if (!context.vkDestroyInstance) {
    context.vkDestroyInstance = reinterpret_cast<PFN_vkDestroyInstance>(
        vkGetInstanceProcAddr(nullptr, "vkDestroyInstance"));
  }
  if (!vkGetPhysicalDeviceQueueFamilyProperties) {
    vkGetPhysicalDeviceQueueFamilyProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceQueueFamilyProperties>(
            vkGetInstanceProcAddr(nullptr,
                                  "vkGetPhysicalDeviceQueueFamilyProperties"));
  }
  if (!vkGetPhysicalDeviceMemoryProperties) {
    vkGetPhysicalDeviceMemoryProperties =
        reinterpret_cast<PFN_vkGetPhysicalDeviceMemoryProperties>(
            vkGetInstanceProcAddr(nullptr,
                                  "vkGetPhysicalDeviceMemoryProperties"));
  }
  if (!vkCreateDevice) {
    vkCreateDevice = reinterpret_cast<PFN_vkCreateDevice>(
        vkGetInstanceProcAddr(nullptr, "vkCreateDevice"));
  }
  if (!vkGetDeviceProcAddr) {
    vkGetDeviceProcAddr = reinterpret_cast<PFN_vkGetDeviceProcAddr>(
        load_symbol(context.library, "vkGetDeviceProcAddr"));
  }

  if (!vkEnumeratePhysicalDevices || !context.vkDestroyInstance ||
      !vkGetPhysicalDeviceQueueFamilyProperties ||
      !vkGetPhysicalDeviceMemoryProperties || !vkCreateDevice ||
      !vkGetDeviceProcAddr) {
    return fail("required Vulkan shared-context entry points are missing.");
  }

  uint32_t physicalDeviceCount = 0;
  result = vkEnumeratePhysicalDevices(context.instance, &physicalDeviceCount,
                                      nullptr);
  if (result != VK_SUCCESS || physicalDeviceCount == 0) {
    std::ostringstream stream;
    stream << "vkEnumeratePhysicalDevices failed or returned no device (VkResult "
           << result << ").";
    return fail(stream.str());
  }

  std::vector<VkPhysicalDevice> physicalDevices(physicalDeviceCount);
  result = vkEnumeratePhysicalDevices(context.instance, &physicalDeviceCount,
                                      physicalDevices.data());
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "physical device enumeration failed on second pass (VkResult "
           << result << ").";
    return fail(stream.str());
  }

  context.physicalDevice = VK_NULL_HANDLE;
  context.queueFamilyIndex = UINT32_MAX;
  for (VkPhysicalDevice candidate : physicalDevices) {
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount,
                                             nullptr);
    std::vector<VkQueueFamilyProperties> families(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(candidate, &queueFamilyCount,
                                             families.data());
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
      if (families[i].queueCount > 0 &&
          (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0) {
        context.physicalDevice = candidate;
        context.queueFamilyIndex = i;
        break;
      }
    }
    if (context.physicalDevice != VK_NULL_HANDLE) {
      break;
    }
  }

  if (context.physicalDevice == VK_NULL_HANDLE) {
    return fail("no compute-capable Vulkan queue family was found.");
  }

  const float queuePriority = 1.0f;
  VkDeviceQueueCreateInfo queueCreateInfo{};
  queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queueCreateInfo.queueFamilyIndex = context.queueFamilyIndex;
  queueCreateInfo.queueCount = 1;
  queueCreateInfo.pQueuePriorities = &queuePriority;

  VkDeviceCreateInfo deviceCreateInfo{};
  deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  deviceCreateInfo.queueCreateInfoCount = 1;
  deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

  result = vkCreateDevice(context.physicalDevice, &deviceCreateInfo, nullptr,
                          &context.device);
  if (result != VK_SUCCESS || context.device == VK_NULL_HANDLE) {
    std::ostringstream stream;
    stream << "vkCreateDevice failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  auto vkGetDeviceQueue = reinterpret_cast<PFN_vkGetDeviceQueue>(
      vkGetDeviceProcAddr(context.device, "vkGetDeviceQueue"));
  context.vkCreateBuffer = reinterpret_cast<PFN_vkCreateBuffer>(
      vkGetDeviceProcAddr(context.device, "vkCreateBuffer"));
  context.vkDestroyBuffer = reinterpret_cast<PFN_vkDestroyBuffer>(
      vkGetDeviceProcAddr(context.device, "vkDestroyBuffer"));
  context.vkGetBufferMemoryRequirements =
      reinterpret_cast<PFN_vkGetBufferMemoryRequirements>(
          vkGetDeviceProcAddr(context.device, "vkGetBufferMemoryRequirements"));
  context.vkAllocateMemory = reinterpret_cast<PFN_vkAllocateMemory>(
      vkGetDeviceProcAddr(context.device, "vkAllocateMemory"));
  context.vkFreeMemory = reinterpret_cast<PFN_vkFreeMemory>(
      vkGetDeviceProcAddr(context.device, "vkFreeMemory"));
  context.vkBindBufferMemory = reinterpret_cast<PFN_vkBindBufferMemory>(
      vkGetDeviceProcAddr(context.device, "vkBindBufferMemory"));
  context.vkMapMemory = reinterpret_cast<PFN_vkMapMemory>(
      vkGetDeviceProcAddr(context.device, "vkMapMemory"));
  context.vkUnmapMemory = reinterpret_cast<PFN_vkUnmapMemory>(
      vkGetDeviceProcAddr(context.device, "vkUnmapMemory"));
  context.vkDestroyDevice = reinterpret_cast<PFN_vkDestroyDevice>(
      vkGetDeviceProcAddr(context.device, "vkDestroyDevice"));
  context.vkCreateDescriptorSetLayout =
      reinterpret_cast<PFN_vkCreateDescriptorSetLayout>(
          vkGetDeviceProcAddr(context.device, "vkCreateDescriptorSetLayout"));
  context.vkDestroyDescriptorSetLayout =
      reinterpret_cast<PFN_vkDestroyDescriptorSetLayout>(
          vkGetDeviceProcAddr(context.device, "vkDestroyDescriptorSetLayout"));
  context.vkCreateDescriptorPool =
      reinterpret_cast<PFN_vkCreateDescriptorPool>(
          vkGetDeviceProcAddr(context.device, "vkCreateDescriptorPool"));
  context.vkDestroyDescriptorPool =
      reinterpret_cast<PFN_vkDestroyDescriptorPool>(
          vkGetDeviceProcAddr(context.device, "vkDestroyDescriptorPool"));
  context.vkAllocateDescriptorSets =
      reinterpret_cast<PFN_vkAllocateDescriptorSets>(
          vkGetDeviceProcAddr(context.device, "vkAllocateDescriptorSets"));
  context.vkUpdateDescriptorSets =
      reinterpret_cast<PFN_vkUpdateDescriptorSets>(
          vkGetDeviceProcAddr(context.device, "vkUpdateDescriptorSets"));
  context.vkCreateShaderModule =
      reinterpret_cast<PFN_vkCreateShaderModule>(
          vkGetDeviceProcAddr(context.device, "vkCreateShaderModule"));
  context.vkDestroyShaderModule =
      reinterpret_cast<PFN_vkDestroyShaderModule>(
          vkGetDeviceProcAddr(context.device, "vkDestroyShaderModule"));
  context.vkCreatePipelineLayout =
      reinterpret_cast<PFN_vkCreatePipelineLayout>(
          vkGetDeviceProcAddr(context.device, "vkCreatePipelineLayout"));
  context.vkDestroyPipelineLayout =
      reinterpret_cast<PFN_vkDestroyPipelineLayout>(
          vkGetDeviceProcAddr(context.device, "vkDestroyPipelineLayout"));
  context.vkCreateComputePipelines =
      reinterpret_cast<PFN_vkCreateComputePipelines>(
          vkGetDeviceProcAddr(context.device, "vkCreateComputePipelines"));
  context.vkDestroyPipeline = reinterpret_cast<PFN_vkDestroyPipeline>(
      vkGetDeviceProcAddr(context.device, "vkDestroyPipeline"));
  auto vkCreateCommandPool = reinterpret_cast<PFN_vkCreateCommandPool>(
      vkGetDeviceProcAddr(context.device, "vkCreateCommandPool"));
  context.vkDestroyCommandPool =
      reinterpret_cast<PFN_vkDestroyCommandPool>(
          vkGetDeviceProcAddr(context.device, "vkDestroyCommandPool"));
  context.vkAllocateCommandBuffers =
      reinterpret_cast<PFN_vkAllocateCommandBuffers>(
          vkGetDeviceProcAddr(context.device, "vkAllocateCommandBuffers"));
  context.vkBeginCommandBuffer =
      reinterpret_cast<PFN_vkBeginCommandBuffer>(
          vkGetDeviceProcAddr(context.device, "vkBeginCommandBuffer"));
  context.vkCmdBindPipeline = reinterpret_cast<PFN_vkCmdBindPipeline>(
      vkGetDeviceProcAddr(context.device, "vkCmdBindPipeline"));
  context.vkCmdBindDescriptorSets =
      reinterpret_cast<PFN_vkCmdBindDescriptorSets>(
          vkGetDeviceProcAddr(context.device, "vkCmdBindDescriptorSets"));
  context.vkCmdPushConstants = reinterpret_cast<PFN_vkCmdPushConstants>(
      vkGetDeviceProcAddr(context.device, "vkCmdPushConstants"));
  context.vkCmdDispatch = reinterpret_cast<PFN_vkCmdDispatch>(
      vkGetDeviceProcAddr(context.device, "vkCmdDispatch"));
  context.vkCmdPipelineBarrier =
      reinterpret_cast<PFN_vkCmdPipelineBarrier>(
          vkGetDeviceProcAddr(context.device, "vkCmdPipelineBarrier"));
  context.vkCmdCopyBuffer = reinterpret_cast<PFN_vkCmdCopyBuffer>(
      vkGetDeviceProcAddr(context.device, "vkCmdCopyBuffer"));
  context.vkCmdFillBuffer = reinterpret_cast<PFN_vkCmdFillBuffer>(
      vkGetDeviceProcAddr(context.device, "vkCmdFillBuffer"));
  context.vkEndCommandBuffer = reinterpret_cast<PFN_vkEndCommandBuffer>(
      vkGetDeviceProcAddr(context.device, "vkEndCommandBuffer"));
  context.vkQueueSubmit = reinterpret_cast<PFN_vkQueueSubmit>(
      vkGetDeviceProcAddr(context.device, "vkQueueSubmit"));
  context.vkQueueWaitIdle = reinterpret_cast<PFN_vkQueueWaitIdle>(
      vkGetDeviceProcAddr(context.device, "vkQueueWaitIdle"));
  context.vkResetCommandPool =
      reinterpret_cast<PFN_vkResetCommandPool>(
          vkGetDeviceProcAddr(context.device, "vkResetCommandPool"));
  context.vkDeviceWaitIdle = reinterpret_cast<PFN_vkDeviceWaitIdle>(
      vkGetDeviceProcAddr(context.device, "vkDeviceWaitIdle"));

  if (!context.vkDestroyDevice || !vkGetDeviceQueue || !context.vkCreateBuffer ||
      !context.vkDestroyBuffer || !context.vkGetBufferMemoryRequirements ||
      !context.vkAllocateMemory || !context.vkFreeMemory ||
      !context.vkBindBufferMemory || !context.vkMapMemory ||
      !context.vkUnmapMemory || !context.vkCreateDescriptorSetLayout ||
      !context.vkDestroyDescriptorSetLayout ||
      !context.vkCreateDescriptorPool || !context.vkDestroyDescriptorPool ||
      !context.vkAllocateDescriptorSets || !context.vkUpdateDescriptorSets ||
      !context.vkCreateShaderModule || !context.vkDestroyShaderModule ||
      !context.vkCreatePipelineLayout || !context.vkDestroyPipelineLayout ||
      !context.vkCreateComputePipelines || !context.vkDestroyPipeline ||
      !vkCreateCommandPool || !context.vkDestroyCommandPool ||
      !context.vkAllocateCommandBuffers || !context.vkBeginCommandBuffer ||
      !context.vkCmdBindPipeline || !context.vkCmdBindDescriptorSets ||
      !context.vkCmdPushConstants || !context.vkCmdDispatch ||
      !context.vkCmdPipelineBarrier || !context.vkCmdCopyBuffer ||
      !context.vkCmdFillBuffer ||
      !context.vkEndCommandBuffer ||
      !context.vkQueueSubmit || !context.vkQueueWaitIdle ||
      !context.vkResetCommandPool || !context.vkDeviceWaitIdle) {
    return fail("one or more required Vulkan device entry points are missing.");
  }

  vkGetDeviceQueue(context.device, context.queueFamilyIndex, 0, &context.queue);
  if (context.queue == VK_NULL_HANDLE) {
    return fail("failed to retrieve a Vulkan compute queue.");
  }

  vkGetPhysicalDeviceMemoryProperties(context.physicalDevice,
                                      &context.memoryProperties);

  VkCommandPoolCreateInfo commandPoolInfo{};
  commandPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  commandPoolInfo.queueFamilyIndex = context.queueFamilyIndex;
  commandPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

  result = vkCreateCommandPool(context.device, &commandPoolInfo, nullptr,
                               &context.commandPool);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "vkCreateCommandPool failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  std::vector<uint32_t> particleShaderSpirv;
  std::vector<uint32_t> coverageShaderSpirv;
  std::vector<uint32_t> activeVoxelCompactShaderSpirv;
  std::vector<uint32_t> activeVoxelPairFillShaderSpirv;
  std::vector<uint32_t> scalarFieldShaderSpirv;
  std::vector<uint32_t> zhuScalarFieldShaderSpirv;
  std::vector<uint32_t> surfaceCellShaderSpirv;
  std::vector<uint32_t> sparseSurfaceCellShaderSpirv;
  std::vector<uint32_t> surfaceCellCompactShaderSpirv;
  std::vector<uint32_t> surfaceTriangleCompactShaderSpirv;
  std::vector<uint32_t> surfaceMeshShaderSpirv;
  std::vector<uint32_t> denseSurfaceMeshShaderSpirv;
  if (!compile_compute_shader_spirv(particleShaderSpirv, outError) ||
      !compile_voxel_coverage_shader_spirv(coverageShaderSpirv, outError) ||
      !compile_active_voxel_compact_shader_spirv(
          activeVoxelCompactShaderSpirv, outError) ||
      !compile_compute_shader_source_spirv(kActiveVoxelPairFillComputeShader,
                                           activeVoxelPairFillShaderSpirv,
                                           outError) ||
      !compile_voxel_scalar_field_shader_spirv(scalarFieldShaderSpirv,
                                               outError) ||
      !compile_zhu_bridson_scalar_field_shader_spirv(
          zhuScalarFieldShaderSpirv,
                                               outError) ||
      !compile_surface_cell_classify_shader_spirv(surfaceCellShaderSpirv,
                                                  outError) ||
      !compile_compute_shader_source_spirv(
          kSparseSurfaceCellFromActiveVoxelComputeShader,
          sparseSurfaceCellShaderSpirv, outError) ||
      !compile_surface_cell_compact_shader_spirv(
          surfaceCellCompactShaderSpirv, outError) ||
      !compile_surface_triangle_compact_shader_spirv(
          surfaceTriangleCompactShaderSpirv, outError) ||
      !compile_surface_mesh_shader_spirv(surfaceMeshShaderSpirv,
                                         outError) ||
      !compile_compute_shader_source_spirv(
          build_dense_surface_mesh_compute_shader_source().c_str(),
          denseSurfaceMeshShaderSpirv, outError)) {
    return fail(outError);
  }

  VkDescriptorSetLayoutBinding particleBindings[5]{};
  particleBindings[0].binding = 0;
  particleBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  particleBindings[0].descriptorCount = 1;
  particleBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  particleBindings[1].binding = 1;
  particleBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  particleBindings[1].descriptorCount = 1;
  particleBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  particleBindings[2].binding = 2;
  particleBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  particleBindings[2].descriptorCount = 1;
  particleBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  particleBindings[3].binding = 3;
  particleBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  particleBindings[3].descriptorCount = 1;
  particleBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  particleBindings[4].binding = 4;
  particleBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  particleBindings[4].descriptorCount = 1;
  particleBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding coverageBindings[3]{};
  coverageBindings[0].binding = 0;
  coverageBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  coverageBindings[0].descriptorCount = 1;
  coverageBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  coverageBindings[1].binding = 1;
  coverageBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  coverageBindings[1].descriptorCount = 1;
  coverageBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  coverageBindings[2].binding = 2;
  coverageBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  coverageBindings[2].descriptorCount = 1;
  coverageBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding activeVoxelCompactBindings[3]{};
  activeVoxelCompactBindings[0].binding = 0;
  activeVoxelCompactBindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  activeVoxelCompactBindings[0].descriptorCount = 1;
  activeVoxelCompactBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  activeVoxelCompactBindings[1].binding = 1;
  activeVoxelCompactBindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  activeVoxelCompactBindings[1].descriptorCount = 1;
  activeVoxelCompactBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  activeVoxelCompactBindings[2].binding = 2;
  activeVoxelCompactBindings[2].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  activeVoxelCompactBindings[2].descriptorCount = 1;
  activeVoxelCompactBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding activeVoxelPairFillBindings[6]{};
  for (uint32_t i = 0; i < 6; ++i) {
    activeVoxelPairFillBindings[i].binding = i;
    activeVoxelPairFillBindings[i].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    activeVoxelPairFillBindings[i].descriptorCount = 1;
    activeVoxelPairFillBindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutBinding scalarFieldBindings[9]{};
  scalarFieldBindings[0].binding = 0;
  scalarFieldBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[0].descriptorCount = 1;
  scalarFieldBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[1].binding = 1;
  scalarFieldBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[1].descriptorCount = 1;
  scalarFieldBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[2].binding = 2;
  scalarFieldBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[2].descriptorCount = 1;
  scalarFieldBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[3].binding = 3;
  scalarFieldBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[3].descriptorCount = 1;
  scalarFieldBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[4].binding = 4;
  scalarFieldBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[4].descriptorCount = 1;
  scalarFieldBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[5].binding = 5;
  scalarFieldBindings[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[5].descriptorCount = 1;
  scalarFieldBindings[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[6].binding = 6;
  scalarFieldBindings[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[6].descriptorCount = 1;
  scalarFieldBindings[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[7].binding = 7;
  scalarFieldBindings[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[7].descriptorCount = 1;
  scalarFieldBindings[7].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  scalarFieldBindings[8].binding = 8;
  scalarFieldBindings[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  scalarFieldBindings[8].descriptorCount = 1;
  scalarFieldBindings[8].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding surfaceCellBindings[3]{};
  surfaceCellBindings[0].binding = 0;
  surfaceCellBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellBindings[0].descriptorCount = 1;
  surfaceCellBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceCellBindings[1].binding = 1;
  surfaceCellBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellBindings[1].descriptorCount = 1;
  surfaceCellBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceCellBindings[2].binding = 2;
  surfaceCellBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellBindings[2].descriptorCount = 1;
  surfaceCellBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding surfaceMeshBindings[5]{};
  surfaceMeshBindings[0].binding = 0;
  surfaceMeshBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceMeshBindings[0].descriptorCount = 1;
  surfaceMeshBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceMeshBindings[1].binding = 1;
  surfaceMeshBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceMeshBindings[1].descriptorCount = 1;
  surfaceMeshBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceMeshBindings[2].binding = 2;
  surfaceMeshBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceMeshBindings[2].descriptorCount = 1;
  surfaceMeshBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceMeshBindings[3].binding = 3;
  surfaceMeshBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceMeshBindings[3].descriptorCount = 1;
  surfaceMeshBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceMeshBindings[4].binding = 4;
  surfaceMeshBindings[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceMeshBindings[4].descriptorCount = 1;
  surfaceMeshBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding sparseSurfaceCellBindings[3]{};
  sparseSurfaceCellBindings[0].binding = 0;
  sparseSurfaceCellBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sparseSurfaceCellBindings[0].descriptorCount = 1;
  sparseSurfaceCellBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  sparseSurfaceCellBindings[1].binding = 1;
  sparseSurfaceCellBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sparseSurfaceCellBindings[1].descriptorCount = 1;
  sparseSurfaceCellBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  sparseSurfaceCellBindings[2].binding = 2;
  sparseSurfaceCellBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  sparseSurfaceCellBindings[2].descriptorCount = 1;
  sparseSurfaceCellBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding surfaceCellCompactBindings[4]{};
  surfaceCellCompactBindings[0].binding = 0;
  surfaceCellCompactBindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellCompactBindings[0].descriptorCount = 1;
  surfaceCellCompactBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceCellCompactBindings[1].binding = 1;
  surfaceCellCompactBindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellCompactBindings[1].descriptorCount = 1;
  surfaceCellCompactBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceCellCompactBindings[2].binding = 2;
  surfaceCellCompactBindings[2].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellCompactBindings[2].descriptorCount = 1;
  surfaceCellCompactBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceCellCompactBindings[3].binding = 3;
  surfaceCellCompactBindings[3].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceCellCompactBindings[3].descriptorCount = 1;
  surfaceCellCompactBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding surfaceTriangleCompactBindings[5]{};
  surfaceTriangleCompactBindings[0].binding = 0;
  surfaceTriangleCompactBindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceTriangleCompactBindings[0].descriptorCount = 1;
  surfaceTriangleCompactBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceTriangleCompactBindings[1].binding = 1;
  surfaceTriangleCompactBindings[1].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceTriangleCompactBindings[1].descriptorCount = 1;
  surfaceTriangleCompactBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceTriangleCompactBindings[2].binding = 2;
  surfaceTriangleCompactBindings[2].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceTriangleCompactBindings[2].descriptorCount = 1;
  surfaceTriangleCompactBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceTriangleCompactBindings[3].binding = 3;
  surfaceTriangleCompactBindings[3].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceTriangleCompactBindings[3].descriptorCount = 1;
  surfaceTriangleCompactBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  surfaceTriangleCompactBindings[4].binding = 4;
  surfaceTriangleCompactBindings[4].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  surfaceTriangleCompactBindings[4].descriptorCount = 1;
  surfaceTriangleCompactBindings[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  VkDescriptorSetLayoutBinding denseSurfaceMeshBindings[4]{};
  denseSurfaceMeshBindings[0].binding = 0;
  denseSurfaceMeshBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  denseSurfaceMeshBindings[0].descriptorCount = 1;
  denseSurfaceMeshBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  denseSurfaceMeshBindings[1].binding = 1;
  denseSurfaceMeshBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  denseSurfaceMeshBindings[1].descriptorCount = 1;
  denseSurfaceMeshBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  denseSurfaceMeshBindings[2].binding = 2;
  denseSurfaceMeshBindings[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  denseSurfaceMeshBindings[2].descriptorCount = 1;
  denseSurfaceMeshBindings[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  denseSurfaceMeshBindings[3].binding = 3;
  denseSurfaceMeshBindings[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  denseSurfaceMeshBindings[3].descriptorCount = 1;
  denseSurfaceMeshBindings[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

  if (!create_shared_compute_pipeline(
          particleBindings, 5, sizeof(PushConstants), particleShaderSpirv,
          context.particleDescriptorSetLayout, context.particlePipelineLayout,
          context.particlePipeline, "particle pipeline") ||
      !create_shared_compute_pipeline(
          coverageBindings, 3, sizeof(CoveragePushConstants),
          coverageShaderSpirv, context.coverageDescriptorSetLayout,
          context.coveragePipelineLayout, context.coveragePipeline,
          "coverage pipeline") ||
      !create_shared_compute_pipeline(
          activeVoxelCompactBindings, 3,
          sizeof(ActiveVoxelCompactPushConstants),
          activeVoxelCompactShaderSpirv,
          context.activeVoxelCompactDescriptorSetLayout,
          context.activeVoxelCompactPipelineLayout,
          context.activeVoxelCompactPipeline,
          "active-voxel compaction pipeline") ||
      !create_shared_compute_pipeline(
          activeVoxelPairFillBindings, 6,
          sizeof(ActiveVoxelPairFillPushConstants),
          activeVoxelPairFillShaderSpirv,
          context.activeVoxelPairFillDescriptorSetLayout,
          context.activeVoxelPairFillPipelineLayout,
          context.activeVoxelPairFillPipeline,
          "active-voxel pair-fill pipeline") ||
      !create_shared_compute_pipeline(
          scalarFieldBindings, 9, sizeof(ScalarFieldPushConstants),
          scalarFieldShaderSpirv, context.scalarFieldDescriptorSetLayout,
          context.scalarFieldPipelineLayout, context.scalarFieldPipeline,
          "scalar-field pipeline") ||
      !create_shared_compute_pipeline_variant(
          zhuScalarFieldShaderSpirv, context.scalarFieldPipelineLayout,
          context.zhuScalarFieldPipeline,
          "zhu scalar-field pipeline") ||
      !create_shared_compute_pipeline(
          surfaceCellBindings, 3, sizeof(SurfaceCellPushConstants),
          surfaceCellShaderSpirv, context.surfaceCellDescriptorSetLayout,
          context.surfaceCellPipelineLayout, context.surfaceCellPipeline,
          "surface-cell pipeline") ||
      !create_shared_compute_pipeline(
          sparseSurfaceCellBindings, 3, sizeof(SurfaceCellPushConstants),
          sparseSurfaceCellShaderSpirv,
          context.sparseSurfaceCellDescriptorSetLayout,
          context.sparseSurfaceCellPipelineLayout,
          context.sparseSurfaceCellPipeline,
          "surface-cell pipeline") ||
      !create_shared_compute_pipeline(
          surfaceCellCompactBindings, 4,
          sizeof(SurfaceCellCompactPushConstants),
          surfaceCellCompactShaderSpirv,
          context.surfaceCellCompactDescriptorSetLayout,
          context.surfaceCellCompactPipelineLayout,
          context.surfaceCellCompactPipeline,
          "surface-cell compaction pipeline") ||
      !create_shared_compute_pipeline(
          surfaceTriangleCompactBindings, 5,
          sizeof(SurfaceMeshPushConstants),
          surfaceTriangleCompactShaderSpirv,
          context.surfaceTriangleCompactDescriptorSetLayout,
          context.surfaceTriangleCompactPipelineLayout,
          context.surfaceTriangleCompactPipeline,
          "surface-triangle compaction pipeline") ||
      !create_shared_compute_pipeline(
          surfaceMeshBindings, 5, sizeof(SurfaceMeshPushConstants),
          surfaceMeshShaderSpirv, context.surfaceMeshDescriptorSetLayout,
          context.surfaceMeshPipelineLayout, context.surfaceMeshPipeline,
          "surface-mesh pipeline") ||
      !create_shared_compute_pipeline(
          denseSurfaceMeshBindings, 4, sizeof(SurfaceMeshPushConstants),
          denseSurfaceMeshShaderSpirv,
          context.denseSurfaceMeshDescriptorSetLayout,
          context.denseSurfaceMeshPipelineLayout,
          context.denseSurfaceMeshPipeline,
          "surface-mesh pipeline")) {
    return fail(outError);
  }

  if (!ensure_shared_compute_dispatch_resources(outError) ||
      !ensure_shared_surface_dispatch_resources(outError)) {
    return fail(outError);
  }

  context.initialized = true;
  context.initError.clear();
  outError.clear();
  return true;
}

} // namespace

bool run_frost_vulkan_compute_particles(const float *inputParticles,
                                        const float *inputVelocities,
                                        std::size_t particleCount,
                                        const VulkanParticleComputeSettings &settings,
                                        VulkanParticleComputeResult &outResult,
                                        std::string &outError) {
  outResult.particles.clear();
  outResult.velocities.clear();
  outResult.minVoxelBounds.clear();
  outResult.maxVoxelBoundsExclusive.clear();
  outResult.voxelCoverageCounts.clear();
  outResult.activeVoxelIndices.clear();
  outResult.activeVoxelCompactLookup.clear();
  outResult.activeVoxelParticleOffsets.clear();
  outResult.activeVoxelParticleIndices.clear();
  outResult.voxelScalarField.clear();
  outResult.scalarFieldResidentOnGpu = false;
  outResult.scalarFieldMode = VulkanScalarFieldMode::coverage_fallback;
  outResult.domainMinVoxel[0] = 0;
  outResult.domainMinVoxel[1] = 0;
  outResult.domainMinVoxel[2] = 0;
  outResult.domainDimensions[0] = 0;
  outResult.domainDimensions[1] = 0;
  outResult.domainDimensions[2] = 0;
  outResult.activeVoxelCount = 0;
  outResult.maxVoxelCoverage = 0;
  outResult.coveredParticleVoxelPairs = 0;
  outResult.voxelLength = settings.voxelLength;
  outResult.planningRadiusScale = settings.planningRadiusScale;
  outResult.fieldRadiusScale = settings.fieldRadiusScale;
  outResult.fieldThreshold = settings.fieldThreshold;
  outResult.surfaceIsoValue = settings.surfaceIsoValue;
  outResult.anisotropyMaxScale = settings.anisotropyMaxScale;
  outResult.kernelSupportRadius = settings.kernelSupportRadius;

  if (!inputParticles || particleCount == 0) {
    outError = "no particle data was provided to the Vulkan compute shader";
    return false;
  }

  if (settings.planningRadiusScale <= 0.0f) {
    outError = "planning radius scale must be greater than zero";
    return false;
  }

  if (settings.voxelLength <= 0.0f) {
    outError = "voxel length must be greater than zero for Vulkan particle planning";
    return false;
  }

  if (particleCount > 65535u * 64u) {
    outError =
        "particle count exceeds the current single-dispatch Vulkan compute limit";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::unique_lock<std::mutex> sharedContextLock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  LibraryHandle library = nullptr;
  VkInstance instance = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  VkBuffer inputBuffer = VK_NULL_HANDLE;
  VkBuffer outputBuffer = VK_NULL_HANDLE;
  VkBuffer velocityBuffer = VK_NULL_HANDLE;
  VkBuffer minBoundsBuffer = VK_NULL_HANDLE;
  VkBuffer maxBoundsBuffer = VK_NULL_HANDLE;
  VkBuffer voxelCoverageBuffer = VK_NULL_HANDLE;
  VkBuffer activeVoxelParticleOffsetBuffer = VK_NULL_HANDLE;
  VkBuffer activeVoxelParticleIndexBuffer = VK_NULL_HANDLE;
  VkBuffer voxelScalarFieldBuffer = VK_NULL_HANDLE;
  VkDeviceMemory inputMemory = VK_NULL_HANDLE;
  VkDeviceMemory outputMemory = VK_NULL_HANDLE;
  VkDeviceMemory minBoundsMemory = VK_NULL_HANDLE;
  VkDeviceMemory maxBoundsMemory = VK_NULL_HANDLE;
  VkDeviceMemory voxelCoverageMemory = VK_NULL_HANDLE;
  VkDeviceMemory voxelScalarFieldMemory = VK_NULL_HANDLE;
  VkCommandPool commandPool = VK_NULL_HANDLE;
  VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipeline pipeline = VK_NULL_HANDLE;
  VkShaderModule shaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout coverageDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool coverageDescriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout coveragePipelineLayout = VK_NULL_HANDLE;
  VkPipeline coveragePipeline = VK_NULL_HANDLE;
  VkShaderModule coverageShaderModule = VK_NULL_HANDLE;
  VkDescriptorSetLayout scalarFieldDescriptorSetLayout = VK_NULL_HANDLE;
  VkDescriptorPool scalarFieldDescriptorPool = VK_NULL_HANDLE;
  VkPipelineLayout scalarFieldPipelineLayout = VK_NULL_HANDLE;
  VkPipeline scalarFieldPipeline = VK_NULL_HANDLE;
  VkShaderModule scalarFieldShaderModule = VK_NULL_HANDLE;

  PFN_vkDestroyInstance pfnDestroyInstance = nullptr;
  PFN_vkDestroyDevice pfnDestroyDevice = nullptr;
  PFN_vkDestroyBuffer pfnDestroyBuffer = nullptr;
  PFN_vkFreeMemory pfnFreeMemory = nullptr;
  PFN_vkDestroyDescriptorSetLayout pfnDestroyDescriptorSetLayout = nullptr;
  PFN_vkDestroyDescriptorPool pfnDestroyDescriptorPool = nullptr;
  PFN_vkDestroyShaderModule pfnDestroyShaderModule = nullptr;
  PFN_vkDestroyPipelineLayout pfnDestroyPipelineLayout = nullptr;
  PFN_vkDestroyPipeline pfnDestroyPipeline = nullptr;
  PFN_vkDestroyCommandPool pfnDestroyCommandPool = nullptr;
  PFN_vkQueueWaitIdle vkQueueWaitIdle = nullptr;
  PFN_vkResetCommandPool vkResetCommandPool = nullptr;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
      if (descriptorPool != VK_NULL_HANDLE && pfnDestroyDescriptorPool) {
        pfnDestroyDescriptorPool(device, descriptorPool, nullptr);
      }
      if (coverageDescriptorPool != VK_NULL_HANDLE &&
          pfnDestroyDescriptorPool) {
        pfnDestroyDescriptorPool(device, coverageDescriptorPool, nullptr);
      }
      if (scalarFieldDescriptorPool != VK_NULL_HANDLE &&
          pfnDestroyDescriptorPool) {
        pfnDestroyDescriptorPool(device, scalarFieldDescriptorPool, nullptr);
      }
      if (pfnDestroyDevice) {
        pfnDestroyDevice(device, nullptr);
      }
    }

    if (instance != VK_NULL_HANDLE && pfnDestroyInstance) {
      pfnDestroyInstance(instance, nullptr);
    }

    if (library) {
      close_vulkan_library(library);
    }
  };

  auto fail = [&](const std::string &message) {
    outError = message;
    cleanup();
    return false;
  };

  device = sharedContext.device;
  queue = sharedContext.queue;
  commandPool = sharedContext.commandPool;
  pfnDestroyBuffer = sharedContext.vkDestroyBuffer;
  pfnFreeMemory = sharedContext.vkFreeMemory;
  pfnDestroyDescriptorSetLayout = sharedContext.vkDestroyDescriptorSetLayout;
  pfnDestroyDescriptorPool = sharedContext.vkDestroyDescriptorPool;
  pfnDestroyShaderModule = sharedContext.vkDestroyShaderModule;
  pfnDestroyPipelineLayout = sharedContext.vkDestroyPipelineLayout;
  pfnDestroyPipeline = sharedContext.vkDestroyPipeline;
  vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto vkCreateBuffer = sharedContext.vkCreateBuffer;
  auto vkGetBufferMemoryRequirements =
      sharedContext.vkGetBufferMemoryRequirements;
  auto vkAllocateMemory = sharedContext.vkAllocateMemory;
  auto vkBindBufferMemory = sharedContext.vkBindBufferMemory;
  auto vkMapMemory = sharedContext.vkMapMemory;
  auto vkUnmapMemory = sharedContext.vkUnmapMemory;
  auto vkCreateDescriptorSetLayout =
      sharedContext.vkCreateDescriptorSetLayout;
  auto vkCreateDescriptorPool = sharedContext.vkCreateDescriptorPool;
  auto vkAllocateDescriptorSets = sharedContext.vkAllocateDescriptorSets;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkCreateShaderModule = sharedContext.vkCreateShaderModule;
  auto vkCreatePipelineLayout = sharedContext.vkCreatePipelineLayout;
  auto vkCreateComputePipelines = sharedContext.vkCreateComputePipelines;
  auto vkAllocateCommandBuffers = sharedContext.vkAllocateCommandBuffers;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  VkResult result = VK_SUCCESS;
  descriptorSetLayout = sharedContext.particleDescriptorSetLayout;
  pipelineLayout = sharedContext.particlePipelineLayout;
  pipeline = sharedContext.particlePipeline;
  coverageDescriptorSetLayout = sharedContext.coverageDescriptorSetLayout;
  coveragePipelineLayout = sharedContext.coveragePipelineLayout;
  coveragePipeline = sharedContext.coveragePipeline;
  scalarFieldDescriptorSetLayout =
      sharedContext.scalarFieldDescriptorSetLayout;
  scalarFieldPipelineLayout = sharedContext.scalarFieldPipelineLayout;
  scalarFieldPipeline = sharedContext.scalarFieldPipeline;

  if (vkResetCommandPool &&
      vkResetCommandPool(device, commandPool, 0) != VK_SUCCESS) {
    return fail("vkResetCommandPool failed for the shared Vulkan command pool.");
  }

  const VkDeviceSize bufferSize =
      static_cast<VkDeviceSize>(particleCount * sizeof(float) * 4);
  const VkDeviceSize voxelBoundsBufferSize =
      static_cast<VkDeviceSize>(particleCount * sizeof(int32_t) * 4);
  if (!ensure_shared_host_buffer(sharedContext, sharedContext.particleInputBuffer,
                                 bufferSize, "particle input buffer",
                                 outError) ||
      !ensure_shared_host_buffer(sharedContext,
                                 sharedContext.particleOutputBuffer,
                                 bufferSize, "particle output buffer",
                                 outError) ||
      !ensure_shared_host_buffer(sharedContext,
                                 sharedContext.particleVelocityBuffer,
                                 bufferSize, "particle velocity buffer",
                                 outError) ||
      !ensure_shared_host_buffer(sharedContext,
                                 sharedContext.particleMinBoundsBuffer,
                                 voxelBoundsBufferSize,
                                 "particle min-bounds buffer", outError) ||
      !ensure_shared_host_buffer(sharedContext,
                                 sharedContext.particleMaxBoundsBuffer,
                                 voxelBoundsBufferSize,
                                 "particle max-bounds buffer", outError)) {
    return fail(outError);
  }

  inputBuffer = sharedContext.particleInputBuffer.buffer;
  outputBuffer = sharedContext.particleOutputBuffer.buffer;
  velocityBuffer = sharedContext.particleVelocityBuffer.buffer;
  minBoundsBuffer = sharedContext.particleMinBoundsBuffer.buffer;
  maxBoundsBuffer = sharedContext.particleMaxBoundsBuffer.buffer;
  std::memcpy(sharedContext.particleInputBuffer.mapped, inputParticles,
              static_cast<size_t>(bufferSize));
  if (inputVelocities) {
    std::memcpy(sharedContext.particleVelocityBuffer.mapped, inputVelocities,
                static_cast<size_t>(bufferSize));
    if (settings.retainParticleData) {
      outResult.velocities.resize(particleCount * 4);
      std::memcpy(outResult.velocities.data(), inputVelocities,
                  static_cast<size_t>(bufferSize));
    }
  } else {
    std::memset(sharedContext.particleVelocityBuffer.mapped, 0,
                static_cast<size_t>(bufferSize));
  }

  VkDescriptorSet descriptorSet = sharedContext.particleDescriptorSet;
  VkCommandBuffer commandBuffer = sharedContext.particleCommandBuffer;
  if (descriptorSet == VK_NULL_HANDLE || commandBuffer == VK_NULL_HANDLE) {
    return fail(
        "Shared Vulkan particle compute resources are unavailable.");
  }

  VkDescriptorBufferInfo bufferInfos[5]{};
  bufferInfos[0].buffer = inputBuffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = bufferSize;
  bufferInfos[1].buffer = outputBuffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = bufferSize;
  bufferInfos[2].buffer = minBoundsBuffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = voxelBoundsBufferSize;
  bufferInfos[3].buffer = maxBoundsBuffer;
  bufferInfos[3].offset = 0;
  bufferInfos[3].range = voxelBoundsBufferSize;
  bufferInfos[4].buffer = velocityBuffer;
  bufferInfos[4].offset = 0;
  bufferInfos[4].range = bufferSize;

  VkWriteDescriptorSet writes[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  result = vkBeginCommandBuffer(commandBuffer, &beginInfo);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "vkBeginCommandBuffer failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

  PushConstants pushConstants{};
  pushConstants.particleCount = static_cast<uint32_t>(particleCount);
  pushConstants.planningRadiusScale = settings.planningRadiusScale;
  pushConstants.voxelLength = settings.voxelLength;
  pushConstants.fieldMode = static_cast<uint32_t>(settings.fieldMode);
  pushConstants.anisotropyMaxScale = settings.anisotropyMaxScale;
  pushConstants.kernelSupportRadius = settings.kernelSupportRadius;
  pushConstants.useVelocityBuffer = inputVelocities != nullptr ? 1u : 0u;
  vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(PushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((particleCount + 63) / 64);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);

  result = vkEndCommandBuffer(commandBuffer);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "vkEndCommandBuffer failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;

  result = vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "vkQueueSubmit failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  result = vkQueueWaitIdle(queue);
  if (result != VK_SUCCESS) {
    std::ostringstream stream;
    stream << "vkQueueWaitIdle failed (VkResult " << result << ").";
    return fail(stream.str());
  }

  if (settings.retainParticleData) {
    outResult.particles.resize(particleCount * 4);
    std::memcpy(outResult.particles.data(),
                sharedContext.particleOutputBuffer.mapped,
                static_cast<size_t>(bufferSize));
  }

  outResult.minVoxelBounds.resize(particleCount * 4);
  std::memcpy(outResult.minVoxelBounds.data(),
              sharedContext.particleMinBoundsBuffer.mapped,
              static_cast<size_t>(voxelBoundsBufferSize));

  outResult.maxVoxelBoundsExclusive.resize(particleCount * 4);
  std::memcpy(outResult.maxVoxelBoundsExclusive.data(),
              sharedContext.particleMaxBoundsBuffer.mapped,
              static_cast<size_t>(voxelBoundsBufferSize));

  int32_t domainMinX = 0;
  int32_t domainMinY = 0;
  int32_t domainMinZ = 0;
  int32_t domainMaxX = 0;
  int32_t domainMaxY = 0;
  int32_t domainMaxZ = 0;
  if (reduce_particle_voxel_bounds(outResult.minVoxelBounds,
                                   outResult.maxVoxelBoundsExclusive,
                                   particleCount, domainMinX, domainMinY,
                                   domainMinZ, domainMaxX, domainMaxY,
                                   domainMaxZ)) {
    domainMinX -= 1;
    domainMinY -= 1;
    domainMinZ -= 1;
    domainMaxX += 1;
    domainMaxY += 1;
    domainMaxZ += 1;

    const int64_t domainDimX =
        static_cast<int64_t>(domainMaxX) - domainMinX;
    const int64_t domainDimY =
        static_cast<int64_t>(domainMaxY) - domainMinY;
    const int64_t domainDimZ =
        static_cast<int64_t>(domainMaxZ) - domainMinZ;

    if (domainDimX > 0 && domainDimY > 0 && domainDimZ > 0) {
      outResult.domainMinVoxel[0] = domainMinX;
      outResult.domainMinVoxel[1] = domainMinY;
      outResult.domainMinVoxel[2] = domainMinZ;
      outResult.domainDimensions[0] = static_cast<int32_t>(domainDimX);
      outResult.domainDimensions[1] = static_cast<int32_t>(domainDimY);
      outResult.domainDimensions[2] = static_cast<int32_t>(domainDimZ);

      const uint64_t voxelCount =
          static_cast<uint64_t>(domainDimX) * static_cast<uint64_t>(domainDimY) *
          static_cast<uint64_t>(domainDimZ);

      if (voxelCount > 0 && voxelCount <= kMaxVoxelCoverageCells) {
        const VkDeviceSize coverageBufferSize =
            static_cast<VkDeviceSize>(voxelCount * sizeof(uint32_t));

        bool coverageReady = ensure_shared_host_buffer(
            sharedContext, sharedContext.voxelCoverageBuffer, coverageBufferSize,
            "voxel coverage buffer", outError);
        if (coverageReady) {
          voxelCoverageBuffer = sharedContext.voxelCoverageBuffer.buffer;
          std::memset(sharedContext.voxelCoverageBuffer.mapped, 0,
                      static_cast<size_t>(coverageBufferSize));
        }

        VkDescriptorSet coverageDescriptorSet = sharedContext.coverageDescriptorSet;
        if (coverageReady) {
          coverageReady =
              coverageDescriptorSetLayout != VK_NULL_HANDLE &&
              coveragePipelineLayout != VK_NULL_HANDLE &&
              coveragePipeline != VK_NULL_HANDLE &&
              coverageDescriptorSet != VK_NULL_HANDLE &&
              sharedContext.coverageCommandBuffer != VK_NULL_HANDLE;
        }

        if (coverageReady) {
          VkDescriptorBufferInfo coverageBufferInfos[3]{};
          coverageBufferInfos[0].buffer = minBoundsBuffer;
          coverageBufferInfos[0].offset = 0;
          coverageBufferInfos[0].range = voxelBoundsBufferSize;
          coverageBufferInfos[1].buffer = maxBoundsBuffer;
          coverageBufferInfos[1].offset = 0;
          coverageBufferInfos[1].range = voxelBoundsBufferSize;
          coverageBufferInfos[2].buffer = voxelCoverageBuffer;
          coverageBufferInfos[2].offset = 0;
          coverageBufferInfos[2].range = coverageBufferSize;

          VkWriteDescriptorSet coverageWrites[3]{};
          for (uint32_t i = 0; i < 3; ++i) {
            coverageWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            coverageWrites[i].dstSet = coverageDescriptorSet;
            coverageWrites[i].dstBinding = i;
            coverageWrites[i].descriptorCount = 1;
            coverageWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            coverageWrites[i].pBufferInfo = &coverageBufferInfos[i];
          }
          vkUpdateDescriptorSets(device, 3, coverageWrites, 0, nullptr);
        }

        if (coverageReady) {
            VkCommandBuffer coverageCommandBuffer =
                sharedContext.coverageCommandBuffer;

            if (coverageReady) {
              VkCommandBufferBeginInfo coverageBeginInfo{};
              coverageBeginInfo.sType =
                  VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
              coverageReady = vkBeginCommandBuffer(coverageCommandBuffer,
                                                   &coverageBeginInfo) ==
                              VK_SUCCESS;
            }

            if (coverageReady) {
              vkCmdBindPipeline(coverageCommandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                coveragePipeline);
              vkCmdBindDescriptorSets(coverageCommandBuffer,
                                      VK_PIPELINE_BIND_POINT_COMPUTE,
                                      coveragePipelineLayout, 0, 1,
                                      &coverageDescriptorSet, 0, nullptr);

              CoveragePushConstants coveragePushConstants{};
              coveragePushConstants.particleCount =
                  static_cast<uint32_t>(particleCount);
              coveragePushConstants.domainMinX = domainMinX;
              coveragePushConstants.domainMinY = domainMinY;
              coveragePushConstants.domainMinZ = domainMinZ;
              coveragePushConstants.domainDimX = static_cast<int32_t>(domainDimX);
              coveragePushConstants.domainDimY = static_cast<int32_t>(domainDimY);
              coveragePushConstants.domainDimZ = static_cast<int32_t>(domainDimZ);
              vkCmdPushConstants(coverageCommandBuffer, coveragePipelineLayout,
                                 VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                 sizeof(CoveragePushConstants),
                                 &coveragePushConstants);
              vkCmdDispatch(coverageCommandBuffer, workgroupCount, 1, 1);

              VkMemoryBarrier coverageMemoryBarrier{};
              coverageMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
              coverageMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              coverageMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
              vkCmdPipelineBarrier(
                  coverageCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &coverageMemoryBarrier, 0,
                  nullptr, 0, nullptr);

              coverageReady =
                  vkEndCommandBuffer(coverageCommandBuffer) == VK_SUCCESS;
            }

            if (coverageReady) {
              VkSubmitInfo coverageSubmitInfo{};
              coverageSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
              coverageSubmitInfo.commandBufferCount = 1;
              coverageSubmitInfo.pCommandBuffers = &coverageCommandBuffer;
              coverageReady =
                  vkQueueSubmit(queue, 1, &coverageSubmitInfo,
                                VK_NULL_HANDLE) == VK_SUCCESS &&
                  vkQueueWaitIdle(queue) == VK_SUCCESS;
            }
        }

        if (coverageReady) {
          uint32_t maxCoverage = 0;
          uint64_t coveredParticleVoxelPairs = 0;
          std::vector<uint32_t> activeVoxelIndices;
          activeVoxelIndices.reserve(
              static_cast<std::size_t>(std::min<uint64_t>(voxelCount, 65536ull)));

          const VkDeviceSize activeVoxelIndexCompactionBufferSize =
              static_cast<VkDeviceSize>(voxelCount * sizeof(uint32_t));
          const VkDeviceSize activeVoxelStatsBufferSize =
              static_cast<VkDeviceSize>(4u * sizeof(uint32_t));
          bool gpuActiveVoxelCompactionReady =
              voxelCount <= std::numeric_limits<uint32_t>::max() &&
              ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelIndexBuffer,
                  activeVoxelIndexCompactionBufferSize,
                  "active voxel-index compaction buffer", outError) &&
              ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelStatsBuffer,
                  activeVoxelStatsBufferSize, "active voxel-stats buffer",
                  outError) &&
              sharedContext.activeVoxelCompactDescriptorSetLayout !=
                  VK_NULL_HANDLE &&
              sharedContext.activeVoxelCompactPipelineLayout != VK_NULL_HANDLE &&
              sharedContext.activeVoxelCompactPipeline != VK_NULL_HANDLE &&
              sharedContext.activeVoxelCompactDescriptorSet != VK_NULL_HANDLE &&
              sharedContext.activeVoxelCompactCommandBuffer != VK_NULL_HANDLE;

          if (gpuActiveVoxelCompactionReady) {
            std::memset(sharedContext.activeVoxelStatsBuffer.mapped, 0,
                        static_cast<size_t>(activeVoxelStatsBufferSize));

            VkDescriptorBufferInfo compactBufferInfos[3]{};
            compactBufferInfos[0].buffer = voxelCoverageBuffer;
            compactBufferInfos[0].offset = 0;
            compactBufferInfos[0].range = coverageBufferSize;
            compactBufferInfos[1].buffer = sharedContext.activeVoxelIndexBuffer.buffer;
            compactBufferInfos[1].offset = 0;
            compactBufferInfos[1].range = activeVoxelIndexCompactionBufferSize;
            compactBufferInfos[2].buffer = sharedContext.activeVoxelStatsBuffer.buffer;
            compactBufferInfos[2].offset = 0;
            compactBufferInfos[2].range = activeVoxelStatsBufferSize;

            VkWriteDescriptorSet compactWrites[3]{};
            for (uint32_t i = 0; i < 3; ++i) {
              compactWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
              compactWrites[i].dstSet = sharedContext.activeVoxelCompactDescriptorSet;
              compactWrites[i].dstBinding = i;
              compactWrites[i].descriptorCount = 1;
              compactWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
              compactWrites[i].pBufferInfo = &compactBufferInfos[i];
            }
            vkUpdateDescriptorSets(device, 3, compactWrites, 0, nullptr);

            VkCommandBuffer compactCommandBuffer =
                sharedContext.activeVoxelCompactCommandBuffer;
            VkCommandBufferBeginInfo compactBeginInfo{};
            compactBeginInfo.sType =
                VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            gpuActiveVoxelCompactionReady =
                vkBeginCommandBuffer(compactCommandBuffer, &compactBeginInfo) ==
                VK_SUCCESS;

            if (gpuActiveVoxelCompactionReady) {
              vkCmdBindPipeline(compactCommandBuffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                sharedContext.activeVoxelCompactPipeline);
              vkCmdBindDescriptorSets(
                  compactCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                  sharedContext.activeVoxelCompactPipelineLayout, 0, 1,
                  &sharedContext.activeVoxelCompactDescriptorSet, 0, nullptr);

              ActiveVoxelCompactPushConstants compactPushConstants{};
              compactPushConstants.voxelCount =
                  static_cast<uint32_t>(voxelCount);
              vkCmdPushConstants(
                  compactCommandBuffer,
                  sharedContext.activeVoxelCompactPipelineLayout,
                  VK_SHADER_STAGE_COMPUTE_BIT, 0,
                  sizeof(ActiveVoxelCompactPushConstants),
                  &compactPushConstants);

              const uint32_t compactWorkgroupCount =
                  static_cast<uint32_t>((voxelCount + 63ull) / 64ull);
              vkCmdDispatch(compactCommandBuffer, compactWorkgroupCount, 1, 1);

              VkMemoryBarrier compactMemoryBarrier{};
              compactMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
              compactMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
              compactMemoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
              vkCmdPipelineBarrier(
                  compactCommandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &compactMemoryBarrier, 0,
                  nullptr, 0, nullptr);

              gpuActiveVoxelCompactionReady =
                  vkEndCommandBuffer(compactCommandBuffer) == VK_SUCCESS;
            }

            if (gpuActiveVoxelCompactionReady) {
              VkSubmitInfo compactSubmitInfo{};
              compactSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
              compactSubmitInfo.commandBufferCount = 1;
              compactSubmitInfo.pCommandBuffers = &compactCommandBuffer;
              gpuActiveVoxelCompactionReady =
                  vkQueueSubmit(queue, 1, &compactSubmitInfo,
                                VK_NULL_HANDLE) == VK_SUCCESS &&
                  vkQueueWaitIdle(queue) == VK_SUCCESS;
            }

            if (gpuActiveVoxelCompactionReady) {
              const uint32_t *stats =
                  static_cast<const uint32_t *>(
                      sharedContext.activeVoxelStatsBuffer.mapped);
              const uint32_t activeVoxelCountFromGpu = stats[0];
              maxCoverage = stats[1];
              const bool pairCountOverflowed = stats[3] != 0u;
              coveredParticleVoxelPairs =
                  pairCountOverflowed
                      ? std::numeric_limits<uint64_t>::max()
                      : static_cast<uint64_t>(stats[2]);
              if (activeVoxelCountFromGpu > 0u) {
                activeVoxelIndices.resize(activeVoxelCountFromGpu);
                std::memcpy(activeVoxelIndices.data(),
                            sharedContext.activeVoxelIndexBuffer.mapped,
                            static_cast<size_t>(activeVoxelCountFromGpu) *
                                sizeof(uint32_t));
                if (settings.sortActiveVoxelIndices &&
                    activeVoxelIndices.size() > 1u) {
                  std::sort(activeVoxelIndices.begin(),
                            activeVoxelIndices.end());
                }
              }
            }
          }

          if (settings.readbackCoverageCounts || !gpuActiveVoxelCompactionReady) {
            outResult.voxelCoverageCounts.resize(
                static_cast<std::size_t>(voxelCount));
            std::memcpy(outResult.voxelCoverageCounts.data(),
                        sharedContext.voxelCoverageBuffer.mapped,
                        static_cast<size_t>(coverageBufferSize));
          } else {
            outResult.voxelCoverageCounts.clear();
          }

          if (!gpuActiveVoxelCompactionReady) {
            maxCoverage = 0;
            coveredParticleVoxelPairs = 0;
            activeVoxelIndices.clear();
            for (std::size_t voxelIndex = 0;
                 voxelIndex < outResult.voxelCoverageCounts.size();
                 ++voxelIndex) {
              const uint32_t count = outResult.voxelCoverageCounts[voxelIndex];
              maxCoverage = std::max(maxCoverage, count);
              coveredParticleVoxelPairs += static_cast<uint64_t>(count);
              if (count > 0) {
                activeVoxelIndices.push_back(static_cast<uint32_t>(voxelIndex));
              }
            }
          }

          if (settings.limitToBoundaryActiveVoxels &&
              !outResult.voxelCoverageCounts.empty() &&
              !activeVoxelIndices.empty()) {
            std::vector<uint32_t> boundaryActiveVoxelIndices;
            uint32_t boundaryMaxCoverage = 0u;
            uint64_t boundaryCoveredParticleVoxelPairs = 0ull;
            filter_boundary_active_voxels(
                activeVoxelIndices, outResult.voxelCoverageCounts,
                static_cast<int32_t>(domainDimX),
                static_cast<int32_t>(domainDimY),
                static_cast<int32_t>(domainDimZ), boundaryActiveVoxelIndices,
                boundaryMaxCoverage, boundaryCoveredParticleVoxelPairs);
            if (!boundaryActiveVoxelIndices.empty() &&
                boundaryActiveVoxelIndices.size() < activeVoxelIndices.size()) {
              activeVoxelIndices.swap(boundaryActiveVoxelIndices);
              maxCoverage = boundaryMaxCoverage;
              coveredParticleVoxelPairs = boundaryCoveredParticleVoxelPairs;
            }
          }

          outResult.activeVoxelCount =
              static_cast<uint32_t>(activeVoxelIndices.size());
          outResult.maxVoxelCoverage = maxCoverage;
          outResult.coveredParticleVoxelPairs = coveredParticleVoxelPairs;
          outResult.activeVoxelIndices = activeVoxelIndices;

          if (maxCoverage > 0 &&
              voxelCount <= kMaxVoxelScalarFieldCells) {
            const uint64_t scalarFieldWorkEstimate =
                coveredParticleVoxelPairs;
            const bool useRequestedParticleField =
                scalarFieldWorkEstimate <=
                kMaxVoxelParticleDistanceEvaluations;
            std::vector<uint32_t> activeVoxelParticleOffsets;
            std::vector<uint32_t> activeVoxelParticleIndices;
            std::vector<int32_t> activeVoxelCompactLookup;
            uint64_t compactedVoxelParticlePairCount = 0;
            bool useCompactedParticlePairs = false;
            bool useGpuCompactedParticlePairs = false;
            if (settings.preferGpuCompactedPairs && useRequestedParticleField &&
                build_active_voxel_lookup_and_offsets(
                    activeVoxelIndices, outResult.voxelCoverageCounts,
                    static_cast<int32_t>(domainDimX),
                    static_cast<int32_t>(domainDimY),
                    static_cast<int32_t>(domainDimZ),
                    kMaxCompactedVoxelParticlePairs,
                    activeVoxelCompactLookup, activeVoxelParticleOffsets,
                    compactedVoxelParticlePairCount)) {
              useCompactedParticlePairs = true;
              useGpuCompactedParticlePairs = true;
            } else if (settings.allowCpuCompactedPairs &&
                       useRequestedParticleField &&
                       build_active_voxel_particle_lists(
                           particleCount, outResult.minVoxelBounds,
                           outResult.maxVoxelBoundsExclusive, domainMinX,
                           domainMinY, domainMinZ,
                           static_cast<int32_t>(domainDimX),
                           static_cast<int32_t>(domainDimY),
                           static_cast<int32_t>(domainDimZ), activeVoxelIndices,
                           kMaxCompactedVoxelParticlePairs,
                           activeVoxelParticleOffsets,
                           activeVoxelParticleIndices,
                           compactedVoxelParticlePairCount,
                           &activeVoxelCompactLookup)) {
              useCompactedParticlePairs = true;
            }
            if (useCompactedParticlePairs && !useGpuCompactedParticlePairs) {
              outResult.activeVoxelCompactLookup = activeVoxelCompactLookup;
              outResult.activeVoxelParticleOffsets = activeVoxelParticleOffsets;
              outResult.activeVoxelParticleIndices = activeVoxelParticleIndices;
            }
            const std::size_t activeVoxelCount = activeVoxelIndices.size();
            const VkDeviceSize activeVoxelIndexBufferSize =
                static_cast<VkDeviceSize>(activeVoxelCount * sizeof(uint32_t));
            const VkDeviceSize activeVoxelCompactLookupBufferSize =
                static_cast<VkDeviceSize>(
                    std::max<std::size_t>(activeVoxelCompactLookup.size(), 1u) *
                    sizeof(int32_t));
            const VkDeviceSize activeVoxelParticleOffsetBufferSize =
                static_cast<VkDeviceSize>(
                    std::max<std::size_t>(activeVoxelParticleOffsets.size(), 1u) *
                    sizeof(uint32_t));
            const VkDeviceSize activeVoxelParticleCursorBufferSize =
                static_cast<VkDeviceSize>(std::max<std::size_t>(activeVoxelCount, 1u) *
                                          sizeof(uint32_t));
            const VkDeviceSize activeVoxelParticleIndexBufferSize =
                static_cast<VkDeviceSize>(
                    std::max<std::size_t>(
                        useGpuCompactedParticlePairs
                            ? static_cast<std::size_t>(compactedVoxelParticlePairCount)
                            : activeVoxelParticleIndices.size(),
                        1u) *
                    sizeof(uint32_t));
            const VkDeviceSize scalarFieldBufferSize =
                static_cast<VkDeviceSize>(voxelCount * sizeof(float));
            bool scalarFieldReady = ensure_shared_host_buffer(
                sharedContext, sharedContext.voxelScalarFieldBuffer,
                scalarFieldBufferSize, "voxel scalar-field buffer", outError);
            if (scalarFieldReady && activeVoxelCount > 0) {
              scalarFieldReady = ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelIndexBuffer,
                  activeVoxelIndexBufferSize, "active voxel-index buffer",
                  outError);
            }
            if (scalarFieldReady) {
              scalarFieldReady = ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelParticleOffsetBuffer,
                  activeVoxelParticleOffsetBufferSize,
                  "active voxel-particle offset buffer", outError);
            }
            if (scalarFieldReady) {
              scalarFieldReady = ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelParticleIndexBuffer,
                  activeVoxelParticleIndexBufferSize,
                  "active voxel-particle index buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelCompactLookupBuffer,
                  activeVoxelCompactLookupBufferSize,
                  "active voxel compact-lookup buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_host_buffer(
                  sharedContext, sharedContext.activeVoxelParticleCursorBuffer,
                  activeVoxelParticleCursorBufferSize,
                  "active voxel-particle cursor buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_device_buffer(
                  sharedContext,
                  sharedContext.deviceActiveVoxelCompactLookupBuffer,
                  activeVoxelCompactLookupBufferSize,
                  "device active voxel compact-lookup buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_device_buffer(
                  sharedContext,
                  sharedContext.deviceActiveVoxelParticleOffsetBuffer,
                  activeVoxelParticleOffsetBufferSize,
                  "device active voxel-particle offset buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_device_buffer(
                  sharedContext,
                  sharedContext.deviceActiveVoxelParticleCursorBuffer,
                  activeVoxelParticleCursorBufferSize,
                  "device active voxel-particle cursor buffer", outError);
            }
            if (scalarFieldReady && useGpuCompactedParticlePairs) {
              scalarFieldReady = ensure_shared_device_buffer(
                  sharedContext,
                  sharedContext.deviceActiveVoxelParticleIndexBuffer,
                  activeVoxelParticleIndexBufferSize,
                  "device active voxel-particle index buffer", outError);
            }
            if (scalarFieldReady) {
              voxelScalarFieldBuffer = sharedContext.voxelScalarFieldBuffer.buffer;
              activeVoxelParticleOffsetBuffer =
                  sharedContext.activeVoxelParticleOffsetBuffer.buffer;
              activeVoxelParticleIndexBuffer =
                  sharedContext.activeVoxelParticleIndexBuffer.buffer;
              std::memset(sharedContext.voxelScalarFieldBuffer.mapped, 0,
                          static_cast<size_t>(scalarFieldBufferSize));
              const float outsideField =
                  compute_vulkan_outside_field(settings);
              float *scalarFieldMapped =
                  static_cast<float *>(sharedContext.voxelScalarFieldBuffer.mapped);
              for (std::size_t i = 0; i < static_cast<std::size_t>(voxelCount);
                   ++i) {
                scalarFieldMapped[i] = outsideField;
              }
              if (activeVoxelCount > 0) {
                std::memcpy(sharedContext.activeVoxelIndexBuffer.mapped,
                            activeVoxelIndices.data(),
                            static_cast<size_t>(activeVoxelIndexBufferSize));
              }
              if (useGpuCompactedParticlePairs &&
                  !activeVoxelCompactLookup.empty()) {
                std::memcpy(sharedContext.activeVoxelCompactLookupBuffer.mapped,
                            activeVoxelCompactLookup.data(),
                            static_cast<size_t>(
                                activeVoxelCompactLookupBufferSize));
              }
              std::memset(sharedContext.activeVoxelParticleOffsetBuffer.mapped, 0,
                          static_cast<size_t>(
                              activeVoxelParticleOffsetBufferSize));
              if (useGpuCompactedParticlePairs) {
                std::memset(sharedContext.activeVoxelParticleCursorBuffer.mapped, 0,
                            static_cast<size_t>(
                                activeVoxelParticleCursorBufferSize));
              }
              std::memset(sharedContext.activeVoxelParticleIndexBuffer.mapped, 0,
                          static_cast<size_t>(
                              activeVoxelParticleIndexBufferSize));
              if (!activeVoxelParticleOffsets.empty()) {
                std::memcpy(sharedContext.activeVoxelParticleOffsetBuffer.mapped,
                            activeVoxelParticleOffsets.data(),
                            static_cast<size_t>(
                                activeVoxelParticleOffsetBufferSize));
              }
              if (!activeVoxelParticleIndices.empty()) {
                std::memcpy(sharedContext.activeVoxelParticleIndexBuffer.mapped,
                            activeVoxelParticleIndices.data(),
                            static_cast<size_t>(
                                activeVoxelParticleIndexBufferSize));
              }
            }

                  if (scalarFieldReady && useGpuCompactedParticlePairs) {
                    scalarFieldReady =
                        sharedContext.activeVoxelPairFillDescriptorSetLayout !=
                            VK_NULL_HANDLE &&
                        sharedContext.activeVoxelPairFillPipelineLayout !=
                            VK_NULL_HANDLE &&
                        sharedContext.activeVoxelPairFillPipeline !=
                            VK_NULL_HANDLE &&
                        sharedContext.activeVoxelPairFillDescriptorSet !=
                            VK_NULL_HANDLE &&
                        sharedContext.activeVoxelPairFillCommandBuffer !=
                            VK_NULL_HANDLE;
                  }

                  if (scalarFieldReady && useGpuCompactedParticlePairs) {
                    VkDescriptorBufferInfo pairFillBufferInfos[6]{};
                    pairFillBufferInfos[0].buffer = minBoundsBuffer;
                    pairFillBufferInfos[0].offset = 0;
                    pairFillBufferInfos[0].range = voxelBoundsBufferSize;
                    pairFillBufferInfos[1].buffer = maxBoundsBuffer;
                    pairFillBufferInfos[1].offset = 0;
                    pairFillBufferInfos[1].range = voxelBoundsBufferSize;
                    pairFillBufferInfos[2].buffer =
                        sharedContext.deviceActiveVoxelCompactLookupBuffer.buffer;
                    pairFillBufferInfos[2].offset = 0;
                    pairFillBufferInfos[2].range =
                        activeVoxelCompactLookupBufferSize;
                    pairFillBufferInfos[3].buffer =
                        sharedContext.deviceActiveVoxelParticleOffsetBuffer.buffer;
                    pairFillBufferInfos[3].offset = 0;
                    pairFillBufferInfos[3].range =
                        activeVoxelParticleOffsetBufferSize;
                    pairFillBufferInfos[4].buffer =
                        sharedContext.deviceActiveVoxelParticleCursorBuffer.buffer;
                    pairFillBufferInfos[4].offset = 0;
                    pairFillBufferInfos[4].range =
                        activeVoxelParticleCursorBufferSize;
                    pairFillBufferInfos[5].buffer =
                        sharedContext.deviceActiveVoxelParticleIndexBuffer.buffer;
                    pairFillBufferInfos[5].offset = 0;
                    pairFillBufferInfos[5].range =
                        activeVoxelParticleIndexBufferSize;

                    VkWriteDescriptorSet pairFillWrites[6]{};
                    for (uint32_t i = 0; i < 6; ++i) {
                      pairFillWrites[i].sType =
                          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                      pairFillWrites[i].dstSet =
                          sharedContext.activeVoxelPairFillDescriptorSet;
                      pairFillWrites[i].dstBinding = i;
                      pairFillWrites[i].descriptorCount = 1;
                      pairFillWrites[i].descriptorType =
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                      pairFillWrites[i].pBufferInfo = &pairFillBufferInfos[i];
                    }
                    vkUpdateDescriptorSets(device, 6, pairFillWrites, 0,
                                           nullptr);

                    VkCommandBuffer pairFillCommandBuffer =
                        sharedContext.activeVoxelPairFillCommandBuffer;
                    VkCommandBufferBeginInfo pairFillBeginInfo{};
                    pairFillBeginInfo.sType =
                        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                    scalarFieldReady =
                        vkBeginCommandBuffer(pairFillCommandBuffer,
                                             &pairFillBeginInfo) == VK_SUCCESS;

                    if (scalarFieldReady) {
                      VkBufferCopy pairFillCopies[3]{};
                      pairFillCopies[0].size = activeVoxelCompactLookupBufferSize;
                      pairFillCopies[1].size = activeVoxelParticleOffsetBufferSize;
                      pairFillCopies[2].size = activeVoxelParticleCursorBufferSize;
                      sharedContext.vkCmdCopyBuffer(
                          pairFillCommandBuffer,
                          sharedContext.activeVoxelCompactLookupBuffer.buffer,
                          sharedContext.deviceActiveVoxelCompactLookupBuffer.buffer, 1,
                          &pairFillCopies[0]);
                      sharedContext.vkCmdCopyBuffer(
                          pairFillCommandBuffer,
                          sharedContext.activeVoxelParticleOffsetBuffer.buffer,
                          sharedContext.deviceActiveVoxelParticleOffsetBuffer.buffer, 1,
                          &pairFillCopies[1]);
                      sharedContext.vkCmdCopyBuffer(
                          pairFillCommandBuffer,
                          sharedContext.activeVoxelParticleCursorBuffer.buffer,
                          sharedContext.deviceActiveVoxelParticleCursorBuffer.buffer, 1,
                          &pairFillCopies[2]);

                      VkMemoryBarrier pairFillUploadBarrier{};
                      pairFillUploadBarrier.sType =
                          VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                      pairFillUploadBarrier.srcAccessMask =
                          VK_ACCESS_TRANSFER_WRITE_BIT;
                      pairFillUploadBarrier.dstAccessMask =
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                      vkCmdPipelineBarrier(
                          pairFillCommandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                          &pairFillUploadBarrier, 0, nullptr, 0, nullptr);

                      vkCmdBindPipeline(pairFillCommandBuffer,
                                        VK_PIPELINE_BIND_POINT_COMPUTE,
                                        sharedContext.activeVoxelPairFillPipeline);
                      vkCmdBindDescriptorSets(
                          pairFillCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.activeVoxelPairFillPipelineLayout, 0, 1,
                          &sharedContext.activeVoxelPairFillDescriptorSet, 0,
                          nullptr);

                      ActiveVoxelPairFillPushConstants pairFillPushConstants{};
                      pairFillPushConstants.particleCount =
                          static_cast<uint32_t>(particleCount);
                      pairFillPushConstants.domainMinX = domainMinX;
                      pairFillPushConstants.domainMinY = domainMinY;
                      pairFillPushConstants.domainMinZ = domainMinZ;
                      pairFillPushConstants.domainDimX =
                          static_cast<int32_t>(domainDimX);
                      pairFillPushConstants.domainDimY =
                          static_cast<int32_t>(domainDimY);
                      pairFillPushConstants.domainDimZ =
                          static_cast<int32_t>(domainDimZ);
                      vkCmdPushConstants(
                          pairFillCommandBuffer,
                          sharedContext.activeVoxelPairFillPipelineLayout,
                          VK_SHADER_STAGE_COMPUTE_BIT, 0,
                          sizeof(ActiveVoxelPairFillPushConstants),
                          &pairFillPushConstants);

                      const uint32_t pairFillWorkgroupCount =
                          static_cast<uint32_t>((particleCount + 63ull) / 64ull);
                      vkCmdDispatch(pairFillCommandBuffer, pairFillWorkgroupCount,
                                    1, 1);

                      VkMemoryBarrier pairFillBarrier{};
                      pairFillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                      pairFillBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
                      pairFillBarrier.dstAccessMask =
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
                      vkCmdPipelineBarrier(
                          pairFillCommandBuffer,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                              VK_PIPELINE_STAGE_HOST_BIT,
                          0, 1, &pairFillBarrier, 0, nullptr, 0, nullptr);

                      scalarFieldReady =
                          vkEndCommandBuffer(pairFillCommandBuffer) == VK_SUCCESS;
                    }

                    if (scalarFieldReady) {
                      VkSubmitInfo pairFillSubmitInfo{};
                      pairFillSubmitInfo.sType =
                          VK_STRUCTURE_TYPE_SUBMIT_INFO;
                      pairFillSubmitInfo.commandBufferCount = 1;
                      pairFillSubmitInfo.pCommandBuffers =
                          &pairFillCommandBuffer;
                      scalarFieldReady =
                          vkQueueSubmit(queue, 1, &pairFillSubmitInfo,
                                        VK_NULL_HANDLE) == VK_SUCCESS &&
                          vkQueueWaitIdle(queue) == VK_SUCCESS;
                    }
                  }

                  VkDescriptorSet scalarFieldDescriptorSet =
                      sharedContext.scalarFieldDescriptorSet;
                  VkBuffer scalarFieldActiveVoxelParticleOffsetBuffer =
                      activeVoxelParticleOffsetBuffer;
                  VkBuffer scalarFieldActiveVoxelParticleIndexBuffer =
                      activeVoxelParticleIndexBuffer;
                  if (useGpuCompactedParticlePairs) {
                    scalarFieldActiveVoxelParticleOffsetBuffer =
                        sharedContext.deviceActiveVoxelParticleOffsetBuffer.buffer;
                    scalarFieldActiveVoxelParticleIndexBuffer =
                        sharedContext.deviceActiveVoxelParticleIndexBuffer.buffer;
                  }

                  if (scalarFieldReady) {
                    scalarFieldReady =
                        scalarFieldDescriptorSetLayout != VK_NULL_HANDLE &&
                        scalarFieldPipelineLayout != VK_NULL_HANDLE &&
                        scalarFieldPipeline != VK_NULL_HANDLE &&
                        scalarFieldDescriptorSet != VK_NULL_HANDLE &&
                        sharedContext.scalarFieldCommandBuffer !=
                            VK_NULL_HANDLE;
                  }

                  if (scalarFieldReady) {
                    VkDescriptorBufferInfo scalarFieldBufferInfos[9]{};
                    scalarFieldBufferInfos[0].buffer = voxelCoverageBuffer;
                    scalarFieldBufferInfos[0].offset = 0;
                    scalarFieldBufferInfos[0].range = coverageBufferSize;
                    scalarFieldBufferInfos[1].buffer =
                        sharedContext.activeVoxelIndexBuffer.buffer;
                    scalarFieldBufferInfos[1].offset = 0;
                    scalarFieldBufferInfos[1].range = activeVoxelIndexBufferSize;
                    scalarFieldBufferInfos[2].buffer = inputBuffer;
                    scalarFieldBufferInfos[2].offset = 0;
                    scalarFieldBufferInfos[2].range = bufferSize;
                    scalarFieldBufferInfos[3].buffer = velocityBuffer;
                    scalarFieldBufferInfos[3].offset = 0;
                    scalarFieldBufferInfos[3].range = bufferSize;
                    scalarFieldBufferInfos[4].buffer = minBoundsBuffer;
                    scalarFieldBufferInfos[4].offset = 0;
                    scalarFieldBufferInfos[4].range = voxelBoundsBufferSize;
                    scalarFieldBufferInfos[5].buffer = maxBoundsBuffer;
                    scalarFieldBufferInfos[5].offset = 0;
                    scalarFieldBufferInfos[5].range = voxelBoundsBufferSize;
                    scalarFieldBufferInfos[6].buffer =
                        scalarFieldActiveVoxelParticleOffsetBuffer;
                    scalarFieldBufferInfos[6].offset = 0;
                    scalarFieldBufferInfos[6].range =
                        activeVoxelParticleOffsetBufferSize;
                    scalarFieldBufferInfos[7].buffer =
                        scalarFieldActiveVoxelParticleIndexBuffer;
                    scalarFieldBufferInfos[7].offset = 0;
                    scalarFieldBufferInfos[7].range =
                        activeVoxelParticleIndexBufferSize;
                    scalarFieldBufferInfos[8].buffer = voxelScalarFieldBuffer;
                    scalarFieldBufferInfos[8].offset = 0;
                    scalarFieldBufferInfos[8].range = scalarFieldBufferSize;

                    VkWriteDescriptorSet scalarFieldWrites[9]{};
                    for (uint32_t i = 0; i < 9; ++i) {
                      scalarFieldWrites[i].sType =
                          VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                      scalarFieldWrites[i].dstSet = scalarFieldDescriptorSet;
                      scalarFieldWrites[i].dstBinding = i;
                      scalarFieldWrites[i].descriptorCount = 1;
                      scalarFieldWrites[i].descriptorType =
                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                      scalarFieldWrites[i].pBufferInfo =
                          &scalarFieldBufferInfos[i];
                    }
                    vkUpdateDescriptorSets(device, 9, scalarFieldWrites, 0,
                                           nullptr);
                  }

                  if (scalarFieldReady) {
                    VkCommandBuffer scalarFieldCommandBuffer =
                        sharedContext.scalarFieldCommandBuffer;

                    if (scalarFieldReady) {
                      VkCommandBufferBeginInfo scalarFieldBeginInfo{};
                      scalarFieldBeginInfo.sType =
                          VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
                      scalarFieldReady =
                          vkBeginCommandBuffer(scalarFieldCommandBuffer,
                                               &scalarFieldBeginInfo) ==
                          VK_SUCCESS;
                    }

                    if (scalarFieldReady) {
                      VkPipeline selectedScalarFieldPipeline =
                          scalarFieldPipeline;
                      if (settings.fieldMode ==
                              VulkanScalarFieldMode::zhu_bridson_blend &&
                          useRequestedParticleField &&
                          sharedContext.zhuScalarFieldPipeline !=
                              VK_NULL_HANDLE) {
                        selectedScalarFieldPipeline =
                            sharedContext.zhuScalarFieldPipeline;
                      }

                      vkCmdBindPipeline(scalarFieldCommandBuffer,
                                        VK_PIPELINE_BIND_POINT_COMPUTE,
                                        selectedScalarFieldPipeline);
                      vkCmdBindDescriptorSets(scalarFieldCommandBuffer,
                                              VK_PIPELINE_BIND_POINT_COMPUTE,
                                              scalarFieldPipelineLayout, 0, 1,
                                              &scalarFieldDescriptorSet, 0,
                                              nullptr);

                      ScalarFieldPushConstants scalarFieldPushConstants{};
                      scalarFieldPushConstants.activeVoxelCount =
                          static_cast<uint32_t>(activeVoxelCount);
                      scalarFieldPushConstants.particleCount =
                          static_cast<uint32_t>(particleCount);
                      scalarFieldPushConstants.domainMinX = domainMinX;
                      scalarFieldPushConstants.domainMinY = domainMinY;
                      scalarFieldPushConstants.domainMinZ = domainMinZ;
                      scalarFieldPushConstants.domainDimX =
                          static_cast<int32_t>(domainDimX);
                      scalarFieldPushConstants.domainDimY =
                          static_cast<int32_t>(domainDimY);
                      scalarFieldPushConstants.fieldMode =
                          static_cast<uint32_t>(settings.fieldMode);
                      scalarFieldPushConstants.voxelLength =
                          settings.voxelLength;
                      scalarFieldPushConstants.fieldRadiusScale =
                          settings.fieldRadiusScale;
                      scalarFieldPushConstants.inverseMaxCoverage =
                          1.0f / static_cast<float>(maxCoverage);
                      scalarFieldPushConstants.fieldThreshold =
                          settings.fieldThreshold;
                      scalarFieldPushConstants.surfaceIsoValue =
                          settings.surfaceIsoValue;
                      scalarFieldPushConstants.anisotropyMaxScale =
                          settings.anisotropyMaxScale;
                      scalarFieldPushConstants.kernelSupportRadius =
                          settings.kernelSupportRadius;
                      scalarFieldPushConstants.useRequestedParticleField =
                          useRequestedParticleField ? 1u : 0u;
                      scalarFieldPushConstants.useVelocityBuffer =
                          inputVelocities != nullptr ? 1u : 0u;
                      scalarFieldPushConstants.useCompactedParticlePairs =
                          useCompactedParticlePairs ? 1u : 0u;
                      vkCmdPushConstants(scalarFieldCommandBuffer,
                                         scalarFieldPipelineLayout,
                                         VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                         sizeof(ScalarFieldPushConstants),
                                         &scalarFieldPushConstants);

                      const uint32_t scalarFieldWorkgroupCount =
                          static_cast<uint32_t>((activeVoxelCount + 63) / 64);
                      vkCmdDispatch(scalarFieldCommandBuffer,
                                    scalarFieldWorkgroupCount, 1, 1);

                      VkMemoryBarrier scalarFieldMemoryBarrier{};
                      scalarFieldMemoryBarrier.sType =
                          VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                      scalarFieldMemoryBarrier.srcAccessMask =
                          VK_ACCESS_SHADER_WRITE_BIT;
                      scalarFieldMemoryBarrier.dstAccessMask =
                          VK_ACCESS_HOST_READ_BIT;
                      vkCmdPipelineBarrier(
                          scalarFieldCommandBuffer,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                          &scalarFieldMemoryBarrier, 0, nullptr, 0, nullptr);

                      scalarFieldReady =
                          vkEndCommandBuffer(scalarFieldCommandBuffer) ==
                          VK_SUCCESS;
                    }

                    if (scalarFieldReady) {
                      VkSubmitInfo scalarFieldSubmitInfo{};
                      scalarFieldSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                      scalarFieldSubmitInfo.commandBufferCount = 1;
                      scalarFieldSubmitInfo.pCommandBuffers =
                          &scalarFieldCommandBuffer;
                      scalarFieldReady =
                          vkQueueSubmit(queue, 1, &scalarFieldSubmitInfo,
                                        VK_NULL_HANDLE) == VK_SUCCESS &&
                          vkQueueWaitIdle(queue) == VK_SUCCESS;
                    }
                  }

            if (scalarFieldReady) {
              outResult.scalarFieldResidentOnGpu = true;
              if (settings.readbackScalarField) {
                outResult.voxelScalarField.resize(
                    static_cast<std::size_t>(voxelCount));
                std::memcpy(outResult.voxelScalarField.data(),
                            sharedContext.voxelScalarFieldBuffer.mapped,
                            static_cast<size_t>(scalarFieldBufferSize));
              } else {
                outResult.voxelScalarField.clear();
              }
              outResult.scalarFieldMode =
                  useRequestedParticleField
                      ? settings.fieldMode
                      : VulkanScalarFieldMode::coverage_fallback;
            }
          }
        }
      }
    }
  }

  cleanup();
  outError.clear();
  return true;
}

bool ensure_frost_vulkan_scalar_field_readback(
    VulkanParticleComputeResult &ioResult, std::string &outError) {
  const int32_t dimX = ioResult.domainDimensions[0];
  const int32_t dimY = ioResult.domainDimensions[1];
  const int32_t dimZ = ioResult.domainDimensions[2];
  if (dimX <= 0 || dimY <= 0 || dimZ <= 0) {
    outError = "Vulkan scalar-field readback requires a valid domain.";
    return false;
  }

  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (!ioResult.voxelScalarField.empty() &&
      ioResult.voxelScalarField.size() == expectedScalarSamples) {
    outError.clear();
    return true;
  }
  if (!ioResult.scalarFieldResidentOnGpu) {
    outError = "Vulkan scalar-field data is not resident on the GPU.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  const VkDeviceSize scalarFieldBufferSize =
      static_cast<VkDeviceSize>(expectedScalarSamples * sizeof(float));
  if (!ensure_shared_host_buffer(sharedContext, sharedContext.voxelScalarFieldBuffer,
                                 scalarFieldBufferSize, "voxel scalar-field buffer",
                                 outError)) {
    return false;
  }

  ioResult.voxelScalarField.resize(expectedScalarSamples);
  std::memcpy(ioResult.voxelScalarField.data(),
              sharedContext.voxelScalarFieldBuffer.mapped,
              static_cast<size_t>(scalarFieldBufferSize));
  outError.clear();
  return true;
}

bool run_frost_vulkan_classify_surface_cells(
    const VulkanParticleComputeResult &computeResult,
    const std::vector<uint32_t> &candidateCellIndices,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    std::string &outError) {
  outActiveCellIndices.clear();
  outActiveCellCubeIndices.clear();

  if (candidateCellIndices.empty()) {
    outError.clear();
    return true;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (dimX < 2 || dimY < 2 || dimZ < 2 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan surface-cell classification requires a valid scalar-field domain.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const std::size_t candidateCount = candidateCellIndices.size();
  const VkDeviceSize candidateCellIndexBufferSize =
      static_cast<VkDeviceSize>(candidateCount * sizeof(uint32_t));
  const VkDeviceSize candidateCubeIndexBufferSize =
      static_cast<VkDeviceSize>(candidateCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));

  bool classifyReady = ensure_shared_host_buffer(
      sharedContext, sharedContext.candidateCellIndexBuffer,
      candidateCellIndexBufferSize, "candidate cell-index buffer", outError);
  if (classifyReady) {
    classifyReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        candidateCubeIndexBufferSize, "candidate cube-index buffer", outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_scalar_field_buffer_for_result(
        sharedContext, computeResult, scalarFieldBufferSize,
        "voxel scalar-field buffer", outError);
  }
  if (!classifyReady) {
    cleanup();
    return false;
  }

  std::memcpy(sharedContext.candidateCellIndexBuffer.mapped,
              candidateCellIndices.data(),
              static_cast<size_t>(candidateCellIndexBufferSize));
  std::memset(sharedContext.candidateCellCubeIndexBuffer.mapped, 0,
              static_cast<size_t>(candidateCubeIndexBufferSize));

  if (sharedContext.surfaceCellDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceCellDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan surface-cell pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo bufferInfos[3]{};
  bufferInfos[0].buffer = sharedContext.candidateCellIndexBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = candidateCellIndexBufferSize;
  bufferInfos[1].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = scalarFieldBufferSize;
  bufferInfos[2].buffer = sharedContext.candidateCellCubeIndexBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = candidateCubeIndexBufferSize;

  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.surfaceCellDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
  VkCommandBuffer commandBuffer = sharedContext.surfaceCellCommandBuffer;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError = "Failed to begin Vulkan surface-cell command buffer.";
    cleanup();
    return false;
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceCellPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceCellPipelineLayout, 0, 1,
                          &sharedContext.surfaceCellDescriptorSet, 0, nullptr);

  SurfaceCellPushConstants pushConstants{};
  pushConstants.candidateCellCount = static_cast<uint32_t>(candidateCount);
  pushConstants.cellDimX = dimX - 1;
  pushConstants.cellDimY = dimY - 1;
  pushConstants.cellDimZ = dimZ - 1;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.surfaceCellPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceCellPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((candidateCount + 63) / 64);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan surface-cell command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan surface-cell classification.";
    cleanup();
    return false;
  }

  std::vector<uint32_t> cubeIndices(candidateCount, 0u);
  std::memcpy(cubeIndices.data(), sharedContext.candidateCellCubeIndexBuffer.mapped,
              static_cast<size_t>(candidateCubeIndexBufferSize));

  outActiveCellIndices.reserve(candidateCount / 2);
  outActiveCellCubeIndices.reserve(candidateCount / 2);
  for (std::size_t i = 0; i < candidateCount; ++i) {
    const uint32_t cubeIndex = cubeIndices[i];
    if (cubeIndex == 0u || cubeIndex == 255u) {
      continue;
    }
    outActiveCellIndices.push_back(candidateCellIndices[i]);
    outActiveCellCubeIndices.push_back(cubeIndex);
  }

  cleanup();
  outError.clear();
  return true;
}

bool run_frost_vulkan_classify_surface_cells_from_active_voxels(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    std::string &outError) {
  outActiveCellIndices.clear();
  outActiveCellCubeIndices.clear();

  if (computeResult.activeVoxelIndices.empty()) {
    outError.clear();
    return true;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const int32_t cellDimX = dimX - 1;
  const int32_t cellDimY = dimY - 1;
  const int32_t cellDimZ = dimZ - 1;
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (cellDimX <= 0 || cellDimY <= 0 || cellDimZ <= 0 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan sparse surface-cell classification requires a valid scalar-field domain.";
    return false;
  }

  const std::size_t activeVoxelCount = computeResult.activeVoxelIndices.size();
  const std::size_t cellCount = static_cast<std::size_t>(cellDimX) *
                                static_cast<std::size_t>(cellDimY) *
                                static_cast<std::size_t>(cellDimZ);

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const VkDeviceSize activeVoxelIndexBufferSize =
      static_cast<VkDeviceSize>(activeVoxelCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize cellCubeIndexBufferSize =
      static_cast<VkDeviceSize>(cellCount * sizeof(uint32_t));
  const VkDeviceSize compactCellIndexBufferSize =
      static_cast<VkDeviceSize>(cellCount * sizeof(uint32_t));
  const VkDeviceSize surfaceCellStatsBufferSize =
      static_cast<VkDeviceSize>(sizeof(uint32_t));

  bool classifyReady = ensure_shared_host_buffer(
      sharedContext, sharedContext.activeVoxelIndexBuffer,
      activeVoxelIndexBufferSize, "sparse active voxel-index buffer", outError);
  if (classifyReady) {
    classifyReady = ensure_shared_scalar_field_buffer_for_result(
        sharedContext, computeResult, scalarFieldBufferSize,
        "sparse surface scalar-field buffer", outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.denseSurfaceCellCubeIndexBuffer,
        cellCubeIndexBufferSize, "sparse dense surface cube-index buffer",
        outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellIndexBuffer,
        compactCellIndexBufferSize, "sparse compact surface cell-index buffer",
        outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        compactCellIndexBufferSize, "sparse compact surface cube-index buffer",
        outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceCellStatsBuffer,
        surfaceCellStatsBufferSize, "surface-cell stats buffer", outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceDenseSurfaceCellCubeIndexBuffer,
        cellCubeIndexBufferSize, "device sparse dense surface cube-index buffer",
        outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceCandidateCellIndexBuffer,
        compactCellIndexBufferSize,
        "device sparse compact surface cell-index buffer", outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceCandidateCellCubeIndexBuffer,
        compactCellIndexBufferSize,
        "device sparse compact surface cube-index buffer", outError);
  }
  if (classifyReady) {
    classifyReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceCellStatsBuffer,
        surfaceCellStatsBufferSize, "device surface-cell stats buffer",
        outError);
  }
  if (!classifyReady) {
    cleanup();
    return false;
  }

  std::memcpy(sharedContext.activeVoxelIndexBuffer.mapped,
              computeResult.activeVoxelIndices.data(),
              static_cast<size_t>(activeVoxelIndexBufferSize));
  std::memset(sharedContext.surfaceCellStatsBuffer.mapped, 0,
              static_cast<size_t>(surfaceCellStatsBufferSize));
  std::memset(sharedContext.candidateCellIndexBuffer.mapped, 0,
              static_cast<size_t>(compactCellIndexBufferSize));
  std::memset(sharedContext.candidateCellCubeIndexBuffer.mapped, 0,
              static_cast<size_t>(compactCellIndexBufferSize));

  if (sharedContext.sparseSurfaceCellDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellPipeline == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan sparse surface-cell pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo bufferInfos[3]{};
  bufferInfos[0].buffer = sharedContext.activeVoxelIndexBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = activeVoxelIndexBufferSize;
  bufferInfos[1].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = scalarFieldBufferSize;
  bufferInfos[2].buffer =
      sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = cellCubeIndexBufferSize;

  VkWriteDescriptorSet writes[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.sparseSurfaceCellDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);
  VkCommandBuffer commandBuffer = sharedContext.sparseSurfaceCellCommandBuffer;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError = "Failed to begin Vulkan sparse surface-cell command buffer.";
    cleanup();
    return false;
  }

  vkCmdFillBuffer(commandBuffer,
                  sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer, 0,
                  cellCubeIndexBufferSize, 0u);
  vkCmdFillBuffer(commandBuffer, sharedContext.deviceSurfaceCellStatsBuffer.buffer,
                  0, surfaceCellStatsBufferSize, 0u);

  VkMemoryBarrier fillBarrier{};
  fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  fillBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier,
                       0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.sparseSurfaceCellPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.sparseSurfaceCellPipelineLayout, 0, 1,
                          &sharedContext.sparseSurfaceCellDescriptorSet, 0,
                          nullptr);

  SurfaceCellPushConstants pushConstants{};
  pushConstants.candidateCellCount = static_cast<uint32_t>(activeVoxelCount);
  pushConstants.cellDimX = cellDimX;
  pushConstants.cellDimY = cellDimY;
  pushConstants.cellDimZ = cellDimZ;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.sparseSurfaceCellPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceCellPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((activeVoxelCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  if (sharedContext.surfaceCellCompactDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan surface-cell compaction pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo compactBufferInfos[4]{};
  compactBufferInfos[0].buffer =
      sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer;
  compactBufferInfos[0].offset = 0;
  compactBufferInfos[0].range = cellCubeIndexBufferSize;
  compactBufferInfos[1].buffer =
      sharedContext.deviceCandidateCellIndexBuffer.buffer;
  compactBufferInfos[1].offset = 0;
  compactBufferInfos[1].range = compactCellIndexBufferSize;
  compactBufferInfos[2].buffer =
      sharedContext.deviceCandidateCellCubeIndexBuffer.buffer;
  compactBufferInfos[2].offset = 0;
  compactBufferInfos[2].range = compactCellIndexBufferSize;
  compactBufferInfos[3].buffer =
      sharedContext.deviceSurfaceCellStatsBuffer.buffer;
  compactBufferInfos[3].offset = 0;
  compactBufferInfos[3].range = surfaceCellStatsBufferSize;

  VkWriteDescriptorSet compactWrites[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    compactWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    compactWrites[i].dstSet = sharedContext.surfaceCellCompactDescriptorSet;
    compactWrites[i].dstBinding = i;
    compactWrites[i].descriptorCount = 1;
    compactWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compactWrites[i].pBufferInfo = &compactBufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 4, compactWrites, 0, nullptr);

  VkMemoryBarrier preCompactBarrier{};
  preCompactBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  preCompactBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  preCompactBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &preCompactBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceCellCompactPipeline);
  vkCmdBindDescriptorSets(commandBuffer,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceCellCompactPipelineLayout, 0, 1,
                          &sharedContext.surfaceCellCompactDescriptorSet, 0,
                          nullptr);

  SurfaceCellCompactPushConstants compactPushConstants{};
  compactPushConstants.cellCount = static_cast<uint32_t>(cellCount);
  vkCmdPushConstants(commandBuffer,
                     sharedContext.surfaceCellCompactPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceCellCompactPushConstants),
                     &compactPushConstants);

  const uint32_t compactWorkgroupCount =
      static_cast<uint32_t>((cellCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, compactWorkgroupCount, 1, 1);

  VkMemoryBarrier compactMemoryBarrier{};
  compactMemoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  compactMemoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  compactMemoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                       &compactMemoryBarrier, 0, nullptr, 0, nullptr);

  VkBufferCopy compactCopies[3]{};
  compactCopies[0].size = compactCellIndexBufferSize;
  compactCopies[1].size = compactCellIndexBufferSize;
  compactCopies[2].size = surfaceCellStatsBufferSize;
  vkCmdCopyBuffer(commandBuffer, sharedContext.deviceCandidateCellIndexBuffer.buffer,
                  sharedContext.candidateCellIndexBuffer.buffer, 1,
                  &compactCopies[0]);
  vkCmdCopyBuffer(commandBuffer,
                  sharedContext.deviceCandidateCellCubeIndexBuffer.buffer,
                  sharedContext.candidateCellCubeIndexBuffer.buffer, 1,
                  &compactCopies[1]);
  vkCmdCopyBuffer(commandBuffer, sharedContext.deviceSurfaceCellStatsBuffer.buffer,
                  sharedContext.surfaceCellStatsBuffer.buffer, 1,
                  &compactCopies[2]);

  VkMemoryBarrier hostReadBarrier{};
  hostReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hostReadBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan sparse surface-cell command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan sparse surface-cell classification.";
    cleanup();
    return false;
  }

  uint32_t activeCellCount = 0u;
  std::memcpy(&activeCellCount, sharedContext.surfaceCellStatsBuffer.mapped,
              static_cast<size_t>(surfaceCellStatsBufferSize));

  if (activeCellCount > cellCount) {
    outError =
        "Vulkan surface-cell compaction produced an invalid active-cell count.";
    cleanup();
    return false;
  }

  outActiveCellIndices.resize(activeCellCount, 0u);
  outActiveCellCubeIndices.resize(activeCellCount, 0u);
  if (activeCellCount > 0u) {
    const std::size_t compactByteSize =
        static_cast<std::size_t>(activeCellCount) * sizeof(uint32_t);
    std::memcpy(outActiveCellIndices.data(),
                sharedContext.candidateCellIndexBuffer.mapped,
                compactByteSize);
    std::memcpy(outActiveCellCubeIndices.data(),
                sharedContext.candidateCellCubeIndexBuffer.mapped,
                compactByteSize);
  }

  cleanup();
  outError.clear();
  return true;
}

bool run_frost_vulkan_generate_sparse_surface_mesh_from_active_voxels(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh, std::string &outError,
    bool copyMeshData) {
  outActiveCellIndices.clear();
  outActiveCellCubeIndices.clear();
  outMesh.triangleCounts.clear();
  outMesh.triangleVertices.clear();
  outMesh.totalTriangleCount = 0;

  if (computeResult.activeVoxelIndices.empty()) {
    outError.clear();
    return true;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const int32_t cellDimX = dimX - 1;
  const int32_t cellDimY = dimY - 1;
  const int32_t cellDimZ = dimZ - 1;
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (cellDimX <= 0 || cellDimY <= 0 || cellDimZ <= 0 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan sparse direct surface path requires a valid scalar-field domain.";
    return false;
  }

  const std::size_t activeVoxelCount = computeResult.activeVoxelIndices.size();
  const std::size_t cellCount = static_cast<std::size_t>(cellDimX) *
                                static_cast<std::size_t>(cellDimY) *
                                static_cast<std::size_t>(cellDimZ);
  if (activeVoxelCount >
      static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    outError =
        "Vulkan sparse direct surface path exceeds addressable active-voxel count.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const VkDeviceSize activeVoxelIndexBufferSize =
      static_cast<VkDeviceSize>(activeVoxelCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize cellCubeIndexBufferSize =
      static_cast<VkDeviceSize>(cellCount * sizeof(uint32_t));
  const VkDeviceSize compactCellIndexBufferSize =
      static_cast<VkDeviceSize>(activeVoxelCount * sizeof(uint32_t));
  const VkDeviceSize surfaceCellStatsBufferSize =
      static_cast<VkDeviceSize>(sizeof(uint32_t));
  const VkDeviceSize triangleCountBufferSize =
      static_cast<VkDeviceSize>(activeVoxelCount * sizeof(uint32_t));
  const VkDeviceSize triangleVertexBufferSize = static_cast<VkDeviceSize>(
      activeVoxelCount * 15ull * 3ull * sizeof(float));

  bool buffersReady = ensure_shared_host_buffer(
      sharedContext, sharedContext.activeVoxelIndexBuffer,
      activeVoxelIndexBufferSize, "sparse direct active voxel-index buffer",
      outError);
  if (buffersReady) {
    buffersReady = ensure_shared_scalar_field_buffer_for_result(
        sharedContext, computeResult, scalarFieldBufferSize,
        "sparse direct surface scalar-field buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellIndexBuffer,
        compactCellIndexBufferSize,
        "sparse direct compact surface cell-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        compactCellIndexBufferSize,
        "sparse direct compact surface cube-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceCellStatsBuffer,
        surfaceCellStatsBufferSize, "sparse direct surface-cell stats buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleCountBuffer,
        triangleCountBufferSize,
        "sparse direct surface triangle-count buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleVertexBuffer,
        triangleVertexBufferSize,
        "sparse direct surface triangle-vertex buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceDenseSurfaceCellCubeIndexBuffer,
        cellCubeIndexBufferSize,
        "device sparse direct dense surface cube-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceCandidateCellIndexBuffer,
        compactCellIndexBufferSize,
        "device sparse direct compact surface cell-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceCandidateCellCubeIndexBuffer,
        compactCellIndexBufferSize,
        "device sparse direct compact surface cube-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceCellStatsBuffer,
        surfaceCellStatsBufferSize,
        "device sparse direct surface-cell stats buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceTriangleCountBuffer,
        triangleCountBufferSize,
        "device sparse direct surface triangle-count buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceTriangleVertexBuffer,
        triangleVertexBufferSize,
        "device sparse direct surface triangle-vertex buffer", outError);
  }
  if (!buffersReady) {
    cleanup();
    return false;
  }

  std::memcpy(sharedContext.activeVoxelIndexBuffer.mapped,
              computeResult.activeVoxelIndices.data(),
              static_cast<size_t>(activeVoxelIndexBufferSize));
  std::memset(sharedContext.candidateCellIndexBuffer.mapped, 0,
              static_cast<size_t>(compactCellIndexBufferSize));
  std::memset(sharedContext.candidateCellCubeIndexBuffer.mapped, 0,
              static_cast<size_t>(compactCellIndexBufferSize));
  std::memset(sharedContext.surfaceCellStatsBuffer.mapped, 0,
              static_cast<size_t>(surfaceCellStatsBufferSize));
  std::memset(sharedContext.surfaceTriangleCountBuffer.mapped, 0,
              static_cast<size_t>(triangleCountBufferSize));
  std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
              static_cast<size_t>(triangleVertexBufferSize));

  if (sharedContext.sparseSurfaceCellDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellPipeline == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceCellCompactDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.sparseSurfaceCellCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan sparse direct surface pipelines are unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo classifyBufferInfos[3]{};
  classifyBufferInfos[0].buffer = sharedContext.activeVoxelIndexBuffer.buffer;
  classifyBufferInfos[0].offset = 0;
  classifyBufferInfos[0].range = activeVoxelIndexBufferSize;
  classifyBufferInfos[1].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  classifyBufferInfos[1].offset = 0;
  classifyBufferInfos[1].range = scalarFieldBufferSize;
  classifyBufferInfos[2].buffer =
      sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer;
  classifyBufferInfos[2].offset = 0;
  classifyBufferInfos[2].range = cellCubeIndexBufferSize;

  VkWriteDescriptorSet classifyWrites[3]{};
  for (uint32_t i = 0; i < 3; ++i) {
    classifyWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    classifyWrites[i].dstSet = sharedContext.sparseSurfaceCellDescriptorSet;
    classifyWrites[i].dstBinding = i;
    classifyWrites[i].descriptorCount = 1;
    classifyWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    classifyWrites[i].pBufferInfo = &classifyBufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 3, classifyWrites, 0, nullptr);

  VkDescriptorBufferInfo compactBufferInfos[4]{};
  compactBufferInfos[0].buffer =
      sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer;
  compactBufferInfos[0].offset = 0;
  compactBufferInfos[0].range = cellCubeIndexBufferSize;
  compactBufferInfos[1].buffer =
      sharedContext.deviceCandidateCellIndexBuffer.buffer;
  compactBufferInfos[1].offset = 0;
  compactBufferInfos[1].range = compactCellIndexBufferSize;
  compactBufferInfos[2].buffer =
      sharedContext.deviceCandidateCellCubeIndexBuffer.buffer;
  compactBufferInfos[2].offset = 0;
  compactBufferInfos[2].range = compactCellIndexBufferSize;
  compactBufferInfos[3].buffer =
      sharedContext.deviceSurfaceCellStatsBuffer.buffer;
  compactBufferInfos[3].offset = 0;
  compactBufferInfos[3].range = surfaceCellStatsBufferSize;

  VkWriteDescriptorSet compactWrites[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    compactWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    compactWrites[i].dstSet = sharedContext.surfaceCellCompactDescriptorSet;
    compactWrites[i].dstBinding = i;
    compactWrites[i].descriptorCount = 1;
    compactWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    compactWrites[i].pBufferInfo = &compactBufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 4, compactWrites, 0, nullptr);

  VkDescriptorBufferInfo meshBufferInfos[5]{};
  meshBufferInfos[0].buffer = sharedContext.deviceCandidateCellIndexBuffer.buffer;
  meshBufferInfos[0].offset = 0;
  meshBufferInfos[0].range = compactCellIndexBufferSize;
  meshBufferInfos[1].buffer =
      sharedContext.deviceCandidateCellCubeIndexBuffer.buffer;
  meshBufferInfos[1].offset = 0;
  meshBufferInfos[1].range = compactCellIndexBufferSize;
  meshBufferInfos[2].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  meshBufferInfos[2].offset = 0;
  meshBufferInfos[2].range = scalarFieldBufferSize;
  meshBufferInfos[3].buffer =
      sharedContext.deviceSurfaceTriangleCountBuffer.buffer;
  meshBufferInfos[3].offset = 0;
  meshBufferInfos[3].range = triangleCountBufferSize;
  meshBufferInfos[4].buffer =
      sharedContext.deviceSurfaceTriangleVertexBuffer.buffer;
  meshBufferInfos[4].offset = 0;
  meshBufferInfos[4].range = triangleVertexBufferSize;

  VkWriteDescriptorSet meshWrites[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    meshWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    meshWrites[i].dstSet = sharedContext.surfaceMeshDescriptorSet;
    meshWrites[i].dstBinding = i;
    meshWrites[i].descriptorCount = 1;
    meshWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    meshWrites[i].pBufferInfo = &meshBufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 5, meshWrites, 0, nullptr);

  VkCommandBuffer commandBuffer = sharedContext.sparseSurfaceCellCommandBuffer;
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError =
        "Failed to begin Vulkan sparse direct surface command buffer.";
    cleanup();
    return false;
  }

  vkCmdFillBuffer(commandBuffer,
                  sharedContext.deviceDenseSurfaceCellCubeIndexBuffer.buffer, 0,
                  cellCubeIndexBufferSize, 0u);
  vkCmdFillBuffer(commandBuffer,
                  sharedContext.deviceCandidateCellIndexBuffer.buffer, 0,
                  compactCellIndexBufferSize, 0u);
  vkCmdFillBuffer(commandBuffer,
                  sharedContext.deviceCandidateCellCubeIndexBuffer.buffer, 0,
                  compactCellIndexBufferSize, 0u);
  vkCmdFillBuffer(commandBuffer, sharedContext.deviceSurfaceCellStatsBuffer.buffer,
                  0, surfaceCellStatsBufferSize, 0u);
  vkCmdFillBuffer(commandBuffer,
                  sharedContext.deviceSurfaceTriangleCountBuffer.buffer, 0,
                  triangleCountBufferSize, 0u);

  VkMemoryBarrier fillBarrier{};
  fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  fillBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier,
                       0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.sparseSurfaceCellPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.sparseSurfaceCellPipelineLayout, 0, 1,
                          &sharedContext.sparseSurfaceCellDescriptorSet, 0,
                          nullptr);

  SurfaceCellPushConstants classifyPushConstants{};
  classifyPushConstants.candidateCellCount =
      static_cast<uint32_t>(activeVoxelCount);
  classifyPushConstants.cellDimX = cellDimX;
  classifyPushConstants.cellDimY = cellDimY;
  classifyPushConstants.cellDimZ = cellDimZ;
  classifyPushConstants.domainDimX = dimX;
  classifyPushConstants.domainDimY = dimY;
  classifyPushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.sparseSurfaceCellPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceCellPushConstants), &classifyPushConstants);

  const uint32_t classifyWorkgroupCount =
      static_cast<uint32_t>((activeVoxelCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, classifyWorkgroupCount, 1, 1);

  VkMemoryBarrier preCompactBarrier{};
  preCompactBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  preCompactBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  preCompactBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &preCompactBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceCellCompactPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceCellCompactPipelineLayout, 0, 1,
                          &sharedContext.surfaceCellCompactDescriptorSet, 0,
                          nullptr);

  SurfaceCellCompactPushConstants compactPushConstants{};
  compactPushConstants.cellCount = static_cast<uint32_t>(cellCount);
  vkCmdPushConstants(commandBuffer,
                     sharedContext.surfaceCellCompactPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceCellCompactPushConstants),
                     &compactPushConstants);

  const uint32_t compactWorkgroupCount =
      static_cast<uint32_t>((cellCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, compactWorkgroupCount, 1, 1);

  VkMemoryBarrier preSurfaceBarrier{};
  preSurfaceBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  preSurfaceBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  preSurfaceBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1,
                       &preSurfaceBarrier, 0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceMeshPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceMeshPipelineLayout, 0, 1,
                          &sharedContext.surfaceMeshDescriptorSet, 0, nullptr);

  SurfaceMeshPushConstants surfacePushConstants{};
  surfacePushConstants.candidateCellCount = static_cast<uint32_t>(activeVoxelCount);
  surfacePushConstants.cellDimX = cellDimX;
  surfacePushConstants.cellDimY = cellDimY;
  surfacePushConstants.domainDimX = dimX;
  surfacePushConstants.domainDimY = dimY;
  surfacePushConstants.domainMinX = computeResult.domainMinVoxel[0];
  surfacePushConstants.domainMinY = computeResult.domainMinVoxel[1];
  surfacePushConstants.domainMinZ = computeResult.domainMinVoxel[2];
  surfacePushConstants.voxelLength = computeResult.voxelLength;
  surfacePushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.surfaceMeshPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceMeshPushConstants), &surfacePushConstants);

  const uint32_t surfaceWorkgroupCount =
      static_cast<uint32_t>((activeVoxelCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, surfaceWorkgroupCount, 1, 1);

  VkMemoryBarrier transferBarrier{};
  transferBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  transferBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  transferBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &transferBarrier, 0,
                       nullptr, 0, nullptr);

  VkBufferCopy outputCopies[5]{};
  outputCopies[0].size = compactCellIndexBufferSize;
  outputCopies[1].size = compactCellIndexBufferSize;
  outputCopies[2].size = surfaceCellStatsBufferSize;
  outputCopies[3].size = triangleCountBufferSize;
  outputCopies[4].size = triangleVertexBufferSize;
  vkCmdCopyBuffer(commandBuffer, sharedContext.deviceCandidateCellIndexBuffer.buffer,
                  sharedContext.candidateCellIndexBuffer.buffer, 1,
                  &outputCopies[0]);
  vkCmdCopyBuffer(commandBuffer,
                  sharedContext.deviceCandidateCellCubeIndexBuffer.buffer,
                  sharedContext.candidateCellCubeIndexBuffer.buffer, 1,
                  &outputCopies[1]);
  vkCmdCopyBuffer(commandBuffer, sharedContext.deviceSurfaceCellStatsBuffer.buffer,
                  sharedContext.surfaceCellStatsBuffer.buffer, 1,
                  &outputCopies[2]);
  vkCmdCopyBuffer(commandBuffer,
                  sharedContext.deviceSurfaceTriangleCountBuffer.buffer,
                  sharedContext.surfaceTriangleCountBuffer.buffer, 1,
                  &outputCopies[3]);
  vkCmdCopyBuffer(commandBuffer,
                  sharedContext.deviceSurfaceTriangleVertexBuffer.buffer,
                  sharedContext.surfaceTriangleVertexBuffer.buffer, 1,
                  &outputCopies[4]);

  VkMemoryBarrier hostReadBarrier{};
  hostReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hostReadBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan sparse direct surface command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan sparse direct surface generation.";
    cleanup();
    return false;
  }

  uint32_t activeCellCount = 0u;
  std::memcpy(&activeCellCount, sharedContext.surfaceCellStatsBuffer.mapped,
              static_cast<size_t>(surfaceCellStatsBufferSize));
  if (activeCellCount > activeVoxelCount) {
    outError =
        "Vulkan sparse direct surface generation produced an invalid active-cell count.";
    cleanup();
    return false;
  }

  outActiveCellIndices.resize(activeCellCount, 0u);
  outActiveCellCubeIndices.resize(activeCellCount, 0u);
  if (activeCellCount > 0u) {
    const std::size_t compactByteSize =
        static_cast<std::size_t>(activeCellCount) * sizeof(uint32_t);
    std::memcpy(outActiveCellIndices.data(),
                sharedContext.candidateCellIndexBuffer.mapped,
                compactByteSize);
    std::memcpy(outActiveCellCubeIndices.data(),
                sharedContext.candidateCellCubeIndexBuffer.mapped,
                compactByteSize);
  }

  const uint32_t *triangleCounts = reinterpret_cast<const uint32_t *>(
      sharedContext.surfaceTriangleCountBuffer.mapped);
  uint32_t totalTriangleCount = 0u;
  for (uint32_t i = 0; i < activeCellCount; ++i) {
    const uint32_t triangleCount = triangleCounts[i];
    if (triangleCount > 5u) {
      outError =
          "Vulkan sparse direct surface generation produced an invalid triangle count.";
      cleanup();
      return false;
    }
    totalTriangleCount += triangleCount;
  }
  outMesh.totalTriangleCount = totalTriangleCount;

  if (copyMeshData) {
    outMesh.triangleCounts.resize(activeCellCount, 0u);
    if (activeCellCount > 0u) {
      const std::size_t triangleCountByteSize =
          static_cast<std::size_t>(activeCellCount) * sizeof(uint32_t);
      std::memcpy(outMesh.triangleCounts.data(),
                  sharedContext.surfaceTriangleCountBuffer.mapped,
                  triangleCountByteSize);
      outMesh.triangleVertices.resize(activeCellCount * 15ull * 3ull, 0.0f);
      std::memcpy(outMesh.triangleVertices.data(),
                  sharedContext.surfaceTriangleVertexBuffer.mapped,
                  static_cast<std::size_t>(activeCellCount) * 15ull * 3ull *
                      sizeof(float));
    }
  }

  cleanup();
  outError.clear();
  return true;
}

bool run_frost_vulkan_generate_surface_mesh(
    const VulkanParticleComputeResult &computeResult,
    const std::vector<uint32_t> &activeCellIndices,
    const std::vector<uint32_t> &activeCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh,
    std::string &outError) {
  outMesh.triangleCounts.clear();
  outMesh.triangleVertices.clear();
  outMesh.totalTriangleCount = 0;

  if (activeCellIndices.empty()) {
    outError.clear();
    return true;
  }
  if (activeCellIndices.size() != activeCellCubeIndices.size()) {
    outError = "Vulkan surface mesh generation requires matching active-cell and cube-index buffers.";
    return false;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (dimX < 2 || dimY < 2 || dimZ < 2 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan surface mesh generation requires a valid scalar-field domain.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const std::size_t activeCellCount = activeCellIndices.size();
  const VkDeviceSize activeCellIndexBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize activeCellCubeIndexBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize triangleCountBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize triangleVertexBufferSize = static_cast<VkDeviceSize>(
      activeCellCount * 15ull * 3ull * sizeof(float));

  sharedContext.residentSurfaceTriangleVerticesCompacted = false;

  bool buffersReady = ensure_shared_host_buffer(
      sharedContext, sharedContext.candidateCellIndexBuffer,
      activeCellIndexBufferSize, "surface-mesh active cell-index buffer",
      outError);
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        activeCellCubeIndexBufferSize, "surface-mesh active cube-index buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_scalar_field_buffer_for_result(
        sharedContext, computeResult, scalarFieldBufferSize,
        "surface-mesh scalar-field buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleCountBuffer,
        triangleCountBufferSize, "surface triangle-count buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleVertexBuffer,
        triangleVertexBufferSize, "surface triangle-vertex buffer", outError);
  }
  if (!buffersReady) {
    cleanup();
    return false;
  }

  std::memcpy(sharedContext.candidateCellIndexBuffer.mapped,
              activeCellIndices.data(),
              static_cast<size_t>(activeCellIndexBufferSize));
  std::memcpy(sharedContext.candidateCellCubeIndexBuffer.mapped,
              activeCellCubeIndices.data(),
              static_cast<size_t>(activeCellCubeIndexBufferSize));
  std::memset(sharedContext.surfaceTriangleCountBuffer.mapped, 0,
              static_cast<size_t>(triangleCountBufferSize));
  std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
              static_cast<size_t>(triangleVertexBufferSize));

  if (sharedContext.surfaceMeshDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan surface-mesh pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo bufferInfos[5]{};
  bufferInfos[0].buffer = sharedContext.candidateCellIndexBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = activeCellIndexBufferSize;
  bufferInfos[1].buffer = sharedContext.candidateCellCubeIndexBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = activeCellCubeIndexBufferSize;
  bufferInfos[2].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = scalarFieldBufferSize;
  bufferInfos[3].buffer = sharedContext.surfaceTriangleCountBuffer.buffer;
  bufferInfos[3].offset = 0;
  bufferInfos[3].range = triangleCountBufferSize;
  bufferInfos[4].buffer = sharedContext.surfaceTriangleVertexBuffer.buffer;
  bufferInfos[4].offset = 0;
  bufferInfos[4].range = triangleVertexBufferSize;

  VkWriteDescriptorSet writes[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.surfaceMeshDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
  VkCommandBuffer commandBuffer = sharedContext.surfaceMeshCommandBuffer;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError = "Failed to begin Vulkan surface-mesh command buffer.";
    cleanup();
    return false;
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceMeshPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceMeshPipelineLayout, 0, 1,
                          &sharedContext.surfaceMeshDescriptorSet, 0, nullptr);

  SurfaceMeshPushConstants pushConstants{};
  pushConstants.candidateCellCount = static_cast<uint32_t>(activeCellCount);
  pushConstants.cellDimX = dimX - 1;
  pushConstants.cellDimY = dimY - 1;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.domainMinX = computeResult.domainMinVoxel[0];
  pushConstants.domainMinY = computeResult.domainMinVoxel[1];
  pushConstants.domainMinZ = computeResult.domainMinVoxel[2];
  pushConstants.voxelLength = computeResult.voxelLength;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.surfaceMeshPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceMeshPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((activeCellCount + 63) / 64);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan surface-mesh command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan surface-mesh generation.";
    cleanup();
    return false;
  }

  outMesh.triangleCounts.resize(activeCellCount, 0u);
  std::memcpy(outMesh.triangleCounts.data(),
              sharedContext.surfaceTriangleCountBuffer.mapped,
              static_cast<size_t>(triangleCountBufferSize));
  outMesh.triangleVertices.resize(
      static_cast<std::size_t>(activeCellCount) * 15ull * 3ull, 0.0f);
  std::memcpy(outMesh.triangleVertices.data(),
              sharedContext.surfaceTriangleVertexBuffer.mapped,
              static_cast<size_t>(triangleVertexBufferSize));

  uint32_t totalTriangleCount = 0;
  for (uint32_t triangleCount : outMesh.triangleCounts) {
    if (triangleCount > 5u) {
      outError =
          "Vulkan surface-mesh generation produced an invalid triangle count.";
      cleanup();
      return false;
    }
    totalTriangleCount += triangleCount;
  }
  outMesh.totalTriangleCount = totalTriangleCount;

  cleanup();
  outError.clear();
  return true;
}

bool run_frost_vulkan_generate_surface_mesh_from_resident_cells(
    const VulkanParticleComputeResult &computeResult, uint32_t activeCellCount,
    VulkanSurfaceMeshResult &outMesh, std::string &outError,
    bool copyMeshData) {
  outMesh.triangleCounts.clear();
  outMesh.triangleVertices.clear();
  outMesh.totalTriangleCount = 0;

  if (activeCellCount == 0u) {
    outError.clear();
    return true;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (dimX < 2 || dimY < 2 || dimZ < 2 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan resident surface mesh generation requires a valid scalar-field domain.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkCmdCopyBuffer = sharedContext.vkCmdCopyBuffer;
  auto vkCmdFillBuffer = sharedContext.vkCmdFillBuffer;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const VkDeviceSize activeCellIndexBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize activeCellCubeIndexBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize triangleCountBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize triangleVertexBufferSize = static_cast<VkDeviceSize>(
      activeCellCount * 15ull * 3ull * sizeof(float));

  bool buffersReady = ensure_shared_host_buffer(
      sharedContext, sharedContext.candidateCellIndexBuffer,
      activeCellIndexBufferSize, "resident surface-mesh active cell-index buffer",
      outError);
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        activeCellCubeIndexBufferSize,
        "resident surface-mesh active cube-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_scalar_field_buffer_for_result(
        sharedContext, computeResult, scalarFieldBufferSize,
        "resident surface-mesh scalar-field buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleCountBuffer,
        triangleCountBufferSize, "resident surface triangle-count buffer",
        outError);
  }
  if (buffersReady && copyMeshData) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleVertexBuffer,
        triangleVertexBufferSize, "resident surface triangle-vertex buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceTriangleCountBuffer,
        triangleCountBufferSize, "device resident surface triangle-count buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_device_buffer(
        sharedContext, sharedContext.deviceSurfaceTriangleVertexBuffer,
        triangleVertexBufferSize,
        "device resident surface triangle-vertex buffer", outError);
  }
  if (!buffersReady) {
    cleanup();
    return false;
  }

  std::memset(sharedContext.surfaceTriangleCountBuffer.mapped, 0,
              static_cast<size_t>(triangleCountBufferSize));
  if (copyMeshData && sharedContext.surfaceTriangleVertexBuffer.mapped != nullptr) {
    std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
                static_cast<size_t>(triangleVertexBufferSize));
  }

  if (sharedContext.surfaceMeshDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceMeshCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan surface-mesh pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo bufferInfos[5]{};
  bufferInfos[0].buffer = sharedContext.candidateCellIndexBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = activeCellIndexBufferSize;
  bufferInfos[1].buffer = sharedContext.candidateCellCubeIndexBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = activeCellCubeIndexBufferSize;
  bufferInfos[2].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = scalarFieldBufferSize;
  bufferInfos[3].buffer = sharedContext.deviceSurfaceTriangleCountBuffer.buffer;
  bufferInfos[3].offset = 0;
  bufferInfos[3].range = triangleCountBufferSize;
  bufferInfos[4].buffer =
      sharedContext.deviceSurfaceTriangleVertexBuffer.buffer;
  bufferInfos[4].offset = 0;
  bufferInfos[4].range = triangleVertexBufferSize;

  VkWriteDescriptorSet writes[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.surfaceMeshDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);
  VkCommandBuffer commandBuffer = sharedContext.surfaceMeshCommandBuffer;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError = "Failed to begin Vulkan resident surface-mesh command buffer.";
    cleanup();
    return false;
  }

  vkCmdFillBuffer(commandBuffer, sharedContext.deviceSurfaceTriangleCountBuffer.buffer,
                  0, triangleCountBufferSize, 0u);

  VkMemoryBarrier fillBarrier{};
  fillBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  fillBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  fillBarrier.dstAccessMask =
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &fillBarrier,
                       0, nullptr, 0, nullptr);

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceMeshPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceMeshPipelineLayout, 0, 1,
                          &sharedContext.surfaceMeshDescriptorSet, 0, nullptr);

  SurfaceMeshPushConstants pushConstants{};
  pushConstants.candidateCellCount = activeCellCount;
  pushConstants.cellDimX = dimX - 1;
  pushConstants.cellDimY = dimY - 1;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.domainMinX = computeResult.domainMinVoxel[0];
  pushConstants.domainMinY = computeResult.domainMinVoxel[1];
  pushConstants.domainMinZ = computeResult.domainMinVoxel[2];
  pushConstants.voxelLength = computeResult.voxelLength;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.surfaceMeshPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceMeshPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((activeCellCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);

  VkBufferCopy surfaceCopies[2]{};
  surfaceCopies[0].size = triangleCountBufferSize;
  surfaceCopies[1].size = triangleVertexBufferSize;
  vkCmdCopyBuffer(commandBuffer, sharedContext.deviceSurfaceTriangleCountBuffer.buffer,
                  sharedContext.surfaceTriangleCountBuffer.buffer, 1,
                  &surfaceCopies[0]);
  if (copyMeshData) {
    vkCmdCopyBuffer(commandBuffer,
                    sharedContext.deviceSurfaceTriangleVertexBuffer.buffer,
                    sharedContext.surfaceTriangleVertexBuffer.buffer, 1,
                    &surfaceCopies[1]);
  }

  VkMemoryBarrier hostReadBarrier{};
  hostReadBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostReadBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
  hostReadBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hostReadBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan resident surface-mesh command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan resident surface-mesh generation.";
    cleanup();
    return false;
  }

  uint32_t totalTriangleCount = 0;
  const uint32_t *residentTriangleCounts =
      reinterpret_cast<const uint32_t *>(
          sharedContext.surfaceTriangleCountBuffer.mapped);
  for (uint32_t i = 0; i < activeCellCount; ++i) {
    const uint32_t triangleCount = residentTriangleCounts[i];
    if (triangleCount > 5u) {
      outError =
          "Vulkan resident surface-mesh generation produced an invalid triangle count.";
      cleanup();
      return false;
    }
    totalTriangleCount += triangleCount;
  }
  outMesh.totalTriangleCount = totalTriangleCount;

  if (copyMeshData) {
    outMesh.triangleCounts.resize(activeCellCount, 0u);
    std::memcpy(outMesh.triangleCounts.data(),
                sharedContext.surfaceTriangleCountBuffer.mapped,
                static_cast<size_t>(triangleCountBufferSize));
    outMesh.triangleVertices.resize(
        static_cast<std::size_t>(activeCellCount) * 15ull * 3ull, 0.0f);
    std::memcpy(outMesh.triangleVertices.data(),
                sharedContext.surfaceTriangleVertexBuffer.mapped,
                static_cast<size_t>(triangleVertexBufferSize));
    sharedContext.residentSurfaceTriangleVerticesCompacted = false;
  } else if (totalTriangleCount > 0u) {
    std::vector<uint32_t> triangleOffsets(activeCellCount, 0u);
    uint32_t runningTriangleOffset = 0u;
    for (uint32_t i = 0; i < activeCellCount; ++i) {
      triangleOffsets[i] = runningTriangleOffset;
      runningTriangleOffset += residentTriangleCounts[i];
    }

    const VkDeviceSize compactTriangleVertexBufferSize = static_cast<VkDeviceSize>(
        totalTriangleCount) * 9ull * sizeof(float);
    if (!ensure_shared_host_buffer(
            sharedContext, sharedContext.surfaceTriangleVertexBuffer,
            compactTriangleVertexBufferSize,
            "resident compact surface triangle-vertex buffer", outError)) {
      cleanup();
      return false;
    }

    std::memcpy(sharedContext.candidateCellIndexBuffer.mapped,
                triangleOffsets.data(),
                static_cast<size_t>(triangleCountBufferSize));
    std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
                static_cast<size_t>(compactTriangleVertexBufferSize));

    if (sharedContext.surfaceTriangleCompactDescriptorSetLayout ==
            VK_NULL_HANDLE ||
        sharedContext.surfaceTriangleCompactPipelineLayout == VK_NULL_HANDLE ||
        sharedContext.surfaceTriangleCompactPipeline == VK_NULL_HANDLE ||
        sharedContext.surfaceTriangleCompactDescriptorSet == VK_NULL_HANDLE ||
        sharedContext.surfaceTriangleCompactCommandBuffer == VK_NULL_HANDLE) {
      outError = "Vulkan surface-triangle compaction pipeline is unavailable.";
      cleanup();
      return false;
    }

    VkDescriptorBufferInfo compactBufferInfos[4]{};
    compactBufferInfos[0].buffer =
        sharedContext.deviceSurfaceTriangleCountBuffer.buffer;
    compactBufferInfos[0].offset = 0;
    compactBufferInfos[0].range = triangleCountBufferSize;
    compactBufferInfos[1].buffer = sharedContext.candidateCellIndexBuffer.buffer;
    compactBufferInfos[1].offset = 0;
    compactBufferInfos[1].range = triangleCountBufferSize;
    compactBufferInfos[2].buffer =
        sharedContext.deviceSurfaceTriangleVertexBuffer.buffer;
    compactBufferInfos[2].offset = 0;
    compactBufferInfos[2].range = triangleVertexBufferSize;
    compactBufferInfos[3].buffer = sharedContext.surfaceTriangleVertexBuffer.buffer;
    compactBufferInfos[3].offset = 0;
    compactBufferInfos[3].range = compactTriangleVertexBufferSize;

    VkWriteDescriptorSet compactWrites[4]{};
    for (uint32_t i = 0; i < 4; ++i) {
      compactWrites[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      compactWrites[i].dstSet = sharedContext.surfaceTriangleCompactDescriptorSet;
      compactWrites[i].dstBinding = i;
      compactWrites[i].descriptorCount = 1;
      compactWrites[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      compactWrites[i].pBufferInfo = &compactBufferInfos[i];
    }
    vkUpdateDescriptorSets(device, 4, compactWrites, 0, nullptr);

    VkCommandBuffer compactCommandBuffer =
        sharedContext.surfaceTriangleCompactCommandBuffer;
    VkCommandBufferBeginInfo compactBeginInfo{};
    compactBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(compactCommandBuffer, &compactBeginInfo) !=
        VK_SUCCESS) {
      outError =
          "Failed to begin Vulkan resident surface-triangle compaction command buffer.";
      cleanup();
      return false;
    }

    vkCmdBindPipeline(compactCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                      sharedContext.surfaceTriangleCompactPipeline);
    vkCmdBindDescriptorSets(
        compactCommandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
        sharedContext.surfaceTriangleCompactPipelineLayout, 0, 1,
        &sharedContext.surfaceTriangleCompactDescriptorSet, 0, nullptr);

    SurfaceTriangleCompactPushConstants compactPushConstants{};
    compactPushConstants.cellCount = activeCellCount;
    vkCmdPushConstants(
        compactCommandBuffer, sharedContext.surfaceTriangleCompactPipelineLayout,
        VK_SHADER_STAGE_COMPUTE_BIT, 0,
        sizeof(SurfaceTriangleCompactPushConstants), &compactPushConstants);

    const uint32_t compactWorkgroupCount =
        static_cast<uint32_t>((activeCellCount + 63u) / 64u);
    vkCmdDispatch(compactCommandBuffer, compactWorkgroupCount, 1, 1);

    VkMemoryBarrier compactHostBarrier{};
    compactHostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    compactHostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    compactHostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(compactCommandBuffer,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT, 0, 1,
                         &compactHostBarrier, 0, nullptr, 0, nullptr);

    if (vkEndCommandBuffer(compactCommandBuffer) != VK_SUCCESS) {
      outError =
          "Failed to end Vulkan resident surface-triangle compaction command buffer.";
      cleanup();
      return false;
    }

    VkSubmitInfo compactSubmitInfo{};
    compactSubmitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    compactSubmitInfo.commandBufferCount = 1;
    compactSubmitInfo.pCommandBuffers = &compactCommandBuffer;
    if (vkQueueSubmit(queue, 1, &compactSubmitInfo, VK_NULL_HANDLE) !=
            VK_SUCCESS ||
        vkQueueWaitIdle(queue) != VK_SUCCESS) {
      outError =
          "Failed to submit Vulkan resident surface-triangle compaction.";
      cleanup();
      return false;
    }

    sharedContext.residentSurfaceTriangleVerticesCompacted = true;
  }

  cleanup();
  outError.clear();
  return true;
}

bool run_frost_vulkan_generate_compact_surface_vertices_from_resident_cells(
    const VulkanParticleComputeResult &computeResult, uint32_t activeCellCount,
    const std::vector<uint32_t> &triangleOffsets,
    uint32_t totalTriangleCount, std::string &outError) {
  if (activeCellCount == 0u || totalTriangleCount == 0u) {
    outError.clear();
    return true;
  }

  if (triangleOffsets.size() != static_cast<std::size_t>(activeCellCount)) {
    outError =
        "Vulkan compact resident surface generation requires triangle offsets matching the active-cell count.";
    return false;
  }

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (dimX < 2 || dimY < 2 || dimZ < 2 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan compact resident surface generation requires a valid scalar-field domain.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const VkDeviceSize activeCellBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize triangleOffsetBufferSize =
      static_cast<VkDeviceSize>(activeCellCount * sizeof(uint32_t));
  const VkDeviceSize compactTriangleVertexBufferSize = static_cast<VkDeviceSize>(
      totalTriangleCount) * 9ull * sizeof(float);

  if (sharedContext.deviceCandidateCellIndexBuffer.buffer == VK_NULL_HANDLE ||
      sharedContext.deviceCandidateCellCubeIndexBuffer.buffer ==
          VK_NULL_HANDLE ||
      sharedContext.deviceCandidateCellIndexBuffer.capacity <
          activeCellBufferSize ||
      sharedContext.deviceCandidateCellCubeIndexBuffer.capacity <
          activeCellBufferSize) {
    outError =
        "Vulkan compact resident surface generation requires resident active-cell buffers from the sparse GPU classification path.";
    cleanup();
    return false;
  }

  bool buffersReady = ensure_shared_scalar_field_buffer_for_result(
      sharedContext, computeResult, scalarFieldBufferSize,
      "compact resident surface scalar-field buffer", outError);
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellIndexBuffer,
        triangleOffsetBufferSize, "compact resident triangle-offset buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleVertexBuffer,
        compactTriangleVertexBufferSize,
        "compact resident surface triangle-vertex buffer", outError);
  }
  if (!buffersReady) {
    cleanup();
    return false;
  }

  if (sharedContext.surfaceTriangleCompactDescriptorSetLayout ==
          VK_NULL_HANDLE ||
      sharedContext.surfaceTriangleCompactPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.surfaceTriangleCompactPipeline == VK_NULL_HANDLE ||
      sharedContext.surfaceTriangleCompactDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.surfaceTriangleCompactCommandBuffer == VK_NULL_HANDLE) {
    outError =
        "Vulkan compact resident surface-triangle pipeline is unavailable.";
    cleanup();
    return false;
  }

  std::memcpy(sharedContext.candidateCellIndexBuffer.mapped,
              triangleOffsets.data(),
              static_cast<size_t>(triangleOffsetBufferSize));
  std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
              static_cast<size_t>(compactTriangleVertexBufferSize));

  VkDescriptorBufferInfo bufferInfos[5]{};
  bufferInfos[0].buffer = sharedContext.deviceCandidateCellIndexBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = activeCellBufferSize;
  bufferInfos[1].buffer =
      sharedContext.deviceCandidateCellCubeIndexBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = activeCellBufferSize;
  bufferInfos[2].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = scalarFieldBufferSize;
  bufferInfos[3].buffer = sharedContext.candidateCellIndexBuffer.buffer;
  bufferInfos[3].offset = 0;
  bufferInfos[3].range = triangleOffsetBufferSize;
  bufferInfos[4].buffer = sharedContext.surfaceTriangleVertexBuffer.buffer;
  bufferInfos[4].offset = 0;
  bufferInfos[4].range = compactTriangleVertexBufferSize;

  VkWriteDescriptorSet writes[5]{};
  for (uint32_t i = 0; i < 5; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.surfaceTriangleCompactDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 5, writes, 0, nullptr);

  VkCommandBuffer commandBuffer =
      sharedContext.surfaceTriangleCompactCommandBuffer;
  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError =
        "Failed to begin Vulkan compact resident surface command buffer.";
    cleanup();
    return false;
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.surfaceTriangleCompactPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.surfaceTriangleCompactPipelineLayout, 0,
                          1, &sharedContext.surfaceTriangleCompactDescriptorSet,
                          0, nullptr);

  SurfaceMeshPushConstants pushConstants{};
  pushConstants.candidateCellCount = activeCellCount;
  pushConstants.cellDimX = dimX - 1;
  pushConstants.cellDimY = dimY - 1;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.domainMinX = computeResult.domainMinVoxel[0];
  pushConstants.domainMinY = computeResult.domainMinVoxel[1];
  pushConstants.domainMinZ = computeResult.domainMinVoxel[2];
  pushConstants.voxelLength = computeResult.voxelLength;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer,
                     sharedContext.surfaceTriangleCompactPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceMeshPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((activeCellCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier hostBarrier{};
  hostBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  hostBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  hostBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &hostBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError =
        "Failed to end Vulkan compact resident surface command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan compact resident surface generation.";
    cleanup();
    return false;
  }

  sharedContext.residentSurfaceTriangleVerticesCompacted = true;
  outError.clear();
  cleanup();
  return true;
}

bool get_frost_vulkan_resident_surface_mesh_view(
    uint32_t activeCellCount, VulkanResidentSurfaceMeshView &outView,
    std::string &outError) {
  outView = VulkanResidentSurfaceMeshView{};
  if (activeCellCount == 0u) {
    outError.clear();
    return true;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  if (sharedContext.surfaceTriangleCountBuffer.mapped == nullptr ||
      sharedContext.surfaceTriangleVertexBuffer.mapped == nullptr) {
    outError = "Vulkan resident surface buffers are unavailable.";
    return false;
  }

  outView.triangleCounts = reinterpret_cast<const uint32_t *>(
      sharedContext.surfaceTriangleCountBuffer.mapped);
  outView.triangleVertices = reinterpret_cast<const float *>(
      sharedContext.surfaceTriangleVertexBuffer.mapped);
  outView.activeCellCount = activeCellCount;
  outView.triangleVerticesCompacted =
      sharedContext.residentSurfaceTriangleVerticesCompacted;
  outError.clear();
  return true;
}

bool get_frost_vulkan_resident_compact_surface_vertex_view(
    uint32_t totalTriangleCount, const float *&outTriangleVertices,
    std::string &outError) {
  outTriangleVertices = nullptr;
  if (totalTriangleCount == 0u) {
    outError.clear();
    return true;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  if (sharedContext.surfaceTriangleVertexBuffer.mapped == nullptr ||
      !sharedContext.residentSurfaceTriangleVerticesCompacted) {
    outError = "Vulkan compact resident surface vertex buffer is unavailable.";
    return false;
  }

  outTriangleVertices = reinterpret_cast<const float *>(
      sharedContext.surfaceTriangleVertexBuffer.mapped);
  outError.clear();
  return true;
}

bool run_frost_vulkan_generate_dense_surface_mesh(
    const VulkanParticleComputeResult &computeResult,
    std::vector<uint32_t> &outActiveCellIndices,
    std::vector<uint32_t> &outActiveCellCubeIndices,
    VulkanSurfaceMeshResult &outMesh,
    std::string &outError) {
  outActiveCellIndices.clear();
  outActiveCellCubeIndices.clear();
  outMesh.triangleCounts.clear();
  outMesh.triangleVertices.clear();
  outMesh.totalTriangleCount = 0;

  const int32_t dimX = computeResult.domainDimensions[0];
  const int32_t dimY = computeResult.domainDimensions[1];
  const int32_t dimZ = computeResult.domainDimensions[2];
  const int32_t cellDimX = dimX - 1;
  const int32_t cellDimY = dimY - 1;
  const int32_t cellDimZ = dimZ - 1;
  const std::size_t expectedScalarSamples =
      static_cast<std::size_t>(dimX) * static_cast<std::size_t>(dimY) *
      static_cast<std::size_t>(dimZ);
  if (cellDimX <= 0 || cellDimY <= 0 || cellDimZ <= 0 ||
      !has_vulkan_scalar_field_samples(computeResult, expectedScalarSamples)) {
    outError =
        "Vulkan dense surface mesh generation requires a valid scalar-field domain.";
    return false;
  }

  const std::size_t cellCount = static_cast<std::size_t>(cellDimX) *
                                static_cast<std::size_t>(cellDimY) *
                                static_cast<std::size_t>(cellDimZ);
  if (cellCount == 0) {
    outError.clear();
    return true;
  }
  if (cellCount > static_cast<std::size_t>(std::numeric_limits<uint32_t>::max())) {
    outError = "Vulkan dense surface mesh generation exceeds addressable cell count.";
    return false;
  }

  auto &sharedContext = get_vulkan_shared_context();
  std::lock_guard<std::mutex> lock(sharedContext.mutex);
  if (!initialize_vulkan_shared_context(sharedContext, outError)) {
    return false;
  }

  VkDevice device = sharedContext.device;
  VkQueue queue = sharedContext.queue;
  VkCommandPool commandPool = sharedContext.commandPool;
  auto vkUpdateDescriptorSets = sharedContext.vkUpdateDescriptorSets;
  auto vkBeginCommandBuffer = sharedContext.vkBeginCommandBuffer;
  auto vkCmdBindPipeline = sharedContext.vkCmdBindPipeline;
  auto vkCmdBindDescriptorSets = sharedContext.vkCmdBindDescriptorSets;
  auto vkCmdPushConstants = sharedContext.vkCmdPushConstants;
  auto vkCmdDispatch = sharedContext.vkCmdDispatch;
  auto vkCmdPipelineBarrier = sharedContext.vkCmdPipelineBarrier;
  auto vkEndCommandBuffer = sharedContext.vkEndCommandBuffer;
  auto vkQueueSubmit = sharedContext.vkQueueSubmit;
  auto vkQueueWaitIdle = sharedContext.vkQueueWaitIdle;
  auto vkResetCommandPool = sharedContext.vkResetCommandPool;

  auto cleanup = [&]() {
    if (device != VK_NULL_HANDLE) {
      if (queue != VK_NULL_HANDLE && vkQueueWaitIdle) {
        vkQueueWaitIdle(queue);
      }
      if (commandPool != VK_NULL_HANDLE && vkResetCommandPool) {
        vkResetCommandPool(device, commandPool, 0);
      }
    }
  };

  const VkDeviceSize scalarFieldBufferSize = static_cast<VkDeviceSize>(
      expectedScalarSamples * sizeof(float));
  const VkDeviceSize cubeIndexBufferSize =
      static_cast<VkDeviceSize>(cellCount * sizeof(uint32_t));
  const VkDeviceSize triangleCountBufferSize =
      static_cast<VkDeviceSize>(cellCount * sizeof(uint32_t));
  const VkDeviceSize triangleVertexBufferSize = static_cast<VkDeviceSize>(
      cellCount * 15ull * 3ull * sizeof(float));

  bool buffersReady = ensure_shared_scalar_field_buffer_for_result(
      sharedContext, computeResult, scalarFieldBufferSize,
      "dense surface-mesh scalar-field buffer", outError);
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.candidateCellCubeIndexBuffer,
        cubeIndexBufferSize, "dense surface cube-index buffer", outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleCountBuffer,
        triangleCountBufferSize, "dense surface triangle-count buffer",
        outError);
  }
  if (buffersReady) {
    buffersReady = ensure_shared_host_buffer(
        sharedContext, sharedContext.surfaceTriangleVertexBuffer,
        triangleVertexBufferSize, "dense surface triangle-vertex buffer",
        outError);
  }
  if (!buffersReady) {
    cleanup();
    return false;
  }

  std::memset(sharedContext.candidateCellCubeIndexBuffer.mapped, 0,
              static_cast<size_t>(cubeIndexBufferSize));
  std::memset(sharedContext.surfaceTriangleCountBuffer.mapped, 0,
              static_cast<size_t>(triangleCountBufferSize));
  std::memset(sharedContext.surfaceTriangleVertexBuffer.mapped, 0,
              static_cast<size_t>(triangleVertexBufferSize));

  if (sharedContext.denseSurfaceMeshDescriptorSetLayout == VK_NULL_HANDLE ||
      sharedContext.denseSurfaceMeshPipelineLayout == VK_NULL_HANDLE ||
      sharedContext.denseSurfaceMeshPipeline == VK_NULL_HANDLE ||
      sharedContext.denseSurfaceMeshDescriptorSet == VK_NULL_HANDLE ||
      sharedContext.denseSurfaceMeshCommandBuffer == VK_NULL_HANDLE) {
    outError = "Vulkan dense surface-mesh pipeline is unavailable.";
    cleanup();
    return false;
  }

  VkDescriptorBufferInfo bufferInfos[4]{};
  bufferInfos[0].buffer = sharedContext.voxelScalarFieldBuffer.buffer;
  bufferInfos[0].offset = 0;
  bufferInfos[0].range = scalarFieldBufferSize;
  bufferInfos[1].buffer = sharedContext.candidateCellCubeIndexBuffer.buffer;
  bufferInfos[1].offset = 0;
  bufferInfos[1].range = cubeIndexBufferSize;
  bufferInfos[2].buffer = sharedContext.surfaceTriangleCountBuffer.buffer;
  bufferInfos[2].offset = 0;
  bufferInfos[2].range = triangleCountBufferSize;
  bufferInfos[3].buffer = sharedContext.surfaceTriangleVertexBuffer.buffer;
  bufferInfos[3].offset = 0;
  bufferInfos[3].range = triangleVertexBufferSize;

  VkWriteDescriptorSet writes[4]{};
  for (uint32_t i = 0; i < 4; ++i) {
    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = sharedContext.denseSurfaceMeshDescriptorSet;
    writes[i].dstBinding = i;
    writes[i].descriptorCount = 1;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].pBufferInfo = &bufferInfos[i];
  }
  vkUpdateDescriptorSets(device, 4, writes, 0, nullptr);
  VkCommandBuffer commandBuffer = sharedContext.denseSurfaceMeshCommandBuffer;

  VkCommandBufferBeginInfo beginInfo{};
  beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS) {
    outError = "Failed to begin Vulkan dense surface-mesh command buffer.";
    cleanup();
    return false;
  }

  vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                    sharedContext.denseSurfaceMeshPipeline);
  vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                          sharedContext.denseSurfaceMeshPipelineLayout, 0, 1,
                          &sharedContext.denseSurfaceMeshDescriptorSet, 0,
                          nullptr);

  SurfaceMeshPushConstants pushConstants{};
  pushConstants.candidateCellCount = static_cast<uint32_t>(cellCount);
  pushConstants.cellDimX = cellDimX;
  pushConstants.cellDimY = cellDimY;
  pushConstants.domainDimX = dimX;
  pushConstants.domainDimY = dimY;
  pushConstants.domainMinX = computeResult.domainMinVoxel[0];
  pushConstants.domainMinY = computeResult.domainMinVoxel[1];
  pushConstants.domainMinZ = computeResult.domainMinVoxel[2];
  pushConstants.voxelLength = computeResult.voxelLength;
  pushConstants.surfaceIsoValue = computeResult.surfaceIsoValue;
  vkCmdPushConstants(commandBuffer, sharedContext.denseSurfaceMeshPipelineLayout,
                     VK_SHADER_STAGE_COMPUTE_BIT, 0,
                     sizeof(SurfaceMeshPushConstants), &pushConstants);

  const uint32_t workgroupCount =
      static_cast<uint32_t>((cellCount + 63u) / 64u);
  vkCmdDispatch(commandBuffer, workgroupCount, 1, 1);

  VkMemoryBarrier memoryBarrier{};
  memoryBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
  memoryBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
  memoryBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
  vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0,
                       nullptr, 0, nullptr);

  if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS) {
    outError = "Failed to end Vulkan dense surface-mesh command buffer.";
    cleanup();
    return false;
  }

  VkSubmitInfo submitInfo{};
  submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submitInfo.commandBufferCount = 1;
  submitInfo.pCommandBuffers = &commandBuffer;
  if (vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS ||
      vkQueueWaitIdle(queue) != VK_SUCCESS) {
    outError = "Failed to submit Vulkan dense surface-mesh generation.";
    cleanup();
    return false;
  }

  std::vector<uint32_t> denseCubeIndices(cellCount, 0u);
  std::vector<uint32_t> denseTriangleCounts(cellCount, 0u);
  std::vector<float> denseTriangleVertices(cellCount * 15ull * 3ull, 0.0f);
  std::memcpy(denseCubeIndices.data(),
              sharedContext.candidateCellCubeIndexBuffer.mapped,
              static_cast<size_t>(cubeIndexBufferSize));
  std::memcpy(denseTriangleCounts.data(),
              sharedContext.surfaceTriangleCountBuffer.mapped,
              static_cast<size_t>(triangleCountBufferSize));
  std::memcpy(denseTriangleVertices.data(),
              sharedContext.surfaceTriangleVertexBuffer.mapped,
              static_cast<size_t>(triangleVertexBufferSize));

  std::size_t activeCellCount = 0;
  uint32_t totalTriangleCount = 0;
  for (std::size_t i = 0; i < cellCount; ++i) {
    const uint32_t cubeIndex = denseCubeIndices[i];
    const uint32_t triangleCount = denseTriangleCounts[i];
    if (triangleCount > 5u) {
      outError =
          "Vulkan dense surface-mesh generation produced an invalid triangle count.";
      cleanup();
      return false;
    }
    if (triangleCount == 0u) {
      continue;
    }
    if (cubeIndex == 0u || cubeIndex == 255u) {
      outError =
          "Vulkan dense surface-mesh generation produced triangles for an inactive cube.";
      cleanup();
      return false;
    }
    ++activeCellCount;
    totalTriangleCount += triangleCount;
  }

  if (activeCellCount == 0 || totalTriangleCount == 0u) {
    outError = "Vulkan dense surface mesh produced no triangles.";
    cleanup();
    return false;
  }

  outActiveCellIndices.reserve(activeCellCount);
  outActiveCellCubeIndices.reserve(activeCellCount);
  outMesh.triangleCounts.reserve(activeCellCount);
  outMesh.triangleVertices.resize(activeCellCount * 15ull * 3ull, 0.0f);

  const std::size_t floatsPerCell = 15ull * 3ull;
  std::size_t compactCellIndex = 0;
  for (std::size_t i = 0; i < cellCount; ++i) {
    const uint32_t triangleCount = denseTriangleCounts[i];
    if (triangleCount == 0u) {
      continue;
    }
    outActiveCellIndices.push_back(static_cast<uint32_t>(i));
    outActiveCellCubeIndices.push_back(denseCubeIndices[i]);
    outMesh.triangleCounts.push_back(triangleCount);
    std::memcpy(&outMesh.triangleVertices[compactCellIndex * floatsPerCell],
                &denseTriangleVertices[i * floatsPerCell],
                static_cast<size_t>(floatsPerCell * sizeof(float)));
    ++compactCellIndex;
  }
  outMesh.totalTriangleCount = totalTriangleCount;

  cleanup();
  outError.clear();
  return true;
}
