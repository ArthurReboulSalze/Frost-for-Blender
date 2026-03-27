#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class FrostInterface {
public:
    FrostInterface();
    ~FrostInterface();

    static bool has_gpu_backend();
    static std::string get_gpu_backend_name();
    std::string get_last_meshing_backend() const;
    std::string get_last_meshing_status() const;
    bool get_last_meshing_used_fallback() const;

    // Set particle data from raw contiguous arrays.
    // positions and velocities use XYZXYZ... layout.
    void set_particles(
        const float* positions,
        size_t particleCount,
        const float* radii,
        const float* velocities = nullptr
    );

    void set_parameter(const std::string& name, bool value);
    void set_parameter(const std::string& name, int value);
    void set_parameter(const std::string& name, float value);

    // Generate mesh into flat XYZ and triangle-index buffers.
    void generate_mesh(std::vector<float>& vertices, std::vector<int>& faces);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
