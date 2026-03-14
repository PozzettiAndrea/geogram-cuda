#ifndef GEOGRAM_VORONOI_GEO_SERIALIZE_H
#define GEOGRAM_VORONOI_GEO_SERIALIZE_H

// ============================================================
// Binary serialization templates for geogram checkpoint system
//
// Follows the pattern from QuadriFlow-cuda/serialize.hpp and
// quadwild-bimdf-cuda/qw_serialize.h:
//   - Template-based Save/Read for common types
//   - Fast POD paths for vectors of primitives
//   - Size-prefixed collections
// ============================================================

#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <cstring>

namespace geo_ser {

// ============================================================
// Scalars
// ============================================================

template<typename T>
inline void Save(FILE* fp, const T& val) {
    fwrite(&val, sizeof(T), 1, fp);
}

template<typename T>
inline void Read(FILE* fp, T& val) {
    fread(&val, sizeof(T), 1, fp);
}

// ============================================================
// Strings
// ============================================================

inline void SaveString(FILE* fp, const std::string& s) {
    int32_t len = (int32_t)s.size();
    fwrite(&len, sizeof(int32_t), 1, fp);
    if (len > 0) fwrite(s.data(), 1, len, fp);
}

inline void ReadString(FILE* fp, std::string& s) {
    int32_t len;
    fread(&len, sizeof(int32_t), 1, fp);
    s.resize(len);
    if (len > 0) fread(&s[0], 1, len, fp);
}

// ============================================================
// Vectors of POD types (fast bulk I/O)
// ============================================================

template<typename T>
inline void SaveVec(FILE* fp, const std::vector<T>& v) {
    int64_t n = (int64_t)v.size();
    fwrite(&n, sizeof(int64_t), 1, fp);
    if (n > 0) fwrite(v.data(), sizeof(T), n, fp);
}

template<typename T>
inline void ReadVec(FILE* fp, std::vector<T>& v) {
    int64_t n;
    fread(&n, sizeof(int64_t), 1, fp);
    v.resize((size_t)n);
    if (n > 0) fread(v.data(), sizeof(T), n, fp);
}

// ============================================================
// Vector of vectors (nested)
// ============================================================

template<typename T>
inline void SaveVecVec(FILE* fp, const std::vector<std::vector<T>>& v) {
    int64_t n = (int64_t)v.size();
    fwrite(&n, sizeof(int64_t), 1, fp);
    for (auto& inner : v) SaveVec(fp, inner);
}

template<typename T>
inline void ReadVecVec(FILE* fp, std::vector<std::vector<T>>& v) {
    int64_t n;
    fread(&n, sizeof(int64_t), 1, fp);
    v.resize((size_t)n);
    for (auto& inner : v) ReadVec(fp, inner);
}

// ============================================================
// Vector of bools (special: stored as uint8_t)
// ============================================================

inline void SaveBoolVec(FILE* fp, const std::vector<bool>& v) {
    int64_t n = (int64_t)v.size();
    fwrite(&n, sizeof(int64_t), 1, fp);
    for (int64_t i = 0; i < n; i++) {
        uint8_t b = v[i] ? 1 : 0;
        fwrite(&b, 1, 1, fp);
    }
}

inline void ReadBoolVec(FILE* fp, std::vector<bool>& v) {
    int64_t n;
    fread(&n, sizeof(int64_t), 1, fp);
    v.resize((size_t)n);
    for (int64_t i = 0; i < n; i++) {
        uint8_t b;
        fread(&b, 1, 1, fp);
        v[i] = (b != 0);
    }
}

// ============================================================
// Raw array I/O (for double*, int* with known sizes)
// ============================================================

inline void SaveArray(FILE* fp, const double* arr, int64_t count) {
    fwrite(&count, sizeof(int64_t), 1, fp);
    if (count > 0) fwrite(arr, sizeof(double), count, fp);
}

inline void ReadArray(FILE* fp, double* arr, int64_t expected_count) {
    int64_t count;
    fread(&count, sizeof(int64_t), 1, fp);
    if (count != expected_count) {
        fprintf(stderr, "[GEO_SER] WARNING: array size mismatch: expected %lld, got %lld\n",
                (long long)expected_count, (long long)count);
    }
    if (count > 0) fread(arr, sizeof(double), count, fp);
}

// Read into newly allocated array (caller must delete[])
inline double* ReadArrayAlloc(FILE* fp, int64_t& out_count) {
    fread(&out_count, sizeof(int64_t), 1, fp);
    double* arr = new double[(size_t)out_count];
    if (out_count > 0) fread(arr, sizeof(double), out_count, fp);
    return arr;
}

} // namespace geo_ser

#endif // GEOGRAM_VORONOI_GEO_SERIALIZE_H
