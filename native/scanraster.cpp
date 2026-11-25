// scanraster: host-side helpers for ./uc scan_* (a shared library loaded with ctypes by src/uncrashed_mcp/scan/native.py).
//
//   sr_raster    top-down z-buffer rasteriser of a ucmesh index (native/ucmesh.cpp format): highest surface (DSM),
//                lowest surface, unit normal of the highest surface and its texture colour, on a regular grid
//   sr_label     connected components of a byte mask (4- or 8-connected), labels 1..n
//   sr_nearest   for every cell the flat index of a nearby set cell (two-pass 8-neighbour propagation)
//   sr_contour   outer boundary of one labelled component (Moore neighbour tracing), pixel coordinates
//   sr_fill_poly even-odd scanline fill of a polygon given in pixel coordinates (cell centres at integers)
//   sr_facades   orthographic images (colour + depth) of many wall planes, for facade detection
//
// Grid. Cell (row r, col c) covers east x0 + (c + .5) * res and south y0 + (r + .5) * res: row 0 is the northern
// edge. World axes are east, south, up metres; the source axes are mapped by (src index, sign) per world axis like
// ucmesh's --axes. Detail levels: tile names that are digit strings form an octree; a parent triangle whose centroid
// lies inside the bounds of an existing child subtree is skipped (the same rule as ucmesh pack).
//
// Build (done on demand by scan/native.py):
//   g++ -O3 -fopenmp -std=c++17 -shared -fPIC native/scanraster.cpp -lturbojpeg -o native/libscanraster.so
#include <turbojpeg.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

typedef uint32_t u32; typedef uint64_t u64;

namespace {

struct Mesh { std::string name, tile; u32 level = 0, tex = 0; u64 v0 = 0, nv = 0, i0 = 0, ni = 0; float lo[3], hi[3]; };
struct Box { float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    void add(const float *a, const float *b) { for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], a[k]); hi[k] = std::max(hi[k], b[k]); } }
    bool in(const float *p, float e) const { for (int k = 0; k < 3; k++) if (p[k] < lo[k] - e || p[k] > hi[k] + e) return false; return true; } };
struct Img { int w = 0, h = 0; std::vector<unsigned char> px; };

struct Map {
    void *p = MAP_FAILED; size_t n = 0;
    bool open_(const std::string &f) {
        int fd = open(f.c_str(), O_RDONLY); if (fd < 0) return false;
        struct stat st; if (fstat(fd, &st)) { close(fd); return false; }
        n = st.st_size; if (n) p = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        return !n || p != MAP_FAILED;
    }
    ~Map() { if (n && p != MAP_FAILED) munmap(p, n); }
};

bool load_index(const std::string &dir, std::vector<std::string> &tex, std::vector<Mesh> &mesh, std::string &why) {
    std::ifstream f(dir + "/index.bin", std::ios::binary); char mg[8];
    if (!f.read(mg, 8) || memcmp(mg, "UCMIDX1", 8)) { why = "not a ucmesh index: " + dir + "/index.bin"; return false; }
    auto r32 = [&] { u32 v = 0; f.read((char *)&v, 4); return v; };
    auto r64 = [&] { u64 v = 0; f.read((char *)&v, 8); return v; };
    auto rs = [&] { u32 n = r32(); if (n > (1u << 20)) { f.setstate(std::ios::failbit); return std::string(); }
                    std::string s(n, '\0'); f.read(s.data(), n); return s; };
    u32 nt = r32(), nm = r32(); if (!f) { why = "truncated index.bin"; return false; }
    tex.resize(nt); mesh.resize(nm);
    for (auto &t : tex) t = rs();
    for (auto &m : mesh) {
        m.name = rs(); m.level = r32(); m.tex = r32(); m.v0 = r64(); m.nv = r64(); m.i0 = r64(); m.ni = r64();
        f.read((char *)m.lo, 12); f.read((char *)m.hi, 12); m.tile = m.name.substr(0, m.name.find('_'));
    }
    if (!f) { why = "truncated index.bin"; return false; }
    return true;
}

bool octree_names(const std::vector<Mesh> &M) {
    for (auto &m : M) { if (m.tile.empty()) return false; for (char c : m.tile) if (c < '0' || c > '7') return false; }
    return !M.empty();
}

}  // namespace

extern "C" {

// Returns 0 on success, else 1 with a message in err. stats (may be NULL) receives
// [triangles drawn, triangles culled as covered by finer tiles, meshes used, texture decode seconds, raster seconds].
int sr_raster(const char *idx_dir, const int *axsrc, const float *axsign, double x0, double y0, double res, int W, int H,
              int cull_octree, float *dsm, float *low, float *nrm, uint8_t *rgb, double *stats, char *err, int ecap) {
    std::string dir = idx_dir, why;
    std::vector<std::string> tex; std::vector<Mesh> M;
    if (!load_index(dir, tex, M, why)) { snprintf(err, ecap, "%s", why.c_str()); return 1; }
    if (W <= 0 || H <= 0 || !(res > 0)) { snprintf(err, ecap, "bad grid %dx%d res %g", W, H, res); return 1; }
    Map vm, tm;
    if (!vm.open_(dir + "/verts.bin") || !tm.open_(dir + "/tris.bin")) { snprintf(err, ecap, "cannot map verts.bin/tris.bin in %s", idx_dir); return 1; }
    const float *VB = (const float *)vm.p; const u32 *TB = (const u32 *)tm.p;
    u64 nverts = vm.n / 20, nidx = tm.n / 4;
    // world bounds of each mesh, grid overlap test
    std::vector<char> use(M.size(), 0); std::vector<double> wlo(M.size() * 2), whi(M.size() * 2);
    double gx1 = x0 + W * res, gy1 = y0 + H * res;
    int used = 0;
    for (size_t i = 0; i < M.size(); i++) {
        const Mesh &m = M[i];
        if (m.v0 + m.nv > nverts || m.i0 + m.ni > nidx) { snprintf(err, ecap, "index out of range in mesh %zu (stale index?)", i); return 1; }
        for (int k = 0; k < 2; k++) {
            double a = axsign[k] * m.lo[axsrc[k]], b = axsign[k] * m.hi[axsrc[k]];
            wlo[i * 2 + k] = std::min(a, b); whi[i * 2 + k] = std::max(a, b);
        }
        if (whi[i * 2] < x0 - res || wlo[i * 2] > gx1 + res || whi[i * 2 + 1] < y0 - res || wlo[i * 2 + 1] > gy1 + res) continue;
        use[i] = 1; used++;
    }
    // octree culling boxes (source axes)
    bool tree = cull_octree && octree_names(M);
    std::unordered_map<std::string, Box> box;
    if (tree) for (auto &m : M) for (size_t n = 1; n <= m.tile.size(); n++) box[m.tile.substr(0, n)].add(m.lo, m.hi);
    std::vector<std::vector<Box>> kids(M.size());
    if (tree) for (size_t i = 0; i < M.size(); i++) if (use[i])
        for (char d = '0'; d <= '7'; d++) { auto it = box.find(M[i].tile + d); if (it != box.end()) kids[i].push_back(it->second); }
    // textures of the meshes used
    std::vector<Img> T(tex.size());
    double t0 = omp_get_wtime();
    if (rgb) {
        std::vector<u32> need; std::vector<char> seen(tex.size(), 0);
        for (size_t i = 0; i < M.size(); i++) if (use[i] && M[i].tex < tex.size() && !seen[M[i].tex]) { seen[M[i].tex] = 1; need.push_back(M[i].tex); }
#pragma omp parallel
        { tjhandle tj = tjInitDecompress();
#pragma omp for schedule(dynamic, 4)
          for (size_t j = 0; j < need.size(); j++) {
              u32 i = need[j];
              std::ifstream g(tex[i], std::ios::binary); std::vector<unsigned char> b((std::istreambuf_iterator<char>(g)), {});
              int w, h, ss, cs; if (b.empty() || tjDecompressHeader3(tj, b.data(), b.size(), &w, &h, &ss, &cs)) continue;
              T[i].px.resize((size_t)w * h * 3);
              if (tjDecompress2(tj, b.data(), b.size(), T[i].px.data(), w, 0, h, TJPF_RGB, 0)) { T[i].px.clear(); continue; }
              T[i].w = w; T[i].h = h;
          }
          tjDestroy(tj); }
    }
    double t1 = omp_get_wtime();
    size_t N = (size_t)W * H;
    for (size_t i = 0; i < N; i++) { dsm[i] = -INFINITY; if (low) low[i] = INFINITY; }
    if (nrm) memset(nrm, 0, N * 3 * sizeof(float));
    if (rgb) memset(rgb, 0, N * 3);
    int nth = omp_get_max_threads();
    std::vector<u64> drawn(nth, 0), culled(nth, 0);
    const int ex = axsrc[0], sy = axsrc[1], up = axsrc[2]; const float se = axsign[0], ss_ = axsign[1], su = axsign[2];
#pragma omp parallel
    { int th = omp_get_thread_num(), nt = omp_get_num_threads();
      int c0 = (int)((long)W * th / nt), c1 = (int)((long)W * (th + 1) / nt);
      double xlo = x0 + c0 * res, xhi = x0 + c1 * res;
      for (size_t mi = 0; mi < M.size(); mi++) {
        if (!use[mi]) continue;
        const Mesh &m = M[mi];
        if (whi[mi * 2] < xlo - res || wlo[mi * 2] > xhi + res) continue;
        const Img *im = (rgb && m.tex < T.size() && T[m.tex].w) ? &T[m.tex] : nullptr;
        for (u64 t = 0; t + 2 < m.ni; t += 3) {
            const float *P[3];
            for (int k = 0; k < 3; k++) P[k] = VB + (m.v0 + TB[m.i0 + t + k]) * 5;
            double X[3], Y[3], U[3];
            for (int k = 0; k < 3; k++) { X[k] = (se * P[k][ex] - x0) / res - .5; Y[k] = (ss_ * P[k][sy] - y0) / res - .5; U[k] = su * P[k][up]; }
            int raw0 = std::min(W - 1, std::max(0, (int)std::ceil(std::min({X[0], X[1], X[2]})))); bool own = raw0 >= c0 && raw0 < c1;
            int cx0 = std::max(c0, (int)std::ceil(std::min({X[0], X[1], X[2]})));
            int cx1 = std::min(c1 - 1, (int)std::floor(std::max({X[0], X[1], X[2]})));
            if (cx0 > cx1) continue;
            int r0 = std::max(0, (int)std::ceil(std::min({Y[0], Y[1], Y[2]})));
            int r1 = std::min(H - 1, (int)std::floor(std::max({Y[0], Y[1], Y[2]})));
            if (r0 > r1) continue;
            if (!kids[mi].empty()) {
                float c[3]; for (int j = 0; j < 3; j++) c[j] = (P[0][j] + P[1][j] + P[2][j]) / 3;
                bool covered = false; for (auto &b : kids[mi]) if (b.in(c, .05f)) { covered = true; break; }
                if (covered) { culled[th] += own; continue; }
            }
            double den = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]);
            if (std::fabs(den) < 1e-12) continue;  // vertical or degenerate: walls show up as height steps instead
            drawn[th] += own;
            double ax = X[1] - X[0], ay = Y[1] - Y[0], az = (U[1] - U[0]) / res, bx = X[2] - X[0], by = Y[2] - Y[0], bz = (U[2] - U[0]) / res;
            double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, nl = std::sqrt(nx * nx + ny * ny + nz * nz) + 1e-12;
            nx /= nl; ny /= nl; nz /= nl; if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
            for (int r = r0; r <= r1; r++) for (int c = cx0; c <= cx1; c++) {
                double l1 = ((c - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (r - Y[0])) / den;
                double l2 = ((X[1] - X[0]) * (r - Y[0]) - (c - X[0]) * (Y[1] - Y[0])) / den;
                double l0 = 1 - l1 - l2; const double e = -1e-6;
                if (l0 < e || l1 < e || l2 < e) continue;
                float z = (float)(l0 * U[0] + l1 * U[1] + l2 * U[2]); size_t i = (size_t)r * W + c;
                if (low && z < low[i]) low[i] = z;
                if (z > dsm[i]) {
                    dsm[i] = z;
                    if (nrm) { nrm[i * 3] = (float)nx; nrm[i * 3 + 1] = (float)ny; nrm[i * 3 + 2] = (float)nz; }
                    if (im) {
                        double u = l0 * P[0][3] + l1 * P[1][3] + l2 * P[2][3], v = l0 * P[0][4] + l1 * P[1][4] + l2 * P[2][4];
                        u -= std::floor(u); v -= std::floor(v);
                        int px = std::min(im->w - 1, std::max(0, (int)(u * im->w))), py = std::min(im->h - 1, std::max(0, (int)((1 - v) * im->h)));
                        memcpy(&rgb[i * 3], &im->px[((size_t)py * im->w + px) * 3], 3);
                    }
                }
            }
        }
      } }
    for (size_t i = 0; i < N; i++) if (std::isinf(dsm[i])) { dsm[i] = NAN; if (low) low[i] = NAN; }
    if (stats) {
        u64 d = 0, c = 0; for (int i = 0; i < nth; i++) { d += drawn[i]; c += culled[i]; }
        stats[0] = (double)d; stats[1] = (double)c; stats[2] = used; stats[3] = t1 - t0; stats[4] = omp_get_wtime() - t1;
    }
    return 0;
}

// Orthographic facade images: nf wall planes rendered in one pass over the index.
// F holds 12 doubles per facade: oe, os (east, south of the wall line at u = 0), ue, us (unit direction along the
// wall), ne, ns (unit outward normal, horizontal), u0 (u of the left image edge), ztop (up of the top edge), res
// (metres per pixel), dmin, dmax (depth slab along the normal, metres; the viewer stands outside at +normal), spare.
// dims: W, H per facade; offs: pixel offset of each facade in the shared rgb (3 bytes / px) and depth buffers.
// Pixel (col i, row j) centre: u = u0 + (i + .5) res, up = ztop - (j + .5) res. The z-buffer keeps the surface
// nearest to the viewer inside the slab; triangles whose normal is closer than min_facing to horizontal-across
// (|n . normal| < min_facing) are skipped (roofs, ground, walls seen edge-on). depth = NAN where nothing was drawn.
// skip (may be NULL): per mesh 1 = ignore (coarser detail level of an area that finer tiles cover).
// stats: [triangles drawn, facades, texture decode seconds, render seconds].
int sr_facades(const char *idx_dir, const int *axsrc, const float *axsign, int nf, const double *F, const int *dims,
               const int64_t *offs, uint8_t *rgb, float *depth, const uint8_t *skip, double min_facing, double *stats,
               char *err, int ecap) {
    std::string dir = idx_dir, why;
    std::vector<std::string> tex; std::vector<Mesh> M;
    if (!load_index(dir, tex, M, why)) { snprintf(err, ecap, "%s", why.c_str()); return 1; }
    Map vm, tm;
    if (!vm.open_(dir + "/verts.bin") || !tm.open_(dir + "/tris.bin")) { snprintf(err, ecap, "cannot map verts.bin/tris.bin in %s", idx_dir); return 1; }
    const float *VB = (const float *)vm.p; const u32 *TB = (const u32 *)tm.p;
    u64 nverts = vm.n / 20, nidx = tm.n / 4;
    const int ex = axsrc[0], sy = axsrc[1], up = axsrc[2]; const float se = axsign[0], ss_ = axsign[1], su = axsign[2];
    // world boxes of meshes and facade slabs
    std::vector<float> mb(M.size() * 6);
    for (size_t i = 0; i < M.size(); i++) {
        const Mesh &m = M[i];
        if (m.v0 + m.nv > nverts || m.i0 + m.ni > nidx) { snprintf(err, ecap, "index out of range in mesh %zu (stale index?)", i); return 1; }
        for (int k = 0; k < 3; k++) {
            float s = axsign[k]; int a = axsrc[k]; float p = s * m.lo[a], q = s * m.hi[a];
            mb[i * 6 + k] = std::min(p, q); mb[i * 6 + 3 + k] = std::max(p, q);
        }
    }
    std::vector<std::vector<u32>> todo(nf);
    std::vector<char> need(tex.size(), 0);
    for (int f = 0; f < nf; f++) {
        const double *P = F + (size_t)f * 12; int W = dims[f * 2], H = dims[f * 2 + 1];
        if (W <= 0 || H <= 0 || !(P[8] > 0)) { snprintf(err, ecap, "bad facade %d: %dx%d res %g", f, W, H, P[8]); return 1; }
        double lo[3] = {1e30, 1e30, P[7] - H * P[8]}, hi[3] = {-1e30, -1e30, P[7]};
        for (double u : {P[6], P[6] + W * P[8]}) for (double d : {P[9], P[10]}) {
            double e = P[0] + u * P[2] + d * P[4], s = P[1] + u * P[3] + d * P[5];
            lo[0] = std::min(lo[0], e); hi[0] = std::max(hi[0], e); lo[1] = std::min(lo[1], s); hi[1] = std::max(hi[1], s);
        }
        for (size_t i = 0; i < M.size(); i++) {
            if (skip && skip[i]) continue;
            const float *b = &mb[i * 6]; bool hit = true;
            for (int k = 0; k < 3; k++) if (b[3 + k] < lo[k] || b[k] > hi[k]) { hit = false; break; }
            if (hit) { todo[f].push_back((u32)i); if (M[i].tex < tex.size()) need[M[i].tex] = 1; }
        }
    }
    std::vector<Img> T(tex.size());
    double t0 = omp_get_wtime();
    { std::vector<u32> list; for (size_t i = 0; i < need.size(); i++) if (need[i]) list.push_back((u32)i);
#pragma omp parallel
      { tjhandle tj = tjInitDecompress();
#pragma omp for schedule(dynamic, 4)
        for (size_t j = 0; j < list.size(); j++) {
            u32 i = list[j];
            std::ifstream g(tex[i], std::ios::binary); std::vector<unsigned char> b((std::istreambuf_iterator<char>(g)), {});
            int w, h, ss, cs; if (b.empty() || tjDecompressHeader3(tj, b.data(), b.size(), &w, &h, &ss, &cs)) continue;
            T[i].px.resize((size_t)w * h * 3);
            if (tjDecompress2(tj, b.data(), b.size(), T[i].px.data(), w, 0, h, TJPF_RGB, 0)) { T[i].px.clear(); continue; }
            T[i].w = w; T[i].h = h;
        }
        tjDestroy(tj); } }
    double t1 = omp_get_wtime();
    std::vector<u64> drawn(nf, 0);
#pragma omp parallel for schedule(dynamic, 1)
    for (int f = 0; f < nf; f++) {
        const double *P = F + (size_t)f * 12; int W = dims[f * 2], H = dims[f * 2 + 1];
        const double oe = P[0], os = P[1], ue = P[2], us = P[3], ne = P[4], ns = P[5], u0 = P[6], zt = P[7], res = P[8], dmin = P[9], dmax = P[10];
        uint8_t *img = rgb + offs[f] * 3; float *dep = depth + offs[f];
        for (size_t i = 0; i < (size_t)W * H; i++) dep[i] = -INFINITY;
        memset(img, 0, (size_t)W * H * 3);
        for (u32 mi : todo[f]) {
            const Mesh &m = M[mi];
            const Img *im = (m.tex < T.size() && T[m.tex].w) ? &T[m.tex] : nullptr;
            for (u64 t = 0; t + 2 < m.ni; t += 3) {
                const float *Q[3]; double X[3], Y[3], D[3], E[3], S[3], U[3];
                for (int k = 0; k < 3; k++) {
                    Q[k] = VB + (m.v0 + TB[m.i0 + t + k]) * 5;
                    E[k] = se * Q[k][ex]; S[k] = ss_ * Q[k][sy]; U[k] = su * Q[k][up];
                    double du = (E[k] - oe) * ue + (S[k] - os) * us;
                    D[k] = (E[k] - oe) * ne + (S[k] - os) * ns;
                    X[k] = (du - u0) / res - .5; Y[k] = (zt - U[k]) / res - .5;
                }
                if (std::max({D[0], D[1], D[2]}) < dmin || std::min({D[0], D[1], D[2]}) > dmax) continue;
                int c0 = std::max(0, (int)std::ceil(std::min({X[0], X[1], X[2]}))), c1 = std::min(W - 1, (int)std::floor(std::max({X[0], X[1], X[2]})));
                int r0 = std::max(0, (int)std::ceil(std::min({Y[0], Y[1], Y[2]}))), r1 = std::min(H - 1, (int)std::floor(std::max({Y[0], Y[1], Y[2]})));
                if (c0 > c1 || r0 > r1) continue;
                double ax = E[1] - E[0], ay = S[1] - S[0], az = U[1] - U[0], bx = E[2] - E[0], by = S[2] - S[0], bz = U[2] - U[0];
                double nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx, nl = std::sqrt(nx * nx + ny * ny + nz * nz);
                if (nl < 1e-12 || std::fabs((nx * ne + ny * ns) / nl) < min_facing) continue;
                double den = (X[1] - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (Y[1] - Y[0]);
                if (std::fabs(den) < 1e-12) continue;
                drawn[f]++;
                for (int r = r0; r <= r1; r++) for (int c = c0; c <= c1; c++) {
                    double l1 = ((c - X[0]) * (Y[2] - Y[0]) - (X[2] - X[0]) * (r - Y[0])) / den;
                    double l2 = ((X[1] - X[0]) * (r - Y[0]) - (c - X[0]) * (Y[1] - Y[0])) / den;
                    double l0 = 1 - l1 - l2; const double e = -1e-6;
                    if (l0 < e || l1 < e || l2 < e) continue;
                    double d = l0 * D[0] + l1 * D[1] + l2 * D[2];
                    if (d < dmin || d > dmax) continue;
                    size_t i = (size_t)r * W + c;
                    if (d > dep[i]) {
                        dep[i] = (float)d;
                        if (im) {
                            double u = l0 * Q[0][3] + l1 * Q[1][3] + l2 * Q[2][3], v = l0 * Q[0][4] + l1 * Q[1][4] + l2 * Q[2][4];
                            u -= std::floor(u); v -= std::floor(v);
                            int px = std::min(im->w - 1, std::max(0, (int)(u * im->w))), py = std::min(im->h - 1, std::max(0, (int)((1 - v) * im->h)));
                            memcpy(&img[i * 3], &im->px[((size_t)py * im->w + px) * 3], 3);
                        } else { img[i * 3] = img[i * 3 + 1] = img[i * 3 + 2] = 128; }
                    }
                }
            }
        }
        for (size_t i = 0; i < (size_t)W * H; i++) if (std::isinf(dep[i])) dep[i] = NAN;
    }
    if (stats) {
        u64 d = 0; for (int f = 0; f < nf; f++) d += drawn[f];
        stats[0] = (double)d; stats[1] = nf; stats[2] = t1 - t0; stats[3] = omp_get_wtime() - t1;
    }
    return 0;
}

// Connected components (two-pass union-find). Returns the number of components; lab gets 1..n (0 = background),
// numbered in raster order of each component's first cell.
int sr_label(const uint8_t *mask, int W, int H, int conn8, int32_t *lab) {
    size_t N = (size_t)W * H; std::vector<int32_t> parent; parent.reserve(1024); parent.push_back(0);
    auto find = [&](int32_t a) { while (parent[a] != a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
    auto unite = [&](int32_t a, int32_t b) { a = find(a); b = find(b); if (a < b) parent[b] = a; else if (b < a) parent[a] = b; };
    for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) {
        size_t i = (size_t)r * W + c; lab[i] = 0; if (!mask[i]) continue;
        int32_t best = 0; int32_t nb[4]; int k = 0;
        if (c > 0 && lab[i - 1]) nb[k++] = lab[i - 1];
        if (r > 0) {
            if (lab[i - W]) nb[k++] = lab[i - W];
            if (conn8 && c > 0 && lab[i - W - 1]) nb[k++] = lab[i - W - 1];
            if (conn8 && c + 1 < W && lab[i - W + 1]) nb[k++] = lab[i - W + 1];
        }
        if (!k) { best = (int32_t)parent.size(); parent.push_back(best); }
        else { best = nb[0]; for (int j = 1; j < k; j++) { if (nb[j] < best) best = nb[j]; } for (int j = 0; j < k; j++) unite(best, nb[j]); }
        lab[i] = best;
    }
    std::vector<int32_t> remap(parent.size(), 0); int32_t n = 0;
    for (size_t i = 0; i < N; i++) if (lab[i]) {
        int32_t root = find(lab[i]); if (!remap[root]) remap[root] = ++n; lab[i] = remap[root];
    }
    return n;
}

// Nearest set cell (flat index, -1 when the mask is empty) by two raster passes propagating the nearest seed.
void sr_nearest(const uint8_t *seed, int W, int H, int32_t *idx) {
    size_t N = (size_t)W * H; std::vector<float> d(N, INFINITY);
    for (size_t i = 0; i < N; i++) { idx[i] = -1; if (seed[i]) { idx[i] = (int32_t)i; d[i] = 0; } }
    auto relax = [&](int r, int c, int rr, int cc) {
        if (rr < 0 || rr >= H || cc < 0 || cc >= W) return;
        int32_t s = idx[(size_t)rr * W + cc]; if (s < 0) return;
        float dr = (float)(r - s / W), dc = (float)(c - s % W), q = dr * dr + dc * dc; size_t i = (size_t)r * W + c;
        if (q < d[i]) { d[i] = q; idx[i] = s; }
    };
    for (int pass = 0; pass < 2; pass++) {
        for (int r = 0; r < H; r++) for (int c = 0; c < W; c++) { relax(r, c, r - 1, c - 1); relax(r, c, r - 1, c); relax(r, c, r - 1, c + 1); relax(r, c, r, c - 1); }
        for (int r = H - 1; r >= 0; r--) for (int c = W - 1; c >= 0; c--) { relax(r, c, r + 1, c + 1); relax(r, c, r + 1, c); relax(r, c, r + 1, c - 1); relax(r, c, r, c + 1); }
    }
}

// Outer boundary of component `label` starting at its first cell in raster order (r0, c0). Writes x,y pairs
// (column, row) of boundary cells in order; returns the count (capped at cap).
int sr_contour(const int32_t *lab, int W, int H, int32_t label, int r0, int c0, int32_t *out, int cap) {
    static const int dr[8] = {0, 1, 1, 1, 0, -1, -1, -1}, dc[8] = {1, 1, 0, -1, -1, -1, 0, 1};  // E SE S SW W NW N NE
    auto at = [&](int r, int c) { return r >= 0 && r < H && c >= 0 && c < W && lab[(size_t)r * W + c] == label; };
    int n = 0, r = r0, c = c0, dir = 7;  // came from the west/north: start searching from NE
    if (cap > 0) { out[0] = c; out[1] = r; n = 1; }
    long guard = 4L * W * H + 16;
    while (guard-- > 0) {
        int found = -1;
        for (int k = 0; k < 8; k++) { int d = (dir + 6 + k) % 8; if (at(r + dr[d], c + dc[d])) { found = d; break; } }
        if (found < 0) break;  // single cell
        r += dr[found]; c += dc[found]; dir = found;
        if (r == r0 && c == c0) break;
        if (n < cap) { out[n * 2] = c; out[n * 2 + 1] = r; n++; } else break;
    }
    return n;
}

// Even-odd fill of a polygon (pixel coordinates, cell centres at integer positions) with value.
void sr_fill_poly(int32_t *img, int W, int H, const double *px, const double *py, int n, int32_t value) {
    if (n < 3) return;
    double ylo = py[0], yhi = py[0]; for (int i = 1; i < n; i++) { ylo = std::min(ylo, py[i]); yhi = std::max(yhi, py[i]); }
    int r0 = std::max(0, (int)std::ceil(ylo)), r1 = std::min(H - 1, (int)std::floor(yhi));
    std::vector<double> xs;
    for (int r = r0; r <= r1; r++) {
        xs.clear();
        for (int i = 0, j = n - 1; i < n; j = i++) {
            double ya = py[j], yb = py[i];
            if ((ya <= r && yb > r) || (yb <= r && ya > r)) xs.push_back(px[j] + (r - ya) * (px[i] - px[j]) / (yb - ya));
        }
        std::sort(xs.begin(), xs.end());
        for (size_t k = 0; k + 1 < xs.size(); k += 2) {
            int a = std::max(0, (int)std::ceil(xs[k])), b = std::min(W - 1, (int)std::floor(xs[k + 1]));
            for (int c = a; c <= b; c++) img[(size_t)r * W + c] = value;
        }
    }
}

// Perspective views (pinhole cameras) of the textured mesh, for street-level detection.
// C holds 8 doubles per camera: east, south, up of the eye; yaw (degrees, 0 = east, 90 = south); pitch (degrees,
// negative = down); horizontal field of view (degrees); near, far (metres along the view axis). All cameras share
// W x H. rgb (may be NULL): 3 bytes per pixel per camera; depth: view-axis distance (NAN = nothing drawn), i.e. the
// first surface hit by each pixel's ray. Triangles crossing the near plane are clipped. skip (may be NULL): per mesh
// 1 = ignore (coarser level of detail). stats: [triangles drawn, cameras, texture decode s, render s].
int sr_views(const char *idx_dir, const int *axsrc, const float *axsign, int nc, const double *C, int W, int H,
             const uint8_t *skip, uint8_t *rgb, float *depth, double *stats, char *err, int ecap) {
    std::string dir = idx_dir, why;
    std::vector<std::string> tex; std::vector<Mesh> M;
    if (!load_index(dir, tex, M, why)) { snprintf(err, ecap, "%s", why.c_str()); return 1; }
    if (W <= 0 || H <= 0 || nc < 0) { snprintf(err, ecap, "bad view size %dx%d", W, H); return 1; }
    Map vm, tm;
    if (!vm.open_(dir + "/verts.bin") || !tm.open_(dir + "/tris.bin")) { snprintf(err, ecap, "cannot map verts.bin/tris.bin in %s", idx_dir); return 1; }
    const float *VB = (const float *)vm.p; const u32 *TB = (const u32 *)tm.p;
    u64 nverts = vm.n / 20, nidx = tm.n / 4;
    const int ex = axsrc[0], sy = axsrc[1], up = axsrc[2]; const float se = axsign[0], ss_ = axsign[1], su = axsign[2];
    std::vector<float> mb(M.size() * 6);
    for (size_t i = 0; i < M.size(); i++) {
        const Mesh &m = M[i];
        if (m.v0 + m.nv > nverts || m.i0 + m.ni > nidx) { snprintf(err, ecap, "index out of range in mesh %zu (stale index?)", i); return 1; }
        for (int k = 0; k < 3; k++) {
            float s = axsign[k]; int a = axsrc[k]; float p = s * m.lo[a], q = s * m.hi[a];
            mb[i * 6 + k] = std::min(p, q); mb[i * 6 + 3 + k] = std::max(p, q);
        }
    }
    // meshes near each camera (box within far of the eye)
    std::vector<std::vector<u32>> todo(nc); std::vector<char> need(tex.size(), 0);
    for (int c = 0; c < nc; c++) {
        const double *P = C + (size_t)c * 8; double far_ = P[7];
        for (size_t i = 0; i < M.size(); i++) {
            if (skip && skip[i]) continue;
            const float *b = &mb[i * 6]; double d2 = 0;
            for (int k = 0; k < 3; k++) { double v = P[k] < b[k] ? b[k] - P[k] : (P[k] > b[3 + k] ? P[k] - b[3 + k] : 0); d2 += v * v; }
            if (d2 <= far_ * far_) { todo[c].push_back((u32)i); if (rgb && M[i].tex < tex.size()) need[M[i].tex] = 1; }
        }
    }
    std::vector<Img> T(tex.size());
    double t0 = omp_get_wtime();
    if (rgb) { std::vector<u32> list; for (size_t i = 0; i < need.size(); i++) if (need[i]) list.push_back((u32)i);
#pragma omp parallel
      { tjhandle tj = tjInitDecompress();
#pragma omp for schedule(dynamic, 4)
        for (size_t j = 0; j < list.size(); j++) {
            u32 i = list[j];
            std::ifstream g(tex[i], std::ios::binary); std::vector<unsigned char> b((std::istreambuf_iterator<char>(g)), {});
            int w, h, ss, cs; if (b.empty() || tjDecompressHeader3(tj, b.data(), b.size(), &w, &h, &ss, &cs)) continue;
            T[i].px.resize((size_t)w * h * 3);
            if (tjDecompress2(tj, b.data(), b.size(), T[i].px.data(), w, 0, h, TJPF_RGB, 0)) { T[i].px.clear(); continue; }
            T[i].w = w; T[i].h = h;
        }
        tjDestroy(tj); } }
    double t1 = omp_get_wtime();
    std::vector<u64> drawn(nc, 0);
    const size_t npx = (size_t)W * H;
#pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < nc; c++) {
        const double *P = C + (size_t)c * 8;
        const double ce = P[0], cs = P[1], cu = P[2], yaw = P[3] * M_PI / 180, pit = P[4] * M_PI / 180;
        const double fx = (W / 2.0) / std::tan(P[5] * M_PI / 360), nearz = P[6], farz = P[7];
        const double f[3] = {std::cos(pit) * std::cos(yaw), std::cos(pit) * std::sin(yaw), std::sin(pit)};
        const double r[3] = {-std::sin(yaw), std::cos(yaw), 0};
        const double u[3] = {-std::sin(pit) * std::cos(yaw), -std::sin(pit) * std::sin(yaw), std::cos(pit)};
        float *dep = depth + npx * c; uint8_t *img = rgb ? rgb + npx * 3 * c : nullptr;
        for (size_t i = 0; i < npx; i++) dep[i] = INFINITY;
        if (img) memset(img, 0, npx * 3);
        for (u32 mi : todo[c]) {
            const Mesh &m = M[mi];
            const Img *im = (img && m.tex < T.size() && T[m.tex].w) ? &T[m.tex] : nullptr;
            for (u64 t = 0; t + 2 < m.ni; t += 3) {
                // view-space vertices: x right, y up, z forward; plus uv
                double V[3][5]; int behind = 0, beyond = 0;
                for (int k = 0; k < 3; k++) {
                    const float *Q = VB + (m.v0 + TB[m.i0 + t + k]) * 5;
                    double d0 = se * Q[ex] - ce, d1 = ss_ * Q[sy] - cs, d2 = su * Q[up] - cu;
                    V[k][0] = d0 * r[0] + d1 * r[1] + d2 * r[2];
                    V[k][1] = d0 * u[0] + d1 * u[1] + d2 * u[2];
                    V[k][2] = d0 * f[0] + d1 * f[1] + d2 * f[2];
                    V[k][3] = Q[3]; V[k][4] = Q[4];
                    behind += V[k][2] < nearz; beyond += V[k][2] > farz;
                }
                if (behind == 3 || beyond == 3) continue;
                // clip against the near plane (Sutherland-Hodgman, one plane): up to 4 vertices
                double Pg[4][5]; int np_ = 0;
                for (int k = 0; k < 3; k++) {
                    const double *A = V[k], *B = V[(k + 1) % 3];
                    bool ain = A[2] >= nearz, bin = B[2] >= nearz;
                    if (ain) { memcpy(Pg[np_++], A, sizeof(double) * 5); }
                    if (ain != bin) { double s = (nearz - A[2]) / (B[2] - A[2]); for (int j = 0; j < 5; j++) Pg[np_][j] = A[j] + s * (B[j] - A[j]); np_++; }
                }
                if (np_ < 3) continue;
                // screen positions, 1/z
                double X[4], Y[4], IZ[4];
                for (int k = 0; k < np_; k++) { IZ[k] = 1.0 / Pg[k][2]; X[k] = W / 2.0 + fx * Pg[k][0] * IZ[k] - .5; Y[k] = H / 2.0 - fx * Pg[k][1] * IZ[k] - .5; }
                for (int k = 1; k + 1 < np_; k++) {
                    int a = 0, b = k, cc = k + 1;
                    double den = (X[b] - X[a]) * (Y[cc] - Y[a]) - (X[cc] - X[a]) * (Y[b] - Y[a]);
                    if (std::fabs(den) < 1e-12) continue;
                    int c0 = std::max(0, (int)std::ceil(std::min({X[a], X[b], X[cc]}))), c1 = std::min(W - 1, (int)std::floor(std::max({X[a], X[b], X[cc]})));
                    int r0 = std::max(0, (int)std::ceil(std::min({Y[a], Y[b], Y[cc]}))), r1 = std::min(H - 1, (int)std::floor(std::max({Y[a], Y[b], Y[cc]})));
                    if (c0 > c1 || r0 > r1) continue;
                    drawn[c]++;
                    for (int y = r0; y <= r1; y++) for (int x = c0; x <= c1; x++) {
                        double l1 = ((x - X[a]) * (Y[cc] - Y[a]) - (X[cc] - X[a]) * (y - Y[a])) / den;
                        double l2 = ((X[b] - X[a]) * (y - Y[a]) - (x - X[a]) * (Y[b] - Y[a])) / den;
                        double l0 = 1 - l1 - l2; const double e = -1e-6;
                        if (l0 < e || l1 < e || l2 < e) continue;
                        double iz = l0 * IZ[a] + l1 * IZ[b] + l2 * IZ[cc];
                        if (iz <= 0) continue;
                        double z = 1.0 / iz; if (z > farz) continue;
                        size_t i = (size_t)y * W + x;
                        if (z < dep[i]) {
                            dep[i] = (float)z;
                            if (im) {
                                double uu = (l0 * Pg[a][3] * IZ[a] + l1 * Pg[b][3] * IZ[b] + l2 * Pg[cc][3] * IZ[cc]) * z;
                                double vv = (l0 * Pg[a][4] * IZ[a] + l1 * Pg[b][4] * IZ[b] + l2 * Pg[cc][4] * IZ[cc]) * z;
                                uu -= std::floor(uu); vv -= std::floor(vv);
                                int px = std::min(im->w - 1, std::max(0, (int)(uu * im->w))), py = std::min(im->h - 1, std::max(0, (int)((1 - vv) * im->h)));
                                memcpy(&img[i * 3], &im->px[((size_t)py * im->w + px) * 3], 3);
                            } else if (img) { img[i * 3] = img[i * 3 + 1] = img[i * 3 + 2] = 128; }
                        }
                    }
                }
            }
        }
        for (size_t i = 0; i < npx; i++) if (std::isinf(dep[i])) dep[i] = NAN;
    }
    if (stats) { u64 d = 0; for (int c = 0; c < nc; c++) d += drawn[c]; stats[0] = (double)d; stats[1] = nc; stats[2] = t1 - t0; stats[3] = omp_get_wtime() - t1; }
    return 0;
}

// TIFF LZW decoder (MSB-first codes of 9..12 bits, Clear 256, EOI 257, early change). Returns the number of
// bytes written to out (at most cap), or -1 for a corrupt stream. Used by the GeoTIFF reader of the data providers.
int64_t sr_lzw(const uint8_t *in, int64_t n, uint8_t *out, int64_t cap) {
    std::vector<int32_t> prefix(4096), first(4096), len(4096); std::vector<uint8_t> last(4096);
    for (int i = 0; i < 256; i++) { prefix[i] = -1; first[i] = i; last[i] = (uint8_t)i; len[i] = 1; }
    int width = 9, next = 258; int32_t prev = -1; int64_t pos = 0; uint64_t bitpos = 0, total = (uint64_t)n * 8;
    std::vector<uint8_t> stack(4096);
    while (bitpos + width <= total) {
        uint32_t code = 0;
        for (int b = 0; b < width; b++, bitpos++) code = (code << 1) | ((in[bitpos >> 3] >> (7 - (bitpos & 7))) & 1);
        if (code == 257) break;
        if (code == 256) { width = 9; next = 258; prev = -1; continue; }
        int32_t c = (int32_t)code; uint8_t f;
        if (c < next) f = (uint8_t)first[c];
        else if (c == next && prev >= 0) f = (uint8_t)first[prev];
        else return -1;
        if (prev >= 0 && next < 4096) { prefix[next] = prev; first[next] = first[prev]; last[next] = c < next ? (uint8_t)first[c] : f;
                                        len[next] = len[prev] + 1; next++; }
        int32_t l = len[c]; if (pos + l > cap) return -1;
        int32_t k = c; for (int32_t i = l - 1; i >= 0; i--) { out[pos + i] = last[k]; k = prefix[k]; }
        pos += l; prev = c;
        if (next + 1 >= (1 << width) && width < 12) width++;
    }
    (void)stack;
    return pos;
}

}  // extern "C"
