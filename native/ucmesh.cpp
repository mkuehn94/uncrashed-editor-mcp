// ucmesh: host-side OBJ+MTL preprocessor for the agent's in-memory textured reference.
//
//   ucmesh index OBJ CACHE_DIR            parse once (v, vt, f, o/g, usemtl, mtllib, map_Kd) into a binary cache
//   ucmesh stats CACHE_DIR                level/coverage summary of an index
//   ucmesh pack  CACHE_DIR OUT_DIR [opts] select, de-duplicate detail levels, atlas textures, write UCMESH2 chunks
//
// Coordinates. The cache keeps source coordinates. pack converts to Unreal (centimetres, X forward/east,
// Y right/south, Z up) with a *proper* change of handedness: a right-handed source seen in the left-handed
// engine keeps its real-world layout (no mirror image) and front-face winding needs no reversal.
//
// Build: g++ -O2 -std=c++17 native/ucmesh.cpp -lturbojpeg -o native/ucmesh
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <turbojpeg.h>
#include <unistd.h>

namespace fs = std::filesystem;
using u32 = uint32_t; using u64 = uint64_t;

[[noreturn]] static void die(const std::string &m) { throw std::runtime_error(m); }

// ---------------------------------------------------------------- index format
// index.bin  : "UCMIDX1\0" u32 ntex u32 nmesh, ntex x (u32 len, bytes), nmesh x MeshRec (+ name)
// verts.bin  : float[5] (x y z u v) per unique (position, uv) pair, mesh-contiguous, source axes
// tris.bin   : u32 per corner, local to the owning mesh
struct Mesh {
    std::string name, tile; u32 level = 0, tex = 0; u64 v0 = 0, nv = 0, i0 = 0, ni = 0;
    float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
};
struct Index { std::vector<std::string> tex; std::vector<Mesh> mesh; };

static void wr32(std::ofstream &f, u32 v) { f.write((const char *)&v, 4); }
static void wr64(std::ofstream &f, u64 v) { f.write((const char *)&v, 8); }
static void wrs(std::ofstream &f, const std::string &s) { wr32(f, (u32)s.size()); f.write(s.data(), s.size()); }

static void save_index(const fs::path &dir, const Index &ix) {
    std::ofstream f(dir / "index.bin", std::ios::binary);
    f.write("UCMIDX1", 8); wr32(f, (u32)ix.tex.size()); wr32(f, (u32)ix.mesh.size());
    for (auto &t : ix.tex) wrs(f, t);
    for (auto &m : ix.mesh) {
        wrs(f, m.name); wr32(f, m.level); wr32(f, m.tex); wr64(f, m.v0); wr64(f, m.nv); wr64(f, m.i0); wr64(f, m.ni);
        f.write((const char *)m.lo, 12); f.write((const char *)m.hi, 12);
    }
    if (!f) die("cannot write index.bin");
}

static Index load_index(const fs::path &dir) {
    std::ifstream f(dir / "index.bin", std::ios::binary); char magic[8]; u32 nt, nm; Index ix;
    if (!f.read(magic, 8) || memcmp(magic, "UCMIDX1", 8)) die("not a ucmesh index: " + (dir / "index.bin").string());
    auto r32 = [&] { u32 v; f.read((char *)&v, 4); return v; };
    auto r64 = [&] { u64 v; f.read((char *)&v, 8); return v; };
    auto rs = [&] { std::string s(r32(), '\0'); f.read(s.data(), s.size()); return s; };
    nt = r32(); nm = r32();
    for (u32 i = 0; i < nt; i++) ix.tex.push_back(rs());
    for (u32 i = 0; i < nm; i++) {
        Mesh m; m.name = rs(); m.level = r32(); m.tex = r32(); m.v0 = r64(); m.nv = r64(); m.i0 = r64(); m.ni = r64();
        f.read((char *)m.lo, 12); f.read((char *)m.hi, 12);
        m.tile = m.name.substr(0, m.name.find('_'));
        ix.mesh.push_back(std::move(m));
    }
    if (!f) die("truncated index.bin");
    return ix;
}

// ---------------------------------------------------------------- OBJ parsing
static std::string trim(std::string s) {
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    size_t i = 0; while (i < s.size() && isspace((unsigned char)s[i])) i++;
    return s.substr(i);
}

static std::string strip_comment(const std::string &s) { return trim(s.substr(0, s.find('#'))); }

static std::string unquote(std::string s) {
    s = trim(s);
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') s = s.substr(1, s.size() - 2);
    return s;
}

// map_Kd may carry options (-s, -o, -bm ...). Accept the plain file form; refuse options instead of misreading them.
static void read_mtl(const fs::path &p, std::map<std::string, std::string> &out) {
    std::ifstream f(p); std::string line, name;
    if (!f) die("cannot read material library " + p.string());
    while (std::getline(f, line)) {
        line = trim(line);
        if (line.rfind("newmtl", 0) == 0 && line.size() > 6 && isspace((unsigned char)line[6])) name = strip_comment(line.substr(7));
        else if (line.rfind("map_Kd", 0) == 0 && line.size() > 6 && isspace((unsigned char)line[6]) && !name.empty()) {
            std::string v = unquote(strip_comment(line.substr(7)));
            if (!v.empty() && v[0] == '-') die("map_Kd options are not supported (" + name + ")");
            out[name] = fs::weakly_canonical(p.parent_path() / v).string();
        }
    }
}

static inline const char *skipws(const char *p) { while (*p == ' ' || *p == '\t') p++; return p; }

static long parse_index(const char *&p, size_t n) {
    char *e; long v = strtol(p, &e, 10);
    if (e == p) die("bad face index");
    p = e;
    long r = v > 0 ? v - 1 : (long)n + v;
    if (v == 0 || r < 0 || (size_t)r >= n) die("face index out of range");
    return r;
}

static void cmd_index(const fs::path &obj, const fs::path &dir) {
    fs::create_directories(dir);
    FILE *in = fopen(obj.c_str(), "rb"); if (!in) die("cannot open " + obj.string());
    std::ofstream vf(dir / "verts.bin", std::ios::binary), tf(dir / "tris.bin", std::ios::binary);
    std::vector<float> V, T; V.reserve(1 << 24); T.reserve(1 << 24);
    std::map<std::string, std::string> mtl; std::map<std::string, u32> texid; Index ix;
    std::string object = "mesh", material; bool open = false;
    Mesh cur; std::unordered_map<u64, u32> remap; std::vector<float> outv; std::vector<u32> outi;
    u64 vtotal = 0, itotal = 0, faces = 0, skipped_untextured = 0, polys = 0;
    auto flush = [&] {
        if (open && !outi.empty()) {
            cur.v0 = vtotal; cur.nv = outv.size() / 5; cur.i0 = itotal; cur.ni = outi.size();
            vf.write((const char *)outv.data(), outv.size() * 4); tf.write((const char *)outi.data(), outi.size() * 4);
            vtotal += cur.nv; itotal += cur.ni; ix.mesh.push_back(cur);
        }
        open = false; outv.clear(); outi.clear(); remap.clear();
    };
    auto begin = [&] {
        flush();
        auto it = mtl.find(material);
        if (it == mtl.end()) return;  // untextured geometry: counted, not indexed
        auto t = texid.find(it->second);
        if (t == texid.end()) { t = texid.emplace(it->second, (u32)ix.tex.size()).first; ix.tex.push_back(it->second); }
        cur = Mesh(); cur.name = object; cur.tex = t->second; cur.tile = object.substr(0, object.find('_'));
        cur.level = (u32)cur.tile.size(); open = true;
    };
    char *line = nullptr; size_t cap = 0; ssize_t len;
    while ((len = getline(&line, &cap, in)) > 0) {
        const char *p = skipws(line);
        if (p[0] == 'v' && (p[1] == ' ' || p[1] == '\t')) {
            char *e; float x = strtof(p + 2, &e), y = strtof(e, &e), z = strtof(e, &e);
            V.push_back(x); V.push_back(y); V.push_back(z);
        } else if (p[0] == 'v' && p[1] == 't' && (p[2] == ' ' || p[2] == '\t')) {
            char *e; float u = strtof(p + 3, &e), v = strtof(e, &e); T.push_back(u); T.push_back(v);
        } else if (p[0] == 'f' && (p[1] == ' ' || p[1] == '\t')) {
            faces++;
            if (!open) { skipped_untextured++; continue; }
            u32 corner[64]; int n = 0; p += 2;
            for (;;) {
                p = skipws(p); if (!*p || *p == '\n' || *p == '\r' || *p == '#') break;
                long vi = parse_index(p, V.size() / 3), ti = -1;
                if (*p == '/') { p++; if (*p != '/') ti = parse_index(p, T.size() / 2); if (*p == '/') { p++; strtol(p, (char **)&p, 10); } }
                if (ti < 0) die("face without texture coordinates in textured mesh " + object);
                while (*p && *p != ' ' && *p != '\t' && *p != '\n' && *p != '\r') p++;
                u64 key = ((u64)vi << 32) | (u64)ti; auto it = remap.find(key);
                if (it == remap.end()) {
                    it = remap.emplace(key, (u32)(outv.size() / 5)).first;
                    float *q = &V[vi * 3];
                    for (int k = 0; k < 3; k++) { outv.push_back(q[k]); cur.lo[k] = std::min(cur.lo[k], q[k]); cur.hi[k] = std::max(cur.hi[k], q[k]); }
                    outv.push_back(T[ti * 2]); outv.push_back(T[ti * 2 + 1]);
                }
                if (n == 64) die("face with more than 64 corners");
                corner[n++] = it->second;
            }
            if (n < 3) die("degenerate face in " + object);
            if (n > 3) polys++;
            for (int k = 1; k + 1 < n; k++) { outi.push_back(corner[0]); outi.push_back(corner[k]); outi.push_back(corner[k + 1]); }
        } else if ((p[0] == 'o' || p[0] == 'g') && (p[1] == ' ' || p[1] == '\t')) {
            object = strip_comment(p + 2); if (open) begin();
        } else if (!strncmp(p, "usemtl", 6) && (p[6] == ' ' || p[6] == '\t')) {
            material = strip_comment(p + 7); begin();
        } else if (!strncmp(p, "mtllib", 6) && (p[6] == ' ' || p[6] == '\t')) {
            std::string v = unquote(strip_comment(p + 7)); fs::path full = obj.parent_path() / v;
            if (fs::exists(full)) read_mtl(full, mtl);
            else { std::istringstream ss(v); std::string w; while (ss >> w) read_mtl(obj.parent_path() / w, mtl); }
        }
    }
    flush(); free(line); fclose(in);
    if (!vf || !tf) die("cache write failed");
    save_index(dir, ix);
    printf("indexed meshes=%zu textures=%zu vertices=%llu triangles=%llu source_faces=%llu untextured_faces=%llu polygons_split=%llu\n",
           ix.mesh.size(), ix.tex.size(), (unsigned long long)vtotal, (unsigned long long)(itotal / 3),
           (unsigned long long)faces, (unsigned long long)skipped_untextured, (unsigned long long)polys);
}

// ---------------------------------------------------------------- cache access
struct Map { void *p = MAP_FAILED; size_t n = 0;
    explicit Map(const fs::path &f) { int fd = open(f.c_str(), O_RDONLY); if (fd < 0) die("cannot open " + f.string());
        struct stat st; fstat(fd, &st); n = st.st_size; if (n) p = mmap(nullptr, n, PROT_READ, MAP_PRIVATE, fd, 0); close(fd);
        if (n && p == MAP_FAILED) die("mmap failed " + f.string()); }
    ~Map() { if (n && p != MAP_FAILED) munmap(p, n); } };

// Detail levels. Tile names that are digit strings form an octree (child = parent + one digit). Exports of
// multi-resolution sources contain a parent next to some of its children; both cover the same space. Drop
// parent triangles whose centroid lies inside the subtree bounds of an existing child.
struct Box { float lo[3] = {1e30f, 1e30f, 1e30f}, hi[3] = {-1e30f, -1e30f, -1e30f};
    void add(const float *a, const float *b) { for (int k = 0; k < 3; k++) { lo[k] = std::min(lo[k], a[k]); hi[k] = std::max(hi[k], b[k]); } }
    bool in(const float *p, float e) const { for (int k = 0; k < 3; k++) if (p[k] < lo[k] - e || p[k] > hi[k] + e) return false; return true; } };

static bool octree_names(const Index &ix) {
    for (auto &m : ix.mesh) { if (m.tile.empty()) return false; for (char c : m.tile) if (c < '0' || c > '7') return false; }
    return true;
}

static std::unordered_map<std::string, Box> subtree_boxes(const Index &ix) {
    std::unordered_map<std::string, Box> box;
    for (auto &m : ix.mesh) for (size_t n = 1; n <= m.tile.size(); n++) box[m.tile.substr(0, n)].add(m.lo, m.hi);
    return box;
}

static void cmd_stats(const fs::path &dir) {
    Index ix = load_index(dir); std::map<u32, std::array<u64, 3>> lv; bool tree = octree_names(ix);
    auto box = subtree_boxes(ix); std::set<std::string> tiles; for (auto &m : ix.mesh) tiles.insert(m.tile);
    for (auto &m : ix.mesh) {
        auto &a = lv[m.level]; a[0]++; a[1] += m.ni / 3;
        bool child = false;
        for (char d = '0'; d <= '7' && tree; d++) if (box.count(m.tile + d)) child = true;
        if (child) a[2]++;
    }
    printf("meshes=%zu textures=%zu octree_names=%d\n", ix.mesh.size(), ix.tex.size(), tree);
    for (auto &[l, a] : lv) printf("level %u meshes=%llu triangles=%llu with_finer_children=%llu\n", l,
                                    (unsigned long long)a[0], (unsigned long long)a[1], (unsigned long long)a[2]);
}

// ---------------------------------------------------------------- axes
// Each output axis (east, south, up) is +/- one source axis. Default matches "x = east, y = up, z = south".
struct Axes { int src[3]; float sign[3]; };
static Axes parse_axes(const std::string &s) {
    // three comma-separated words naming what source x, y, z point to: east west north south up down
    Axes a{{-1, -1, -1}, {1, 1, 1}}; std::stringstream ss(s); std::string w; int i = 0;
    while (std::getline(ss, w, ',')) {
        if (i > 2) die("--axes takes three words");
        int out; float sg;
        if (w == "east") out = 0, sg = 1; else if (w == "west") out = 0, sg = -1;
        else if (w == "south") out = 1, sg = 1; else if (w == "north") out = 1, sg = -1;
        else if (w == "up") out = 2, sg = 1; else if (w == "down") out = 2, sg = -1;
        else die("unknown axis word " + w);
        if (a.src[out] >= 0) die("--axes names one direction twice");
        a.src[out] = i; a.sign[out] = sg; i++;
    }
    if (i != 3) die("--axes takes three words");
    return a;
}
// source point -> (east, south, up) metres. east/south/up is a left-handed triple, exactly Unreal's X, Y, Z.
static inline void to_world(const Axes &a, const float *p, double *o) { for (int k = 0; k < 3; k++) o[k] = a.sign[k] * p[a.src[k]]; }
// For a right-handed source (every OBJ convention named by --axes), this mapping has determinant -1 in numbers but
// preserves the picture. Unreal's front faces are clockwise as seen by the viewer; right-handed CCW triangles thus
// need no reversal. (Verified live: the opposite winding renders back faces.)

// ---------------------------------------------------------------- textures
struct Image { int w = 0, h = 0; std::vector<unsigned char> px; };
static tjhandle TJD = nullptr, TJC = nullptr;

static Image load_jpeg_or_die(const std::string &path) {
    std::ifstream f(path, std::ios::binary); std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)), {});
    if (buf.empty()) die("cannot read texture " + path);
    int w, h, ss, cs; Image im;
    if (tjDecompressHeader3(TJD, buf.data(), buf.size(), &w, &h, &ss, &cs)) die("not a JPEG texture (PNG sources: convert first): " + path);
    im.w = w; im.h = h; im.px.resize((size_t)w * h * 3);
    if (tjDecompress2(TJD, buf.data(), buf.size(), im.px.data(), w, 0, h, TJPF_RGB, TJFLAG_ACCURATEDCT)) die("JPEG decode failed: " + path);
    return im;
}

static Image downscale(const Image &a, int f) {
    if (f <= 1) return a;
    Image b; b.w = std::max(1, a.w / f); b.h = std::max(1, a.h / f); b.px.resize((size_t)b.w * b.h * 3);
    for (int y = 0; y < b.h; y++) for (int x = 0; x < b.w; x++) for (int c = 0; c < 3; c++) {
        int s = 0, n = 0;
        for (int dy = 0; dy < f; dy++) for (int dx = 0; dx < f; dx++) {
            int sx = std::min(a.w - 1, x * f + dx), sy = std::min(a.h - 1, y * f + dy); s += a.px[((size_t)sy * a.w + sx) * 3 + c]; n++; }
        b.px[((size_t)y * b.w + x) * 3 + c] = (unsigned char)(s / n);
    }
    return b;
}

struct Slot { int page, x, y, w, h; };

// ---------------------------------------------------------------- pack
struct Opts {
    double cx = 0, cy = 0, radius = 0; bool all = false;          // region in east/south metres (source origin)
    double datum = 0;                                             // source height mapped to output z = 0
    int page = 4096, pad = 4, quality = 90, far_scale = 2;        // atlas page size, gutter, JPEG quality
    double near_m = 160, chunk_m = 96;                            // full-resolution radius, chunk grid
    bool keep_overlap = false; Axes axes = parse_axes("east,up,south");
    u64 max_tris = 12000000; double max_mpx = 900;                 // refusal limits (whole pack)
};

static std::string winpath(const fs::path &p) {
    std::string s = "Z:" + fs::absolute(p).string(); for (char &c : s) if (c == '/') c = '\\'; return s;
}

static void cmd_pack(const fs::path &dir, const fs::path &out, const Opts &o) {
    Index ix = load_index(dir); Map vm(dir / "verts.bin"), tm(dir / "tris.bin");
    const float *VB = (const float *)vm.p; const u32 *IB = (const u32 *)tm.p;
    bool tree = octree_names(ix) && !o.keep_overlap; auto box = subtree_boxes(ix);
    std::unordered_map<std::string, bool> has; for (auto &m : ix.mesh) has[m.tile] = true;
    fs::create_directories(out);
    // 1. per-mesh triangle selection
    struct Sel { u32 mesh; std::vector<u32> tris; double c[2]; };
    std::vector<Sel> sel; u64 culled = 0, outside = 0, kept = 0;
    for (u32 mi = 0; mi < ix.mesh.size(); mi++) {
        const Mesh &m = ix.mesh[mi];
        double lo[3], hi[3]; to_world(o.axes, m.lo, lo); to_world(o.axes, m.hi, hi);
        double bx0 = std::min(lo[0], hi[0]), bx1 = std::max(lo[0], hi[0]), by0 = std::min(lo[1], hi[1]), by1 = std::max(lo[1], hi[1]);
        if (!o.all) { double dx = std::max({bx0 - o.cx, 0.0, o.cx - bx1}), dy = std::max({by0 - o.cy, 0.0, o.cy - by1});
            if (dx * dx + dy * dy > o.radius * o.radius) { outside += m.ni / 3; continue; } }
        std::vector<const Box *> kids;
        if (tree) for (char d = '0'; d <= '7'; d++) { auto it = box.find(m.tile + d); if (it != box.end()) kids.push_back(&it->second); }
        Sel s{mi, {}, {(bx0 + bx1) / 2, (by0 + by1) / 2}};
        for (u64 t = 0; t < m.ni / 3; t++) {
            float c[3] = {0, 0, 0};
            for (int k = 0; k < 3; k++) { const float *v = VB + (m.v0 + IB[m.i0 + t * 3 + k]) * 5; for (int j = 0; j < 3; j++) c[j] += v[j] / 3; }
            bool covered = false; for (auto *b : kids) if (b->in(c, .05f)) { covered = true; break; }
            if (covered) { culled++; continue; }
            if (!o.all) { double w[3]; to_world(o.axes, c, w); double dx = w[0] - o.cx, dy = w[1] - o.cy;
                if (dx * dx + dy * dy > o.radius * o.radius) { outside++; continue; } }
            s.tris.push_back((u32)t); kept++;
        }
        if (!s.tris.empty()) sel.push_back(std::move(s));
    }
    if (!kept) die("nothing selected (check --center/--radius and --axes)");
    if (kept > o.max_tris) die("selection has " + std::to_string(kept) + " triangles, over --max-tris; shrink --radius");
    // 2. chunks: a mesh (and its texture) belongs to the chunk holding its bounds centre
    std::map<std::pair<long, long>, std::vector<u32>> chunks;
    for (u32 i = 0; i < sel.size(); i++) chunks[{(long)std::floor(sel[i].c[0] / o.chunk_m), (long)std::floor(sel[i].c[1] / o.chunk_m)}].push_back(i);
    // pixel budget, before any decode
    double mpx = 0;
    for (auto &[k, list] : chunks) {
        double ccx = (k.first + .5) * o.chunk_m - o.cx, ccy = (k.second + .5) * o.chunk_m - o.cy;
        int f = std::hypot(ccx, ccy) <= o.near_m ? 1 : o.far_scale; std::set<u32> t;
        for (u32 i : list) t.insert(ix.mesh[sel[i].mesh].tex);
        for (u32 ti : t) { int w, h, ss, cs; std::ifstream fi(ix.tex[ti], std::ios::binary);
            std::vector<unsigned char> b((std::istreambuf_iterator<char>(fi)), {});
            if (b.empty() || tjDecompressHeader3(TJD, b.data(), b.size(), &w, &h, &ss, &cs)) die("unreadable texture " + ix.tex[ti]);
            mpx += (double)(w / f) * (h / f) / 1e6; }
    }
    if (mpx > o.max_mpx) die("textures need " + std::to_string((int)mpx) + " Mpx, over --max-mpx; raise --far-scale or shrink --radius");
    // 3. per chunk: atlas, sections, payload
    std::ofstream man(out / "manifest.json"); man << "{\n \"format\": \"UCMESH2\",\n \"chunks\": [\n";
    u64 tot_tris = 0, tot_verts = 0; int nchunk = 0; double tot_px = 0;
    for (auto &[key, list] : chunks) {
        double ccx = (key.first + .5) * o.chunk_m - o.cx, ccy = (key.second + .5) * o.chunk_m - o.cy;
        int f = std::hypot(ccx, ccy) <= o.near_m ? 1 : o.far_scale;
        std::map<u32, Image> img; for (u32 i : list) { u32 t = ix.mesh[sel[i].mesh].tex; if (!img.count(t)) img[t] = downscale(load_jpeg_or_die(ix.tex[t]), f); }
        // shelf packing, tallest first
        std::vector<u32> order; for (auto &[t, im] : img) order.push_back(t);
        std::sort(order.begin(), order.end(), [&](u32 a, u32 b) { return img[a].h != img[b].h ? img[a].h > img[b].h : a < b; });
        std::map<u32, Slot> slot; std::vector<int> page_h; int x = 0, y = 0, shelf = 0, page = -1;
        for (u32 t : order) {
            int w = img[t].w + 2 * o.pad, h = img[t].h + 2 * o.pad;
            if (w > o.page || h > o.page) die("texture larger than --page: " + ix.tex[t]);
            if (page < 0 || x + w > o.page) { y += shelf; x = 0; shelf = 0; }
            if (page < 0 || y + h > o.page) { page++; page_h.push_back(0); x = y = shelf = 0; }
            slot[t] = {page, x + o.pad, y + o.pad, img[t].w, img[t].h};
            x += w; shelf = std::max(shelf, h); page_h[page] = std::max(page_h[page], y + shelf);
        }
        std::vector<std::string> pages;
        for (int p = 0; p < (int)page_h.size(); p++) {
            int H = (page_h[p] + 3) & ~3, W = o.page; std::vector<unsigned char> px((size_t)W * H * 3, 0);
            for (auto &[t, s] : slot) if (s.page == p) {
                const Image &im = img[t];
                for (int yy = -o.pad; yy < im.h + o.pad; yy++) for (int xx = -o.pad; xx < im.w + o.pad; xx++) {
                    int sx = std::clamp(xx, 0, im.w - 1), sy = std::clamp(yy, 0, im.h - 1);
                    memcpy(&px[((size_t)(s.y + yy) * W + s.x + xx) * 3], &im.px[((size_t)sy * im.w + sx) * 3], 3);
                }
            }
            unsigned char *jpg = nullptr; unsigned long jn = 0;
            if (tjCompress2(TJC, px.data(), W, 0, H, TJPF_RGB, &jpg, &jn, TJSAMP_444, o.quality, TJFLAG_ACCURATEDCT)) die("JPEG encode failed");
            char name[96]; snprintf(name, sizeof name, "chunk_%+04ld_%+04ld_p%d.jpg", key.first, key.second, p);
            std::ofstream jf(out / name, std::ios::binary); jf.write((const char *)jpg, jn); tjFree(jpg);
            if (!jf) die("cannot write atlas");
            pages.push_back(name); tot_px += (double)W * H;
            for (auto &[t, s] : slot) if (s.page == p) { (void)s; }
            page_h[p] = H;
        }
        // sections: one per atlas page
        char pname[96]; snprintf(pname, sizeof pname, "chunk_%+04ld_%+04ld.ucmesh", key.first, key.second);
        std::ofstream pf(out / pname, std::ios::binary); pf.write("UCMESH2", 8);
        wr32(pf, (u32)pages.size()); wr32(pf, (u32)pages.size());
        for (auto &pg : pages) wrs(pf, winpath(out / pg));
        u64 ctris = 0, cverts = 0; double lo[3] = {1e30, 1e30, 1e30}, hi[3] = {-1e30, -1e30, -1e30}; u64 uv_clamped = 0;
        for (int p = 0; p < (int)pages.size(); p++) {
            std::vector<float> xyz, uv; std::vector<int32_t> idx;
            for (u32 i : list) {
                const Mesh &m = ix.mesh[sel[i].mesh]; const Slot &s = slot[m.tex]; if (s.page != p) continue;
                std::vector<int32_t> local(m.nv, -1);
                for (u32 t : sel[i].tris) for (int k = 0; k < 3; k++) {
                    u32 li = IB[m.i0 + (u64)t * 3 + k]; int32_t &dst = local[li];
                    if (dst < 0) {
                        const float *v = VB + (m.v0 + li) * 5; double w[3]; to_world(o.axes, v, w); w[2] -= o.datum;
                        double ox[3] = {(w[0] - o.cx) * 100, (w[1] - o.cy) * 100, w[2] * 100};
                        for (int j = 0; j < 3; j++) { xyz.push_back((float)ox[j]); lo[j] = std::min(lo[j], ox[j]); hi[j] = std::max(hi[j], ox[j]); }
                        float uu = v[3], vv = v[4];
                        if (uu < -.001f || uu > 1.001f || vv < -.001f || vv > 1.001f) uv_clamped++;
                        uu = std::clamp(uu, 0.f, 1.f); vv = std::clamp(vv, 0.f, 1.f);
                        uv.push_back((s.x + uu * s.w) / o.page); uv.push_back((s.y + (1 - vv) * s.h) / (float)page_h[p]);
                        dst = (int32_t)(xyz.size() / 3 - 1);
                    }
                    idx.push_back(dst);
                }
            }
            wr32(pf, (u32)p); wr32(pf, (u32)(xyz.size() / 3)); wr32(pf, (u32)idx.size());
            pf.write((const char *)xyz.data(), xyz.size() * 4); pf.write((const char *)uv.data(), uv.size() * 4);
            pf.write((const char *)idx.data(), idx.size() * 4);
            ctris += idx.size() / 3; cverts += xyz.size() / 3;
        }
        if (!pf) die("cannot write payload");
        tot_tris += ctris; tot_verts += cverts;
        man << (nchunk++ ? ",\n" : "") << "  {\"file\": \"" << pname << "\", \"triangles\": " << ctris << ", \"vertices\": " << cverts
            << ", \"pages\": " << pages.size() << ", \"texture_scale\": " << f << ", \"textures\": " << img.size()
            << ", \"uv_clamped\": " << uv_clamped << ", \"centre_m\": [" << ccx << ", " << ccy << "]"
            << ", \"bounds_cm\": [[" << lo[0] << ", " << lo[1] << ", " << lo[2] << "], [" << hi[0] << ", " << hi[1] << ", " << hi[2] << "]]}";
        fprintf(stderr, "\rchunk %d/%zu", nchunk, chunks.size());
    }
    man << "\n ],\n \"triangles\": " << tot_tris << ", \"vertices\": " << tot_verts << ", \"atlas_mpx\": " << tot_px / 1e6
        << ",\n \"overlap_triangles_dropped\": " << culled << ", \"outside_triangles\": " << outside
        << ",\n \"origin_m\": {\"east\": " << o.cx << ", \"south\": " << o.cy << ", \"up\": " << o.datum << "}"
        << ",\n \"radius_m\": " << (o.all ? 0 : o.radius) << ", \"frame\": \"unreal cm: x east, y south, z up\"\n}\n";
    if (!man) die("cannot write manifest");
    fprintf(stderr, "\n");
    printf("packed chunks=%d triangles=%llu vertices=%llu atlas_mpx=%.1f overlap_dropped=%llu outside=%llu\n", nchunk,
           (unsigned long long)tot_tris, (unsigned long long)tot_verts, tot_px / 1e6, (unsigned long long)culled, (unsigned long long)outside);
}

static void usage() {
    fputs("usage: ucmesh index OBJ CACHE | stats CACHE | pack CACHE OUT [--center EAST SOUTH --radius M | --all]\n"
          "       [--datum SOURCE_UP_M] [--axes east,up,south] [--near M] [--far-scale N] [--chunk M] [--page PX]\n"
          "       [--quality Q] [--keep-overlap] [--max-tris N] [--max-mpx N]\n", stderr);
}

int main(int argc, char **argv) {
    try {
        TJD = tjInitDecompress(); TJC = tjInitCompress();
        if (argc >= 4 && !strcmp(argv[1], "index")) cmd_index(argv[2], argv[3]);
        else if (argc == 3 && !strcmp(argv[1], "stats")) cmd_stats(argv[2]);
        else if (argc >= 4 && !strcmp(argv[1], "pack")) {
            Opts o; bool region = false;
            for (int i = 4; i < argc; i++) {
                std::string a = argv[i]; auto num = [&] { if (i + 1 >= argc) die(a + " needs a value"); char *e; double v = strtod(argv[++i], &e);
                    if (*e || !std::isfinite(v)) die(a + ": not a number");
                    return v; };
                if (a == "--center") { o.cx = num(); o.cy = num(); }
                else if (a == "--radius") { o.radius = num(); region = o.radius > 0; }
                else if (a == "--all") o.all = region = true;
                else if (a == "--datum") o.datum = num();
                else if (a == "--axes") { if (++i >= argc) die("--axes needs a value"); o.axes = parse_axes(argv[i]); }
                else if (a == "--near") o.near_m = num();
                else if (a == "--far-scale") o.far_scale = (int)num();
                else if (a == "--chunk") o.chunk_m = num();
                else if (a == "--page") o.page = (int)num();
                else if (a == "--quality") o.quality = (int)num();
                else if (a == "--keep-overlap") o.keep_overlap = true;
                else if (a == "--max-tris") o.max_tris = (u64)num();
                else if (a == "--max-mpx") o.max_mpx = num();
                else die("unknown option " + a);
            }
            if (!region) die("pack needs --radius or --all");
            if (o.far_scale < 1 || o.far_scale > 8 || o.page < 256 || o.page > 8192 || o.quality < 30 || o.quality > 100 || o.chunk_m < 8)
                die("option out of range");
            cmd_pack(argv[2], argv[3], o);
        } else { usage(); return 2; }
    } catch (const std::exception &e) { fprintf(stderr, "ucmesh: %s\n", e.what()); return 1; }
    return 0;
}
