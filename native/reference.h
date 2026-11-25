/* Temporary textured reference held in game memory: one untagged, non-colliding actor with one
   ProceduralMeshComponent per UCMESH2 chunk (native/ucmesh.cpp writes them). Not a catalogue object, never
   saved, invisible to every editor-object query. Game thread only.

   reference add X Y Z YAW PATH   add one chunk; the first creates the actor at that pose (cm, degrees), later
                                  chunks must name the same pose. PATH is the rest of the line (spaces allowed).
   reference move X Y Z YAW | hide | show | status | clear */
#define REF_MAX_SECTIONS 16
#define REF_MAX_VERTS (6u * 1000 * 1000)      /* per chunk */
#define REF_MAX_INDICES (24u * 1000 * 1000)   /* per chunk */
#define REF_MAX_COMPONENTS 1024
#define REF_PATH 1024

struct ref_array { void *data; int32_t num, max; };
struct ref_section { uint32_t tex; int32_t nv, ni; float *xyz, *uv; int32_t *idx; };
struct ref_chunk { uint32_t ntex, nsec; wchar_t tex[REF_MAX_SECTIONS][REF_PATH]; struct ref_section s[REF_MAX_SECTIONS]; };
static struct {
    uint8_t *actor, *world; struct xf pose; int components, hidden;
    unsigned sections, textures; uint64_t triangles, vertices, texels;
} ref_state;

static void ref_free(struct ref_chunk *c) {
    unsigned i;
    if (!c) return;
    for (i = 0; i < c->nsec; i++) { free(c->s[i].xyz); free(c->s[i].uv); free(c->s[i].idx); }
    free(c);
}

static int ref_read(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int ref_wide(const char *utf8, wchar_t *out, int cap) {
    return MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, utf8, -1, out, cap) > 0;
}

static struct ref_chunk *ref_parse(const char *path, char *why, int wcap) {
    wchar_t wpath[REF_PATH]; char magic[8], text[REF_PATH]; uint32_t n[3], i, j, len, nsec;
    struct ref_chunk *c = 0; FILE *f = 0; uint64_t verts = 0, idx = 0;
#define REF_BAD(msg) do { snprintf(why, wcap, "%s", msg); goto bad; } while (0)
    if (!ref_wide(path, wpath, REF_PATH) || !(f = _wfopen(wpath, L"rb"))) REF_BAD("cannot open chunk file");
    if (!ref_read(f, magic, 8) || memcmp(magic, "UCMESH2\0", 8)) REF_BAD("not a UCMESH2 chunk (rebuild with native/ucmesh)");
    if (!ref_read(f, n, 8) || !n[0] || n[0] > REF_MAX_SECTIONS || !n[1] || n[1] > REF_MAX_SECTIONS) REF_BAD("bad chunk header");
    if (!(c = calloc(1, sizeof *c))) REF_BAD("out of memory");
    c->ntex = n[0]; nsec = n[1];
    for (i = 0; i < c->ntex; i++) {
        if (!ref_read(f, &len, 4) || !len || len >= REF_PATH || !ref_read(f, text, len) || memchr(text, 0, len)) REF_BAD("bad texture path");
        text[len] = 0;
        if (!ref_wide(text, c->tex[i], REF_PATH)) REF_BAD("texture path is not UTF-8");
    }
    for (i = 0; i < nsec; i++) {
        struct ref_section *s = &c->s[i];
        if (!ref_read(f, n, 12) || n[0] >= c->ntex || !n[1] || !n[2] || n[2] % 3) REF_BAD("bad section header");
        verts += n[1]; idx += n[2];
        if (verts > REF_MAX_VERTS || idx > REF_MAX_INDICES) REF_BAD("chunk too large; repack with a smaller --chunk");
        s->tex = n[0]; s->nv = (int32_t)n[1]; s->ni = (int32_t)n[2]; c->nsec = i + 1;
        s->xyz = malloc((size_t)s->nv * 12); s->uv = malloc((size_t)s->nv * 8); s->idx = malloc((size_t)s->ni * 4);
        if (!s->xyz || !s->uv || !s->idx) REF_BAD("out of memory");
        if (!ref_read(f, s->xyz, (size_t)s->nv * 12) || !ref_read(f, s->uv, (size_t)s->nv * 8) || !ref_read(f, s->idx, (size_t)s->ni * 4))
            REF_BAD("truncated chunk");
        for (j = 0; j < (uint32_t)s->nv * 3; j++) if (!isfinite(s->xyz[j]) || fabsf(s->xyz[j]) > 1e7f) REF_BAD("vertex out of range");
        for (j = 0; j < (uint32_t)s->nv * 2; j++) if (!isfinite(s->uv[j]) || s->uv[j] < -.01f || s->uv[j] > 1.01f) REF_BAD("UV out of atlas");
        for (j = 0; j < (uint32_t)s->ni; j++) if (s->idx[j] < 0 || s->idx[j] >= s->nv) REF_BAD("index out of range");
    }
    if (fgetc(f) != EOF) REF_BAD("trailing bytes in chunk");
    fclose(f); return c;
bad:
    if (f) fclose(f);
    ref_free(c); return 0;
#undef REF_BAD
}

static int ref_live(void) {
    return ref_state.actor && ref_state.world == current_world() && alive(ref_state.actor);
}

static int ref_clear(void) {
    if (ref_live() && !destroy_actor(ref_state.actor)) return 0;
    memset(&ref_state, 0, sizeof ref_state); return 1;
}

static int ref_ret(uint8_t *fn, const char *name, int cap) {
    int r = fn ? param_offset(fn, name) : -1;
    return r >= 0 && rd16(fn + UFUNC_PARMSSIZE) <= cap && r + 8 <= rd16(fn + UFUNC_PARMSSIZE) ? r : -1;
}

static uint8_t *ref_texture(uint8_t *world, const wchar_t *path, uint64_t *texels) {
    uint8_t p[0x80] = {0}, *lib = class_cdo("KismetRenderingLibrary"), *fn = object_function(lib, "ImportFileAsTexture2D"), *t;
    int32_t n = (int32_t)wcslen(path) + 1; struct fstr s = {(uint16_t *)path, n, n}; int r = ref_ret(fn, "ReturnValue", sizeof p);
    uint8_t *g = 0;
    if (r < 0 || !np_put(fn, p, sizeof p, "WorldContextObject", &world, 8) || !np_put(fn, p, sizeof p, "Filename", &s, 16)) return 0;
    call(lib, fn, p); t = rdp(p + r);
    if (!t || !alive(t)) return 0;
    /* Texel count only for status: GetSizeX/Y exist on UTexture2D (Blueprint pure). */
    if ((g = object_function(t, "Blueprint_GetSizeX"))) {
        uint8_t q[16] = {0}; int32_t w, h; call(t, g, q); w = rd32(q);
        g = object_function(t, "Blueprint_GetSizeY"); memset(q, 0, sizeof q); if (g) { call(t, g, q); h = rd32(q); *texels += (uint64_t)w * h; }
    }
    return t;
}

static uint8_t *ref_get_tex(uint8_t *mat, uint64_t name) {
    uint8_t p[0x80] = {0}, *fn = object_function(mat, "K2_GetTextureParameterValue"); int r = ref_ret(fn, "ReturnValue", sizeof p);
    if (r < 0 || !np_put(fn, p, sizeof p, "ParameterName", &name, 8)) return 0;
    call(mat, fn, p); return rdp(p + r);
}

static int ref_section(uint8_t *comp, struct ref_section *s, int i) {
    uint8_t p[0x100] = {0}, zero = 0, *fn = object_function(comp, "CreateMeshSection");
    struct ref_array xyz = {s->xyz, s->nv, s->nv}, uv = {s->uv, s->nv, s->nv}, idx = {s->idx, s->ni, s->ni};
    if (!np_put(fn, p, sizeof p, "SectionIndex", &i, 4) || !np_put(fn, p, sizeof p, "Vertices", &xyz, 16) ||
        !np_put(fn, p, sizeof p, "Triangles", &idx, 16) || !np_put(fn, p, sizeof p, "UV0", &uv, 16) ||
        !np_put(fn, p, sizeof p, "bCreateCollision", &zero, 1)) return 0;
    call(comp, fn, p); return 1;
}

static int ref_material(uint8_t *comp, uint8_t *parent, uint8_t *tex, uint64_t slot, int i) {
    uint8_t p[0x80] = {0}, *mid, *fn = object_function(comp, "CreateDynamicMaterialInstance"); int r = ref_ret(fn, "ReturnValue", sizeof p);
    if (r < 0 || !np_put(fn, p, sizeof p, "ElementIndex", &i, 4) || !np_put(fn, p, sizeof p, "SourceMaterial", &parent, 8)) return 0;
    call(comp, fn, p); mid = rdp(p + r);
    if (!mid || !alive(mid)) return 0;
    fn = object_function(mid, "SetTextureParameterValue"); memset(p, 0, sizeof p);
    if (!np_put(fn, p, sizeof p, "ParameterName", &slot, 8) || !np_put(fn, p, sizeof p, "Value", &tex, 8)) return 0;
    call(mid, fn, p); return ref_get_tex(mid, slot) == tex;
}

static int ref_call1(uint8_t *o, const char *func, const char *param, const void *v, int n) {
    uint8_t p[0x40] = {0}, *fn = object_function(o, func);
    if (!np_put(fn, p, sizeof p, param, v, n)) return 0;
    call(o, fn, p); return 1;
}

static void ref_destroy_component(uint8_t *comp) {
    uint8_t p[0x20] = {0}, *fn = object_function(comp, "K2_DestroyComponent");
    if (fn && np_put(fn, p, sizeof p, "Object", &comp, 8)) call(comp, fn, p);
}

static int ref_spawn(const struct xf *t, char *err, int ecap) {
    uint8_t p[0x100], one = 1, zero = 0, *game = class_cdo("GameplayStatics"), *cls = find_struct("Actor"), *world = current_world(), *a, *fn;
    float tr[12]; int r;
    make_ftransform(t, tr); memset(p, 0, sizeof p); fn = object_function(game, "BeginDeferredActorSpawnFromClass"); r = ref_ret(fn, "ReturnValue", sizeof p);
    if (r < 0 || !cls || !np_put(fn, p, sizeof p, "WorldContextObject", &world, 8) || !np_put(fn, p, sizeof p, "ActorClass", &cls, 8) ||
        !np_put(fn, p, sizeof p, "SpawnTransform", tr, 48) || !np_put(fn, p, sizeof p, "CollisionHandlingOverride", &one, 1)) goto fail;
    call(game, fn, p); a = rdp(p + r);
    if (!a || !alive(a)) goto fail;
    ref_state.actor = a; ref_state.world = world; ref_state.pose = *t;
    fn = object_function(game, "FinishSpawningActor"); memset(p, 0, sizeof p);
    if (!np_put(fn, p, sizeof p, "Actor", &a, 8) || !np_put(fn, p, sizeof p, "SpawnTransform", tr, 48)) goto fail;
    call(game, fn, p);
    if (!ref_call1(a, "SetActorEnableCollision", "bNewActorEnableCollision", &zero, 1)) goto fail;
    return 1;
fail:
    ref_clear(); snprintf(err, ecap, "could not create the reference actor"); return 0;
}

static int ref_same_pose(const struct xf *a, const struct xf *b) {
    int i; for (i = 0; i < 3; i++) if (fabsf(a->loc[i] - b->loc[i]) > 1.f) return 0;
    return fabsf(remainderf(a->rot[1] - b->rot[1], 360.f)) < .01f;
}

static int ref_add(const struct xf *t, const char *path, char *err, int ecap) {
    struct ref_chunk *c; char why[160] = "";
    uint8_t p[0x100], zero = 0, *comp, *fn, *world = current_world(), *mesh = find_struct("ProceduralMeshComponent");
    uint8_t *mat = find_object("Widget3DPassThrough_Opaque", "MaterialInstanceConstant"), *tex[REF_MAX_SECTIONS] = {0};
    float identity[12] = {0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0}; uint64_t slot, texels = 0, tris = 0, verts = 0; int r; unsigned i;
    if (!world || !editor_tag()) { snprintf(err, ecap, "open the track editor first"); return 0; }
    if (!mesh || !mat || !make_fname("SlateUI", &slot) || !ref_get_tex(mat, slot)) { snprintf(err, ecap, "renderer reflection unavailable"); return 0; }
    if (ref_live() && !ref_same_pose(t, &ref_state.pose)) { snprintf(err, ecap, "reference already placed at another pose; use that pose or clear"); return 0; }
    if (ref_live() && ref_state.components >= REF_MAX_COMPONENTS) { snprintf(err, ecap, "reference holds %d chunks already", REF_MAX_COMPONENTS); return 0; }
    if (!(c = ref_parse(path, why, sizeof why))) { snprintf(err, ecap, "%s", why); return 0; }
    if (!ref_live()) { memset(&ref_state, 0, sizeof ref_state); if (!ref_spawn(t, err, ecap)) { ref_free(c); return 0; } }
    for (i = 0; i < c->ntex; i++)
        if (!(tex[i] = ref_texture(world, c->tex[i], &texels))) { ref_free(c); snprintf(err, ecap, "texture import failed (missing atlas file?)"); return 0; }
    fn = object_function(ref_state.actor, "AddComponentByClass"); memset(p, 0, sizeof p); r = ref_ret(fn, "ReturnValue", sizeof p);
    if (r < 0 || !np_put(fn, p, sizeof p, "Class", &mesh, 8) || !np_put(fn, p, sizeof p, "RelativeTransform", identity, 48)) {
        ref_free(c); snprintf(err, ecap, "AddComponentByClass unavailable"); return 0;
    }
    call(ref_state.actor, fn, p); comp = rdp(p + r);
    if (!comp || !alive(comp)) { ref_free(c); snprintf(err, ecap, "could not add a mesh component"); return 0; }
    ref_call1(comp, "SetCollisionEnabled", "NewType", &zero, 1);
    ref_call1(comp, "SetCastShadow", "NewCastShadow", &zero, 1);
    for (i = 0; i < c->nsec; i++) {
        if (!ref_section(comp, &c->s[i], (int)i) || !ref_material(comp, mat, tex[c->s[i].tex], slot, (int)i)) {
            ref_destroy_component(comp); ref_free(c); snprintf(err, ecap, "mesh section %u failed; chunk rolled back", i); return 0;
        }
        tris += (uint64_t)c->s[i].ni / 3; verts += (uint64_t)c->s[i].nv;
    }
    if (ref_state.components == 0 && !actor_set(ref_state.actor, &ref_state.pose)) {
        ref_clear(); ref_free(c); snprintf(err, ecap, "could not place the reference"); return 0;
    }
    if (ref_state.hidden) { uint8_t one = 1; ref_call1(ref_state.actor, "SetActorHiddenInGame", "bNewHidden", &one, 1); }
    ref_state.components++; ref_state.sections += c->nsec; ref_state.textures += c->ntex;
    ref_state.triangles += tris; ref_state.vertices += verts; ref_state.texels += texels;
    outf("reference added sections=%u triangles=%llu texels=%llu total_chunks=%d total_triangles=%llu", c->nsec,
         (unsigned long long)tris, (unsigned long long)texels, ref_state.components, (unsigned long long)ref_state.triangles);
    ref_free(c); return 1;
}

static int ref_pose(const struct args *a, int at, struct xf *t) {
    int i; char *e;
    memset(t, 0, sizeof *t);
    for (i = 0; i < 4; i++) {
        float v = strtof(a->v[at + i], &e);
        if (e == a->v[at + i] || *e || !isfinite(v) || fabsf(v) > 1e8f) return 0;
        if (i < 3) t->loc[i] = v; else t->rot[1] = v;
    }
    t->scale[0] = t->scale[1] = t->scale[2] = 1; return 1;
}

static void ref_status(void) {
    if (!ref_live()) { outf("reference empty"); return; }
    outf("reference loaded chunks=%d sections=%u triangles=%llu vertices=%llu textures=%u texels=%llu hidden=%d "
         "pose=%.1f,%.1f,%.1f,%.3f collision=off saved=no", ref_state.components, ref_state.sections,
         (unsigned long long)ref_state.triangles, (unsigned long long)ref_state.vertices, ref_state.textures,
         (unsigned long long)ref_state.texels, ref_state.hidden, ref_state.pose.loc[0], ref_state.pose.loc[1],
         ref_state.pose.loc[2], ref_state.pose.rot[1]);
}

static uint8_t *ref_capture_actor(void) { return ref_live() ? ref_state.actor : 0; }

/* line is the full command line: the chunk path for `add` is everything after the fifth word. */
static int verb_reference(const struct args *a, const char *line, char *err, int ecap) {
    struct xf t; const char *sub = a->n > 1 ? a->v[1] : "";
    if (a->n == 2 && ieq(sub, "status")) { ref_status(); return 1; }
    if (a->n == 2 && ieq(sub, "clear")) {
        if (!ref_clear()) { snprintf(err, ecap, "could not destroy the reference actor"); return 0; }
        outf("reference cleared"); return 1;
    }
    if (a->n == 2 && (ieq(sub, "hide") || ieq(sub, "show"))) {
        uint8_t h = ieq(sub, "hide");
        if (!ref_live() || !ref_call1(ref_state.actor, "SetActorHiddenInGame", "bNewHidden", &h, 1)) { snprintf(err, ecap, "no reference loaded"); return 0; }
        ref_state.hidden = h; outf("reference %s", h ? "hidden" : "shown"); return 1;
    }
    if (a->n == 6 && ieq(sub, "move") && ref_pose(a, 2, &t)) {
        if (!ref_live() || !actor_set(ref_state.actor, &t)) { snprintf(err, ecap, "no reference loaded"); return 0; }
        ref_state.pose = t; outf("reference moved"); return 1;
    }
    if (a->n >= 7 && ieq(sub, "add") && ref_pose(a, 2, &t)) {
        const char *p = line; int w;
        for (w = 0; w < 6; w++) { p += strspn(p, " \t"); p += strcspn(p, " \t"); }
        p += strspn(p, " \t");
        return ref_add(&t, p, err, ecap);
    }
    snprintf(err, ecap, "usage: reference status | clear | hide | show | move X Y Z YAW | add X Y Z YAW PATH (cm, degrees)");
    return 0;
}
