/* Read-only performance diagnostics. Included by agent.c after zones.h.
   Nothing is modified. Native units: cm, seconds. Offsets, bitfields and functions
   are reflected by name and cached per class; anything not found is reported as -1.

   perf <file> [x0 y0 x1 y1]
     A <actor> <editor> <row|-> <class> <x> <y> <z> <ex> <ey> <ez> prim sm ism inst shadow coll phys lights
       dynlights shadowlights nevercull hidden skel fx tris movable actorhidden
         nevercull = visible primitives whose CachedMaxDrawDistance and LDMaxDrawDistance are both <= 0.
         one line per level actor (all loaded levels) whose location is in the box; editor=1 for placed objects
         and gates; x y z = bounds centre, e = half extent. tris = sum of reflected LOD0 triangles (-1 if the engine exposes no count).
     I <actor> <component class> <mesh> <instances>     every InstancedStaticMeshComponent (detects engine instancing)
     L <actor> <light class> <mobility> <castshadows> <intensity> <x> <y> <z>
     end actors=<n> components=<n> truncated=<0|1> trisfn=<0|1> instancefn=<0|1>
   rowflags <file>
     R\t<row>\t<field>\t<type>\t<value>...      simple fields of every DT_MapEditorAssets row (S_ME_Asset)
   frametime
     frame delta=<s> real=<s> frames=<n|-1> clock=<s>   GameplayStatics world delta / real time + QPC clock */

#define PF_ACTOR_CAP 131072
#define PF_HASH (1 << 18)
#define PF_CLASS_CAP 1024
#define PF_MESH_CAP 4096

enum { PF_PRIM = 1, PF_SM = 2, PF_ISM = 4, PF_LIGHT = 8, PF_SKEL = 16, PF_FX = 32 };

struct pf_bit { int off; uint8_t mask; };
struct pf_class {
    uint8_t *cls; int kind;
    struct pf_bit shadow, visible, hidden, lshadow;
    int mobility, cached_draw, ld_draw, mesh, intensity;
    uint8_t *fcoll, *fphys, *finst, *floc;
    int coll_ret, phys_ret, phys_bone, inst_ret, loc_ret;
};
struct pf_actor {
    uint8_t *a; int editor, in_box, actor_hidden; float loc[3];
    int prim, sm, ism, inst, shadow, coll, phys, lights, dynlights, shadowlights, nevercull, hidden, skel, fx, movable;
    long long tris;
};
struct pf_mesh { uint8_t *mesh; int tris; };

static struct pf_actor pf_actors[PF_ACTOR_CAP];
static int pf_hash[PF_HASH];
static struct pf_class pf_classes[PF_CLASS_CAP];
static int pf_nclasses;
static struct pf_mesh pf_meshes[PF_MESH_CAP];
static int pf_nmeshes, pf_trisfn, pf_instfn;

static int pf_slot(uint8_t *a) {
    uint64_t h = ((uint64_t)a >> 4) * 0x9E3779B97F4A7C15ull;
    return (int)(h >> 46) & (PF_HASH - 1);
}

static int pf_find(uint8_t *a) {
    int s = pf_slot(a), k;
    for (k = 0; k < PF_HASH; k++, s = (s + 1) & (PF_HASH - 1)) {
        if (pf_hash[s] < 0) return -1;
        if (pf_actors[pf_hash[s]].a == a) return pf_hash[s];
    }
    return -1;
}

static void pf_insert(uint8_t *a, int index) {
    int s = pf_slot(a), guard;
    for (guard = 0; guard < PF_HASH; guard++, s = (s + 1) & (PF_HASH - 1))   /* bounded: never spin the game thread */
        if (pf_hash[s] < 0) { pf_hash[s] = index; return; }
}

static int pf_derives(uint8_t *cls, uint8_t *base) {
    int depth = 0;
    if (!base) return 0;
    while (cls && readable(cls, 0x60) && depth++ < 48) {
        if (cls == base) return 1;
        cls = rdp(cls + USTRUCT_SUPER);
    }
    return 0;
}

static struct pf_bit pf_bitfield(uint8_t *cls, const char *name) {
    struct pf_bit b = {-1, 0};
    uint8_t *p = find_property(cls, name); char type[64], inner[128];
    if (!p) return b;
    prop_type(p, type, sizeof type, inner, sizeof inner);
    if (!ieq(type, "BoolProperty")) return b;
    b.off = rd32(p + FPROP_OFFSET) + p[FPROP_EXTRA + 1]; b.mask = p[FPROP_EXTRA + 2];
    if (!b.mask) b.off = -1;
    return b;
}

static int pf_bit_get(uint8_t *o, struct pf_bit b) {
    return b.off > 0 && readable(o + b.off, 1) ? (o[b.off] & b.mask) != 0 : -1;
}

static uint8_t *pf_fn(uint8_t *cls, const char *name, const char *parm, int *off) {
    uint8_t *fn = find_function(cls, name);
    *off = fn ? param_offset(fn, parm) : -1;
    if (!fn || *off < 0 || rd16(fn + UFUNC_PARMSSIZE) > 0x80) return 0;
    return fn;
}

static struct pf_class *pf_class_of(uint8_t *cls) {
    static uint8_t *b_prim, *b_sm, *b_ism, *b_light, *b_skin, *b_fx, *b_particle;
    static int bases;
    struct pf_class *c; int i;
    for (i = 0; i < pf_nclasses; i++) if (pf_classes[i].cls == cls) return &pf_classes[i];
    if (!bases) {
        b_prim = find_struct("PrimitiveComponent"); b_sm = find_struct("StaticMeshComponent");
        b_ism = find_struct("InstancedStaticMeshComponent"); b_light = find_struct("LightComponentBase");
        b_skin = find_struct("SkinnedMeshComponent"); b_fx = find_struct("FXSystemComponent");
        b_particle = find_struct("ParticleSystemComponent"); bases = 1;
    }
    if (pf_nclasses == PF_CLASS_CAP) return 0;
    c = &pf_classes[pf_nclasses++]; memset(c, 0, sizeof *c); c->cls = cls;
    if (pf_derives(cls, b_prim)) c->kind |= PF_PRIM;
    if (pf_derives(cls, b_sm)) c->kind |= PF_SM;
    if (pf_derives(cls, b_ism)) c->kind |= PF_ISM;
    if (pf_derives(cls, b_light)) c->kind |= PF_LIGHT;
    if (pf_derives(cls, b_skin)) c->kind |= PF_SKEL;
    if (pf_derives(cls, b_fx) || pf_derives(cls, b_particle)) c->kind |= PF_FX;
    if (!c->kind) return c;
    c->visible = pf_bitfield(cls, "bVisible"); c->hidden = pf_bitfield(cls, "bHiddenInGame");
    c->mobility = prop_offset(cls, "Mobility");
    c->floc = pf_fn(cls, "K2_GetComponentLocation", "ReturnValue", &c->loc_ret);
    if (c->kind & PF_PRIM) {
        c->shadow = pf_bitfield(cls, "CastShadow"); c->cached_draw = prop_offset(cls, "CachedMaxDrawDistance");
        c->ld_draw = prop_offset(cls, "LDMaxDrawDistance");
        c->fcoll = pf_fn(cls, "GetCollisionEnabled", "ReturnValue", &c->coll_ret);
        c->fphys = pf_fn(cls, "IsSimulatingPhysics", "ReturnValue", &c->phys_ret);
        c->phys_bone = c->fphys ? param_offset(c->fphys, "BoneName") : -1;
    }
    if (c->kind & PF_SM) c->mesh = prop_offset(cls, "StaticMesh");
    if (c->kind & PF_ISM) c->finst = pf_fn(cls, "GetInstanceCount", "ReturnValue", &c->inst_ret);
    if (c->kind & PF_LIGHT) { c->lshadow = pf_bitfield(cls, "CastShadows"); c->intensity = prop_offset(cls, "Intensity"); }
    return c;
}

/* LOD0 triangles of a static mesh through a reflected getter, if this engine build has one. */
static int pf_mesh_tris(uint8_t *mesh) {
    uint8_t p[0x80], *fn; int i, ol, orv, t = -1;
    if (!mesh || !alive(mesh)) return -1;
    for (i = 0; i < pf_nmeshes; i++) if (pf_meshes[i].mesh == mesh) return pf_meshes[i].tris;
    fn = object_function(mesh, "GetNumTriangles");
    ol = fn ? param_offset(fn, "LODIndex") : -1; orv = fn ? param_offset(fn, "ReturnValue") : -1;
    if (fn && ol >= 0 && orv >= 0 && rd16(fn + UFUNC_PARMSSIZE) <= (int)sizeof p) {
        memset(p, 0, sizeof p); call(mesh, fn, p); memcpy(&t, p + orv, 4); pf_trisfn = 1;
        if (t < 0 || t > 100000000) t = -1;
    }
    if (pf_nmeshes < PF_MESH_CAP) { pf_meshes[pf_nmeshes].mesh = mesh; pf_meshes[pf_nmeshes++].tris = t; }
    return t;
}

/* zeroed parameters: BoneName None = whole body */
static int pf_call_ret(uint8_t *obj, uint8_t *fn, int ret, int bytes) {
    uint8_t p[0x80]; int v = 0;
    if (!fn) return -1;
    memset(p, 0, sizeof p); call(obj, fn, p); memcpy(&v, p + ret, (size_t)bytes); return v;
}

static int verb_perf(const struct args *a, char *err, int ecap) {
    uint8_t *levels[ENV_LEVEL_CAP]; uint64_t tag = editor_tag();
    float x0 = argf(a, 2, -1e30f), y0 = argf(a, 3, -1e30f), x1 = argf(a, 4, 1e30f), y1 = argf(a, 5, 1e30f);
    int truncated = 0, nl, i, j, n, na = 0, ncomp = 0, nobj = object_count(), written = 0;
    struct pf_bit ahidden; struct pf_actor *r; FILE *f;
    if ((a->n != 2 && a->n != 6) || !(f = fopen(a->v[1], "w"))) {
        snprintf(err, ecap, "usage: perf <writable_file> [x0 y0 x1 y1]"); return 0;
    }
    if (x0 > x1) { float t = x0; x0 = x1; x1 = t; }
    if (y0 > y1) { float t = y0; y0 = y1; y1 = t; }
    memset(pf_hash, 0xff, sizeof pf_hash); pf_nclasses = 0; pf_nmeshes = 0; pf_trisfn = 0; pf_instfn = 0;
    ahidden = pf_bitfield(find_struct("Actor"), "bHidden");
    nl = env_levels(levels, ENV_LEVEL_CAP, &truncated);
    for (j = 0; j < nl; j++) {
        uint8_t **actors = env_level_actors(levels[j], &n);
        for (i = 0; i < n; i++) {
            uint8_t *act = actors[i];
            if (!act || !alive(act) || pf_find(act) >= 0) continue;
            if (na == PF_ACTOR_CAP) { truncated = 1; break; }
            r = &pf_actors[na]; memset(r, 0, sizeof *r);
            r->a = act; r->editor = is_editor_actor(act, tag); r->actor_hidden = pf_bit_get(act, ahidden);
            r->in_box = vec_getter(act, "K2_GetActorLocation", r->loc) && r->loc[0] >= x0 && r->loc[0] <= x1 &&
                        r->loc[1] >= y0 && r->loc[1] <= y1;
            pf_insert(act, na++);
        }
    }
    /* One pass over GUObjectArray: a component's Outer is its owning actor. */
    for (i = 0; i < nobj; i++) {
        uint8_t *o = object_at(i), *owner; struct pf_class *c; int idx, vis, hid, mob;
        if (!o || !valid_object(o)) continue;
        owner = rdp(o + UOBJ_OUTER);
        if (!owner || (idx = pf_find(owner)) < 0) continue;
        c = pf_class_of(rdp(o + UOBJ_CLASS));
        if (!c || !c->kind || !alive(o)) continue;
        r = &pf_actors[idx]; ncomp++;
        vis = pf_bit_get(o, c->visible); hid = pf_bit_get(o, c->hidden);
        mob = c->mobility > 0 ? o[c->mobility] : -1;
        if (c->kind & PF_LIGHT) {
            int ls = pf_bit_get(o, c->lshadow); float inten = -1, loc[3] = {0, 0, 0};
            if (vis == 0 || hid == 1) continue;
            if (c->intensity > 0) memcpy(&inten, o + c->intensity, 4);
            if (c->floc) { uint8_t p[0x80]; memset(p, 0, sizeof p); call(o, c->floc, p); memcpy(loc, p + c->loc_ret, 12); }
            r->lights++; if (mob >= 1) r->dynlights++; if (ls == 1) r->shadowlights++;
            if (r->in_box) {
                char cn[128]; obj_class_name(o, cn, sizeof cn);
                fprintf(f, "L %llx %s %d %d %g %.1f %.1f %.1f\n", (unsigned long long)owner, cn, mob, ls, inten, loc[0], loc[1], loc[2]);
            }
            continue;
        }
        if (!(c->kind & PF_PRIM)) continue;
        if (vis == 0 || hid == 1) { r->hidden++; continue; }
        r->prim++;
        if (mob == 2) r->movable++;
        if (pf_bit_get(o, c->shadow) == 1) r->shadow++;
        /* culled when either the cull-volume cache or the desired (LD) max draw distance is positive */
        if (c->cached_draw > 0 || c->ld_draw > 0) {
            float d = 0, ld = 0;
            if (c->cached_draw > 0) memcpy(&d, o + c->cached_draw, 4);
            if (c->ld_draw > 0) memcpy(&ld, o + c->ld_draw, 4);
            if (d <= 0 && ld <= 0) r->nevercull++;
        }
        if (c->fcoll && (pf_call_ret(o, c->fcoll, c->coll_ret, 1) & 0xff) > 0) r->coll++;
        if (c->fphys && (pf_call_ret(o, c->fphys, c->phys_ret, 1) & 0xff) > 0) r->phys++;
        if (c->kind & PF_SKEL) r->skel++;
        if (c->kind & PF_FX) r->fx++;
        if (c->kind & PF_SM) {
            uint8_t *mesh = c->mesh > 0 ? rdp(o + c->mesh) : 0; int t = pf_mesh_tris(mesh), count = 1;
            if (c->kind & PF_ISM) {
                char cn[128], mn[256];
                count = c->finst ? pf_call_ret(o, c->finst, c->inst_ret, 4) : -1;
                if (c->finst) pf_instfn = 1;
                r->ism++; r->inst += count > 0 ? count : 0;
                obj_class_name(o, cn, sizeof cn);
                if (mesh && alive(mesh)) obj_name(mesh, mn, sizeof mn); else snprintf(mn, sizeof mn, "-");
                if (r->in_box) fprintf(f, "I %llx %s %s %d\n", (unsigned long long)owner, cn, mn, count);
            } else r->sm++;
            if (t > 0 && count > 0) r->tris += (long long)t * count;
        }
    }
    for (i = 0; i < na; i++) {
        float o[3], e[3]; char row[64], cn[128];
        r = &pf_actors[i];
        if (!r->in_box) continue;
        if (!actor_bounds(r->a, o, e)) { memcpy(o, r->loc, 12); memset(e, 0, sizeof e); }
        actor_row(r->a, row, sizeof row); obj_class_name(r->a, cn, sizeof cn);
        fprintf(f, "A %llx %d %s %s %.1f %.1f %.1f %.1f %.1f %.1f %d %d %d %d %d %d %d %d %d %d %d %d %d %d %lld %d %d\n",
                (unsigned long long)r->a, r->editor, row[0] ? row : "-", cn, o[0], o[1], o[2], e[0], e[1], e[2],
                r->prim, r->sm, r->ism, r->inst, r->shadow, r->coll, r->phys, r->lights, r->dynlights, r->shadowlights,
                r->nevercull, r->hidden, r->skel, r->fx, r->tris, r->movable, r->actor_hidden);
        written++;
    }
    fprintf(f, "end actors=%d components=%d truncated=%d trisfn=%d instancefn=%d\n", written, ncomp, truncated, pf_trisfn, pf_instfn);
    if (ferror(f)) { fclose(f); snprintf(err, ecap, "perf write failed"); return 0; }
    fclose(f);
    outf("perf actors=%d level_actors=%d components=%d truncated=%d trisfn=%d", written, na, ncomp, truncated, pf_trisfn);
    return 1;
}

/* Every simple field of every catalogue row, so Python can read cull / shadow / physics flags. */
static int verb_rowflags(const struct args *a, char *err, int ecap) {
    uint8_t *dt = data_table("DT_MapEditorAssets"), *st = find_struct("S_ME_Asset"), *elems;
    int32_t num, i, rows = 0; FILE *f;
    if (!dt || !st) { snprintf(err, ecap, "DT_MapEditorAssets / S_ME_Asset not found (is the editor open?)"); return 0; }
    if (a->n != 2 || !(f = fopen(a->v[1], "w"))) { snprintf(err, ecap, "usage: rowflags <writable_file>"); return 0; }
    elems = rdp(dt + DT_ROWMAP); num = rd32(dt + DT_ROWMAP + 8);
    if (num <= 0 || num > 100000 || !readable(elems, (size_t)num * ROWMAP_ELEM)) { fclose(f); snprintf(err, ecap, "row map unreadable"); return 0; }
    for (i = 0; i < num; i++) {
        uint8_t *e = elems + (size_t)i * ROWMAP_ELEM, *data = rdp(e + 8), *p; char nm[300]; int guard = 0;
        if (!readable(data, rd32(st + USTRUCT_SIZE))) continue;
        fname_str(rd64(e), nm, sizeof nm); fprintf(f, "R\t%s", nm);
        for (p = rdp(st + USTRUCT_CHILDPROPS); p && readable(p, 0x80) && guard++ < 200; p = rdp(p + FFIELD_NEXT)) {
            char type[64], inner[128], pname[256], v[300]; int off = rd32(p + FPROP_OFFSET); uint8_t *q = data + off;
            prop_type(p, type, sizeof type, inner, sizeof inner); fname_str(rd64(p + FFIELD_NAME), pname, sizeof pname);
            if (ieq(type, "BoolProperty")) snprintf(v, sizeof v, "%d", (q[p[FPROP_EXTRA + 1]] & (p[FPROP_EXTRA + 2] ? p[FPROP_EXTRA + 2] : 1)) != 0);
            else if (ieq(type, "FloatProperty")) snprintf(v, sizeof v, "%g", *(float *)q);
            else if (ieq(type, "IntProperty")) snprintf(v, sizeof v, "%d", rd32(q));
            else if (ieq(type, "ByteProperty") || ieq(type, "EnumProperty")) snprintf(v, sizeof v, "%u", q[0]);
            else if (ieq(type, "NameProperty")) fname_str(rd64(q), v, sizeof v);
            else if (ieq(type, "ObjectProperty") || ieq(type, "ClassProperty")) {
                uint8_t *o = rdp(q); if (o && valid_object(o)) obj_name(o, v, sizeof v); else snprintf(v, sizeof v, "-");
            } else if (ieq(type, "StructProperty") && ieq(inner, "Vector")) {
                float vec[3]; memcpy(vec, q, 12); snprintf(v, sizeof v, "%g,%g,%g", vec[0], vec[1], vec[2]);
            } else continue;
            fprintf(f, "\t%s\t%s\t%s", pname, type, v);
        }
        fputc('\n', f); rows++;
    }
    if (ferror(f)) { fclose(f); snprintf(err, ecap, "rowflags write failed"); return 0; }
    fclose(f); outf("rowflags rows=%d", rows); return 1;
}

/* World delta (dilated, last frame) and real time through GameplayStatics; frame count if exposed. */
static int verb_frametime(char *err, int ecap) {
    uint8_t *gs = class_cdo("GameplayStatics"), *ks = class_cdo("KismetSystemLibrary"), *world = current_world(), *fn;
    uint8_t p[0x80]; float delta = -1, real = -1; long long frames = -1; int o; LARGE_INTEGER c, fq;
    if (!gs || !world) { snprintf(err, ecap, "GameplayStatics or world unavailable"); return 0; }
    fn = object_function(gs, "GetWorldDeltaSeconds"); memset(p, 0, sizeof p);
    if (np_put(fn, p, sizeof p, "WorldContextObject", &world, 8) && (o = param_offset(fn, "ReturnValue")) >= 0) { call(gs, fn, p); memcpy(&delta, p + o, 4); }
    fn = object_function(gs, "GetRealTimeSeconds"); memset(p, 0, sizeof p);
    if (np_put(fn, p, sizeof p, "WorldContextObject", &world, 8) && (o = param_offset(fn, "ReturnValue")) >= 0) { call(gs, fn, p); memcpy(&real, p + o, 4); }
    fn = ks ? object_function(ks, "GetFrameCount") : 0;
    if (fn && (o = param_offset(fn, "ReturnValue")) >= 0 && rd16(fn + UFUNC_PARMSSIZE) <= (int)sizeof p) {
        memset(p, 0, sizeof p); call(ks, fn, p); memcpy(&frames, p + o, 8);
    }
    if (delta < 0 && real < 0) { snprintf(err, ecap, "GetWorldDeltaSeconds/GetRealTimeSeconds not found"); return 0; }
    QueryPerformanceCounter(&c); QueryPerformanceFrequency(&fq);
    outf("frame delta=%.6f real=%.6f frames=%lld clock=%.6f", delta, real, frames, (double)c.QuadPart / (double)fq.QuadPart);
    return 1;
}
