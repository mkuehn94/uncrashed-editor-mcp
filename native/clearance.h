/* Route clearance for aircraft and ground vehicles. Read-only. Included by agent.c after kismet helpers
   (K, k_types, weak_actor, actor_row, env_component_box). Native units: cm, degrees.

   sweep <request> <output>
     request lines (one header each, then samples in path order):
       B hx hy hz                  craft half extents: x along heading, y sideways, z up
       R probe_cm ndirs            radial probes: ndirs (0, 4..32) horizontal directions relative to the
                                   sample yaw (k=0 forward, k=ndirs/4 right: Unreal Y is right), + up + down
       C channel                   ECollisionChannel the craft collides as (2 = Pawn); every hit component
                                   is asked GetCollisionResponseToChannel(channel)
       M sweep | M drape up_cm down_cm [skip]
                                   skip=1: drape ignores foliage (InstancedFoliageActor) and race gates
       I actor                     extra actor to ignore (at most 64; must be a live Actor)
       S idx x y z pitch yaw sweep probe   sample; sweep=1 sweeps the box from it to the next S line
     sweep output:
       H seg actor comp start x y z time   swept-box hit (BoxTraceMultiForObjects, complex collision)
       P idx d actor comp ... (ndirs+2)    nearest blocking distance per probe, -1 = free to probe_cm
     drape output (no sweeps or probes):
       D idx z actor comp | D idx miss     highest blocking surface under the sample
     always:
       K comp actor class resp otype enabled   every component seen (resp -1 = unknown)
       A actor row class                       every actor seen
       sweep mode=.. samples=.. segments=.. hits=.. overlap_only=.. truncated=.. seen_full=.. failed=..
             skipped=.. channel=.. ms=..   (skipped: drape hits on foliage / gates passed over)
   Overlap-only and QueryOnly / NoCollision components (triggers, gate opening planes, flythrough) never
   shorten probes or drape; swept hits are reported with response + enabled so the caller decides.

   gatebox <output> [channel]     channel: ECollisionChannel index 0..31 (default 2 = Pawn)
     G actor x y z pitch yaw roll sx sy sz lo_x lo_y lo_z hi_x hi_y hi_z row start   every race gate, level order
        (row: BP_Gate_C Row or -, start: its Start? flag, -1 unknown)
     GC gate comp class resp otype enabled x y z pitch yaw roll bx by bz lo_x lo_y lo_z hi_x hi_y hi_z
        each primitive component of it; b = scaled half extent (BoxComponent, or StaticMeshComponent mesh
        bounds; x y z is then the box centre) or -1 -1 -1
     gatebox gates=.. components=.. channel=.. */

#define CL_SAMPLES 4096
#define CL_PROBES 34
#define CL_SEEN 16384
#define CL_SEG_HITS 48

static struct {
    int ok;
    uint8_t *fm, *f_resp, *f_otype, *f_enabled;
    int m_wco, m_s, m_e, m_types, m_cx, m_ign, m_hits, m_self, m_ret, m_size;
    int r_ch, r_ret, o_ret, e_ret;
} CL;

static int cl_channel = 2;
static uint8_t *cl_ign[65];
static int cl_nign;

struct cl_seen { uint8_t *p; int8_t resp, otype, enabled, kind, skip; };
static struct cl_seen cl_seen[CL_SEEN];
static int cl_nseen, cl_seen_full, cl_drape_skip, cl_skipped;

static int cl_init(char *err, int ecap) {
    uint8_t *prim;
    if (CL.ok) return 1;
    if (!kismet_init(err, ecap)) return 0;
    CL.fm = find_function(rdp(K.ksl + UOBJ_CLASS), "LineTraceMultiForObjects");
    prim = find_struct("PrimitiveComponent");
    if (!CL.fm || !prim || K.h_component < 0 || K.h_time < 0) {
        snprintf(err, ecap, "LineTraceMultiForObjects, PrimitiveComponent or HitResult.Component/Time not found");
        return 0;
    }
    CL.m_wco = param_offset(CL.fm, "WorldContextObject"); CL.m_s = param_offset(CL.fm, "Start");
    CL.m_e = param_offset(CL.fm, "End"); CL.m_types = param_offset(CL.fm, "ObjectTypes");
    CL.m_cx = param_offset(CL.fm, "bTraceComplex"); CL.m_ign = param_offset(CL.fm, "ActorsToIgnore");
    CL.m_hits = param_offset(CL.fm, "OutHits"); CL.m_self = param_offset(CL.fm, "bIgnoreSelf");
    CL.m_ret = param_offset(CL.fm, "ReturnValue"); CL.m_size = rd16(CL.fm + UFUNC_PARMSSIZE);
    if (CL.m_wco < 0 || CL.m_s < 0 || CL.m_e < 0 || CL.m_types < 0 || CL.m_cx < 0 || CL.m_ign < 0 || CL.m_hits < 0 ||
        CL.m_ret < 0 || CL.m_size > 0x400) {
        snprintf(err, ecap, "multi line trace parameter layout not understood");
        return 0;
    }
    /* Optional: without them every hit is reported with resp=-1 (unknown) and counts as blocking. */
    CL.f_resp = find_function(prim, "GetCollisionResponseToChannel");
    CL.f_otype = find_function(prim, "GetCollisionObjectType");
    CL.f_enabled = find_function(prim, "GetCollisionEnabled");
    CL.r_ch = CL.f_resp ? param_offset(CL.f_resp, "Channel") : -1;
    CL.r_ret = CL.f_resp ? param_offset(CL.f_resp, "ReturnValue") : -1;
    CL.o_ret = CL.f_otype ? param_offset(CL.f_otype, "ReturnValue") : -1;
    CL.e_ret = CL.f_enabled ? param_offset(CL.f_enabled, "ReturnValue") : -1;
    if (CL.r_ch < 0 || CL.r_ret < 0 || rd16(CL.f_resp + UFUNC_PARMSSIZE) > 0x20) CL.f_resp = 0;
    if (CL.o_ret < 0 || rd16(CL.f_otype + UFUNC_PARMSSIZE) > 0x20) CL.f_otype = 0;
    if (CL.e_ret < 0 || rd16(CL.f_enabled + UFUNC_PARMSSIZE) > 0x20) CL.f_enabled = 0;
    CL.ok = 1;
    return 1;
}

static int8_t cl_byte_call(uint8_t *comp, uint8_t *fn, int ret, int arg, int value) {
    uint8_t p[0x20];
    if (!fn || !comp || !alive(comp) || !is_a(comp, "PrimitiveComponent")) return -1;
    memset(p, 0, sizeof p);
    if (arg >= 0) p[arg] = (uint8_t)value;
    call(comp, fn, p);
    return (int8_t)p[ret];
}

/* record a component or actor once; returns its entry (0 when the table is full) */
static struct cl_seen *cl_note(uint8_t *p, int kind) {
    uint32_t h = (uint32_t)(((uint64_t)p >> 4) * 2654435761u) % CL_SEEN, i;
    if (!p) return 0;
    for (i = 0; i < CL_SEEN; i++, h = (h + 1) % CL_SEEN) {
        if (cl_seen[h].p == p) return &cl_seen[h];
        if (!cl_seen[h].p) break;
    }
    if (i == CL_SEEN || cl_nseen >= CL_SEEN * 3 / 4) { cl_seen_full = 1; return 0; }
    cl_seen[h].p = p; cl_seen[h].kind = (int8_t)kind; cl_nseen++;
    if (kind == 1) {
        cl_seen[h].resp = cl_byte_call(p, CL.f_resp, CL.r_ret, CL.r_ch, cl_channel);
        cl_seen[h].otype = cl_byte_call(p, CL.f_otype, CL.o_ret, -1, 0);
        cl_seen[h].enabled = cl_byte_call(p, CL.f_enabled, CL.e_ret, -1, 0);
    } else {
        cl_seen[h].skip = (int8_t)(alive(p) && (is_a(p, "InstancedFoliageActor") || is_gate(p)));
    }
    return &cl_seen[h];
}

/* blocking unless the component answers Ignore (0) or Overlap (1) for the craft's channel, or has no
   physics collision: ECollisionEnabled NoCollision (0) / QueryOnly (1). Race gates prove the point: their
   opening is a QueryOnly plane that "blocks" Pawn for checkpoint queries and is flown through. */
static int cl_blocks(uint8_t *comp) {
    struct cl_seen *s = cl_note(comp, 1);
    return !s || ((s->resp < 0 || s->resp == 2) && s->enabled != 0 && s->enabled != 1);
}

/* half extents of a static mesh component's mesh bounds times its world scale; 0 if unknown */
static int cl_mesh_half(uint8_t *comp, float h[3]) {
    static int off = -2;
    uint8_t parms[0x40], *mesh, *fn;
    float sc[3];
    int o, i;
    if (!is_a(comp, "StaticMeshComponent")) return 0;
    if (off == -2) off = field_offset(comp, "StaticMesh");
    if (off <= 0 || !(mesh = rdp(comp + off)) || !valid_object(mesh) || !vec_getter(comp, "K2_GetComponentScale", sc)) return 0;
    if (!(fn = object_function(mesh, "GetBoundingBox")) || (o = param_offset(fn, "ReturnValue")) < 0 || o + 24 > (int)sizeof parms) return 0;
    memset(parms, 0, sizeof parms);
    call(mesh, fn, parms);
    for (i = 0; i < 3; i++) h[i] = (((float *)(parms + o))[3 + i] - ((float *)(parms + o))[i]) * 0.5f * fabsf(sc[i]);
    return 1;
}

/* the world-context actor first, then the request's I actors (already stored from cl_ign[1]) */
static void cl_ignore(struct tarr *ign, uint8_t *self) {
    cl_ign[0] = self;
    ign->data = cl_ign; ign->num = ign->max = 1 + cl_nign;
}

static struct tarr cl_line_hits, cl_box_hits; /* engine-owned storage, reused between calls */

static int cl_hit_list(struct tarr *hits) {
    if (hits->num <= 0) return 0;
    if (hits->num > 4096 || !readable(hits->data, (size_t)hits->num * K.h_size)) return -1;
    return hits->num;
}

/* nearest blocking hit along s->e: distance in cm (0 = starts inside), -1 none, -2 trace failed */
static float cl_probe(uint8_t *self, const float s[3], const float e[3], uint8_t **actor, uint8_t **comp, int *overlap) {
    static uint8_t parms[0x400];
    struct tarr types, ignore;
    float len = sqrtf((e[0]-s[0])*(e[0]-s[0]) + (e[1]-s[1])*(e[1]-s[1]) + (e[2]-s[2])*(e[2]-s[2])), best = -1;
    int i, n;
    *actor = *comp = 0;
    types.data = k_types; types.num = types.max = k_ntypes;
    cl_ignore(&ignore, self);
    memset(parms, 0, (size_t)CL.m_size);
    memcpy(parms + CL.m_wco, &self, 8);
    memcpy(parms + CL.m_s, s, 12);
    memcpy(parms + CL.m_e, e, 12);
    memcpy(parms + CL.m_types, &types, 16);
    parms[CL.m_cx] = 1;
    memcpy(parms + CL.m_ign, &ignore, 16);
    if (CL.m_self >= 0) parms[CL.m_self] = 1;
    cl_line_hits.num = 0;
    memcpy(parms + CL.m_hits, &cl_line_hits, 16);
    call(K.ksl, CL.fm, parms);
    memcpy(&cl_line_hits, parms + CL.m_hits, 16);
    if ((n = cl_hit_list(&cl_line_hits)) < 0) return -2;
    for (i = 0; i < n; i++) {
        uint8_t *hr = (uint8_t *)cl_line_hits.data + (size_t)i * K.h_size, *c, *a;
        float t = 1, d;
        int start = 0;
        memcpy(&t, hr + K.h_time, 4);
        if (K.h_spen >= 0) start = (hr[K.h_spen] & K.h_spen_mask) != 0;
        c = weak_actor(hr + K.h_component);
        a = weak_actor(hr + K.h_actor);
        if (!cl_blocks(c)) { if (overlap) (*overlap)++; continue; }
        if (cl_drape_skip && a) {
            struct cl_seen *sa = cl_note(a, 0);
            if (sa && sa->skip) { cl_skipped++; continue; }
        }
        d = start ? 0 : t * len;
        if (!isfinite(d)) continue;
        if (best < 0 || d < best) { best = d; *actor = a; *comp = c; }
    }
    if (*actor) cl_note(*actor, 0);
    return best;
}

/* every hit of the oriented box swept from s to e; 1 ok, 0 trace failed */
static int cl_sweep(FILE *out, int seg, uint8_t *self, const float s[3], const float e[3], const float half[3],
                    const float rot[3], int *count, int *overlap, int *truncated) {
    static uint8_t parms[0x400];
    struct tarr types, ignore;
    int i, n, shown = 0;
    types.data = k_types; types.num = types.max = k_ntypes;
    cl_ignore(&ignore, self);
    memset(parms, 0, (size_t)K.b_size);
    memcpy(parms + K.b_wco, &self, 8);
    memcpy(parms + K.b_s, s, 12);
    memcpy(parms + K.b_e, e, 12);
    memcpy(parms + K.b_half, half, 12);
    memcpy(parms + K.b_rot, rot, 12);
    memcpy(parms + K.b_types, &types, 16);
    parms[K.b_cx] = 1;
    memcpy(parms + K.b_ign, &ignore, 16);
    if (K.b_self >= 0) parms[K.b_self] = 1;
    cl_box_hits.num = 0;
    memcpy(parms + K.b_hits, &cl_box_hits, 16);
    call(K.ksl, K.fb, parms);
    memcpy(&cl_box_hits, parms + K.b_hits, 16);
    if ((n = cl_hit_list(&cl_box_hits)) < 0) return 0;
    for (i = 0; i < n; i++) {
        uint8_t *hr = (uint8_t *)cl_box_hits.data + (size_t)i * K.h_size, *c, *a;
        float t = 0, p[3];
        int start = 0;
        memcpy(&t, hr + K.h_time, 4);
        memcpy(p, hr + K.h_imp, 12);
        if (K.h_spen >= 0) start = (hr[K.h_spen] & K.h_spen_mask) != 0;
        c = weak_actor(hr + K.h_component);
        a = weak_actor(hr + K.h_actor);
        if (!cl_blocks(c)) (*overlap)++;
        if (a) cl_note(a, 0);
        if (shown == CL_SEG_HITS) { *truncated = 1; break; }
        fprintf(out, "H %d %llx %llx %d %.1f %.1f %.1f %.4f\n", seg, (unsigned long long)a, (unsigned long long)c, start,
                p[0], p[1], p[2], t);
        shown++; (*count)++;
    }
    return 1;
}

static void cl_dump_seen(FILE *out) {
    int i;
    for (i = 0; i < CL_SEEN; i++) {
        uint8_t *p = cl_seen[i].p;
        char cls[128], row[64];
        if (!p || !alive(p)) continue;
        obj_class_name(p, cls, sizeof cls);
        if (cl_seen[i].kind == 1) {
            uint8_t *owner = rdp(p + UOBJ_OUTER);
            fprintf(out, "K %llx %llx %s %d %d %d\n", (unsigned long long)p,
                    (unsigned long long)(owner && alive(owner) && is_a(owner, "Actor") ? owner : 0), cls,
                    cl_seen[i].resp, cl_seen[i].otype, cl_seen[i].enabled);
        } else {
            actor_row(p, row, sizeof row);
            fprintf(out, "A %llx %s %s\n", (unsigned long long)p, row[0] ? row : "-", cls);
        }
    }
}

static uint8_t *cl_context(void) {
    int32_t n, i;
    uint8_t **actors = level_actors(&n);
    for (i = 0; i < n; i++) if (actors[i] && valid_object(actors[i])) return actors[i];
    return 0;
}

static int verb_sweep(const struct args *a, char *err, int ecap) {
    static float pos[CL_SAMPLES][3], rot[CL_SAMPLES][3];
    static int idx[CL_SAMPLES];
    static uint8_t do_sweep[CL_SAMPLES], do_probe[CL_SAMPLES];
    FILE *in, *out;
    char line[256], mode[16] = "sweep";
    float half[3] = {-1, -1, -1}, probe = 0, up = 200, down = 5000;
    int ndirs = 0, n = 0, i, k, bad = 0, hits = 0, overlap = 0, truncated = 0, segments = 0, failed = 0, skip = 0;
    DWORD t0 = GetTickCount();
    uint8_t *ctx;
    if (a->n != 3) { snprintf(err, ecap, "usage: sweep request_file output_file"); return 0; }
    if (!cl_init(err, ecap)) return 0;
    if (!(in = fopen(a->v[1], "r"))) { snprintf(err, ecap, "cannot read sweep request %s", a->v[1]); return 0; }
    cl_nign = 0; cl_channel = 2;
    while (fgets(line, sizeof line, in)) {
        unsigned long long addr;
        int sw, pr, ch;
        float v[5];
        if (line[0] == '\n' || line[0] == '\r' || !line[0]) continue;
        if (line[0] == 'B' && sscanf(line, "B %f %f %f", &half[0], &half[1], &half[2]) == 3) continue;
        if (line[0] == 'R' && sscanf(line, "R %f %d", &probe, &ndirs) == 2) continue;
        if (line[0] == 'C' && sscanf(line, "C %d", &ch) == 1 && ch >= 0 && ch < 32) { cl_channel = ch; continue; }
        if (line[0] == 'M' && sscanf(line, "M %15s %f %f %d", mode, &up, &down, &skip) >= 1) continue;
        if (line[0] == 'I' && sscanf(line, "I %llx", &addr) == 1 && cl_nign < 64) {
            uint8_t *p = (uint8_t *)(uintptr_t)addr;
            if (!alive(p) || !is_a(p, "Actor")) { fclose(in); snprintf(err, ecap, "I %llx is not a live actor", addr); return 0; }
            cl_ign[1 + cl_nign++] = p;
            continue;
        }
        if (line[0] == 'S' && n < CL_SAMPLES &&
            sscanf(line, "S %d %f %f %f %f %f %d %d", &idx[n], &v[0], &v[1], &v[2], &v[3], &v[4], &sw, &pr) == 8 &&
            isfinite(v[0]) && isfinite(v[1]) && isfinite(v[2]) && isfinite(v[3]) && isfinite(v[4])) {
            pos[n][0] = v[0]; pos[n][1] = v[1]; pos[n][2] = v[2];
            rot[n][0] = v[3]; rot[n][1] = v[4]; rot[n][2] = 0;
            do_sweep[n] = (uint8_t)(sw != 0); do_probe[n] = (uint8_t)(pr != 0); n++;
            continue;
        }
        bad = 1; break;
    }
    fclose(in);
    if (bad || !n || !(ieq(mode, "sweep") || ieq(mode, "drape")) || !isfinite(probe) || probe < 0 || probe > 100000 ||
        ndirs < 0 || ndirs > 32 || (ndirs && ndirs < 4) || !isfinite(up) || !isfinite(down) || up < 0 || down <= 0 ||
        (ieq(mode, "sweep") && (half[0] <= 0 || half[1] <= 0 || half[2] <= 0 || half[0] > 5000 || half[1] > 5000 || half[2] > 5000))) {
        snprintf(err, ecap, "sweep request invalid, empty or over %d samples", CL_SAMPLES);
        return 0;
    }
    if (!(ctx = cl_context())) { snprintf(err, ecap, "no actor for world context"); return 0; }
    if (!(out = fopen(a->v[2], "w"))) { snprintf(err, ecap, "cannot write sweep output"); return 0; }
    memset(cl_seen, 0, sizeof cl_seen); cl_nseen = cl_seen_full = cl_skipped = 0;
    cl_drape_skip = ieq(mode, "drape") && skip;
    for (i = 0; i < n; i++) {
        if (ieq(mode, "drape")) {
            float s[3] = {pos[i][0], pos[i][1], pos[i][2] + up}, e[3] = {pos[i][0], pos[i][1], pos[i][2] - down}, d;
            uint8_t *ha, *hc;
            d = cl_probe(ctx, s, e, &ha, &hc, &overlap);
            if (d < -1.5f) failed++;
            if (d >= 0) fprintf(out, "D %d %.2f %llx %llx\n", idx[i], s[2] - d, (unsigned long long)ha, (unsigned long long)hc);
            else fprintf(out, "D %d miss\n", idx[i]);
            continue;
        }
        if (do_sweep[i] && i + 1 < n && idx[i + 1] == idx[i] + 1) {
            float e[3];
            memcpy(e, pos[i + 1], 12);
            if (fabsf(e[0]-pos[i][0]) + fabsf(e[1]-pos[i][1]) + fabsf(e[2]-pos[i][2]) < 0.5f) e[2] += 0.5f;
            if (!cl_sweep(out, idx[i], ctx, pos[i], e, half, rot[i], &hits, &overlap, &truncated)) failed++;
            segments++;
        }
        if (do_probe[i] && probe > 0) {
            double y0 = rot[i][1] * M_PI / 180.0;
            fprintf(out, "P %d", idx[i]);
            for (k = 0; k < ndirs + 2; k++) {
                float e[3], d;
                uint8_t *ha, *hc;
                if (k < ndirs) {
                    double ang = y0 + 2.0 * M_PI * k / ndirs;
                    e[0] = pos[i][0] + probe * (float)cos(ang); e[1] = pos[i][1] + probe * (float)sin(ang); e[2] = pos[i][2];
                } else {
                    e[0] = pos[i][0]; e[1] = pos[i][1]; e[2] = pos[i][2] + (k == ndirs ? probe : -probe);
                }
                d = cl_probe(ctx, pos[i], e, &ha, &hc, &overlap);
                if (d < -1.5f) failed++;
                fprintf(out, " %.1f %llx %llx", d < 0 ? -1.0f : d, (unsigned long long)ha, (unsigned long long)hc);
            }
            fputc('\n', out);
        }
    }
    cl_drape_skip = 0;
    cl_dump_seen(out);
    fprintf(out, "sweep mode=%s samples=%d segments=%d hits=%d overlap_only=%d truncated=%d seen_full=%d failed=%d skipped=%d channel=%d ms=%lu\n",
            mode, n, segments, hits, overlap, truncated, cl_seen_full, failed, cl_skipped, cl_channel, (unsigned long)(GetTickCount() - t0));
    bad = ferror(out);
    fclose(out);
    if (bad) { snprintf(err, ecap, "sweep output write failed"); return 0; }
    outf("sweep samples=%d segments=%d hits=%d failed=%d", n, segments, hits, failed);
    return 1;
}

/* gatebox <output> [channel]: race gates and their collision components (the opening / trigger volume) */
static int verb_gatebox(const struct args *a, char *err, int ecap) {
    enum { GCAP = 512 };
    static uint8_t *gates[GCAP];
    int32_t n, i, total = object_count();
    int ng = 0, nc = 0, j, bad;
    uint8_t **actors = level_actors(&n);
    FILE *out;
    if (a->n < 2 || a->n > 3) { snprintf(err, ecap, "usage: gatebox output_file [channel]"); return 0; }
    if (!cl_init(err, ecap)) return 0;
    cl_channel = 2;
    if (a->n > 2) {
        char *end;
        long ch = strtol(a->v[2], &end, 10);
        if (end == a->v[2] || *end || ch < 0 || ch > 31) { snprintf(err, ecap, "gatebox channel must be an index 0..31, got %s", a->v[2]); return 0; }
        cl_channel = (int)ch;
    }
    if (!(out = fopen(a->v[1], "w"))) { snprintf(err, ecap, "cannot write gatebox output"); return 0; }
    for (i = 0; i < n && ng < GCAP; i++) {
        struct xf t;
        float o[3], e[3];
        char row[128] = "-";
        int start = -1, roff;
        if (!actors[i] || !alive(actors[i]) || !is_gate(actors[i]) || !actor_get(actors[i], &t) || !actor_bounds(actors[i], o, e)) continue;
        gates[ng++] = actors[i];
        if ((roff = field_offset(actors[i], "Row")) > 0) { fname_str(rd64(actors[i] + roff), row, sizeof row); if (!row[0] || ieq(row, "None")) strcpy(row, "-"); }
        if (!np_bool(actors[i], "Start?", -1, &start)) start = -1;
        fprintf(out, "G %llx %.2f %.2f %.2f %.3f %.3f %.3f %.4f %.4f %.4f %.2f %.2f %.2f %.2f %.2f %.2f %s %d\n",
                (unsigned long long)actors[i], t.loc[0], t.loc[1], t.loc[2], t.rot[0], t.rot[1], t.rot[2],
                t.scale[0], t.scale[1], t.scale[2], o[0]-e[0], o[1]-e[1], o[2]-e[2], o[0]+e[0], o[1]+e[1], o[2]+e[2], row, start);
    }
    for (i = 0; i < total && ng; i++) {
        uint8_t *o = object_at(i), *outer;
        float loc[3], r[3], bx[3] = {-1, -1, -1}, c[3], e[3];
        char cls[128];
        if (!o || !valid_object(o) || !(outer = rdp(o + UOBJ_OUTER))) continue;
        for (j = 0; j < ng && gates[j] != outer; j++) {}
        if (j == ng || !alive(o) || !is_a(o, "PrimitiveComponent")) continue;
        if (!vec_getter(o, "K2_GetComponentLocation", loc) || !vec_getter(o, "K2_GetComponentRotation", r) ||
            !env_component_box(o, c, e)) continue;
        if (is_a(o, "BoxComponent")) vec_getter(o, "GetScaledBoxExtent", bx);
        else if (cl_mesh_half(o, bx)) { loc[0] = c[0]; loc[1] = c[1]; loc[2] = c[2]; } /* centre of the mesh box */
        obj_class_name(o, cls, sizeof cls);
        fprintf(out, "GC %llx %llx %s %d %d %d %.2f %.2f %.2f %.3f %.3f %.3f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n",
                (unsigned long long)outer, (unsigned long long)o, cls,
                cl_byte_call(o, CL.f_resp, CL.r_ret, CL.r_ch, cl_channel), cl_byte_call(o, CL.f_otype, CL.o_ret, -1, 0),
                cl_byte_call(o, CL.f_enabled, CL.e_ret, -1, 0), loc[0], loc[1], loc[2], r[0], r[1], r[2],
                bx[0], bx[1], bx[2], c[0]-e[0], c[1]-e[1], c[2]-e[2], c[0]+e[0], c[1]+e[1], c[2]+e[2]);
        nc++;
    }
    fprintf(out, "gatebox gates=%d components=%d channel=%d truncated=%d\n", ng, nc, cl_channel, ng == GCAP);
    bad = ferror(out);
    fclose(out);
    if (bad) { snprintf(err, ecap, "gatebox output write failed"); return 0; }
    outf("gatebox gates=%d components=%d", ng, nc);
    return 1;
}
