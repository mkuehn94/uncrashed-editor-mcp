/* Collision / physics toggles for placed editor objects (never base landscape,
   foliage or other non-editor actors; gates only with explicit force).

   Modes (strength order): on < nophysics < flythrough < off
     nophysics   SetSimulatePhysics(false) where a component simulates.
     flythrough  per primitive component: response Ignore to the selected channels
                 (default Pawn, PhysicsBody, Vehicle) and, unless keepbody,
                 QueryAndPhysics -> QueryOnly / PhysicsOnly -> NoCollision, so no rigid
                 contact regardless of the body's channel. Object type, query and
                 Visibility/Camera stay, so ForObjects traces, audits and editor picking
                 keep working. Also stops simulation.
     off         SetActorEnableCollision(false) + stops simulation. Traces miss it too.
     on          exact restore of the recorded originals (profile name, enabled type,
                 per-channel responses, simulate, actor flag).
   Originals are recorded per component BEFORE any setter runs. Restoring only
   rewrites values that differ, so a non-Custom profile name survives.

   Persistence: UncrashedMCP.collision next to the map (written by Python) holds
   portable rules; the resident DLL re-applies them from the ~1 s poll after a map
   (re)load. The stock game without this DLL ignores the file.
     UNCRASHED_COLLISION_V1 <base>
     O <mode> <mask hex> <flags|-> <row> <x> <y> <z> <tol>   one object (cm)
     G <mode> <mask hex> <flags|-> <group>                   editor group tag
     R <mode> <mask hex> <flags|-> <row>                     every instance of a row
   flags: F = may touch gates, K = keepbody. Precedence O > G > R; mode "on" is
   an explicit exception to a broader rule. Commands without persistence create
   "manual" records the poll never overrides. "collision refresh" re-reads the
   file and keeps manual records; "refresh <sel>" hands only the selected manual
   records back to matching rules, "refresh all" every one. Rules are re-applied
   only while the editor pawn exists (test flight / menu: state is left as is).
   Records identify actors and components by address + FName + class + lazily
   assigned GUObjectArray serial, so a reused address is never mistaken for ours.

   The part between COLL_PURE markers needs only libc and is unit tested on the
   host (tests/test_scenery.py). */

/* ---- COLL_PURE begin ---- */
#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COLL_RULE_CAP 20000
#define COLL_BUCKETS 32768
#define COLL_CELL 100.0f
enum { CM_ON = 0, CM_NOPHYS = 1, CM_FLY = 2, CM_OFF = 3 };
static const char *coll_mode_names[] = {"on", "nophysics", "flythrough", "off"};
#define COLL_DEFAULT_MASK ((1u << 2) | (1u << 5) | (1u << 6)) /* Pawn, PhysicsBody, Vehicle */

struct coll_rule {
    char kind; /* O, G, R */
    uint8_t mode, force, keepbody;
    uint32_t mask;
    char key[100];
    float loc[3], tol;
    int next, hits;
};

static int coll_ieq(const char *a, const char *b) {
    while (*a && *b && tolower((unsigned char)*a) == tolower((unsigned char)*b)) a++, b++;
    return *a == *b;
}

static int coll_name_ok(const char *s) {
    int i, n = (int)strlen(s);
    if (n < 1 || n > 96 || coll_ieq(s, "None") || coll_ieq(s, "Group") || !strcmp(s, "-") || !strcmp(s, ".")) return 0;
    for (i = 0; i < n; i++) if (!isalnum((unsigned char)s[i]) && !strchr("_.-", s[i])) return 0;
    return 1;
}

static int coll_mode_parse(const char *s) {
    int i;
    for (i = 0; i < 4; i++) if (coll_ieq(s, coll_mode_names[i])) return i;
    return -1;
}

static int coll_prefix_num(const char *tok, const char *prefix, int lo, int hi) {
    size_t n = strlen(prefix);
    char *end;
    long v;
    if (strlen(tok) <= n) return -1;
    {
        size_t i;
        for (i = 0; i < n; i++) if (tolower((unsigned char)tok[i]) != prefix[i]) return -1;
    }
    v = strtol(tok + n, &end, 10);
    return (!*end && v >= lo && v <= hi) ? (int)v : -1;
}

/* "2,5,6" | "pawn,physicsbody,vehicle" | "game3" (=ECC_GameTraceChannel3=16) | "engine1" (=8) */
static int coll_channels_parse(const char *text, uint32_t *mask) {
    static const char *names[] = {"worldstatic", "worlddynamic", "pawn", "visibility", "camera", "physicsbody", "vehicle", "destructible"};
    char tok[32];
    uint32_t m = 0;
    if (!text || !*text) return 0;
    while (*text) {
        int len = (int)strcspn(text, ","), ch = -1, i;
        char *end;
        long v;
        if (len < 1 || len >= (int)sizeof tok) return 0;
        memcpy(tok, text, (size_t)len); tok[len] = 0;
        text += len + (text[len] == ',');
        v = strtol(tok, &end, 10);
        if (!*end && v >= 0 && v <= 31) ch = (int)v;
        for (i = 0; ch < 0 && i < 8; i++) if (coll_ieq(tok, names[i])) ch = i;
        if (ch < 0 && (v = coll_prefix_num(tok, "engine", 1, 6)) > 0) ch = 7 + (int)v;
        if (ch < 0 && (v = coll_prefix_num(tok, "game", 1, 18)) > 0) ch = 13 + (int)v;
        if (ch < 0) return 0;
        m |= 1u << ch;
    }
    if (!m) return 0;
    *mask = m;
    return 1;
}

static int coll_rule_parse(const char *line, struct coll_rule *r) {
    char kind, mode[16], flags[8], key[128], extra;
    unsigned mask;
    int n, i;
    memset(r, 0, sizeof *r);
    r->next = -1;
    kind = line[0];
    if (kind == 'O') {
        n = sscanf(line, "O %15s %x %7s %127s %f %f %f %f %c", mode, &mask, flags, key, &r->loc[0], &r->loc[1], &r->loc[2], &r->tol, &extra);
        if (n != 8 || !isfinite(r->tol) || r->tol <= 0 || r->tol > 50) return 0;
        for (i = 0; i < 3; i++) if (!isfinite(r->loc[i]) || fabsf(r->loc[i]) > 1e8f) return 0;
    } else if (kind == 'G' || kind == 'R') {
        char fmt[] = "? %15s %x %7s %127s %c";
        fmt[0] = kind;
        if (sscanf(line, fmt, mode, &mask, flags, key, &extra) != 4) return 0;
    } else return 0;
    r->kind = kind;
    if (coll_mode_parse(mode) < 0 || !coll_name_ok(key)) return 0;
    if (kind != 'G' && strncmp(key, "NewRow", 6)) return 0;
    r->mode = (uint8_t)coll_mode_parse(mode);
    if (r->mode == CM_FLY && !mask) return 0;
    if (strcmp(flags, "-")) for (i = 0; flags[i]; i++) {
        if (flags[i] == 'F') r->force = 1;
        else if (flags[i] == 'K') r->keepbody = 1;
        else return 0;
    }
    r->mask = mask;
    snprintf(r->key, sizeof r->key, "%s", key);
    return 1;
}

static uint32_t coll_hash(char kind, const char *key, int cx, int cy) {
    uint32_t h = 2166136261u ^ (uint8_t)kind;
    for (; *key; key++) h = (h ^ (uint8_t)tolower((unsigned char)*key)) * 16777619u;
    h = (h ^ (uint32_t)cx) * 16777619u;
    h = (h ^ (uint32_t)cy) * 16777619u;
    return h % COLL_BUCKETS;
}

static int coll_cell(float v) { return (int)floorf(v / COLL_CELL); }

/* Chains rules into buckets (O by row + plan cell; G/R by key). Later duplicates win. */
static void coll_index(struct coll_rule *rules, int n, int *buckets) {
    int i;
    for (i = 0; i < COLL_BUCKETS; i++) buckets[i] = -1;
    for (i = 0; i < n; i++) {
        struct coll_rule *r = &rules[i];
        uint32_t b = r->kind == 'O' ? coll_hash('O', r->key, coll_cell(r->loc[0]), coll_cell(r->loc[1])) : coll_hash(r->kind, r->key, 0, 0);
        r->next = buckets[b];
        buckets[b] = i;
    }
}

/* Index of the governing rule or -1. Gates only match rules with force. */
static int coll_match(const struct coll_rule *rules, const int *buckets, const char *row, const char *group,
                      const float loc[3], int gate) {
    int dx, dy, i, best = -1;
    float bestd = 1e30f;
    if (row && row[0] && loc) {
        for (dx = -1; dx <= 1; dx++) for (dy = -1; dy <= 1; dy++) {
            for (i = buckets[coll_hash('O', row, coll_cell(loc[0]) + dx, coll_cell(loc[1]) + dy)]; i >= 0; i = rules[i].next) {
                const struct coll_rule *r = &rules[i];
                float ex = loc[0] - r->loc[0], ey = loc[1] - r->loc[1], ez = loc[2] - r->loc[2], d = sqrtf(ex * ex + ey * ey + ez * ez);
                if (r->kind != 'O' || !coll_ieq(r->key, row) || d > r->tol || (gate && !r->force)) continue;
                if (d < bestd || (d == bestd && i > best)) { best = i; bestd = d; }
            }
        }
        if (best >= 0) return best;
    }
    if (group && group[0] && strcmp(group, "-"))
        for (i = buckets[coll_hash('G', group, 0, 0)]; i >= 0; i = rules[i].next)
            if (rules[i].kind == 'G' && coll_ieq(rules[i].key, group) && (!gate || rules[i].force)) return i;
    if (row && row[0])
        for (i = buckets[coll_hash('R', row, 0, 0)]; i >= 0; i = rules[i].next)
            if (rules[i].kind == 'R' && coll_ieq(rules[i].key, row) && (!gate || rules[i].force)) return i;
    return -1;
}
/* ---- COLL_PURE end ---- */
#ifndef COLL_PURE_ONLY

#define COLL_REC_CAP 20000
#define COLL_COMP_CAP 64
#define COLL_TARGET_CAP 20000

/* identity = address + FName + class + GUObjectArray serial (see coll_ident) */
struct coll_comp {
    uint8_t *comp, *cls;
    int serial;
    uint8_t enabled, sim, resp[32];
    uint64_t profile, name;
};
struct coll_rec {
    uint8_t *actor, *cls;
    int serial, ncomp;
    uint8_t actor_coll, mode, manual, keepbody;
    uint32_t mask;
    uint64_t name;
    struct coll_comp *comps;
};

static struct coll_rec coll_recs[COLL_REC_CAP];
static int coll_nrec, coll_table[COLL_REC_CAP * 2 + 1];
static int coll_table_ready;
static void coll_rebuild(void);
static struct coll_rule coll_rules[COLL_RULE_CAP], coll_pending[COLL_RULE_CAP];
static int coll_nrules, coll_buckets[COLL_BUCKETS], coll_base_checked, coll_config, coll_failures;
static int coll_gone, coll_replaced, coll_comp_replaced, coll_kept_unknown; /* cumulative, for status */
static char coll_error[220], coll_seen_session[160], coll_bad_session[160];
static uint64_t coll_seen_stamp, coll_seen_fp, coll_bad_stamp;
static uint8_t *coll_gate_class;

/* ---- reflected functions (resolved once from the base classes) ---- */
static struct {
    int ok;
    uint8_t *prim, *get_en, *set_en, *get_resp, *set_resp, *get_obj, *is_sim, *set_sim, *get_prof, *set_prof,
        *a_get, *a_set, *comps, *loc;
    int get_en_r, set_en_v, get_resp_c, get_resp_r, set_resp_c, set_resp_v, get_obj_r, is_sim_b, is_sim_r, set_sim_v,
        get_prof_r, set_prof_v, set_prof_u, a_get_r, a_set_v, comps_c, comps_r, loc_r;
} CF;

static uint8_t *coll_fn(uint8_t *cls, const char *a, const char *b) {
    uint8_t *f = find_function(cls, a);
    if (!f && b) f = find_function(cls, b);
    return f && rd16(f + UFUNC_PARMSSIZE) <= 0x100 ? f : 0;
}

static int coll_psize(uint8_t *fn, const char *name) {
    uint8_t *p = find_property(fn, name);
    return p ? rd32(p + FPROP_ELEMSIZE) : -1;
}

static int coll_init(char *err, int ecap) {
    uint8_t *actor = find_struct("Actor");
    if (CF.ok) return 1;
    CF.prim = find_struct("PrimitiveComponent");
    if (!CF.prim || !actor) { snprintf(err, ecap, "PrimitiveComponent/Actor class not found"); return 0; }
    CF.get_en = coll_fn(CF.prim, "GetCollisionEnabled", "K2_GetCollisionEnabled");
    CF.set_en = coll_fn(CF.prim, "SetCollisionEnabled", 0);
    CF.get_resp = coll_fn(CF.prim, "GetCollisionResponseToChannel", 0);
    CF.set_resp = coll_fn(CF.prim, "SetCollisionResponseToChannel", 0);
    CF.get_obj = coll_fn(CF.prim, "GetCollisionObjectType", 0);
    CF.is_sim = coll_fn(CF.prim, "IsSimulatingPhysics", 0);
    CF.set_sim = coll_fn(CF.prim, "SetSimulatePhysics", 0);
    CF.get_prof = coll_fn(CF.prim, "GetCollisionProfileName", 0);
    CF.set_prof = coll_fn(CF.prim, "SetCollisionProfileName", 0);
    CF.a_get = coll_fn(actor, "GetActorEnableCollision", 0);
    CF.a_set = coll_fn(actor, "SetActorEnableCollision", 0);
    CF.comps = coll_fn(actor, "GetComponentsByClass", "K2_GetComponentsByClass");
    CF.loc = coll_fn(actor, "K2_GetActorLocation", 0);
    if (!CF.get_en || !CF.set_en || !CF.get_resp || !CF.set_resp || !CF.get_obj || !CF.is_sim || !CF.set_sim ||
        !CF.get_prof || !CF.set_prof || !CF.a_get || !CF.a_set || !CF.loc) {
        snprintf(err, ecap, "collision UFunctions missing (en=%d/%d resp=%d/%d obj=%d sim=%d/%d prof=%d/%d actor=%d/%d)",
                 !!CF.get_en, !!CF.set_en, !!CF.get_resp, !!CF.set_resp, !!CF.get_obj, !!CF.is_sim, !!CF.set_sim,
                 !!CF.get_prof, !!CF.set_prof, !!CF.a_get, !!CF.a_set);
        return 0;
    }
    if (!coll_table_ready) coll_rebuild();
    CF.get_en_r = param_offset(CF.get_en, "ReturnValue"); CF.set_en_v = param_offset(CF.set_en, "NewType");
    CF.get_resp_c = param_offset(CF.get_resp, "Channel"); CF.get_resp_r = param_offset(CF.get_resp, "ReturnValue");
    CF.set_resp_c = param_offset(CF.set_resp, "Channel"); CF.set_resp_v = param_offset(CF.set_resp, "NewResponse");
    CF.get_obj_r = param_offset(CF.get_obj, "ReturnValue");
    CF.is_sim_b = param_offset(CF.is_sim, "BoneName"); CF.is_sim_r = param_offset(CF.is_sim, "ReturnValue");
    CF.set_sim_v = param_offset(CF.set_sim, "bSimulate");
    CF.get_prof_r = param_offset(CF.get_prof, "ReturnValue");
    CF.set_prof_v = param_offset(CF.set_prof, "InCollisionProfileName"); CF.set_prof_u = param_offset(CF.set_prof, "bUpdateOverlaps");
    CF.a_get_r = param_offset(CF.a_get, "ReturnValue"); CF.a_set_v = param_offset(CF.a_set, "bNewActorEnableCollision");
    CF.loc_r = param_offset(CF.loc, "ReturnValue");
    CF.comps_c = CF.comps ? param_offset(CF.comps, "ComponentClass") : -1;
    CF.comps_r = CF.comps ? param_offset(CF.comps, "ReturnValue") : -1;
    if (CF.get_en_r < 0 || CF.set_en_v < 0 || CF.get_resp_c < 0 || CF.get_resp_r < 0 || CF.set_resp_c < 0 ||
        CF.set_resp_v < 0 || CF.get_obj_r < 0 || CF.is_sim_r < 0 || CF.set_sim_v < 0 || CF.get_prof_r < 0 ||
        CF.set_prof_v < 0 || CF.a_get_r < 0 || CF.a_set_v < 0 || CF.loc_r < 0) {
        snprintf(err, ecap, "collision UFunction parameter layout not understood");
        return 0;
    }
    if (coll_psize(CF.get_en, "ReturnValue") != 1 || coll_psize(CF.set_en, "NewType") != 1 || coll_psize(CF.get_resp, "Channel") != 1 ||
        coll_psize(CF.get_resp, "ReturnValue") != 1 || coll_psize(CF.set_resp, "Channel") != 1 || coll_psize(CF.set_resp, "NewResponse") != 1 ||
        coll_psize(CF.get_obj, "ReturnValue") != 1 || coll_psize(CF.get_prof, "ReturnValue") != 8 || coll_psize(CF.set_prof, "InCollisionProfileName") != 8) {
        snprintf(err, ecap, "collision UFunction parameter sizes not understood");
        return 0;
    }
    if (CF.comps_c < 0 || CF.comps_r < 0) CF.comps = 0; /* fall back to the attachment tree */
    CF.ok = 1;
    return 1;
}

static int cf_get(uint8_t *o, uint8_t *fn, int ret, int arg, int value) {
    uint8_t p[0x100];
    memset(p, 0, sizeof p);
    if (arg >= 0) p[arg] = (uint8_t)value;
    call(o, fn, p);
    return p[ret];
}
static void cf_set(uint8_t *o, uint8_t *fn, int a, int av, int b, int bv) {
    uint8_t p[0x100];
    memset(p, 0, sizeof p);
    p[a] = (uint8_t)av;
    if (b >= 0) p[b] = (uint8_t)bv;
    call(o, fn, p);
}
static int cf_enabled(uint8_t *c) { return cf_get(c, CF.get_en, CF.get_en_r, -1, 0); }
static int cf_resp(uint8_t *c, int ch) { return cf_get(c, CF.get_resp, CF.get_resp_r, CF.get_resp_c, ch); }
static int cf_sim(uint8_t *c) { return cf_get(c, CF.is_sim, CF.is_sim_r, -1, 0); } /* BoneName None */
static int cf_actor(uint8_t *a) { return cf_get(a, CF.a_get, CF.a_get_r, -1, 0); }
static uint64_t cf_profile(uint8_t *c) {
    uint8_t p[0x100];
    memset(p, 0, sizeof p);
    call(c, CF.get_prof, p);
    return rd64(p + CF.get_prof_r);
}
static void cf_set_profile(uint8_t *c, uint64_t name) {
    uint8_t p[0x100];
    memset(p, 0, sizeof p);
    memcpy(p + CF.set_prof_v, &name, 8);
    if (CF.set_prof_u >= 0) p[CF.set_prof_u] = 1;
    call(c, CF.set_prof, p);
}

static int coll_serial(uint8_t *o) {
    uint8_t *item = object_item(rd32(o + UOBJ_INDEX));
    return item ? rd32(item + 0x10) : 0;
}

/* Is o still the object recorded as (serial, name, class)? UE assigns
   FUObjectItem.SerialNumber lazily (first weak pointer) and zeroes it when the
   slot is freed, so a recorded 0 adopts the current value (as zones.h does) and
   FName + class guard against a new object reusing address and index. */
enum { COLL_GONE = 0, COLL_SAME = 1, COLL_OTHER = 2 };
static int coll_ident(uint8_t *o, int *serial, uint64_t name, uint8_t *cls) {
    int now;
    if (!o || !alive(o)) return COLL_GONE;
    if (rd64(o + UOBJ_NAME) != name || rdp(o + UOBJ_CLASS) != cls) return COLL_OTHER;
    now = coll_serial(o);
    if (*serial && now != *serial) return COLL_OTHER;
    if (!*serial) *serial = now;
    return COLL_SAME;
}
static int coll_rec_ident(struct coll_rec *r) { return coll_ident(r->actor, &r->serial, r->name, r->cls); }
static void coll_lost(int ident) { if (ident == COLL_OTHER) coll_replaced++; else coll_gone++; }

static int coll_loc(uint8_t *a, float out[3]) {
    uint8_t p[0x100];
    memset(p, 0, sizeof p);
    call(a, CF.loc, p);
    memcpy(out, p + CF.loc_r, 12);
    return isfinite(out[0]) && isfinite(out[1]) && isfinite(out[2]);
}

static int coll_is_prim(uint8_t *o) {
    uint8_t *cls = o ? rdp(o + UOBJ_CLASS) : 0;
    int depth = 0;
    while (cls && readable(cls, 0x60) && depth++ < 48) {
        if (cls == CF.prim) return 1;
        cls = rdp(cls + USTRUCT_SUPER);
    }
    return 0;
}

static int coll_tree(uint8_t *act, uint8_t *c, uint8_t **out, int n, int cap, int depth, int *trunc) {
    static int off = -2;
    uint8_t **kids; int k, i;
    if (!c || !alive(c) || depth > 16) return n;
    if (rdp(c + UOBJ_OUTER) == act && coll_is_prim(c)) {
        for (i = 0; i < n && out[i] != c; i++) {}
        if (i == n) { if (n == cap) { *trunc = 1; return n; } out[n++] = c; }
    }
    if (off == -2) off = prop_offset(find_struct("SceneComponent"), "AttachChildren");
    if (off <= 0) return n;
    kids = (uint8_t **)rdp(c + off); k = rd32(c + off + 8);
    if (k <= 0 || k > 256 || !readable(kids, (size_t)k * 8)) return n;
    for (i = 0; i < k; i++) n = coll_tree(act, kids[i], out, n, cap, depth + 1, trunc);
    return n;
}

/* the actor's own primitive components; *trunc set when more than cap */
static int coll_components(uint8_t *act, uint8_t **out, int cap, int *trunc) {
    static struct tarr arr; /* engine-owned storage, reused (move-assigned) between calls */
    int n = 0, i;
    *trunc = 0;
    if (CF.comps) {
        uint8_t p[0x100];
        memset(p, 0, sizeof p);
        memcpy(p + CF.comps_c, &CF.prim, 8);
        memcpy(p + CF.comps_r, &arr, 16);
        call(act, CF.comps, p);
        memcpy(&arr, p + CF.comps_r, 16);
        if (arr.num < 0 || arr.num > 4096 || arr.max < arr.num || (arr.num && !readable(arr.data, (size_t)arr.num * 8))) {
            memset(&arr, 0, sizeof arr); /* never hand an unexpected buffer back to the engine: leak instead */
            *trunc = 1;
            return 0;
        }
        for (i = 0; i < arr.num; i++) {
            uint8_t *c = ((uint8_t **)arr.data)[i];
            if (!c || !alive(c) || !coll_is_prim(c)) continue;
            if (n == cap) { *trunc = 1; break; }
            out[n++] = c;
        }
        return n;
    }
    return coll_tree(act, root_component(act), out, 0, cap, 0, trunc);
}

static int coll_is_gate(uint8_t *a) {
    uint8_t *cls = rdp(a + UOBJ_CLASS);
    int depth = 0;
    if (!coll_gate_class) return 0;
    while (cls && readable(cls, 0x60) && depth++ < 48) {
        if (cls == coll_gate_class) return 1;
        cls = rdp(cls + USTRUCT_SUPER);
    }
    return 0;
}

static int coll_has_named_tag(uint8_t *a, const char *name) {
    uint64_t *tags; int n = actor_tag_list(a, &tags), i; char s[128];
    for (i = 0; i < n; i++) { fname_str(tags[i], s, sizeof s); if (ieq(s, name)) return 1; }
    return 0;
}

/* placed editor object (or gate). Without an editor pawn (unverified: test flight)
   require the catalogue row tag and the editor's ME_StaticMesh/ME_Blueprint marker. */
static int coll_placed(uint8_t *a, uint64_t tag, int *gate) {
    char row[64];
    *gate = 0;
    if (!a || !alive(a)) return 0;
    if (tag && actor_has_tag(a, tag)) { *gate = coll_is_gate(a); return 1; }
    if (coll_is_gate(a)) { *gate = 1; return 1; }
    if (tag) return 0;
    actor_row(a, row, sizeof row);
    return row[0] && (coll_has_named_tag(a, "ME_StaticMesh") || coll_has_named_tag(a, "ME_Blueprint"));
}

/* ---- record table (actor pointer -> record index, linear probing) ---- */
/* Empty slots are -1. The array is static (all zeros at injection), so it is only valid after
   coll_rebuild(); coll_table_ready guards every user. A zeroed table looks completely full: an
   unguarded probe then never ends and freezes the game thread (seen live: fresh DLL + persisted
   rule, first apply from the background poll). Every probe loop is bounded as well. */
/* ---- COLL_TABLE begin (host-tested: tests/test_scenery.py) ---- */
static int coll_slot(uint8_t *a) { return (int)((((uint64_t)a >> 4) * 11400714819323198485ull) >> 40) % (COLL_REC_CAP * 2 + 1); }
static int coll_table_put(uint8_t *a, int index) {
    int s = coll_slot(a), guard;
    for (guard = 0; guard < COLL_REC_CAP * 2 + 1; guard++, s = (s + 1) % (COLL_REC_CAP * 2 + 1))
        if (coll_table[s] < 0) { coll_table[s] = index; return 1; }
    return 0;
}
static void coll_rebuild(void) {
    int i;
    for (i = 0; i < COLL_REC_CAP * 2 + 1; i++) coll_table[i] = -1;
    coll_table_ready = 1;
    for (i = 0; i < coll_nrec; i++) coll_table_put(coll_recs[i].actor, i); /* nrec <= CAP < table size: always fits */
}
/* ---- COLL_TABLE end ---- */
static struct coll_rec *coll_find(uint8_t *a) {
    int s, guard = 0;
    if (!coll_nrec || !coll_table_ready) return 0;
    for (s = coll_slot(a); coll_table[s] >= 0 && guard++ < COLL_REC_CAP * 2 + 1; s = (s + 1) % (COLL_REC_CAP * 2 + 1)) {
        struct coll_rec *r = &coll_recs[coll_table[s]];
        if (r->actor == a && coll_rec_ident(r) == COLL_SAME) return r; /* stale reused pointer: keep probing */
    }
    return 0;
}
/* drops records that were cleared (actor=0) or whose actor is gone / was replaced
   (nothing of ours left on it; counted for status) */
static void coll_compact(void) {
    int i, kept = 0, id;
    for (i = 0; i < coll_nrec; i++) {
        struct coll_rec *r = &coll_recs[i];
        if (!r->actor || (id = coll_rec_ident(r)) != COLL_SAME) {
            if (r->actor) coll_lost(id);
            free(r->comps);
            continue;
        }
        coll_recs[kept++] = *r;
    }
    coll_nrec = kept;
    coll_rebuild();
}

static int coll_comp_ok(struct coll_comp *c) {
    int id = coll_ident(c->comp, &c->serial, c->name, c->cls);
    if (id == COLL_OTHER) coll_comp_replaced++;
    return id == COLL_SAME;
}

/* exact restore; returns 1 when everything recorded is back. Record is cleared on success. */
static int coll_restore(struct coll_rec *r) {
    int i, ch, ok = 1, id;
    char prof[64];
    if (!r->actor) return 1;
    if ((id = coll_rec_ident(r)) != COLL_SAME) { coll_lost(id); r->actor = 0; return 1; } /* not ours any more */
    for (i = 0; i < r->ncomp; i++) {
        struct coll_comp *c = &r->comps[i];
        if (!coll_comp_ok(c)) continue; /* component rebuilt by its Blueprint: nothing of ours left (counted) */
        if (r->mode == CM_FLY) {
            fname_str(c->profile, prof, sizeof prof);
            if (!ieq(prof, "Custom") && cf_profile(c->comp) != c->profile) cf_set_profile(c->comp, c->profile);
            if (cf_enabled(c->comp) != c->enabled) cf_set(c->comp, CF.set_en, CF.set_en_v, c->enabled, -1, 0);
            for (ch = 0; ch < 32; ch++)
                if ((r->mask >> ch & 1) && cf_resp(c->comp, ch) != c->resp[ch])
                    cf_set(c->comp, CF.set_resp, CF.set_resp_c, ch, CF.set_resp_v, c->resp[ch]);
            if (cf_enabled(c->comp) != c->enabled) ok = 0;
            for (ch = 0; ch < 32; ch++) if ((r->mask >> ch & 1) && cf_resp(c->comp, ch) != c->resp[ch]) ok = 0;
        }
        if (c->sim && !cf_sim(c->comp)) {
            cf_set(c->comp, CF.set_sim, CF.set_sim_v, 1, -1, 0);
            if (!cf_sim(c->comp)) ok = 0;
        }
    }
    if (r->mode == CM_OFF && cf_actor(r->actor) != r->actor_coll) {
        cf_set(r->actor, CF.a_set, CF.a_set_v, r->actor_coll, -1, 0);
        if (cf_actor(r->actor) != r->actor_coll) ok = 0;
    }
    if (ok) r->actor = 0;
    return ok;
}

/* records the originals, then applies mode. 1 = applied and verified, 0 = failed
   (record kept so restore still works), -1 = refused before any change. */
/* reuse: a slot of this actor that coll_restore just cleared (keeps the index table valid) */
static int coll_apply(uint8_t *a, int mode, uint32_t mask, int keepbody, int manual, struct coll_rec *reuse, char *why, int wcap) {
    uint8_t *comps[COLL_COMP_CAP];
    int n, trunc, i, ch, ok = 1;
    struct coll_rec *r;
    struct coll_comp *mem;
    if (!reuse && coll_nrec == COLL_REC_CAP) { snprintf(why, wcap, "record capacity %d reached", COLL_REC_CAP); return -1; }
    n = coll_components(a, comps, COLL_COMP_CAP, &trunc);
    if (trunc) { snprintf(why, wcap, "component list unavailable or over %d", COLL_COMP_CAP); return -1; }
    mem = n ? (struct coll_comp *)calloc((size_t)n, sizeof *mem) : 0;
    if (n && !mem) { snprintf(why, wcap, "out of memory"); return -1; }
    r = reuse ? reuse : &coll_recs[coll_nrec];
    if (reuse) free(reuse->comps);
    memset(r, 0, sizeof *r);
    r->comps = mem;
    r->actor = a; r->serial = coll_serial(a); r->name = rd64(a + UOBJ_NAME); r->cls = rdp(a + UOBJ_CLASS); r->ncomp = n; r->mode = (uint8_t)mode; r->manual = (uint8_t)manual;
    r->keepbody = (uint8_t)keepbody; r->mask = mode == CM_FLY ? mask : 0; r->actor_coll = (uint8_t)cf_actor(a);
    for (i = 0; i < n; i++) {
        struct coll_comp *c = &r->comps[i];
        c->comp = comps[i]; c->serial = coll_serial(comps[i]); c->name = rd64(comps[i] + UOBJ_NAME);
        c->cls = rdp(comps[i] + UOBJ_CLASS); c->enabled = (uint8_t)cf_enabled(comps[i]);
        c->sim = (uint8_t)cf_sim(comps[i]); c->profile = cf_profile(comps[i]);
        for (ch = 0; ch < 32; ch++) if (r->mask >> ch & 1) c->resp[ch] = (uint8_t)cf_resp(comps[i], ch);
    }
    /* Retain originals BEFORE any setter, including a possible fault. */
    if (!reuse) {
        coll_nrec++;
        if (!coll_table_ready || !coll_table_put(a, coll_nrec - 1)) coll_rebuild(); /* rebuild indexes every record */
    }
    if (mode == CM_ON) return 1; /* override pin: rules must not touch this actor */
    for (i = 0; i < n; i++) {
        struct coll_comp *c = &r->comps[i];
        if (c->sim) { cf_set(c->comp, CF.set_sim, CF.set_sim_v, 0, -1, 0); if (cf_sim(c->comp)) ok = 0; }
        if (mode != CM_FLY) continue;
        for (ch = 0; ch < 32; ch++) if (r->mask >> ch & 1) {
            cf_set(c->comp, CF.set_resp, CF.set_resp_c, ch, CF.set_resp_v, 0);
            if (cf_resp(c->comp, ch) != 0) ok = 0;
        }
        if (!keepbody && (c->enabled == 2 || c->enabled == 3)) {
            int want = c->enabled == 3 ? 1 : 0;
            cf_set(c->comp, CF.set_en, CF.set_en_v, want, -1, 0);
            if (cf_enabled(c->comp) != want) ok = 0;
        }
    }
    if (mode == CM_OFF) { cf_set(a, CF.a_set, CF.a_set_v, 0, -1, 0); if (cf_actor(a)) ok = 0; }
    if (!ok) snprintf(why, wcap, "engine did not accept every setter");
    return ok;
}

/* ---- iteration over placed actors in all loaded levels ---- */
struct coll_target { uint8_t *actor; char row[64], group[100]; float loc[3]; int gate; };
static struct coll_target coll_targets[COLL_TARGET_CAP];

static uint64_t coll_fingerprint(void) {
    uint8_t *levels[ENV_LEVEL_CAP];
    int trunc = 0, nl = env_levels(levels, ENV_LEVEL_CAP, &trunc), i, j, n;
    uint64_t h = 1469598103934665603ull, sum = 0;
    for (i = 0; i < nl; i++) {
        uint8_t **actors = env_level_actors(levels[i], &n);
        h = (h ^ (uint64_t)levels[i]) * 1099511628211ull;
        h = (h ^ (uint64_t)n) * 1099511628211ull;
        for (j = 0; j < n; j++) sum += (uint64_t)actors[j] * 0x9E3779B97F4A7C15ull;
    }
    return h ^ sum ^ (uint64_t)trunc;
}

static void coll_resolve_gate_class(void) {
    if (!coll_gate_class || !valid_object(coll_gate_class)) coll_gate_class = find_struct("BP_Gate_C");
}

/* collects every placed actor (needloc: also its location). Returns count, -1 on overflow. */
static int coll_collect(int needloc) {
    uint8_t *levels[ENV_LEVEL_CAP];
    int trunc = 0, nl = env_levels(levels, ENV_LEVEL_CAP, &trunc), i, j, n, count = 0;
    uint64_t tag = editor_tag();
    coll_resolve_gate_class();
    for (i = 0; i < nl; i++) {
        uint8_t **actors = env_level_actors(levels[i], &n);
        for (j = 0; j < n; j++) {
            struct coll_target *t; int gate;
            if (!coll_placed(actors[j], tag, &gate)) continue;
            if (count == COLL_TARGET_CAP) return -1;
            t = &coll_targets[count];
            t->actor = actors[j]; t->gate = gate;
            actor_row(t->actor, t->row, sizeof t->row);
            actor_group(t->actor, t->group, sizeof t->group);
            if (!needloc || !coll_loc(t->actor, t->loc)) t->loc[0] = t->loc[1] = t->loc[2] = NAN;
            count++;
        }
    }
    return count;
}
/* ---- selection (verb selectors; also "collision refresh <sel>" adoption) ---- */
enum { SEL_NONE, SEL_ALL, SEL_ACTOR, SEL_GROUP, SEL_ROW, SEL_BOX, SEL_LIST };
struct coll_sel { int kind, nset; char key[100]; float lo[3], hi[3]; };
struct coll_opts { int preview, force, keepbody, limit, brief; uint32_t mask; const char *out; };

#define COLL_SET 16384
static uint8_t *coll_set_store[COLL_SET];
static uint8_t coll_set_seen[COLL_SET];
static int coll_set_slot(uint8_t *a) { return (int)(((((uint64_t)a >> 4) * 11400714819323198485ull) >> 50) % COLL_SET); }
static int coll_set_add(struct coll_sel *s, uint8_t *a) {
    int k = coll_set_slot(a), guard = 0;
    while (coll_set_store[k] && coll_set_store[k] != a) { if (++guard >= COLL_SET) return 0; k = (k + 1) % COLL_SET; }
    if (coll_set_store[k] == a) return 1;
    if (s->nset >= COLL_SET / 2) return 0;
    coll_set_store[k] = a; s->nset++;
    return 1;
}
static int coll_set_has(uint8_t *a) {
    int k = coll_set_slot(a), guard = 0;
    while (coll_set_store[k] && guard++ < COLL_SET) { if (coll_set_store[k] == a) { coll_set_seen[k] = 1; return 1; } k = (k + 1) % COLL_SET; }
    return 0;
}

static int coll_selected(const struct coll_sel *s, const struct coll_target *t) {
    int i;
    switch (s->kind) {
    case SEL_ALL: return 1;
    case SEL_ACTOR: case SEL_LIST: return coll_set_has(t->actor);
    case SEL_GROUP: return ieq(t->group, s->key);
    case SEL_ROW: return ieq(t->row, s->key);
    case SEL_BOX:
        for (i = 0; i < 3; i++) if (!(t->loc[i] >= s->lo[i] && t->loc[i] <= s->hi[i])) return 0;
        return 1;
    }
    return 0;
}

/* options and at most one selector from a->v[first..] */
static int coll_parse(const struct args *a, int first, struct coll_sel *sel, struct coll_opts *o, char *err, int ecap) {
    int i, k;
    memset(sel, 0, sizeof *sel);
    memset(o, 0, sizeof *o);
    memset(coll_set_store, 0, sizeof coll_set_store);
    memset(coll_set_seen, 0, sizeof coll_set_seen);
    o->mask = COLL_DEFAULT_MASK; o->limit = 50;
    for (i = first; i < a->n; i++) {
        const char *v = a->v[i];
        if (ieq(v, "preview") || ieq(v, "dry")) o->preview = 1;
        else if (ieq(v, "force")) o->force = 1;
        else if (ieq(v, "keepbody")) o->keepbody = 1;
        else if (ieq(v, "brief")) o->brief = 1;
        else if (!strncmp(v, "limit=", 6)) o->limit = atoi(v + 6);
        else if (!strncmp(v, "out=", 4)) o->out = v + 4;
        else if (!strncmp(v, "channels=", 9)) {
            if (!coll_channels_parse(v + 9, &o->mask)) { snprintf(err, ecap, "invalid channels %s", v + 9); return 0; }
        } else if (sel->kind) { snprintf(err, ecap, "one selector only (%s)", v); return 0; }
        else if (ieq(v, "all")) sel->kind = SEL_ALL;
        else if (!strncmp(v, "actor=", 6)) {
            uint8_t *p = (uint8_t *)(uintptr_t)strtoull(v + 6, 0, 16);
            sel->kind = SEL_ACTOR;
            if (!p || !coll_set_add(sel, p)) { snprintf(err, ecap, "bad actor"); return 0; }
        } else if (!strncmp(v, "group=", 6) || !strncmp(v, "row=", 4)) {
            sel->kind = v[0] == 'g' ? SEL_GROUP : SEL_ROW;
            snprintf(sel->key, sizeof sel->key, "%s", strchr(v, '=') + 1);
            if (!coll_name_ok(sel->key)) { snprintf(err, ecap, "invalid group/row name"); return 0; }
        } else if (!strncmp(v, "box=", 4)) {
            sel->kind = SEL_BOX;
            if (sscanf(v + 4, "%f,%f,%f,%f,%f,%f", &sel->lo[0], &sel->lo[1], &sel->lo[2], &sel->hi[0], &sel->hi[1], &sel->hi[2]) != 6) {
                snprintf(err, ecap, "box needs x0,y0,z0,x1,y1,z1 (cm)"); return 0;
            }
            for (k = 0; k < 3; k++) if (!isfinite(sel->lo[k]) || !isfinite(sel->hi[k]) || sel->lo[k] > sel->hi[k]) { snprintf(err, ecap, "invalid box"); return 0; }
        } else if (!strncmp(v, "list=", 5)) {
            char line[64];
            FILE *f = fopen(v + 5, "r");
            sel->kind = SEL_LIST;
            if (!f) { snprintf(err, ecap, "cannot read actor list"); return 0; }
            while (fgets(line, sizeof line, f)) {
                uint8_t *p;
                if (!line[strspn(line, " \t\r\n")]) continue;
                p = (uint8_t *)(uintptr_t)strtoull(line, 0, 16);
                if (!p || !coll_set_add(sel, p)) { fclose(f); snprintf(err, ecap, "invalid actor list (max %d)", COLL_SET / 2); return 0; }
            }
            fclose(f);
            if (!sel->nset) { snprintf(err, ecap, "empty actor list"); return 0; }
        } else { snprintf(err, ecap, "unexpected argument %s", v); return 0; }
    }
    if (o->limit < 0 || o->limit > 5000) o->limit = 50;
    return 1;
}

/* ---- editor pawn with a negative cache ----
   Without a BP_MapEditor_Pawn_C (test flight, menu) cached_instance rescans the
   whole GUObjectArray on every call; the 1 s poll asks at most every 5 s then.
   Explicit commands clear the cache. */
static DWORD coll_nopawn_at;
static int coll_nopawn;
static uint8_t *coll_pawn(void) {
    uint8_t *p;
    if (coll_nopawn && GetTickCount() - coll_nopawn_at < 5000) return 0;
    p = editor_pawn();
    coll_nopawn = !p;
    if (!p) coll_nopawn_at = GetTickCount();
    return p;
}

/* ---- rules: config file and re-application ---- */
static int coll_load(const char *path, const char *base, int *n) {
    char line[512], saved[128];
    FILE *f = fopen(path, "r");
    int bad = 0;
    *n = 0;
    if (!f) return 0;
    if (!fgets(line, sizeof line, f) || sscanf(line, "UNCRASHED_COLLISION_V1 %127s", saved) != 1 || strcmp(saved, base)) bad = 1;
    while (!bad && fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0]) continue;
        if (*n == COLL_RULE_CAP || !coll_rule_parse(line, &coll_pending[*n])) { bad = 1; break; }
        (*n)++;
    }
    if (ferror(f)) bad = 1;
    fclose(f);
    return !bad;
}

/* Rules own every matching actor without a manual record. A manual record is
   handed to the rules only when adopted: adopt_all (map-wide) or adopt (selected). */
static int coll_apply_rules(int adopt_all, const struct coll_sel *adopt) {
    int needloc = adopt && adopt->kind == SEL_BOX, n, i, failed = 0;
    char why[120];
    for (i = 0; i < coll_nrules && !needloc; i++) needloc = coll_rules[i].kind == 'O';
    for (i = 0; i < coll_nrules; i++) coll_rules[i].hits = 0;
    if (!coll_nrules) { /* nothing to match: only hand back rule-owned records */
        for (i = 0; i < coll_nrec; i++) if (coll_recs[i].actor && !coll_recs[i].manual && !coll_restore(&coll_recs[i])) failed++;
        coll_compact();
        coll_failures = failed;
        if (failed) { snprintf(coll_error, sizeof coll_error, "%d actors could not be restored; originals retained", failed); return 0; }
        return 1;
    }
    n = coll_collect(needloc);
    if (n < 0) { snprintf(coll_error, sizeof coll_error, "over %d placed actors; rules not applied", COLL_TARGET_CAP); return 0; }
    for (i = 0; i < n; i++) {
        struct coll_target *t = &coll_targets[i];
        struct coll_rec *r = coll_find(t->actor);
        int k = coll_match(coll_rules, coll_buckets, t->row, t->group, isfinite(t->loc[0]) ? t->loc : 0, t->gate);
        const struct coll_rule *rule = k >= 0 ? &coll_rules[k] : 0;
        if (rule) coll_rules[k].hits++;
        if (r && r->manual && !(rule && (adopt_all || (adopt && coll_selected(adopt, t))))) continue;
        if (!rule || rule->mode == CM_ON) {
            if (r && !coll_restore(r)) failed++;
            continue;
        }
        if (r && r->mode == rule->mode && r->mask == (rule->mode == CM_FLY ? rule->mask : 0) && r->keepbody == rule->keepbody) {
            r->manual = 0;
            continue;
        }
        if (r && !coll_restore(r)) { failed++; continue; }
        if (coll_apply(t->actor, rule->mode, rule->mask, rule->keepbody, 0, r, why, sizeof why) != 1) failed++;
    }
    coll_compact();
    coll_failures = failed;
    if (failed) { snprintf(coll_error, sizeof coll_error, "%d actors could not be set/restored; originals retained", failed); return 0; }
    return 1;
}

/* force: re-read even if nothing changed. Rules are (re)applied only while the
   editor pawn exists (base and map directory come from it / the game instance);
   without it (test flight, menu) every record stays exactly as it is. */
static int coll_refresh(int force, int adopt_all, const struct coll_sel *adopt) {
    char directory[1024], path[1200], base[128], session[160];
    WIN32_FILE_ATTRIBUTE_DATA attrs;
    uint64_t stamp = 0, fp;
    int n = 0, changed;
    if (!coll_init(coll_error, sizeof coll_error)) return 0;
    if (!coll_pawn()) {
        coll_kept_unknown = 1;
        if (!force) return 1;
        snprintf(coll_error, sizeof coll_error, "no editor pawn (test flight or menu): rules are only (re)applied in the editor; state kept");
        return 0;
    }
    coll_kept_unknown = 0;
    map_directory(directory, sizeof directory); current_base(base, sizeof base); managed_session(session, sizeof session);
    snprintf(path, sizeof path, "%s\\UncrashedMCP.collision", directory);
    if (directory[0] && GetFileAttributesExA(path, GetFileExInfoStandard, &attrs))
        stamp = ((uint64_t)attrs.ftLastWriteTime.dwHighDateTime << 32) | attrs.ftLastWriteTime.dwLowDateTime;
    fp = coll_fingerprint();
    changed = strcmp(session, coll_seen_session) != 0;
    if (changed) { /* never carry old-map rules into a different map, even when its file is invalid */
        coll_nrules = 0; coll_config = 0; coll_base_checked = 0;
        coll_index(coll_rules, 0, coll_buckets);
        snprintf(coll_seen_session, sizeof coll_seen_session, "%s", session);
        coll_seen_stamp = coll_seen_fp = 0;
    }
    if (!force && !changed && stamp == coll_seen_stamp && fp == coll_seen_fp) return 1;
    if (!force && !changed && stamp && stamp == coll_bad_stamp && !strcmp(session, coll_bad_session)) return 0; /* already reported */
    if (!base[0] || ieq(base, "None") || (stamp && !coll_load(path, base, &n))) {
        int unknown = !base[0] || ieq(base, "None");
        coll_bad_stamp = stamp;
        snprintf(coll_bad_session, sizeof coll_bad_session, "%s", session);
        if (changed) coll_apply_rules(0, 0); /* zero rules: hands back rule-owned records */
        if (unknown) snprintf(coll_error, sizeof coll_error, "terrain base unknown; collision rules not applied");
        else snprintf(coll_error, sizeof coll_error, "invalid collision file or base mismatch (%s); %s", path,
                      changed ? "no rules active on this map" : "existing state unchanged");
        return 0;
    }
    memcpy(coll_rules, coll_pending, (size_t)n * sizeof coll_pending[0]);
    coll_nrules = n; coll_config = stamp != 0; coll_base_checked = 1; coll_error[0] = 0;
    coll_index(coll_rules, coll_nrules, coll_buckets);
    coll_seen_stamp = stamp; coll_seen_fp = fp;
    return coll_apply_rules(adopt_all, adopt);
}

static int coll_restore_all(void) {
    int i, failed = 0;
    for (i = 0; i < coll_nrec; i++) if (coll_recs[i].actor && !coll_restore(&coll_recs[i])) failed++;
    coll_compact();
    if (failed) snprintf(coll_error, sizeof coll_error, "%d actors could not be restored; originals retained", failed);
    return !failed;
}

/* ~1 s resident poll (agent.c wndproc). Always returns 1; problems go to the log once. */
static uint64_t collision_tick(void *unused) {
    static char logged[sizeof coll_error];
    (void)unused;
    flush_readable_cache();
    if (!coll_refresh(0, 0, 0) && coll_error[0] && strcmp(logged, coll_error)) {
        logf_("collision: %s", coll_error);
        snprintf(logged, sizeof logged, "%s", coll_error);
    }
    return 1;
}

/* ---- verb ---- */
/* every line goes to the out file (if any); the reply shows the first `limit` */
static void coll_line(FILE *f, int show, const char *fmt, ...) {
    char buf[700];
    va_list ap;
    va_start(ap, fmt); vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    if (f) fprintf(f, "%s\n", buf);
    if (show) outf("%s", buf);
}

/* channels (of mask) this component blocks: Block response and some collision enabled */
static uint32_t coll_comp_blocks(uint8_t *c, uint32_t mask) {
    uint32_t b = 0;
    int ch;
    if (!cf_enabled(c)) return 0;
    for (ch = 0; ch < 32; ch++) if ((mask >> ch & 1) && cf_resp(c, ch) == 2) b |= 1u << ch;
    return b;
}

static int coll_verb_status(const struct args *a) {
    int manual = 0, unmatched = 0, i, k;
    coll_compact();
    for (i = 0; i < coll_nrec; i++) manual += coll_recs[i].manual;
    for (i = 0; i < coll_nrules; i++) unmatched += !coll_rules[i].hits;
    outf("collision rules=%d records=%d manual=%d unmatched_rules=%d config=%d base_checked=%d editor=%d failures=%d "
         "gone=%d replaced=%d components_replaced=%d editor_absent_kept=%d",
         coll_nrules, coll_nrec, manual, unmatched, coll_config, coll_base_checked, editor_pawn() != 0, coll_failures,
         coll_gone, coll_replaced, coll_comp_replaced, coll_kept_unknown);
    for (i = 0, k = 0; a->n > 2 && ieq(a->v[2], "unmatched") && i < coll_nrules && k < 200; i++) {
        struct coll_rule *r = &coll_rules[i];
        if (r->hits) continue;
        k++;
        if (r->kind == 'O') outf("unmatched %d O %s %s %.1f,%.1f,%.1f", i, coll_mode_names[r->mode], r->key, r->loc[0], r->loc[1], r->loc[2]);
        else outf("unmatched %d %c %s %s", i, r->kind, coll_mode_names[r->mode], r->key);
    }
    if (coll_replaced || coll_comp_replaced)
        outf("note replaced = recorded object/component no longer the same UObject (destroyed or rebuilt); nothing of ours left on it");
    if (coll_error[0]) outf("warning %s", coll_error);
    return 1;
}

/* which object channel moving pawns (drone, vehicles, editor camera) use: run while flying */
static int coll_verb_channels(void) {
    uint8_t *levels[ENV_LEVEL_CAP];
    int trunc = 0, nl = env_levels(levels, ENV_LEVEL_CAP, &trunc), i, j, n, count = 0;
    for (i = 0; i < nl; i++) {
        uint8_t **actors = env_level_actors(levels[i], &n);
        for (j = 0; j < n; j++) {
            uint8_t *p = actors[j], *root;
            char cls[128], prof[64];
            if (!p || !alive(p) || !is_a(p, "Pawn")) continue;
            root = root_component(p); obj_class_name(p, cls, sizeof cls); count++;
            if (!root || !alive(root) || !coll_is_prim(root)) { outf("pawn %llx class=%s root=none", (unsigned long long)p, cls); continue; }
            fname_str(cf_profile(root), prof, sizeof prof);
            outf("pawn %llx class=%s object=%d enabled=%d profile=%s sim=%d actor_collision=%d vs_worldstatic=%d vs_worlddynamic=%d",
                 (unsigned long long)p, cls, cf_get(root, CF.get_obj, CF.get_obj_r, -1, 0), cf_enabled(root), prof,
                 cf_sim(root), cf_actor(p), cf_resp(root, 0), cf_resp(root, 1));
        }
    }
    outf("collision pawns=%d truncated=%d channels: 0 WorldStatic 1 WorldDynamic 2 Pawn 3 Visibility 4 Camera 5 PhysicsBody 6 Vehicle 7 Destructible 8-13 engine1-6 14-31 game1-18", count, trunc);
    return 1;
}

#define COLL_C_FMT "C %llx row=%s group=%s loc=%.2f,%.2f,%.2f gate=%d mode=%s owner=%s rmask=%x kb=%d"
#define COLL_C_ARGS(t, m1, r) (unsigned long long)(t)->actor, (t)->row[0] ? (t)->row : "-", (t)->group, (t)->loc[0], (t)->loc[1], (t)->loc[2], \
    (t)->gate, m1, (r) ? ((r)->manual ? "manual" : "rule") : "-", (r) ? (unsigned)(r)->mask : 0u, (r) ? (int)(r)->keepbody : 0

static int verb_collision(const struct args *a, char *err, int ecap) {
    const char *sub = a->n > 1 ? a->v[1] : "";
    struct coll_sel sel;
    struct coll_opts o;
    int mode = CM_ON, query = 0, physics = 0, first = 2, n, i, k;
    int matched = 0, gates = 0, changed = 0, unchanged = 0, failed = 0, bound = 0, shown = 0, blocking = 0, sim = 0, off = 0;
    FILE *out = 0;
    char why[120];
    coll_nopawn = 0; /* explicit command: look for the editor pawn again */
    if (!coll_init(err, ecap)) return 0;
    if (ieq(sub, "status")) return coll_verb_status(a);
    if (ieq(sub, "channels")) return coll_verb_channels();
    if (ieq(sub, "refresh")) {
        /* refresh: re-read + apply, manual records kept | refresh all: rules adopt every manual record they match |
           refresh <sel>: only the selected manual records are adopted */
        int unmatched = 0, all;
        if (!coll_parse(a, 2, &sel, &o, err, ecap)) return 0;
        all = sel.kind == SEL_ALL;
        if (!coll_refresh(1, all, sel.kind && !all ? &sel : 0)) { snprintf(err, ecap, "%s", coll_error); return 0; }
        for (k = 0; k < coll_nrules; k++) unmatched += !coll_rules[k].hits;
        outf("collision refreshed adopt=%s rules=%d records=%d unmatched_rules=%d config=%d base_checked=%d",
             all ? "all" : sel.kind ? "selected" : "none", coll_nrules, coll_nrec, unmatched, coll_config, coll_base_checked);
        return 1;
    }
    if (ieq(sub, "query")) query = 1;
    else if (ieq(sub, "physics") && a->n > 2 && (ieq(a->v[2], "off") || ieq(a->v[2], "on"))) {
        physics = 1; mode = ieq(a->v[2], "off") ? CM_NOPHYS : CM_ON; first = 3;
    } else if ((mode = coll_mode_parse(sub)) < 0) {
        snprintf(err, ecap, "usage: collision status [unmatched]|refresh [all|<sel>]|channels | query|on|off|flythrough|nophysics <sel> | physics off|on <sel> "
                            "[preview] [force] [keepbody] [brief] [channels=2,5,6] [limit=n] [out=file]; sel: actor=hex group=tag row=row box=x0,y0,z0,x1,y1,z1 list=file all");
        return 0;
    }
    if (!coll_parse(a, first, &sel, &o, err, ecap)) return 0;
    if (!sel.kind) { snprintf(err, ecap, "a selector is required: actor= group= row= box= list= or all"); return 0; }
    if (sel.kind == SEL_ALL && !query && mode != CM_ON) { snprintf(err, ecap, "'all' is only allowed for query and on"); return 0; }
    n = coll_collect(1);
    if (n < 0) { snprintf(err, ecap, "over %d placed actors", COLL_TARGET_CAP); return 0; }
    /* pass 1: selection and preconditions; nothing changes on failure */
    for (i = 0; i < n; i++) if (coll_selected(&sel, &coll_targets[i])) { matched++; gates += coll_targets[i].gate; }
    if (sel.kind == SEL_ACTOR || sel.kind == SEL_LIST) {
        int missing = 0;
        for (k = 0; k < COLL_SET; k++) missing += coll_set_store[k] && !coll_set_seen[k];
        if (missing) { snprintf(err, ecap, "%d listed actors are not placed editor objects in the loaded level; nothing changed", missing); return 0; }
        if (gates && !o.force && !query) { snprintf(err, ecap, "%d selected actors are race gates (checkpoints); pass force to change them", gates); return 0; }
    }
    if (!matched && sel.kind != SEL_ALL) { snprintf(err, ecap, "selector matched no placed editor objects"); return 0; }
    if (o.out && !(out = fopen(o.out, "w"))) { snprintf(err, ecap, "cannot write %s", o.out); return 0; }
    /* pass 2 */
    for (i = 0; i < n; i++) {
        struct coll_target *t = &coll_targets[i];
        struct coll_rec *r;
        const char *m1;
        int cur;
        if (!coll_selected(&sel, t)) continue;
        r = coll_find(t->actor);
        cur = r ? r->mode : CM_ON;
        m1 = coll_mode_names[cur];
        if (query && o.brief) { /* identity + ownership only, no component calls */
            int ac = cf_actor(t->actor);
            off += !ac;
            coll_line(out, shown++ < o.limit, COLL_C_FMT " actor_collision=%d", COLL_C_ARGS(t, m1, r), ac);
            continue;
        }
        if (query) {
            uint8_t *comps[COLL_COMP_CAP];
            uint32_t blocks = 0;
            int trunc, any_sim = 0, ac = cf_actor(t->actor), nc = coll_components(t->actor, comps, COLL_COMP_CAP, &trunc);
            for (k = 0; k < nc; k++) { if (ac) blocks |= coll_comp_blocks(comps[k], o.mask); any_sim |= cf_sim(comps[k]); }
            blocking += blocks != 0; sim += any_sim; off += !ac;
            coll_line(out, shown < o.limit, COLL_C_FMT " actor_collision=%d comps=%d truncated=%d sim=%d blocks=%x",
                      COLL_C_ARGS(t, m1, r), ac, nc, trunc, any_sim, (unsigned)blocks);
            for (k = 0; k < nc; k++) {
                char cls[96], prof[64];
                uint8_t *c = comps[k];
                obj_class_name(c, cls, sizeof cls);
                fname_str(cf_profile(c), prof, sizeof prof);
                coll_line(out, shown < o.limit, "P %llx comp=%llx class=%s enabled=%d object=%d profile=%s sim=%d r0=%d r1=%d r2=%d r3=%d r4=%d r5=%d r6=%d blocks=%x",
                          (unsigned long long)t->actor, (unsigned long long)c, cls, cf_enabled(c), cf_get(c, CF.get_obj, CF.get_obj_r, -1, 0), prof,
                          cf_sim(c), cf_resp(c, 0), cf_resp(c, 1), cf_resp(c, 2), cf_resp(c, 3), cf_resp(c, 4), cf_resp(c, 5), cf_resp(c, 6),
                          (unsigned)coll_comp_blocks(c, o.mask));
            }
            shown++;
            continue;
        }
        if (t->gate && !o.force) {
            coll_line(out, shown++ < o.limit, COLL_C_FMT " to=%s change=0 ok=1 skipped=gate", COLL_C_ARGS(t, m1, r), m1);
            continue;
        }
        {
            int target = mode, change, ok = 1;
            const char *note = "";
            char before[160];
            snprintf(before, sizeof before, "owner=%s rmask=%x kb=%d", r ? (r->manual ? "manual" : "rule") : "-",
                     r ? (unsigned)r->mask : 0u, r ? (int)r->keepbody : 0);
            if (physics && r && ((mode == CM_ON && r->mode != CM_NOPHYS) || (mode == CM_NOPHYS && r->mode > CM_NOPHYS))) {
                bound++; target = cur; note = " skipped=physics_part_of_mode";
                change = 0;
            } else
                change = target != cur || (target == CM_FLY && r && (r->mask != o.mask || r->keepbody != o.keepbody)) || (r && r->manual == 0);
            if (change && !o.preview) {
                int rule = coll_match(coll_rules, coll_buckets, t->row, t->group, isfinite(t->loc[0]) ? t->loc : 0, t->gate);
                int covered = rule >= 0 && coll_rules[rule].mode != CM_ON;
                if (r && !coll_restore(r)) { ok = 0; note = " restore_failed"; }
                else if (target != CM_ON || covered) {
                    /* "on" over a rule leaves a manual pin so the poll does not re-apply it */
                    int res = coll_apply(t->actor, target, o.mask, o.keepbody, 1, r, why, sizeof why);
                    if (res != 1) { ok = 0; note = res < 0 ? " refused" : " partial"; }
                }
            }
            if (!ok) failed++;
            else if (change) changed++;
            else unchanged++;
            coll_line(out, shown++ < o.limit, "C %llx row=%s group=%s loc=%.2f,%.2f,%.2f gate=%d mode=%s %s to=%s change=%d ok=%d%s",
                      (unsigned long long)t->actor, t->row[0] ? t->row : "-", t->group, t->loc[0], t->loc[1], t->loc[2], t->gate, m1,
                      before, coll_mode_names[target], change, ok, note);
        }
    }
    coll_compact();
    if (query)
        coll_line(out, 1, "collision query matched=%d gates=%d blocking=%d actor_collision_off=%d simulating=%d mask=%x brief=%d shown=%d",
                  matched, gates, blocking, off, sim, (unsigned)o.mask, o.brief, shown < o.limit ? shown : o.limit);
    else
        coll_line(out, 1, "collision mode=%s preview=%d matched=%d changed=%d unchanged=%d gates=%d gates_forced=%d failed=%d physics_bound=%d mask=%x keepbody=%d records=%d",
                  physics ? (mode == CM_ON ? "physics-on" : "physics-off") : coll_mode_names[mode], o.preview, matched, changed, unchanged, gates, o.force,
                  failed, bound, (unsigned)o.mask, o.keepbody, coll_nrec);
    if (out) {
        k = ferror(out);
        fclose(out);
        if (k) { snprintf(err, ecap, "write of %s failed", o.out); return 0; }
    }
    if (failed) { snprintf(err, ecap, "%d actors could not be changed; originals retained (collision on restores)", failed); return 0; }
    return 1;
}
#endif
