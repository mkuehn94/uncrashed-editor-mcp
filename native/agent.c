/* uncrashed_agent.dll - resident in-game agent for Uncrashed (UE 4.27, Win64 shipping).

   Everything the MCP server does to the open track editor goes through here, by
   calling the game's own UFunctions with ProcessEvent. Nothing is simulated with
   key presses.

   Surviving game updates
   ----------------------
   No game address is trusted as a constant:
     - FNamePool and GUObjectArray are found by scanning .data for their shape.
     - The GWorld slot is the .data pointer that holds a live UWorld.
     - UObject::ProcessEvent is the vtable slot next to ScriptCore's ProcessInternal
       that really computes KismetMathLibrary::Add_IntInt(2, 3) == 5.
     - Every Blueprint/game field (MapEditorTag, LoadedMapDirectory, TrackLap, ...)
       and every engine UPROPERTY is located by name through reflection.
     - Actor transforms are read and written with K2_* UFunctions, so collision and
       rendering stay in sync.
   What remains fixed is the UE 4.27 core layout (UObject/UStruct/FProperty fields,
   ULevel::Actors, the FTransform layout). That changes only with an engine upgrade,
   and resolution reports "layout mismatch" if it ever does.

   Protocol
   --------
   The pump thread polls Z:\tmp\uncrashed-agent-cmd.txt every 50 ms. A new line
   "<seq> <verb> [args]" is executed on the game thread (the game window is
   subclassed; WM_NULL wakes it). The reply is written to
   Z:\tmp\uncrashed-agent-out.txt.tmp and renamed to ...-out.txt, so a reader that
   sees "<seq> " at the start has the complete reply. Its last line is
   "end <seq> ok|failed". A job runs on a private stack under a vectored exception
   handler; a fault is reported as "<seq> fault <code> rip=<addr>" instead of
   crashing the game.

   Units: centimetres and degrees (Unreal's own). The Python side converts.

   Editor verbs
     spawn <row> <x> <y> <z> [pitch yaw roll] [sx sy sz] [group=<tag>]   editor "Spawn Asset"
     gate <row> <x> <y> <z> [pitch yaw roll] [noui]         editor "SpawnTrackAsset" (+ Gate list)
     get <actor>                                       location, rotation, scale, row
     transform <actor> <x> <y> <z> <pitch> <yaw> <roll> <sx> <sy> <sz>   ('-' keeps a value)
     replace <actor> <row>                             same transform, other catalogue row
     destroy <actor> [force]
     destroyat <x> <y> <z> [tolerance] [count]
     destroybox <x0> <y0> <x1> <y1> [z0|-] [z1|-] [row|-]
     floating [gap] [limit] [x0 y0 x1 y1]              objects with nothing under them
     bounds <actor>                                    world AABB, transform, optional mesh OBB
     worldbounds placed|terrain|playable               loaded extents (playable requires explicit tag)
     inventory <file>                                 placed-object bounds for read-only audits
     audit <request> <output> <mode> [spacing gap depth canopies]  environment.h diagnostics
     actors <x> <y> <z> <radius> [limit] [all]         placed editor actors near a point
     clear                                             destroy every placed object and gate
     base <row>                                        editor "LoadLevel": switch base level
     setmap <folder|absolute directory>                editor's current map folder (for Save)
     terrain x y z_from z_to                           bare landscape, ignoring foliage / structures
     laps <n>
     save [folder|-] [title...]                        editor "SaveMap"
     mapinfo
     camera get|release|set x y z pitch yaw [hold]     native camera (holds expire after 120s)
     capture start w h x y z pitch yaw roll fov        independent render-target capture
     capture render|stop <token> | export <token> <directory> <filename>
     groups <file> | edit <request> <output>           snapshots and validated reversible edits
     contacts <request> <output> <gap_cm>              assembly contact facts
     anchors <request> <output> <gap_cm>               measured foundation anchors
     site <file> x0 y0 z0 x1 y1 z1 minheight_cm         recognised loaded base flora
     zones status|refresh                             portable exclusion intent / reversible masks
     batch <windows path> [quiet]                      one verb per line, all in this job
                                                       (quiet: only errors and the summary)
   Infrastructure
     ping | scan | resolve | unload
   Debug
     names <s> [n] | objects <s> [n] | funcs <s> [n] | func <addr> | props <struct>
     rows <s> [n] [table] | fname <text> | read <addr> <n> | write <addr> <hex>
     call <obj> <func> <hexparms>
*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <ctype.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CMD_FILE "Z:\\tmp\\uncrashed-agent-cmd.txt"
#define OUT_FILE "Z:\\tmp\\uncrashed-agent-out.txt"
#define LOG_FILE "Z:\\tmp\\uncrashed-agent.log"
#define GAME_MODULE L"Uncrashed-Win64-Shipping.exe"
#define JOB_STACK_BYTES 0x400000u

/* Hints from the build this was written against. Used only when they validate. */
#ifndef HINT_GWORLD_RVA
#define HINT_GWORLD_RVA 0x4FBD3D0u
#endif
#ifndef HINT_PROCESS_EVENT_RVA
#define HINT_PROCESS_EVENT_RVA 0x1507cf0u
#endif

/* ---- UE 4.27 core layout (engine-version constants) ---- */
#define UOBJ_INDEX 0x0C
#define UOBJ_CLASS 0x10
#define UOBJ_NAME 0x18
#define UOBJ_OUTER 0x20
#define UFIELD_NEXT 0x28
#define USTRUCT_SUPER 0x40
#define USTRUCT_CHILDREN 0x48
#define USTRUCT_CHILDPROPS 0x50
#define USTRUCT_SIZE 0x58
#define UFUNC_FLAGS 0xB0
#define UFUNC_NUMPARMS 0xB4
#define UFUNC_PARMSSIZE 0xB6
#define UFUNC_FUNC 0xD8
#define UCLASS_CDO 0x118
#define FFIELD_CLASS 0x08
#define FFIELD_NEXT 0x20
#define FFIELD_NAME 0x28
#define FPROP_ELEMSIZE 0x3C
#define FPROP_FLAGS 0x40
#define FPROP_OFFSET 0x4C
#define FPROP_EXTRA 0x78
#define OBJITEM_SIZE 0x18
#define OBJITEM_FLAGS 0x08
#define LEVEL_ACTORS 0x98
#define DT_ROWMAP 0x30
#define ROWMAP_ELEM 0x18
#define FUNC_NATIVE 0x400u
#define CPF_PARM 0x80ull
#define CPF_OUTPARM 0x100ull
#define CPF_REFPARM 0x8000000ull
#define OBJFLAG_DEAD ((1u << 29) | (1u << 28)) /* PendingKill | Unreachable */

/* ======================================================================
   fault guard: guarded(fn, arg) runs fn(arg) on a private stack
   ====================================================================== */

DWORD64 resume_rip, resume_rsp, resume_rbp;
volatile int tripped, armed;
volatile DWORD64 fault_rip;
volatile DWORD fault_code;
uint8_t *job_hi;

static LONG CALLBACK veh(EXCEPTION_POINTERS *xp) {
    DWORD code = xp->ExceptionRecord->ExceptionCode;
    if (!armed) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    if (code == 0xC0000005 || code == 0xC000001D || code == 0xC0000094 || code == 0xC0000096 ||
        code == 0xC0000409 || code == 0x80000003 || code == 0xC00000FD) {
        if (tripped) {
            armed = 0;
            return EXCEPTION_CONTINUE_SEARCH;
        }
        tripped = 1;
        armed = 0;
        fault_code = code;
        fault_rip = (DWORD64)xp->ExceptionRecord->ExceptionAddress;
        xp->ContextRecord->Rip = resume_rip;
        xp->ContextRecord->Rsp = resume_rsp;
        xp->ContextRecord->Rbp = resume_rbp;
        xp->ContextRecord->Rax = 0;
        return EXCEPTION_CONTINUE_EXECUTION;
    }
    return EXCEPTION_CONTINUE_SEARCH;
}

__asm__(
    ".text\n"
    ".globl guarded\n"
    "guarded:\n\t"
    "push %rbp\n\t"
    "mov %rsp, %rbp\n\t"
    "push %rbx\n\t"
    "push %rsi\n\t"
    "push %rdi\n\t"
    "push %r12\n\t"
    "push %r13\n\t"
    "push %r14\n\t"
    "push %r15\n\t"
    "sub $8, %rsp\n\t"
    "mov %rsp, resume_rsp(%rip)\n\t"
    "mov %rbp, resume_rbp(%rip)\n\t"
    "lea 1f(%rip), %rax\n\t"
    "mov %rax, resume_rip(%rip)\n\t"
    "mov %rcx, %r13\n\t"
    "mov job_hi(%rip), %rax\n\t"
    "and $-16, %rax\n\t"
    "sub $0x40, %rax\n\t"
    "mov %rax, %rsp\n\t"
    "mov %rdx, %rcx\n\t"
    "movl $0, tripped(%rip)\n\t"
    "movq $0, fault_rip(%rip)\n\t"
    "movl $0, fault_code(%rip)\n\t"
    "movl $1, armed(%rip)\n\t"
    "call *%r13\n\t"
    "mov resume_rsp(%rip), %rsp\n\t"
    "1:\n\t"
    "movl $0, armed(%rip)\n\t"
    "add $8, %rsp\n\t"
    "pop %r15\n\t"
    "pop %r14\n\t"
    "pop %r13\n\t"
    "pop %r12\n\t"
    "pop %rdi\n\t"
    "pop %rsi\n\t"
    "pop %rbx\n\t"
    "pop %rbp\n\t"
    "ret\n");
extern uint64_t guarded(void *fn, void *arg);

/* ======================================================================
   output
   ====================================================================== */

static FILE *out_file;
static char out_prefix[32];

static void logf_(const char *fmt, ...) {
    FILE *f = fopen(LOG_FILE, "a");
    va_list ap;
    if (!f) {
        return;
    }
    va_start(ap, fmt);
    vfprintf(f, fmt, ap);
    va_end(ap);
    fputc('\n', f);
    fclose(f);
}

static int quiet_batch;

static void outf(const char *fmt, ...) {
    va_list ap;
    if (!out_file || (quiet_batch && strncmp(fmt, "error", 5) != 0)) {
        return;
    }
    if (out_prefix[0]) {
        fputs(out_prefix, out_file);
    }
    va_start(ap, fmt);
    vfprintf(out_file, fmt, ap);
    va_end(ap);
    fputc('\n', out_file);
}

/* ======================================================================
   memory
   ====================================================================== */

static uint8_t *game_base;
static uint64_t game_end;

#define RCACHE 64
static uint64_t rc_lo[RCACHE], rc_hi[RCACHE];
static int rc_next;

static void flush_readable_cache(void) {
    memset(rc_lo, 0, sizeof rc_lo);
    memset(rc_hi, 0, sizeof rc_hi);
}

/* VirtualQuery is a slow syscall under Wine; readable regions are cached per job. */
static int readable(const void *p, size_t n) {
    MEMORY_BASIC_INFORMATION mbi;
    uint64_t a = (uint64_t)p;
    int i;
    if (a < 0x10000 || a > 0x7FFFFFFFFFFFull) {
        return 0;
    }
    for (i = 0; i < RCACHE; i++) {
        if (a >= rc_lo[i] && a + n <= rc_hi[i]) {
            return 1;
        }
    }
    while (n) {
        size_t span;
        if (!VirtualQuery((void *)a, &mbi, sizeof mbi)) {
            return 0;
        }
        if (mbi.State != MEM_COMMIT || (mbi.Protect & (PAGE_GUARD | PAGE_NOACCESS)) ||
            !(mbi.Protect & (PAGE_READONLY | PAGE_READWRITE | PAGE_EXECUTE_READ | PAGE_EXECUTE_READWRITE |
                             PAGE_WRITECOPY | PAGE_EXECUTE_WRITECOPY))) {
            return 0;
        }
        rc_lo[rc_next] = (uint64_t)mbi.BaseAddress;
        rc_hi[rc_next] = (uint64_t)mbi.BaseAddress + mbi.RegionSize;
        rc_next = (rc_next + 1) % RCACHE;
        span = (size_t)((uint64_t)mbi.BaseAddress + mbi.RegionSize - a);
        if (span >= n) {
            return 1;
        }
        n -= span;
        a += span;
    }
    return 1;
}

static uint64_t rd64(const void *p) { uint64_t v; memcpy(&v, p, 8); return v; }
static int32_t rd32(const void *p) { int32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd16(const void *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static uint8_t *rdp(const void *p) { return (uint8_t *)rd64(p); }

static int in_game(uint64_t v) {
    return v >= (uint64_t)game_base && v < game_end;
}

static int pe_section(const char *name, uint8_t **start, uint8_t **end) {
    IMAGE_DOS_HEADER *dos = (IMAGE_DOS_HEADER *)game_base;
    IMAGE_NT_HEADERS64 *nt = (IMAGE_NT_HEADERS64 *)(game_base + dos->e_lfanew);
    IMAGE_SECTION_HEADER *sec = IMAGE_FIRST_SECTION(nt);
    int i;
    for (i = 0; i < nt->FileHeader.NumberOfSections; i++) {
        if (strncmp((const char *)sec[i].Name, name, 8) == 0) {
            *start = game_base + sec[i].VirtualAddress;
            *end = *start + sec[i].Misc.VirtualSize;
            return 1;
        }
    }
    return 0;
}

/* ======================================================================
   strings
   ====================================================================== */

static int ieq(const char *a, const char *b) {
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return 0;
        }
        a++;
        b++;
    }
    return *a == *b;
}

static int icontains(const char *hay, const char *needle) {
    size_t n = strlen(needle);
    if (!n) {
        return 1;
    }
    for (; *hay; hay++) {
        size_t i;
        for (i = 0; i < n; i++) {
            if (!hay[i] || tolower((unsigned char)hay[i]) != tolower((unsigned char)needle[i])) {
                break;
            }
        }
        if (i == n) {
            return 1;
        }
    }
    return 0;
}

/* ======================================================================
   FNamePool
   ====================================================================== */

static uint8_t **g_name_blocks; /* FNamePool::Blocks */
static uint32_t *g_name_cursor; /* CurrentBlock, CurrentByteCursor */

static int name_entry(uint32_t index, char *buf, int cap) {
    uint32_t block = index >> 16, off = (index & 0xFFFF) * 2;
    uint8_t *e;
    uint16_t h;
    int len, wide, i;
    if (!g_name_blocks || block > 8191 || !readable(g_name_blocks[block] + off, 2)) {
        return -1;
    }
    e = g_name_blocks[block] + off;
    h = rd16(e);
    len = h >> 6;
    wide = h & 1;
    if (len <= 0 || len > 1023 || !readable(e + 2, (size_t)len * (wide ? 2 : 1))) {
        return -1;
    }
    if (len >= cap) {
        len = cap - 1;
    }
    for (i = 0; i < len; i++) {
        buf[i] = wide ? (char)(rd16(e + 2 + i * 2) & 0x7F) : (char)e[2 + i];
    }
    buf[len] = 0;
    return len;
}

static void fname_str(uint64_t fname, char *buf, int cap) {
    uint32_t idx = (uint32_t)fname, num = (uint32_t)(fname >> 32);
    int n = name_entry(idx, buf, cap);
    if (n < 0) {
        snprintf(buf, cap, "<bad-name-%x>", idx);
        return;
    }
    if (num) {
        snprintf(buf + n, cap - n, "_%u", num - 1);
    }
}

typedef int (*name_cb)(uint32_t index, const char *text, void *ctx);

static void walk_names(name_cb cb, void *ctx) {
    uint32_t cur_block = g_name_cursor[0], cur_cursor = g_name_cursor[1], b;
    char text[1100];
    for (b = 0; b <= cur_block && b < 8192; b++) {
        uint8_t *blk = g_name_blocks[b];
        uint32_t end = (b == cur_block) ? cur_cursor : 0x20000u, off = 0;
        if (!readable(blk, 2)) {
            break;
        }
        while (off + 2 <= end) {
            uint16_t h = rd16(blk + off);
            int len = h >> 6, wide = h & 1;
            uint32_t size;
            if (!len) {
                break;
            }
            size = (2 + (uint32_t)len * (wide ? 2 : 1) + 1) & ~1u;
            if (off + size > end) {
                break;
            }
            if (name_entry((b << 16) | (off / 2), text, sizeof text) > 0 && cb((b << 16) | (off / 2), text, ctx)) {
                return;
            }
            off += size;
        }
    }
}

struct find_name_ctx {
    const char *want;
    uint32_t index;
    int found;
};

static int find_name_cb(uint32_t index, const char *text, void *p) {
    struct find_name_ctx *c = (struct find_name_ctx *)p;
    if (ieq(text, c->want)) {
        c->index = index;
        c->found = 1;
        return 1;
    }
    return 0;
}

/* FName for an existing name, with Unreal's "_N" number split. 0 if not interned. */
static int make_fname(const char *text, uint64_t *out) {
    size_t n = strlen(text), i = n;
    struct find_name_ctx c;
    while (i > 0 && isdigit((unsigned char)text[i - 1])) {
        i--;
    }
    if (i > 1 && i < n && text[i - 1] == '_' && (text[i] != '0' || n - i == 1) && n - i < 10) {
        char base[1024];
        memcpy(base, text, i - 1);
        base[i - 1] = 0;
        c.want = base;
        c.found = 0;
        walk_names(find_name_cb, &c);
        if (c.found) {
            *out = (uint64_t)c.index | ((uint64_t)(strtoul(text + i, NULL, 10) + 1) << 32);
            return 1;
        }
    }
    c.want = text;
    c.found = 0;
    walk_names(find_name_cb, &c);
    if (c.found) {
        *out = c.index;
    }
    return c.found;
}

/* ======================================================================
   GUObjectArray
   ====================================================================== */

static uint8_t ***g_objects; /* FChunkedFixedUObjectArray::Objects */
static int32_t *g_objnum;

static int32_t object_count(void) {
    return g_objnum ? *g_objnum : 0;
}

static uint8_t *object_item(int32_t i) {
    uint8_t **chunks = (uint8_t **)g_objects[0];
    uint8_t *chunk;
    if (i < 0 || !readable(chunks + (i >> 16), 8)) {
        return 0;
    }
    chunk = chunks[i >> 16];
    if (!readable(chunk + (i & 0xFFFF) * OBJITEM_SIZE, OBJITEM_SIZE)) {
        return 0;
    }
    return chunk + (i & 0xFFFF) * OBJITEM_SIZE;
}

static uint8_t *object_at(int32_t i) {
    uint8_t *item = object_item(i);
    return item ? rdp(item) : 0;
}

static int valid_object(uint8_t *o) {
    return readable(o, 0x28) && readable(rdp(o + UOBJ_CLASS), 0x28);
}

/* valid, and not pending kill or unreachable */
static int alive(uint8_t *o) {
    uint8_t *item;
    if (!valid_object(o)) {
        return 0;
    }
    item = object_item(rd32(o + UOBJ_INDEX));
    return item && rdp(item) == o && !((uint32_t)rd32(item + OBJITEM_FLAGS) & OBJFLAG_DEAD);
}

static void obj_name(uint8_t *o, char *buf, int cap) {
    if (!o || !readable(o, 0x28)) {
        snprintf(buf, cap, "null");
        return;
    }
    fname_str(rd64(o + UOBJ_NAME), buf, cap);
}

static void obj_class_name(uint8_t *o, char *buf, int cap) {
    obj_name(o ? rdp(o + UOBJ_CLASS) : 0, buf, cap);
}

static void obj_path(uint8_t *o, char *buf, int cap) {
    uint8_t *chain[32];
    int n = 0, i, used = 0;
    char part[1100];
    while (o && n < 32 && readable(o, 0x28)) {
        chain[n++] = o;
        o = rdp(o + UOBJ_OUTER);
    }
    buf[0] = 0;
    for (i = n - 1; i >= 0 && used < cap - 1; i--) {
        obj_name(chain[i], part, sizeof part);
        used += snprintf(buf + used, cap - used, "%s%s", i == n - 1 ? "" : (i == n - 2 ? "." : ":"), part);
    }
}

static int is_a(uint8_t *o, const char *classname) {
    uint8_t *cls = rdp(o + UOBJ_CLASS);
    char nm[256];
    int depth = 0;
    while (cls && readable(cls, 0x60) && depth++ < 48) {
        obj_name(cls, nm, sizeof nm);
        if (ieq(nm, classname)) {
            return 1;
        }
        cls = rdp(cls + USTRUCT_SUPER);
    }
    return 0;
}

/* ======================================================================
   reflection
   ====================================================================== */

#define SCACHE 96
static char sc_name[SCACHE][96];
static uint8_t *sc_ptr[SCACHE];
static int sc_next;

/* UStruct/UClass/UScriptStruct by short name. Types live for the process, so cached. */
static uint8_t *find_struct(const char *name) {
    int32_t n = object_count(), i;
    char nm[256];
    for (i = 0; i < SCACHE; i++) {
        if (sc_ptr[i] && ieq(sc_name[i], name)) {
            return sc_ptr[i];
        }
    }
    for (i = 0; i < n; i++) {
        uint8_t *o = object_at(i);
        if (!o || !valid_object(o)) {
            continue;
        }
        obj_name(o, nm, sizeof nm);
        if (!ieq(nm, name) || !is_a(o, "Struct")) {
            continue;
        }
        snprintf(sc_name[sc_next], sizeof sc_name[0], "%s", name);
        sc_ptr[sc_next] = o;
        sc_next = (sc_next + 1) % SCACHE;
        return o;
    }
    return 0;
}

static uint8_t *find_object(const char *name, const char *classname) {
    int32_t n = object_count(), i;
    char nm[256], cn[256];
    for (i = 0; i < n; i++) {
        uint8_t *o = object_at(i);
        if (!o || !valid_object(o)) {
            continue;
        }
        obj_name(o, nm, sizeof nm);
        if (!ieq(nm, name)) {
            continue;
        }
        obj_class_name(o, cn, sizeof cn);
        if (!classname || ieq(cn, classname)) {
            return o;
        }
    }
    return 0;
}

/* first live non-CDO instance of exactly classname */
static uint8_t *find_instance(const char *classname) {
    int32_t n = object_count(), i;
    char nm[256], cn[256];
    for (i = 0; i < n; i++) {
        uint8_t *o = object_at(i);
        if (!o || !valid_object(o)) {
            continue;
        }
        obj_class_name(o, cn, sizeof cn);
        if (!ieq(cn, classname)) {
            continue;
        }
        obj_name(o, nm, sizeof nm);
        if (strncmp(nm, "Default__", 9) == 0 || !alive(o)) {
            continue;
        }
        return o;
    }
    return 0;
}

static uint8_t *class_cdo(const char *classname) {
    uint8_t *cls = find_struct(classname), *cdo;
    if (!cls) {
        return 0;
    }
    cdo = rdp(cls + UCLASS_CDO);
    return valid_object(cdo) ? cdo : 0;
}

static uint8_t *find_function(uint8_t *cls, const char *fname) {
    char nm[256];
    int depth = 0;
    while (cls && readable(cls, 0x60) && depth++ < 48) {
        uint8_t *f = rdp(cls + USTRUCT_CHILDREN);
        int guard = 0;
        while (f && readable(f, 0x30) && guard++ < 8000) {
            obj_name(f, nm, sizeof nm);
            if (ieq(nm, fname)) {
                return f;
            }
            f = rdp(f + UFIELD_NEXT);
        }
        cls = rdp(cls + USTRUCT_SUPER);
    }
    return 0;
}

static uint8_t *object_function(uint8_t *obj, const char *fname) {
    return obj ? find_function(rdp(obj + UOBJ_CLASS), fname) : 0;
}

/* FProperty named `name` on struct s (supers included), or 0 */
static uint8_t *find_property(uint8_t *s, const char *name) {
    char pname[256];
    int depth = 0;
    while (s && readable(s, 0x60) && depth++ < 48) {
        uint8_t *p = rdp(s + USTRUCT_CHILDPROPS);
        int guard = 0;
        while (p && readable(p, 0x80) && guard++ < 4000) {
            fname_str(rd64(p + FFIELD_NAME), pname, sizeof pname);
            if (ieq(pname, name)) {
                return p;
            }
            p = rdp(p + FFIELD_NEXT);
        }
        s = rdp(s + USTRUCT_SUPER);
    }
    return 0;
}

static int prop_offset(uint8_t *s, const char *name) {
    uint8_t *p = find_property(s, name);
    return p ? rd32(p + FPROP_OFFSET) : -1;
}

/* offset of a field on an object's class, or -1 */
static int field_offset(uint8_t *obj, const char *name) {
    return obj ? prop_offset(rdp(obj + UOBJ_CLASS), name) : -1;
}

static void prop_type(uint8_t *p, char *cname, int ccap, char *inner, int icap) {
    uint8_t *fclass = rdp(p + FFIELD_CLASS), *in;
    inner[0] = 0;
    if (readable(fclass, 8)) {
        fname_str(rd64(fclass), cname, ccap);
    } else {
        snprintf(cname, ccap, "?");
    }
    in = readable(p + FPROP_EXTRA, 8) ? rdp(p + FPROP_EXTRA) : 0;
    if (in && valid_object(in)) {
        obj_name(in, inner, icap);
    }
}

/* ======================================================================
   resolution - once per process, re-checked before every job
   ====================================================================== */

typedef void (*pe_fn)(void *, void *, void *);

static uint8_t **g_world_slot;
static int g_pe_slot = -1;
static uint64_t g_pe_bad[16];
static int g_pe_bad_n;
static uint64_t g_pe_testing;
static int g_layout_ok;
static int g_world_level_off = 0x30;
static char g_scan_error[128];

static int looks_like_block0(uint8_t *p) {
    if (!readable(p, 32)) {
        return 0;
    }
    return rd16(p) >> 6 == 4 && !(rd16(p) & 1) && memcmp(p + 2, "None", 4) == 0 && rd16(p + 6) >> 6 == 12 &&
           memcmp(p + 8, "ByteProperty", 12) == 0;
}

static int resolve_names(uint8_t *ds, uint8_t *de) {
    uint8_t *p;
    for (p = ds; p + 8 <= de; p += 8) {
        if (looks_like_block0(rdp(p))) {
            g_name_blocks = (uint8_t **)p;
            g_name_cursor = (uint32_t *)(p - 8);
            return 1;
        }
    }
    return 0;
}

/* FChunkedFixedUObjectArray: every sampled item's object carries its own index */
static int resolve_objects(uint8_t *ds, uint8_t *de) {
    static const int32_t samples[] = {0, 1, 2, 10, 100, 1000, 5000};
    uint8_t *p;
    for (p = ds; p + 0x20 <= de; p += 8) {
        uint8_t **chunks = (uint8_t **)rd64(p);
        int32_t num = rd32(p + 0x14), maxc = rd32(p + 0x18);
        unsigned k;
        int ok = 1;
        if (!chunks || num < 6000 || num > 50000000 || maxc < 1 || maxc > 8192 || !readable(chunks, 8)) {
            continue;
        }
        for (k = 0; k < sizeof samples / sizeof samples[0] && ok; k++) {
            int32_t i = samples[k];
            uint8_t *chunk = chunks[i >> 16], *obj;
            if (!readable(chunk + (i & 0xFFFF) * OBJITEM_SIZE, 8)) {
                ok = 0;
                break;
            }
            obj = rdp(chunk + (i & 0xFFFF) * OBJITEM_SIZE);
            ok = readable(obj, 0x28) && rd32(obj + UOBJ_INDEX) == i;
        }
        if (ok) {
            g_objects = (uint8_t ***)p;
            g_objnum = (int32_t *)(p + 0x14);
            return 1;
        }
    }
    return 0;
}

static int is_live_world(uint8_t *w) {
    char cn[64], nm[64];
    if (!w || !valid_object(w)) {
        return 0;
    }
    obj_class_name(w, cn, sizeof cn);
    obj_name(w, nm, sizeof nm);
    return ieq(cn, "World") && strncmp(nm, "Default__", 9) != 0;
}

static int resolve_world(uint8_t *ds, uint8_t *de) {
    uint8_t *p, *world_cls = find_struct("World");
    if (world_cls && prop_offset(world_cls, "PersistentLevel") > 0) {
        g_world_level_off = prop_offset(world_cls, "PersistentLevel");
    }
    if (is_live_world(rdp(game_base + HINT_GWORLD_RVA))) {
        g_world_slot = (uint8_t **)(game_base + HINT_GWORLD_RVA);
        return 1;
    }
    for (p = ds; p + 8 <= de; p += 8) {
        uint8_t *w = rdp(p);
        if (is_live_world(w) && readable(rdp(w + g_world_level_off), LEVEL_ACTORS + 0x10)) {
            g_world_slot = (uint8_t **)p;
            return 1;
        }
    }
    return 0;
}

static uint8_t *current_world(void) {
    uint8_t *w = g_world_slot ? *g_world_slot : 0;
    return is_live_world(w) ? w : 0;
}

static uint8_t *current_level(void) {
    uint8_t *w = current_world();
    uint8_t *l = w ? rdp(w + g_world_level_off) : 0;
    return readable(l, LEVEL_ACTORS + 0x10) ? l : 0;
}

/* the native thunk every Blueprint function runs through */
static uint64_t process_internal(void) {
    int32_t n = object_count(), i;
    char cn[64];
    for (i = 0; i < n; i++) {
        uint8_t *o = object_at(i);
        if (!o || !valid_object(o)) {
            continue;
        }
        obj_class_name(o, cn, sizeof cn);
        if (ieq(cn, "Function") && !((uint32_t)rd32(o + UFUNC_FLAGS) & FUNC_NATIVE) && in_game(rd64(o + UFUNC_FUNC))) {
            return rd64(o + UFUNC_FUNC);
        }
    }
    return 0;
}

static int pe_bad(uint64_t a) {
    int i;
    for (i = 0; i < g_pe_bad_n; i++) {
        if (g_pe_bad[i] == a) {
            return 1;
        }
    }
    return 0;
}

static int test_pe_slot(int slot) {
    uint8_t *cdo = class_cdo("KismetMathLibrary");
    uint8_t *fn = cdo ? find_function(rdp(cdo + UOBJ_CLASS), "Add_IntInt") : 0;
    uint8_t parms[0x20];
    int oa, ob, orv;
    if (!fn) {
        return 0;
    }
    oa = prop_offset(fn, "A");
    ob = prop_offset(fn, "B");
    orv = prop_offset(fn, "ReturnValue");
    if (oa < 0 || ob < 0 || orv < 0 || orv + 4 > (int)sizeof parms) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    parms[oa] = 2;
    parms[ob] = 3;
    ((pe_fn)(((uint64_t *)rd64(cdo))[slot]))(cdo, fn, parms);
    return rd32(parms + orv) == 5;
}

static uint64_t pe_distance(uint64_t a, uint64_t pi, uint64_t hint) {
    return a == hint ? 0 : (a > pi ? a - pi : pi - a);
}

/* A candidate that faults is recorded by ensure_resolved (g_pe_testing) and skipped. */
static int resolve_process_event(void) {
    uint8_t *obj_cdo = class_cdo("Object");
    uint64_t *vt, pi, hint = (uint64_t)(game_base + HINT_PROCESS_EVENT_RVA);
    int order[160], n = 0, i, j;
    if (!obj_cdo) {
        return 0;
    }
    vt = (uint64_t *)rd64(obj_cdo);
    pi = process_internal();
    if (!readable(vt, 160 * 8) || !pi) {
        return 0;
    }
    for (i = 0; i < 160; i++) {
        if (vt[i] == hint || (in_game(vt[i]) && pe_distance(vt[i], pi, 0) < 0x8000)) {
            order[n++] = i;
        }
    }
    for (i = 0; i < n; i++) {
        for (j = i + 1; j < n; j++) {
            if (pe_distance(vt[order[j]], pi, hint) < pe_distance(vt[order[i]], pi, hint)) {
                int t = order[i];
                order[i] = order[j];
                order[j] = t;
            }
        }
    }
    for (i = 0; i < n && i < 8; i++) {
        if (pe_bad(vt[order[i]])) {
            continue;
        }
        g_pe_testing = vt[order[i]];
        if (test_pe_slot(order[i])) {
            g_pe_slot = order[i];
            g_pe_testing = 0;
            return 1;
        }
        if (g_pe_bad_n < 16) {
            g_pe_bad[g_pe_bad_n++] = vt[order[i]];
        }
    }
    g_pe_testing = 0;
    return 0;
}

static int resolved(void) {
    return g_name_blocks && g_objects && g_layout_ok && current_world() && g_pe_slot >= 0;
}

static uint64_t do_resolve(void *unused) {
    uint8_t *ds, *de;
    (void)unused;
    g_scan_error[0] = 0;
    if (!pe_section(".data", &ds, &de)) {
        snprintf(g_scan_error, sizeof g_scan_error, "no .data section");
        return 0;
    }
    if (!g_name_blocks && !resolve_names(ds, de)) {
        snprintf(g_scan_error, sizeof g_scan_error, "FNamePool not found");
        return 0;
    }
    if (!g_objects && !resolve_objects(ds, de)) {
        snprintf(g_scan_error, sizeof g_scan_error, "GUObjectArray not found");
        return 0;
    }
    if (!g_layout_ok) {
        /* a correct CDO pointer, name and reflected field prove the 4.27 layout */
        uint8_t *actor = find_struct("Actor");
        char nm[64] = "";
        if (actor && valid_object(rdp(actor + UCLASS_CDO))) {
            obj_name(rdp(actor + UCLASS_CDO), nm, sizeof nm);
        }
        if (strcmp(nm, "Default__Actor") != 0 || prop_offset(actor, "RootComponent") <= 0) {
            snprintf(g_scan_error, sizeof g_scan_error, "layout mismatch (not UE 4.27?)");
            return 0;
        }
        g_layout_ok = 1;
    }
    if (!current_world() && !resolve_world(ds, de)) {
        snprintf(g_scan_error, sizeof g_scan_error, "GWorld not found");
        return 0;
    }
    if (g_pe_slot < 0 && !resolve_process_event()) {
        snprintf(g_scan_error, sizeof g_scan_error, "ProcessEvent not found");
        return 0;
    }
    return 1;
}

static void reset_resolution(void) {
    g_name_blocks = 0;
    g_objects = 0;
    g_objnum = 0;
    g_layout_ok = 0;
    g_world_slot = 0;
    g_pe_slot = -1;
    g_pe_bad_n = 0;
    memset(sc_ptr, 0, sizeof sc_ptr);
}

/* Resolve under the guard; a ProcessEvent candidate that crashes is marked bad. */
static int ensure_resolved(void) {
    int attempt;
    if (resolved()) {
        return 1;
    }
    for (attempt = 0; attempt < 10; attempt++) {
        uint64_t ok = guarded((void *)do_resolve, 0);
        if (!tripped) {
            return (int)ok;
        }
        logf_("resolve faulted %lx at %llx (candidate %llx)", (unsigned long)fault_code,
              (unsigned long long)fault_rip, (unsigned long long)g_pe_testing);
        if (!g_pe_testing || g_pe_bad_n >= 16) {
            snprintf(g_scan_error, sizeof g_scan_error, "fault during resolution");
            return 0;
        }
        g_pe_bad[g_pe_bad_n++] = g_pe_testing;
        g_pe_testing = 0;
    }
    return 0;
}

/* ======================================================================
   calling UFunctions
   ====================================================================== */

static void call(uint8_t *obj, uint8_t *fn, void *parms) {
    ((pe_fn)(((uint64_t *)rd64(obj))[g_pe_slot]))(obj, fn, parms);
}

/* offset of the parameter (CPF_Parm) named `name`, or -1 */
static int param_offset(uint8_t *fn, const char *name) {
    uint8_t *p = find_property(fn, name);
    return (p && (rd64(p + FPROP_FLAGS) & CPF_PARM)) ? rd32(p + FPROP_OFFSET) : -1;
}

struct fstr {
    uint16_t *data;
    int32_t num;
    int32_t max;
};

static void fstring_to_utf8(const struct fstr *s, char *out, int cap) {
    int i, n = 0;
    out[0] = 0;
    if (!s->data || s->num <= 0 || s->num > 65536 || !readable(s->data, (size_t)s->num * 2)) {
        return;
    }
    for (i = 0; i < s->num && n < cap - 1 && s->data[i]; i++) {
        out[n++] = s->data[i] < 0x80 ? (char)s->data[i] : '?';
    }
    out[n] = 0;
}

/* An FString on the engine heap: KismetStringLibrary::Concat_StrStr(text, ""). */
static int engine_fstring(const char *text, struct fstr *out) {
    static uint16_t wide[2048];
    uint8_t parms[0x80];
    uint8_t *cdo = class_cdo("KismetStringLibrary");
    uint8_t *fn = object_function(cdo, "Concat_StrStr");
    struct fstr a;
    int i, n = (int)strlen(text), oa, orv;
    if (!fn || n >= 2047) {
        return 0;
    }
    oa = param_offset(fn, "A");
    orv = param_offset(fn, "ReturnValue");
    if (oa < 0 || orv < 0 || orv + 16 > (int)sizeof parms) {
        return 0;
    }
    for (i = 0; i < n; i++) {
        wide[i] = (uint8_t)text[i];
    }
    wide[n] = 0;
    a.data = wide;
    a.num = n + 1;
    a.max = n + 1;
    memset(parms, 0, sizeof parms);
    memcpy(parms + oa, &a, sizeof a);
    call(cdo, fn, parms);
    memcpy(out, parms + orv, sizeof *out);
    return out->data != 0 || n == 0;
}

/* An FText (24 bytes, engine-owned) through KismetTextLibrary::Conv_StringToText. */
static int engine_ftext(const char *text, uint8_t out[24]) {
    uint8_t parms[0x80];
    uint8_t *cdo = class_cdo("KismetTextLibrary");
    uint8_t *fn = object_function(cdo, "Conv_StringToText");
    struct fstr s;
    int oi, orv;
    if (!fn) {
        return 0;
    }
    oi = param_offset(fn, "InString");
    orv = param_offset(fn, "ReturnValue");
    if (oi < 0 || orv < 0 || orv + 24 > (int)sizeof parms || !engine_fstring(text, &s)) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    memcpy(parms + oi, &s, sizeof s);
    call(cdo, fn, parms);
    memcpy(out, parms + orv, 24);
    return rd64(out) != 0;
}

/* An FName for any text (created if new) through KismetStringLibrary::Conv_StringToName. */
static int engine_fname(const char *text, uint64_t *out) {
    uint8_t parms[0x40];
    uint8_t *cdo = class_cdo("KismetStringLibrary");
    uint8_t *fn = object_function(cdo, "Conv_StringToName");
    struct fstr s;
    int oi, orv;
    if (!fn) {
        return 0;
    }
    oi = param_offset(fn, "InString");
    orv = param_offset(fn, "ReturnValue");
    if (oi < 0 || orv < 0 || orv + 8 > (int)sizeof parms || !engine_fstring(text, &s)) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    memcpy(parms + oi, &s, sizeof s);
    call(cdo, fn, parms);
    *out = rd64(parms + orv);
    return 1;
}

/* ======================================================================
   game objects (cached, validated on use)
   ====================================================================== */

static uint8_t *c_pawn, *c_gi, *c_gm;

static uint8_t *cached_instance(uint8_t **cache, const char *classname) {
    char cn[128];
    if (*cache && alive(*cache)) {
        obj_class_name(*cache, cn, sizeof cn);
        if (ieq(cn, classname)) {
            return *cache;
        }
    }
    *cache = find_instance(classname);
    return *cache;
}

static uint8_t *editor_pawn(void) { return cached_instance(&c_pawn, "BP_MapEditor_Pawn_C"); }
static uint8_t *game_instance(void) { return cached_instance(&c_gi, "BP_GameInstanceFPV_C"); }
static uint8_t *editor_game_mode(void) { return cached_instance(&c_gm, "GM_MapEditor_C"); }

static uint64_t editor_tag(void) {
    uint8_t *pawn = editor_pawn();
    int off = field_offset(pawn, "MapEditorTag");
    return off > 0 ? rd64(pawn + off) : 0;
}

/* ---- data tables ---- */

struct row {
    uint64_t fname;
    uint8_t *data;
};

static uint8_t *data_table(const char *name) {
    static uint8_t *cache[8];
    static char names[8][64];
    int i;
    for (i = 0; i < 8; i++) {
        if (cache[i] && ieq(names[i], name) && valid_object(cache[i])) {
            return cache[i];
        }
    }
    for (i = 0; i < 8; i++) {
        if (!cache[i] || ieq(names[i], name)) {
            cache[i] = find_object(name, "DataTable");
            snprintf(names[i], sizeof names[i], "%s", name);
            return cache[i];
        }
    }
    return find_object(name, "DataTable");
}

static int table_row(const char *table, const char *rowname, struct row *out) {
    uint8_t *dt = data_table(table), *elems;
    int32_t num, i;
    char nm[300];
    if (!dt) {
        return 0;
    }
    elems = rdp(dt + DT_ROWMAP);
    num = rd32(dt + DT_ROWMAP + 8);
    if (num <= 0 || num > 100000 || !readable(elems, (size_t)num * ROWMAP_ELEM)) {
        return 0;
    }
    for (i = 0; i < num; i++) {
        uint8_t *e = elems + (size_t)i * ROWMAP_ELEM;
        if (!readable(rdp(e + 8), 8)) {
            continue;
        }
        fname_str(rd64(e), nm, sizeof nm);
        if (ieq(nm, rowname)) {
            out->fname = rd64(e);
            out->data = rdp(e + 8);
            return 1;
        }
    }
    return 0;
}

/* ---- actors ---- */

static uint8_t **level_actors(int32_t *count) {
    uint8_t *level = current_level();
    uint8_t **data;
    *count = 0;
    if (!level) {
        return 0;
    }
    data = (uint8_t **)rdp(level + LEVEL_ACTORS);
    *count = rd32(level + LEVEL_ACTORS + 8);
    if (*count <= 0 || *count > 500000 || !readable(data, (size_t)*count * 8)) {
        *count = 0;
        return 0;
    }
    return data;
}

static int is_level_actor(uint8_t *a) {
    int32_t n, i;
    uint8_t **actors = level_actors(&n);
    for (i = 0; i < n; i++) {
        if (actors[i] == a) {
            return 1;
        }
    }
    return 0;
}

static int actor_tag_list(uint8_t *a, uint64_t **data) {
    static int off = -2;
    int32_t n;
    if (off == -2) {
        off = prop_offset(find_struct("Actor"), "Tags");
    }
    if (off <= 0) {
        return 0;
    }
    *data = (uint64_t *)rdp(a + off);
    n = rd32(a + off + 8);
    return (n > 0 && n <= 64 && readable(*data, (size_t)n * 8)) ? n : 0;
}

static int actor_has_tag(uint8_t *a, uint64_t tag) {
    uint64_t *data;
    int n = actor_tag_list(a, &data), i;
    for (i = 0; tag && i < n; i++) {
        if ((uint32_t)data[i] == (uint32_t)tag) {
            return 1;
        }
    }
    return 0;
}

/* the catalogue row an editor actor was spawned from (its NewRow* tag) */
static void actor_row(uint8_t *a, char *out, int cap) {
    uint64_t *data;
    int n = actor_tag_list(a, &data), i;
    out[0] = 0;
    for (i = 0; i < n; i++) {
        char nm[128];
        fname_str(data[i], nm, sizeof nm);
        if (strncmp(nm, "NewRow", 6) == 0) {
            snprintf(out, cap, "%s", nm);
            return;
        }
    }
}

static void actor_tags(uint8_t *a, char *out, int cap) {
    uint64_t *data;
    int n = actor_tag_list(a, &data), i, used = 0;
    out[0] = 0;
    for (i = 0; i < n && used < cap - 1; i++) {
        char nm[128];
        fname_str(data[i], nm, sizeof nm);
        if (strcmp(nm, "None") != 0) {
            used += snprintf(out + used, cap - used, "%s%s", used ? "," : "", nm);
        }
    }
}

static int is_gate(uint8_t *a) {
    return is_a(a, "BP_Gate_C");
}

/* placed through the editor: carries its tag, or is a race gate */
static int is_editor_actor(uint8_t *a, uint64_t tag) {
    return alive(a) && (actor_has_tag(a, tag) || is_gate(a));
}

/* ---- transforms through UFunctions ---- */

struct xf {
    float loc[3];
    float rot[3]; /* pitch, yaw, roll */
    float scale[3];
};

static int vec_getter(uint8_t *a, const char *fname, float out[3]) {
    uint8_t parms[0x40];
    uint8_t *fn = object_function(a, fname);
    int orv;
    if (!fn || (orv = param_offset(fn, "ReturnValue")) < 0 || orv + 12 > (int)sizeof parms) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    call(a, fn, parms);
    memcpy(out, parms + orv, 12);
    return 1;
}

static int actor_get(uint8_t *a, struct xf *t) {
    return vec_getter(a, "K2_GetActorLocation", t->loc) && vec_getter(a, "K2_GetActorRotation", t->rot) &&
           vec_getter(a, "GetActorScale3D", t->scale);
}

static uint8_t *root_component(uint8_t *a) {
    static int off = -2;
    if (off == -2) {
        off = prop_offset(find_struct("Actor"), "RootComponent");
    }
    return off > 0 ? rdp(a + off) : 0;
}

static int component_mobility(uint8_t *comp) {
    int off = field_offset(comp, "Mobility");
    return off > 0 ? comp[off] : -1;
}

static void set_mobility(uint8_t *comp, int mobility) {
    uint8_t parms[0x10];
    uint8_t *fn = object_function(comp, "SetMobility");
    int o = fn ? param_offset(fn, "NewMobility") : -1;
    if (o < 0) {
        return;
    }
    memset(parms, 0, sizeof parms);
    parms[o] = (uint8_t)mobility;
    call(comp, fn, parms);
}

static int actor_set(uint8_t *a, const struct xf *t) {
    uint8_t parms[0x200];
    uint8_t *fn = object_function(a, "K2_SetActorLocationAndRotation");
    uint8_t *fs = object_function(a, "SetActorScale3D");
    uint8_t *root = root_component(a);
    int ol, orr, ot, osc;
    if (!fn || !fs || rd16(fn + UFUNC_PARMSSIZE) > (int)sizeof parms) {
        return 0;
    }
    ol = param_offset(fn, "NewLocation");
    orr = param_offset(fn, "NewRotation");
    ot = param_offset(fn, "bTeleport");
    osc = param_offset(fs, "NewScale3D");
    if (ol < 0 || orr < 0 || osc < 0) {
        return 0;
    }
    /* A Static root ignores runtime moves. The editor's own gizmo moves objects,
       so making a placed object Movable matches what the editor allows. */
    if (root && valid_object(root) && component_mobility(root) == 0) {
        set_mobility(root, 2);
    }
    memset(parms, 0, sizeof parms);
    memcpy(parms + ol, t->loc, 12);
    memcpy(parms + orr, t->rot, 12);
    if (ot >= 0) {
        parms[ot] = 1;
    }
    call(a, fn, parms);
    memset(parms, 0, sizeof parms);
    memcpy(parms + osc, t->scale, 12);
    call(a, fs, parms);
    return 1;
}

static void rotator_quat(const float rot[3], float q[4]) {
    double p = rot[0] * M_PI / 360.0, y = rot[1] * M_PI / 360.0, r = rot[2] * M_PI / 360.0;
    double sp = sin(p), cp = cos(p), sy = sin(y), cy = cos(y), sr = sin(r), cr = cos(r);
    q[0] = (float)(cr * sp * sy - sr * cp * cy);
    q[1] = (float)(-cr * sp * cy - sr * cp * sy);
    q[2] = (float)(cr * cp * sy - sr * sp * cy);
    q[3] = (float)(cr * cp * cy + sr * sp * sy);
}

/* FTransform (4.27, float): rotation quat, translation (+pad), scale (+pad) */
static void make_ftransform(const struct xf *t, float out[12]) {
    memset(out, 0, 48);
    rotator_quat(t->rot, out);
    memcpy(out + 4, t->loc, 12);
    memcpy(out + 8, t->scale, 12);
}

/* ======================================================================
   editor operations
   ====================================================================== */

/* Fill a Blueprint editor function's parameters by type: the first row-ish name,
   every transform, the row struct. Everything else (bools, group tags) stays zero.
   Returns the offset of the first out Actor parameter, or -1. */
static int fill_editor_params(uint8_t *fn, uint8_t *parms, int size, const struct row *row, const char *row_struct,
                              const float transform[12], const char *const *true_bools, uint64_t group) {
    uint8_t *p;
    int guard = 0, out_actor = -1, name_done = 0;
    char cname[64], inner[128], pname[128];
    for (p = rdp(fn + USTRUCT_CHILDPROPS); p && readable(p, 0x80) && guard++ < 400; p = rdp(p + FFIELD_NEXT)) {
        uint64_t flags = rd64(p + FPROP_FLAGS);
        int off = rd32(p + FPROP_OFFSET), esz = rd32(p + FPROP_ELEMSIZE);
        if (!(flags & CPF_PARM) || off < 0 || off + esz > size) {
            continue;
        }
        prop_type(p, cname, sizeof cname, inner, sizeof inner);
        fname_str(rd64(p + FFIELD_NAME), pname, sizeof pname);
        if ((flags & CPF_OUTPARM) && !(flags & CPF_REFPARM)) {
            if (out_actor < 0 && ieq(cname, "ObjectProperty")) {
                out_actor = off;
            }
            continue;
        }
        if (ieq(cname, "NameProperty") && icontains(pname, "group")) {
            memcpy(parms + off, &group, 8);
        } else if (ieq(cname, "NameProperty") && !name_done && (icontains(pname, "row") || icontains(pname, "id"))) {
            memcpy(parms + off, &row->fname, 8);
            name_done = 1;
        } else if (ieq(cname, "StructProperty") && ieq(inner, "Transform")) {
            memcpy(parms + off, transform, 48);
        } else if (ieq(cname, "StructProperty") && ieq(inner, row_struct)) {
            uint8_t *rs = find_struct(row_struct);
            int rsz = rs ? rd32(rs + USTRUCT_SIZE) : esz;
            memcpy(parms + off, row->data, rsz < esz ? rsz : esz);
        } else if (ieq(cname, "BoolProperty") && true_bools) {
            const char *const *b;
            for (b = true_bools; *b; b++) {
                if (ieq(pname, *b)) {
                    parms[off] = 1;
                }
            }
        }
    }
    return out_actor;
}

/* Spawn through a pawn function whose parameters are (row name, row struct, transform, ..., out Actor). */
static uint8_t *editor_place(const char *func, const char *table, const char *row_struct, const char *rowname,
                             const struct xf *t, const char *const *true_bools, uint64_t group, char *err,
                             int ecap) {
    static uint8_t parms[0x1000];
    uint8_t *pawn = editor_pawn(), *fn, *actor;
    struct row row;
    float tr[12];
    int size, out;
    if (!pawn) {
        snprintf(err, ecap, "no editor pawn (is the track editor open?)");
        return 0;
    }
    if (!table_row(table, rowname, &row)) {
        snprintf(err, ecap, "no row %s in %s", rowname, table);
        return 0;
    }
    fn = object_function(pawn, func);
    size = fn ? rd16(fn + UFUNC_PARMSSIZE) : 0;
    if (!fn || size > (int)sizeof parms) {
        snprintf(err, ecap, "editor function '%s' not found", func);
        return 0;
    }
    make_ftransform(t, tr);
    memset(parms, 0, sizeof parms);
    out = fill_editor_params(fn, parms, size, &row, row_struct, tr, true_bools, group);
    call(pawn, fn, parms);
    actor = out >= 0 ? rdp(parms + out) : 0;
    if (!actor || !alive(actor)) {
        snprintf(err, ecap, "the editor did not place %s", rowname);
        return 0;
    }
    return actor;
}

static uint8_t *editor_spawn(const char *rowname, const struct xf *t, uint64_t group, char *err, int ecap) {
    return editor_place("Spawn Asset", "DT_MapEditorAssets", "S_ME_Asset", rowname, t, 0, group, err, ecap);
}

/* "UI event?" makes the editor add the gate to its Gate list panel, as a click does */
static uint8_t *editor_gate(const char *rowname, const struct xf *t, int ui_event, char *err, int ecap) {
    static const char *const ui[] = {"UI event?", 0};
    return editor_place("SpawnTrackAsset", "DT_MapEditorGates", "S_ME_RaceGates", rowname, t, ui_event ? ui : 0, 0,
                        err, ecap);
}

static int destroy_actor(uint8_t *a) {
    uint8_t parms[0x10];
    uint8_t *fn = object_function(a, "K2_DestroyActor");
    if (!fn) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    call(a, fn, parms);
    return 1;
}

/* The editor's gate panel is rebuilt from the BP_Gate_C actors by
   UI_MapEditor::UpdateGateListUI(CheckIfRaceIsOK? = true). Called after any gate change. */
static int refresh_gate_ui(void) {
    uint8_t parms[0x10];
    uint8_t *pawn = editor_pawn(), *ui, *fn;
    int off = field_offset(pawn, "UI_MapEditor"), o;
    ui = off > 0 ? rdp(pawn + off) : 0;
    fn = ui && alive(ui) ? object_function(ui, "UpdateGateListUI") : 0;
    if (!fn) {
        return 0;
    }
    memset(parms, 0, sizeof parms);
    o = param_offset(fn, "CheckIfRaceIsOK?");
    if (o >= 0 && o < (int)sizeof parms) {
        parms[o] = 1;
    }
    call(ui, fn, parms);
    return 1;
}

static void print_actor(const char *verb, uint8_t *a, const char *extra) {
    struct xf t;
    char row[64], cls[128];
    obj_class_name(a, cls, sizeof cls);
    actor_row(a, row, sizeof row);
    if (actor_get(a, &t)) {
        outf("%s %llx row=%s class=%s loc=%.2f,%.2f,%.2f rot=%.3f,%.3f,%.3f scale=%.4f,%.4f,%.4f%s", verb,
             (unsigned long long)a, row[0] ? row : "-", cls, t.loc[0], t.loc[1], t.loc[2], t.rot[0], t.rot[1], t.rot[2],
             t.scale[0], t.scale[1], t.scale[2], extra ? extra : "");
    } else {
        outf("%s %llx row=%s class=%s%s", verb, (unsigned long long)a, row[0] ? row : "-", cls, extra ? extra : "");
    }
}

static uint8_t *resolve_actor(const char *hex, int force, char *err, int ecap) {
    uint8_t *a = (uint8_t *)strtoull(hex, NULL, 16);
    if (!a || !alive(a) || !is_level_actor(a)) {
        snprintf(err, ecap, "%s is not a live actor in the open level", hex);
        return 0;
    }
    if (!force && !is_editor_actor(a, editor_tag())) {
        snprintf(err, ecap, "%s is not a placed map object", hex);
        return 0;
    }
    return a;
}

/* ======================================================================
   verbs
   ====================================================================== */

struct args {
    int n;
    char *v[24];
};

static void split_args(char *s, struct args *a) {
    a->n = 0;
    while (*s && a->n < 24) {
        while (*s == ' ' || *s == '\t') {
            s++;
        }
        if (!*s) {
            break;
        }
        a->v[a->n++] = s;
        while (*s && *s != ' ' && *s != '\t') {
            s++;
        }
        if (*s) {
            *s++ = 0;
        }
    }
}

static float argf(const struct args *a, int i, float dflt) {
    return (i < a->n && strcmp(a->v[i], "-") != 0) ? (float)atof(a->v[i]) : dflt;
}

static void args_xf(const struct args *a, int first, struct xf *t) {
    int i;
    for (i = 0; i < 3; i++) {
        t->loc[i] = argf(a, first + i, 0);
        t->rot[i] = argf(a, first + 3 + i, 0);
        t->scale[i] = argf(a, first + 6 + i, 1);
    }
}

static uint8_t *g_last_spawned; /* for "drop last" right after a spawn in a batch */

static int verb_spawn(const struct args *a, char *err, int ecap) {
    struct xf t;
    uint8_t *actor;
    if (a->n < 5) {
        snprintf(err, ecap, "usage: spawn <row> <x> <y> <z> [pitch yaw roll] [sx sy sz]");
        return 0;
    }
    struct args nums = *a;
    uint64_t group = 0;
    int i;
    nums.n = 0;
    for (i = 0; i < a->n; i++) {
        if (strncmp(a->v[i], "group=", 6) == 0) {
            if (a->v[i][6] && !ieq(a->v[i] + 6, "None") && !engine_fname(a->v[i] + 6, &group)) {
                snprintf(err, ecap, "could not make group name %s", a->v[i] + 6);
                return 0;
            }
        } else {
            nums.v[nums.n++] = a->v[i];
        }
    }
    args_xf(&nums, 2, &t);
    if (t.scale[0] <= 0 || t.scale[1] <= 0 || t.scale[2] <= 0) {
        snprintf(err, ecap, "scale must be positive");
        return 0;
    }
    if (!(actor = editor_spawn(a->v[1], &t, group, err, ecap))) {
        return 0;
    }
    g_last_spawned = actor;
    print_actor("spawned", actor, 0);
    return 1;
}

static int verb_gate(const struct args *a, char *err, int ecap) {
    struct xf t;
    uint8_t *actor;
    if (a->n < 5) {
        snprintf(err, ecap, "usage: gate <row> <x> <y> <z> [pitch yaw roll] [noui]");
        return 0;
    }
    args_xf(a, 2, &t);
    t.scale[0] = t.scale[1] = t.scale[2] = 1;
    if (!(actor = editor_gate(a->v[1], &t, !(a->n > 8 && ieq(a->v[8], "noui")), err, ecap))) {
        return 0;
    }
    refresh_gate_ui();
    print_actor("gate", actor, 0);
    return 1;
}

static int verb_get(const struct args *a, char *err, int ecap) {
    uint8_t *actor;
    if (a->n < 2) {
        snprintf(err, ecap, "usage: get <actor>");
        return 0;
    }
    if (!(actor = resolve_actor(a->v[1], 1, err, ecap))) {
        return 0;
    }
    print_actor("actor", actor, 0);
    return 1;
}

static int verb_transform(const struct args *a, char *err, int ecap) {
    uint8_t *actor;
    struct xf t;
    int i;
    if (a->n < 2) {
        snprintf(err, ecap, "usage: transform <actor> <x> <y> <z> <pitch> <yaw> <roll> <sx> <sy> <sz> ('-' keeps)");
        return 0;
    }
    if (!(actor = resolve_actor(a->v[1], 0, err, ecap))) {
        return 0;
    }
    if (!actor_get(actor, &t)) {
        snprintf(err, ecap, "cannot read the transform");
        return 0;
    }
    for (i = 0; i < 3; i++) {
        t.loc[i] = argf(a, 2 + i, t.loc[i]);
        t.rot[i] = argf(a, 5 + i, t.rot[i]);
        t.scale[i] = argf(a, 8 + i, t.scale[i]);
        if (t.scale[i] <= 0) {
            snprintf(err, ecap, "scale must be positive");
            return 0;
        }
    }
    if (!actor_set(actor, &t)) {
        snprintf(err, ecap, "the actor setters were not found");
        return 0;
    }
    print_actor("moved", actor, 0);
    return 1;
}

static int verb_replace(const struct args *a, char *err, int ecap) {
    uint8_t *actor, *fresh;
    struct xf t;
    if (a->n < 3) {
        snprintf(err, ecap, "usage: replace <actor> <row>");
        return 0;
    }
    if (!(actor = resolve_actor(a->v[1], 0, err, ecap))) {
        return 0;
    }
    if (is_gate(actor)) {
        snprintf(err, ecap, "gates cannot be replaced by catalogue rows");
        return 0;
    }
    if (!actor_get(actor, &t)) {
        snprintf(err, ecap, "cannot read the transform");
        return 0;
    }
    /* spawn first: if the new row fails, the old object stays */
    if (!(fresh = editor_spawn(a->v[2], &t, 0, err, ecap))) {
        return 0;
    }
    destroy_actor(actor);
    print_actor("replaced", fresh, 0);
    return 1;
}

static int verb_destroy(const struct args *a, char *err, int ecap) {
    uint8_t *actor;
    char row[64];
    int gate;
    if (a->n < 2) {
        snprintf(err, ecap, "usage: destroy <actor> [force]");
        return 0;
    }
    if (!(actor = resolve_actor(a->v[1], a->n > 2 && ieq(a->v[2], "force"), err, ecap))) {
        return 0;
    }
    actor_row(actor, row, sizeof row);
    gate = is_gate(actor);
    destroy_actor(actor);
    if (gate) {
        refresh_gate_ui();
    }
    outf("destroyed %llx row=%s%s", (unsigned long long)actor, row[0] ? row : "-", gate ? " gate" : "");
    return 1;
}

static float dist3(const float p[3], float x, float y, float z) {
    return sqrtf((p[0] - x) * (p[0] - x) + (p[1] - y) * (p[1] - y) + (p[2] - z) * (p[2] - z));
}

static int verb_destroyat(const struct args *a, char *err, int ecap) {
    float x, y, z, tol;
    int count, done = 0, gates = 0;
    uint64_t tag = editor_tag();
    if (a->n < 4) {
        snprintf(err, ecap, "usage: destroyat <x> <y> <z> [tolerance] [count]");
        return 0;
    }
    x = argf(a, 1, 0);
    y = argf(a, 2, 0);
    z = argf(a, 3, 0);
    tol = argf(a, 4, 50);
    count = a->n > 5 ? atoi(a->v[5]) : 1;
    while (done < count) {
        int32_t n, i;
        uint8_t **actors = level_actors(&n), *best = 0;
        float best_d = tol, loc[3];
        char row[64];
        for (i = 0; i < n; i++) {
            float d;
            if (!actors[i] || !is_editor_actor(actors[i], tag) || !vec_getter(actors[i], "K2_GetActorLocation", loc)) {
                continue;
            }
            d = dist3(loc, x, y, z);
            if (d <= best_d) {
                best = actors[i];
                best_d = d;
            }
        }
        if (!best) {
            break;
        }
        actor_row(best, row, sizeof row);
        if (is_gate(best)) {
            gates++;
        }
        outf("destroyed %llx row=%s d=%.1f%s", (unsigned long long)best, row[0] ? row : "-", best_d,
             is_gate(best) ? " gate" : "");
        destroy_actor(best);
        done++;
    }
    if (gates) {
        refresh_gate_ui();
    }
    outf("count %d", done);
    return 1;
}

/* world bounds (centre, half extent) through AActor::GetActorBounds */
static int actor_bounds(uint8_t *a, float origin[3], float extent[3]) {
    uint8_t parms[0x40];
    uint8_t *fn = object_function(a, "GetActorBounds");
    int oo, oe;
    if (!fn || rd16(fn + UFUNC_PARMSSIZE) > (int)sizeof parms) {
        return 0;
    }
    oo = param_offset(fn, "Origin");
    oe = param_offset(fn, "BoxExtent");
    if (oo < 0 || oe < 0) {
        return 0;
    }
    memset(parms, 0, sizeof parms); /* bOnlyCollidingComponents = false, bIncludeFromChildActors = false */
    call(a, fn, parms);
    memcpy(origin, parms + oo, 12);
    memcpy(extent, parms + oe, 12);
    return 1;
}

struct box {
    uint8_t *actor;
    float lo[3], hi[3];
};

/* floating [gap_cm] [limit] [x0 y0 x1 y1]: placed objects whose bounds bottom is above the
   ground by more than gap and that no other object's bounds hold up (top within gap below
   their bottom, overlapping in plan). Hanging signs and neons show up too; judge by eye. */
static int verb_floating(const struct args *a, char *err, int ecap) {
    static struct box boxes[60000];
    float gap = argf(a, 1, 40), rx0 = argf(a, 3, -1e30f), ry0 = argf(a, 4, -1e30f), rx1 = argf(a, 5, 1e30f),
          ry1 = argf(a, 6, 1e30f);
    int limit = a->n > 2 ? atoi(a->v[2]) : 200, nb = 0, shown = 0, floating = 0, i, j;
    int32_t n;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag();
    for (i = 0; i < n && nb < 60000; i++) {
        float o[3], e[3];
        if (!actors[i] || !is_editor_actor(actors[i], tag) || is_gate(actors[i]) || !actor_bounds(actors[i], o, e)) {
            continue;
        }
        boxes[nb].actor = actors[i];
        for (j = 0; j < 3; j++) {
            boxes[nb].lo[j] = o[j] - e[j];
            boxes[nb].hi[j] = o[j] + e[j];
        }
        nb++;
    }
    for (i = 0; i < nb; i++) {
        struct box *b = &boxes[i];
        float cx = (b->lo[0] + b->hi[0]) / 2, cy = (b->lo[1] + b->hi[1]) / 2;
        int held = b->lo[2] <= gap;
        if (cx < rx0 || cx > rx1 || cy < ry0 || cy > ry1) {
            continue;
        }
        for (j = 0; j < nb && !held; j++) {
            struct box *o = &boxes[j];
            if (j == i) {
                continue;
            }
            /* overlaps in plan (shrunk a little) and reaches up to just under our bottom */
            if (o->lo[0] < b->hi[0] - 5 && o->hi[0] > b->lo[0] + 5 && o->lo[1] < b->hi[1] - 5 && o->hi[1] > b->lo[1] + 5 &&
                o->hi[2] >= b->lo[2] - gap && o->lo[2] < b->lo[2]) {
                held = 1;
            }
        }
        if (held) {
            continue;
        }
        floating++;
        if (shown < limit) {
            char row[64], extra[160];
            actor_row(b->actor, row, sizeof row);
            snprintf(extra, sizeof extra, " bottom=%.1f top=%.1f size=%.0fx%.0fx%.0f", b->lo[2], b->hi[2],
                     b->hi[0] - b->lo[0], b->hi[1] - b->lo[1], b->hi[2] - b->lo[2]);
            print_actor("floating", b->actor, extra);
            shown++;
        }
    }
    outf("floating %d of %d objects", floating, nb);
    (void)err;
    (void)ecap;
    return 1;
}

/* ======================================================================
   collision checks through the engine's own physics queries
   (real collision geometry of every object, not catalogue bounds)
   ====================================================================== */

struct tarr {
    void *data;
    int32_t num;
    int32_t max;
};

static struct {
    int ok;
    uint8_t *ksl, *fl, *fb;
    int l_wco, l_s, l_e, l_types, l_cx, l_ign, l_dbg, l_hit, l_self, l_ret, l_size;
    int b_wco, b_s, b_e, b_half, b_rot, b_types, b_cx, b_ign, b_dbg, b_hits, b_self, b_ret, b_size;
    int h_size, h_imp, h_actor, h_component, h_item, h_time, h_spen;
    uint8_t h_spen_mask;
} K;

/* object types to test against: WorldStatic, WorldDynamic, PhysicsBody, Vehicle, Destructible (no Pawn) */
static uint8_t k_types[32] = {0, 1, 3, 4, 5};
static int k_ntypes = 5;

static int kismet_init(char *err, int ecap) {
    uint8_t *cls, *hr, *bp;
    if (K.ok) {
        return 1;
    }
    K.ksl = class_cdo("KismetSystemLibrary");
    if (!K.ksl) {
        snprintf(err, ecap, "KismetSystemLibrary not found");
        return 0;
    }
    cls = rdp(K.ksl + UOBJ_CLASS);
    K.fl = find_function(cls, "LineTraceSingleForObjects");
    K.fb = find_function(cls, "BoxTraceMultiForObjects");
    hr = find_struct("HitResult");
    if (!K.fl || !K.fb || !hr) {
        snprintf(err, ecap, "trace functions or HitResult not found");
        return 0;
    }
    K.l_wco = param_offset(K.fl, "WorldContextObject"); K.l_s = param_offset(K.fl, "Start");
    K.l_e = param_offset(K.fl, "End"); K.l_types = param_offset(K.fl, "ObjectTypes");
    K.l_cx = param_offset(K.fl, "bTraceComplex"); K.l_ign = param_offset(K.fl, "ActorsToIgnore");
    K.l_dbg = param_offset(K.fl, "DrawDebugType"); K.l_hit = param_offset(K.fl, "OutHit");
    K.l_self = param_offset(K.fl, "bIgnoreSelf"); K.l_ret = param_offset(K.fl, "ReturnValue");
    K.l_size = rd16(K.fl + UFUNC_PARMSSIZE);
    K.b_wco = param_offset(K.fb, "WorldContextObject"); K.b_s = param_offset(K.fb, "Start");
    K.b_e = param_offset(K.fb, "End"); K.b_half = param_offset(K.fb, "HalfSize");
    K.b_rot = param_offset(K.fb, "Orientation"); K.b_types = param_offset(K.fb, "ObjectTypes");
    K.b_cx = param_offset(K.fb, "bTraceComplex"); K.b_ign = param_offset(K.fb, "ActorsToIgnore");
    K.b_dbg = param_offset(K.fb, "DrawDebugType"); K.b_hits = param_offset(K.fb, "OutHits");
    K.b_self = param_offset(K.fb, "bIgnoreSelf"); K.b_ret = param_offset(K.fb, "ReturnValue");
    K.b_size = rd16(K.fb + UFUNC_PARMSSIZE);
    K.h_size = rd32(hr + USTRUCT_SIZE);
    K.h_imp = prop_offset(hr, "ImpactPoint");
    K.h_actor = prop_offset(hr, "Actor");
    K.h_component = prop_offset(hr, "Component");
    K.h_item = prop_offset(hr, "Item");
    K.h_time = prop_offset(hr, "Time");
    bp = find_property(hr, "bStartPenetrating");
    if (bp) {
        K.h_spen = rd32(bp + FPROP_OFFSET);
        K.h_spen_mask = bp[0x7A];
    } else {
        K.h_spen = -1;
    }
    if (K.l_wco < 0 || K.l_s < 0 || K.l_e < 0 || K.l_types < 0 || K.l_cx < 0 || K.l_ign < 0 || K.l_hit < 0 ||
        K.l_ret < 0 || K.b_wco < 0 || K.b_s < 0 || K.b_e < 0 || K.b_half < 0 || K.b_rot < 0 || K.b_types < 0 ||
        K.b_cx < 0 || K.b_ign < 0 || K.b_hits < 0 || K.b_ret < 0 || K.h_size <= 0 || K.h_size > 0x200 ||
        K.h_imp < 0 || K.h_actor < 0 || K.l_size > 0x400 || K.b_size > 0x400) {
        snprintf(err, ecap, "trace parameter layout not understood");
        return 0;
    }
    K.ok = 1;
    return 1;
}

static uint8_t *weak_actor(const uint8_t *p) {
    int32_t idx = rd32(p);
    uint8_t *o;
    if (idx <= 0) {
        return 0;
    }
    o = object_at(idx);
    return (o && alive(o)) ? o : 0;
}

/* actors every ray also ignores (set by drop: what rests on the object being dropped) */
static uint8_t *k_ign_extra[64];
static int k_nign_extra;

/* first hit of a ray (complex collision), self ignored. 1 = hit. */
static int k_line(uint8_t *self, const float s[3], const float e[3], float hit[3], uint8_t **hit_actor) {
    static uint8_t parms[0x400];
    uint8_t *ign[65];
    struct tarr types, ignore;
    int i;
    ign[0] = self;
    for (i = 0; i < k_nign_extra; i++) ign[1 + i] = k_ign_extra[i];
    types.data = k_types; types.num = types.max = k_ntypes;
    ignore.data = ign; ignore.num = ignore.max = 1 + k_nign_extra;
    memset(parms, 0, (size_t)K.l_size);
    memcpy(parms + K.l_wco, &self, 8);
    memcpy(parms + K.l_s, s, 12);
    memcpy(parms + K.l_e, e, 12);
    memcpy(parms + K.l_types, &types, 16);
    parms[K.l_cx] = 1;
    memcpy(parms + K.l_ign, &ignore, 16);
    if (K.l_self >= 0) parms[K.l_self] = 1;
    call(K.ksl, K.fl, parms);
    if (!parms[K.l_ret]) {
        return 0;
    }
    memcpy(hit, parms + K.l_hit + K.h_imp, 12);
    if (hit_actor) {
        *hit_actor = weak_actor(parms + K.l_hit + K.h_actor);
    }
    return 1;
}

/* actors whose collision intersects an oriented box (zero-length sweep, complex collision), self excluded */
static int k_box(uint8_t *self, const float c[3], const float half[3], const float rot[3], uint8_t **out, int cap) {
    static uint8_t parms[0x400];
    static struct tarr hits; /* engine-owned storage, reused between calls */
    uint8_t *ign[1];
    struct tarr types, ignore;
    float e[3];
    int i, j, n = 0;
    ign[0] = self;
    types.data = k_types; types.num = types.max = k_ntypes;
    ignore.data = ign; ignore.num = ignore.max = 1;
    e[0] = c[0]; e[1] = c[1]; e[2] = c[2] + 0.5f;
    memset(parms, 0, (size_t)K.b_size);
    memcpy(parms + K.b_wco, &self, 8);
    memcpy(parms + K.b_s, c, 12);
    memcpy(parms + K.b_e, e, 12);
    memcpy(parms + K.b_half, half, 12);
    memcpy(parms + K.b_rot, rot, 12);
    memcpy(parms + K.b_types, &types, 16);
    parms[K.b_cx] = 1;
    memcpy(parms + K.b_ign, &ignore, 16);
    if (K.b_self >= 0) parms[K.b_self] = 1;
    hits.num = 0;
    memcpy(parms + K.b_hits, &hits, 16);
    call(K.ksl, K.fb, parms);
    memcpy(&hits, parms + K.b_hits, 16);
    if (hits.num <= 0 || hits.num > 4096 || !readable(hits.data, (size_t)hits.num * K.h_size)) {
        return 0;
    }
    for (i = 0; i < hits.num && n < cap; i++) {
        uint8_t *hr = (uint8_t *)hits.data + (size_t)i * K.h_size, *act;
        float t = 0;
        int start = 1;
        if (K.h_time >= 0) memcpy(&t, hr + K.h_time, 4);
        if (K.h_spen >= 0) start = (hr[K.h_spen] & K.h_spen_mask) != 0;
        if (!start && t > 0.01f) {
            continue; /* only what the box already intersects */
        }
        act = weak_actor(hr + K.h_actor);
        if (!act || act == self) {
            continue;
        }
        for (j = 0; j < n && out[j] != act; j++) {
        }
        if (j == n) {
            out[n++] = act;
        }
    }
    return n;
}

static void rot_axes(const float r[3], float ax[3][3]) {
    double p = r[0] * M_PI / 180.0, y = r[1] * M_PI / 180.0, ro = r[2] * M_PI / 180.0;
    double SP = sin(p), CP = cos(p), SY = sin(y), CY = cos(y), SR = sin(ro), CR = cos(ro);
    ax[0][0] = (float)(CP * CY); ax[0][1] = (float)(CP * SY); ax[0][2] = (float)SP;
    ax[1][0] = (float)(SR * SP * CY - CR * SY); ax[1][1] = (float)(SR * SP * SY + CR * CY); ax[1][2] = (float)(-SR * CP);
    ax[2][0] = (float)(-(CR * SP * CY + SR * SY)); ax[2][1] = (float)(CY * SR - CR * SP * SY); ax[2][2] = (float)(CR * CP);
}

/* oriented box of an object whose root is a static mesh component: world centre, half size, rotation */
static int actor_obb(uint8_t *a, float c[3], float h[3], float rot[3]) {
    static int off_mesh = -2;
    static uint8_t *bb_mesh[512];
    static float bb_lo[512][3], bb_hi[512][3];
    uint8_t *root = root_component(a), *mesh, *fn;
    struct xf t;
    float lo[3], hi[3], ax[3][3], lc[3];
    int i, slot;
    if (!root || !valid_object(root) || !is_a(root, "StaticMeshComponent")) {
        return 0;
    }
    if (off_mesh == -2) {
        off_mesh = field_offset(root, "StaticMesh");
    }
    if (off_mesh <= 0 || !(mesh = rdp(root + off_mesh)) || !valid_object(mesh) || !actor_get(a, &t)) {
        return 0;
    }
    slot = (int)(((uint64_t)mesh >> 4) % 512);
    if (bb_mesh[slot] == mesh) {
        memcpy(lo, bb_lo[slot], 12);
        memcpy(hi, bb_hi[slot], 12);
    } else {
        uint8_t parms[0x40];
        int o;
        if (!(fn = object_function(mesh, "GetBoundingBox")) || (o = param_offset(fn, "ReturnValue")) < 0) {
            return 0;
        }
        memset(parms, 0, sizeof parms);
        call(mesh, fn, parms);
        memcpy(lo, parms + o, 12);
        memcpy(hi, parms + o + 12, 12);
        bb_mesh[slot] = mesh;
        memcpy(bb_lo[slot], lo, 12);
        memcpy(bb_hi[slot], hi, 12);
    }
    rot_axes(t.rot, ax);
    for (i = 0; i < 3; i++) {
        lc[i] = (lo[i] + hi[i]) * 0.5f * t.scale[i];
        h[i] = fabsf((hi[i] - lo[i]) * 0.5f * t.scale[i]);
        rot[i] = t.rot[i];
    }
    for (i = 0; i < 3; i++) {
        c[i] = t.loc[i] + lc[0] * ax[0][i] + lc[1] * ax[1][i] + lc[2] * ax[2][i];
    }
    return h[0] > 0.5f || h[1] > 0.5f || h[2] > 0.5f;
}

/* trace <x> <y> <z_from> <z_to> (cm): what a vertical ray hits, once per object type 0..31, then with the
   set the collision check uses */
static int verb_trace(const struct args *a, char *err, int ecap) {
    int32_t n;
    uint8_t **actors = level_actors(&n), *ctx = 0, save[32];
    float s[3], e[3], hp[3];
    int t, i, ns = k_ntypes;
    if (a->n < 5) {
        snprintf(err, ecap, "usage: trace x y z_from z_to");
        return 0;
    }
    if (!kismet_init(err, ecap)) return 0;
    for (i = 0; i < n && !ctx; i++) if (actors[i] && valid_object(actors[i])) ctx = actors[i];
    if (!ctx) {
        snprintf(err, ecap, "no actor for world context");
        return 0;
    }
    s[0] = e[0] = argf(a, 1, 0); s[1] = e[1] = argf(a, 2, 0); s[2] = argf(a, 3, 10000); e[2] = argf(a, 4, -1000);
    memcpy(save, k_types, 32);
    for (t = -1; t < 32; t++) {
        uint8_t *ha = 0;
        char nm[96], cl[96];
        if (t >= 0) {
            k_types[0] = (uint8_t)t;
            k_ntypes = 1;
        } else {
            k_ntypes = ns;
        }
        if (k_line(ctx, s, e, hp, &ha)) {
            nm[0] = cl[0] = 0;
            if (ha) { obj_name(ha, nm, sizeof nm); obj_class_name(ha, cl, sizeof cl); }
            outf("%s type=%d z=%.0f %s %s", t < 0 ? "check-set" : "single", t, hp[2], nm, cl);
        }
        memcpy(k_types, save, 32);
    }
    k_ntypes = ns;
    return 1;
}

/* terrain x y z_from z_to: landscape only; foliage and placed structures are ignored. */
static int verb_terrain(const struct args *a, char *err, int ecap) {
    int32_t n;
    uint8_t **actors = level_actors(&n), *ctx = 0, *ha = 0;
    uint8_t *saved_ign[64];
    int saved_n = k_nign_extra, i, found = 0;
    float s[3], e[3], hp[3];
    if (a->n < 5) {
        snprintf(err, ecap, "usage: terrain x y z_from z_to");
        return 0;
    }
    if (!kismet_init(err, ecap)) return 0;
    for (i = 0; i < n && !ctx; i++) if (actors[i] && valid_object(actors[i])) ctx = actors[i];
    if (!ctx) {
        snprintf(err, ecap, "no actor for world context");
        return 0;
    }
    s[0] = e[0] = argf(a, 1, 0); s[1] = e[1] = argf(a, 2, 0);
    s[2] = argf(a, 3, 100000); e[2] = argf(a, 4, -200000);
    memcpy(saved_ign, k_ign_extra, sizeof saved_ign);
    k_nign_extra = 0;
    while (k_line(ctx, s, e, hp, &ha)) {
        if (ha && is_a(ha, "LandscapeProxy")) { found = 1; break; }
        if (!ha || k_nign_extra == 64) break;
        for (i = 0; i < k_nign_extra && k_ign_extra[i] != ha; i++) {}
        if (i < k_nign_extra) break;
        k_ign_extra[k_nign_extra++] = ha;
    }
    memcpy(k_ign_extra, saved_ign, sizeof saved_ign);
    k_nign_extra = saved_n;
    if (found) outf("terrain x=%.2f y=%.2f z=%.2f", hp[0], hp[1], hp[2]);
    else outf("terrain miss x=%.2f y=%.2f", s[0], s[1]);
    return 1;
}

#include "environment.h"

/* collide <out_path> [x0 y0 x1 y1 (cm)] [gap_cm]
   For every placed object (static mesh root, not a gate or flat ground tile) whose bounds centre is in the box:
     I a b level   its oriented box, shrunk (level 1: by 25 cm or 35 %; level 2: to half), intersects b's collision
     S a sunk z    it stands sunk below the surface under its centre by more than 3 cm (what it sinks into)
     B a row box   an object without a single static mesh (blueprint building): its bounds only
     R a b         it rests on b (first support ray hit within gap; 0 = landscape or base level)
     T a b...      everything its box grown by 15 cm touches (0 = landscape or base level)
     F a gap side  nothing under its footprint within gap (it may still hang on something it touches);
                   gap = nearest thing below any footprint point, side = 1 if something is within 60 cm sideways
   Whether something floats is then a graph question: is it connected to the ground through R and T edges.
   plus one O line per object tested. Written to out_path; the reply is a summary. */
static int verb_collide(const struct args *a, char *err, int ecap) {
    static uint8_t *subj[60000];
    static uint8_t *hit[256];
    float rx0 = argf(a, 2, -1e30f), ry0 = argf(a, 3, -1e30f), rx1 = argf(a, 4, 1e30f), ry1 = argf(a, 5, 1e30f);
    float gap = argf(a, 6, 30);
    int32_t n, i;
    int ns = 0, pairs = 0, deep = 0, sunk = 0, floating = 0, skipped = 0, k, j;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag(), ground = 0;
    FILE *f;
    DWORD t0 = GetTickCount();
    if (a->n < 2) {
        snprintf(err, ecap, "usage: collide <out_path> [x0 y0 x1 y1] [gap_cm]");
        return 0;
    }
    if (!kismet_init(err, ecap)) {
        return 0;
    }
    if (!(f = fopen(a->v[1], "w"))) {
        snprintf(err, ecap, "cannot write %s", a->v[1]);
        return 0;
    }
    make_fname("Ground", &ground);
    for (i = 0; i < n && ns < 60000; i++) {
        uint8_t *act = actors[i];
        float o[3], e[3];
        if (!act || !is_editor_actor(act, tag) || is_gate(act) || !actor_bounds(act, o, e)) continue;
        if (o[0] < rx0 || o[0] > rx1 || o[1] < ry0 || o[1] > ry1) continue;
        if (ground && actor_has_tag(act, ground) && e[2] < 60.0f) continue;
        subj[ns++] = act;
    }
    for (k = 0; k < ns; k++) {
        uint8_t *act = subj[k], *below = 0;
        float o[3], e[3], c[3], h[3], r[3], hs[3], top, bot;
        char row[64];
        int nh, supported = 0;
        if (!actor_bounds(act, o, e)) {
            skipped++;
            continue;
        }
        if (!actor_obb(act, c, h, r)) {
            /* no single static mesh (blueprint buildings): only its bounds, some of their parts have no query
               collision, so the report uses the box as solid */
            actor_row(act, row, sizeof row);
            fprintf(f, "B %llx %s %.0f %.0f %.0f %.0f %.0f %.0f\n", (unsigned long long)act, row[0] ? row : "-", o[0] - e[0],
                    o[1] - e[1], o[2] - e[2], o[0] + e[0], o[1] + e[1], o[2] + e[2]);
            skipped++;
            continue;
        }
        actor_row(act, row, sizeof row);
        bot = o[2] - e[2];
        top = o[2] + e[2];
        fprintf(f, "O %llx %s %.0f %.0f %.0f %.0f %.0f %.0f\n", (unsigned long long)act, row[0] ? row : "-", o[0] - e[0],
                o[1] - e[1], bot, o[0] + e[0], o[1] + e[1], top);
        /* intersections */
        for (j = 0; j < 3; j++) {
            float s = h[j] - 25.0f;
            hs[j] = s > h[j] * 0.65f ? s : h[j] * 0.65f;
        }
        nh = k_box(act, c, hs, r, hit, 256);
        if (nh > 0) {
            uint8_t *deep_hit[256];
            int nd;
            for (j = 0; j < 3; j++) hs[j] = h[j] * 0.5f;
            nd = k_box(act, c, hs, r, deep_hit, 256);
            for (j = 0; j < nh; j++) {
                char brow[64];
                int lvl = 1, m;
                if (!is_editor_actor(hit[j], tag)) continue;
                for (m = 0; m < nd; m++) if (deep_hit[m] == hit[j]) lvl = 2;
                actor_row(hit[j], brow, sizeof brow);
                fprintf(f, "I %llx %llx %s %d\n", (unsigned long long)act, (unsigned long long)hit[j], brow[0] ? brow : "-", lvl);
                pairs++;
                deep += lvl == 2;
            }
        }
        /* support: 13 rays down from inside the object through its oriented footprint (centre, inner and outer
           corners, edge midpoints); the centre ray also measures how deep it sits in what is under it */
        {
            static const float pu[13] = {0, -0.35f, 0.35f, -0.35f, 0.35f, -0.48f, 0.48f, -0.48f, 0.48f, 0, 0, -0.48f, 0.48f};
            static const float pv[13] = {0, -0.35f, -0.35f, 0.35f, 0.35f, -0.48f, -0.48f, 0.48f, 0.48f, -0.48f, 0.48f, 0, 0};
            float ax[3][3], zs = bot + (e[2] < 200.0f ? e[2] : 200.0f), mingap = 1e30f;
            rot_axes(r, ax);
            for (j = 0; j < 13; j++) {
                float s[3], en[3], hp[3];
                uint8_t *ha = 0;
                if (j == 0) {
                    s[0] = o[0];
                    s[1] = o[1];
                } else {
                    s[0] = c[0] + pu[j] * 2 * h[0] * ax[0][0] + pv[j] * 2 * h[1] * ax[1][0];
                    s[1] = c[1] + pu[j] * 2 * h[0] * ax[0][1] + pv[j] * 2 * h[1] * ax[1][1];
                }
                en[0] = s[0];
                en[1] = s[1];
                s[2] = zs;
                en[2] = bot - gap;
                if (env_line(act, s, en, hp, &ha, 1, 0)) {
                    supported = 1;
                    if (j == 0 && hp[2] > bot + 3.0f) {
                        char srow[64], scls[96];
                        srow[0] = 0;
                        if (ha) actor_row(ha, srow, sizeof srow);
                        if (ha) obj_class_name(ha, scls, sizeof scls); else snprintf(scls, sizeof scls, "-");
                        fprintf(f, "S %llx %.1f %s %s %.0f %.0f\n", (unsigned long long)act, hp[2] - bot,
                                srow[0] ? srow : "-", scls, o[0], o[1]);
                        sunk++;
                    }
                    below = ha;
                    break;
                }
            }
            if (supported) {
                fprintf(f, "R %llx %llx\n", (unsigned long long)act, (unsigned long long)(below && is_editor_actor(below, tag) ? below : 0));
            }
            /* everything it touches (its box grown by 15 cm); 0 = landscape / base level geometry */
            for (j = 0; j < 3; j++) hs[j] = h[j] + 15.0f;
            {
                int nc = k_box(act, c, hs, r, hit, 256), m;
                if (nc > 0) {
                    fprintf(f, "T %llx", (unsigned long long)act);
                    for (m = 0; m < nc; m++) if (!env_base_foliage(hit[m])) fprintf(f, " %llx", (unsigned long long)(is_editor_actor(hit[m], tag) ? hit[m] : 0));
                    fputc('\n', f);
                }
            }
            if (!supported) {
                int side;
                /* how far is the nearest thing below any of the 13 points, and does anything touch it from the
                   side within 60 cm (a sign on a wall, a platform next to a silo) */
                for (j = 0; j < 13; j++) {
                    float s[3], en[3], hp[3];
                    if (j == 0) {
                        s[0] = o[0];
                        s[1] = o[1];
                    } else {
                        s[0] = c[0] + pu[j] * 2 * h[0] * ax[0][0] + pv[j] * 2 * h[1] * ax[1][0];
                        s[1] = c[1] + pu[j] * 2 * h[0] * ax[0][1] + pv[j] * 2 * h[1] * ax[1][1];
                    }
                    en[0] = s[0];
                    en[1] = s[1];
                    s[2] = bot;
                    en[2] = bot - 50000.0f;
                    if (env_line(act, s, en, hp, 0, 1, 0) && bot - hp[2] < mingap) mingap = bot - hp[2];
                }
                hs[0] = h[0] + 60.0f;
                hs[1] = h[1] + 60.0f;
                hs[2] = h[2] + 5.0f;
                side = k_box(act, c, hs, r, hit, 4) > 0;
                fprintf(f, "F %llx %s %.0f %.0f %.0f %.0f %d\n", (unsigned long long)act, row[0] ? row : "-", o[0], o[1], bot,
                        mingap < 1e29f ? mingap : -1.0f, side);
                floating++;
            }
        }
        (void)below;
    }
    fclose(f);
    outf("collide objects=%d skipped=%d intersect_hits=%d deep=%d sunk=%d floating=%d ms=%lu", ns, skipped, pairs, deep,
         sunk, floating, (unsigned long)(GetTickCount() - t0));
    return 1;
}


/* ---- drop: rest an object on exactly what is under it ----
   drop <actor|last> [step_cm=8] [max_cm=1000] [embed_cm=0] [dry]   (last = what the last spawn made)
   Samples the object's oriented footprint on a step_cm grid. At each point a ray against the object alone
   (Line Trace Component, complex) finds its underside, and a ray down from there against everything else finds
   the surface below. The smallest gap over all points is how far it can move down until it touches; a negative
   gap (it is sunk in or overlaps what is under it) lifts it. embed_cm sinks it that much further on purpose.
   dry only measures. Rotation and x, y stay. */
static struct {
    int ok;
    uint8_t *fn;
    int s, e, cx, show, pshow, loc, ret, size;
} LT;

static int ltc_init(uint8_t *comp) {
    if (LT.ok) return 1;
    LT.fn = object_function(comp, "K2_LineTraceComponent");
    if (!LT.fn) return 0;
    LT.s = param_offset(LT.fn, "TraceStart"); LT.e = param_offset(LT.fn, "TraceEnd");
    LT.cx = param_offset(LT.fn, "bTraceComplex"); LT.show = param_offset(LT.fn, "bShowTrace");
    LT.pshow = param_offset(LT.fn, "bPersistentShowTrace"); LT.loc = param_offset(LT.fn, "HitLocation");
    LT.ret = param_offset(LT.fn, "ReturnValue"); LT.size = rd16(LT.fn + UFUNC_PARMSSIZE);
    if (LT.s < 0 || LT.e < 0 || LT.cx < 0 || LT.loc < 0 || LT.ret < 0 || LT.size > 0x400) return 0;
    LT.ok = 1;
    return 1;
}

/* first hit of a segment against one component only */
static int ltc(uint8_t *comp, const float s[3], const float e[3], float hit[3]) {
    static uint8_t parms[0x400];
    memset(parms, 0, (size_t)LT.size);
    memcpy(parms + LT.s, s, 12);
    memcpy(parms + LT.e, e, 12);
    parms[LT.cx] = 1;
    call(comp, LT.fn, parms);
    if (!parms[LT.ret]) return 0;
    memcpy(hit, parms + LT.loc, 12);
    return 1;
}

static int drop_measure(uint8_t *act, float step, float maxd, float *gap_out, uint8_t **on, int *samples) {
    uint8_t *root = root_component(act);
    float o[3], e[3], c[3], h[3], r[3], ax[3][3], bot, top, pin, best = 1e30f;
    int i, j, nu, nv, n = 0;
    *on = 0;
    if (!root || !valid_object(root) || !ltc_init(root) || !actor_bounds(act, o, e) || !actor_obb(act, c, h, r)) return 0;
    rot_axes(r, ax);
    bot = o[2] - e[2];
    top = o[2] + e[2];
    pin = e[2] < 60.0f ? e[2] : 60.0f; /* how deep something under it may already reach into it */
    /* what rests on it (its bottom is in the object's upper half) is not support: rays ignore it */
    {
        static uint8_t *hit[256];
        float hs[3], xo[3], xe[3];
        int m, nh;
        for (m = 0; m < 3; m++) hs[m] = h[m] + 20.0f;
        nh = k_box(act, c, hs, r, hit, 256);
        k_nign_extra = 0;
        for (m = 0; m < nh && k_nign_extra < 64; m++) {
            if (actor_bounds(hit[m], xo, xe) && xo[2] - xe[2] > bot + 0.5f * (top - bot)) k_ign_extra[k_nign_extra++] = hit[m];
        }
    }
    nu = (int)(2 * h[0] / step) + 1;
    nv = (int)(2 * h[1] / step) + 1;
    if (nu > 200) nu = 200;
    if (nv > 200) nv = 200;
    for (i = 0; i <= nu; i++) {
        for (j = 0; j <= nv; j++) {
            float u = -h[0] + 2 * h[0] * i / nu, v = -h[1] + 2 * h[1] * j / nv;
            float s[3], en[3], hb[3], ha[3];
            uint8_t *ga = 0;
            s[0] = en[0] = c[0] + u * ax[0][0] + v * ax[1][0];
            s[1] = en[1] = c[1] + u * ax[0][1] + v * ax[1][1];
            s[2] = bot - 20.0f;
            en[2] = top + 20.0f;
            if (!ltc(root, s, en, hb)) continue; /* the object has no underside here */
            s[2] = hb[2] + pin;
            en[2] = hb[2] - maxd;
            if (!k_line(act, s, en, ha, &ga)) continue;
            if (ha[2] > s[2] - 0.5f) {
                /* started inside something: measure from just above its underside instead */
                s[2] = hb[2] + 1.0f;
                if (!k_line(act, s, en, ha, &ga) || ha[2] > s[2] - 0.5f) continue;
            }
            n++;
            if (hb[2] - ha[2] < best) {
                best = hb[2] - ha[2];
                *on = ga;
            }
        }
    }
    k_nign_extra = 0;
    *samples = n;
    *gap_out = best;
    return n > 0;
}

static int verb_drop(const struct args *a, char *err, int ecap) {
    uint8_t *actor, *on;
    struct xf t;
    float step = argf(a, 2, 8), maxd = argf(a, 3, 1000), embed = argf(a, 4, 0), gap, after;
    int n, dry = a->n > 5 && ieq(a->v[5], "dry");
    char onrow[64];
    if (a->n < 2) {
        snprintf(err, ecap, "usage: drop <actor|last> [step_cm=8] [max_cm=1000] [embed_cm=0] [dry]");
        return 0;
    }
    if (step < 1) step = 1;
    if (!kismet_init(err, ecap)) return 0;
    if (ieq(a->v[1], "last")) {
        actor = g_last_spawned;
        if (!actor || !alive(actor) || !is_level_actor(actor)) {
            snprintf(err, ecap, "nothing spawned yet (drop last means the object the last spawn made)");
            return 0;
        }
    } else if (!(actor = resolve_actor(a->v[1], 0, err, ecap))) {
        return 0;
    }
    if (!actor_get(actor, &t)) {
        snprintf(err, ecap, "cannot read the transform");
        return 0;
    }
    if (!drop_measure(actor, step, maxd, &gap, &on, &n)) {
        snprintf(err, ecap, "nothing under it within %.0f cm, or it is not a single static mesh", maxd);
        return 0;
    }
    onrow[0] = 0;
    if (on && is_editor_actor(on, editor_tag())) actor_row(on, onrow, sizeof onrow);
    else snprintf(onrow, sizeof onrow, on ? "level" : "-");
    if (dry) {
        outf("gap %llx %.1f cm over %s (%d samples)", (unsigned long long)actor, gap, onrow, n);
        return 1;
    }
    t.loc[2] -= gap + embed;
    if (!actor_set(actor, &t)) {
        snprintf(err, ecap, "the actor setters were not found");
        return 0;
    }
    /* measure again from the new place: what is left is the sampling error */
    if (!drop_measure(actor, step, maxd, &after, &on, &n)) after = 0;
    outf("dropped %llx by %.1f cm onto %s, now z=%.1f, residual %.1f cm (%d samples)", (unsigned long long)actor,
         gap + embed, onrow, t.loc[2], after, n);
    return 1;
}
static int verb_bounds(const struct args *a, char *err, int ecap) {
    uint8_t *actor;
    float o[3], e[3];
    if (a->n < 2 || !(actor = resolve_actor(a->v[1], 1, err, ecap))) {
        if (a->n < 2) {
            snprintf(err, ecap, "usage: bounds <actor>");
        }
        return 0;
    }
    if (!actor_bounds(actor, o, e)) {
        snprintf(err, ecap, "GetActorBounds not found");
        return 0;
    }
    outf("bounds %llx lo=%.3f,%.3f,%.3f hi=%.3f,%.3f,%.3f", (unsigned long long)actor, o[0] - e[0], o[1] - e[1],
         o[2] - e[2], o[0] + e[0], o[1] + e[1], o[2] + e[2]);
    {
        float c[3], h[3], r[3];
        print_actor("actor", actor, "");
        if (actor_obb(actor,c,h,r)) outf("obb center=%.3f,%.3f,%.3f half=%.3f,%.3f,%.3f rotation=%.3f,%.3f,%.3f",
                                     c[0],c[1],c[2],h[0],h[1],h[2],r[0],r[1],r[2]);
    }
    return 1;
}

/* destroybox <x0> <y0> <x1> <y1> [z0|-] [z1|-] [row|-] : every placed object whose pivot is inside */
static int verb_destroybox(const struct args *a, char *err, int ecap) {
    static uint8_t *doomed[100000];
    float x0, y0, x1, y1, z0, z1, t;
    const char *row_filter;
    int32_t n, i, done = 0, gates = 0;
    uint8_t **actors;
    uint64_t tag = editor_tag();
    if (a->n < 5) {
        snprintf(err, ecap, "usage: destroybox <x0> <y0> <x1> <y1> [z0|-] [z1|-] [row|-]");
        return 0;
    }
    x0 = argf(a, 1, 0);
    y0 = argf(a, 2, 0);
    x1 = argf(a, 3, 0);
    y1 = argf(a, 4, 0);
    z0 = argf(a, 5, -1e30f);
    z1 = argf(a, 6, 1e30f);
    row_filter = (a->n > 7 && strcmp(a->v[7], "-") != 0) ? a->v[7] : 0;
    if (x0 > x1) { t = x0; x0 = x1; x1 = t; }
    if (y0 > y1) { t = y0; y0 = y1; y1 = t; }
    if (z0 > z1) { t = z0; z0 = z1; z1 = t; }
    actors = level_actors(&n);
    for (i = 0; i < n; i++) {
        uint8_t *act = actors[i];
        float loc[3];
        char row[64];
        if (!act || !is_editor_actor(act, tag) || !vec_getter(act, "K2_GetActorLocation", loc)) {
            continue;
        }
        if (loc[0] < x0 || loc[0] > x1 || loc[1] < y0 || loc[1] > y1 || loc[2] < z0 || loc[2] > z1) {
            continue;
        }
        actor_row(act, row, sizeof row);
        if (row_filter && !ieq(row, row_filter)) {
            continue;
        }
        if (is_gate(act)) {
            gates++;
        }
        if (done < 50) {
            outf("destroyed %llx row=%s%s", (unsigned long long)act, row[0] ? row : "-", is_gate(act) ? " gate" : "");
        }
        if (done < (int)(sizeof doomed / sizeof doomed[0])) {
            doomed[done++] = act;
        }
    }
    /* destroying removes actors from the level array, so only after the walk */
    for (i = 0; i < done; i++) {
        destroy_actor(doomed[i]);
    }
    if (gates) {
        refresh_gate_ui();
    }
    outf("count %d", done);
    return 1;
}

static int verb_actors(const struct args *a) {
    float x = argf(a, 1, 0), y = argf(a, 2, 0), z = argf(a, 3, 0), r = argf(a, 4, 0);
    int limit = a->n > 5 ? atoi(a->v[5]) : 50, all = a->n > 6 && ieq(a->v[6], "all");
    int32_t n, i, shown = 0, matched = 0;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag();
    for (i = 0; i < n; i++) {
        uint8_t *act = actors[i];
        float loc[3], d;
        if (!act || !alive(act) || (!all && !is_editor_actor(act, tag)) || !vec_getter(act, "K2_GetActorLocation", loc)) {
            continue;
        }
        d = dist3(loc, x, y, z);
        if (r > 0 && d > r) {
            continue;
        }
        matched++;
        if (shown < limit) {
            char tags[400], extra[460];
            actor_tags(act, tags, sizeof tags);
            snprintf(extra, sizeof extra, " d=%.2f tags=%s", d, tags);
            print_actor("actor", act, extra);
            shown++;
        }
    }
    outf("matched %d shown %d level %d", matched, shown, n);
    return 1;
}

/* gates : every race gate in the level, in level order */
static int verb_gates(void) {
    int32_t n, i, count = 0;
    uint8_t **actors = level_actors(&n);
    for (i = 0; i < n; i++) {
        if (actors[i] && alive(actors[i]) && is_gate(actors[i])) {
            print_actor("gate", actors[i], "");
            count++;
        }
    }
    outf("gates %d", count);
    return 1;
}

/* cells <cell_m> <row,row,...|!row,row,...|*> [zmax_m] : occupancy grid from actor bounds, one line per occupied cell
   ("cell <ix> <iy> <n> <top_m>", cell centre = (ix+0.5)*cell). zmax skips actors whose bottom is above it. */
static int verb_cells(const struct args *a, char *err, int ecap) {
    enum { G = 256 };
    static uint16_t cnt[G][G];
    static float top[G][G];
    float cell = argf(a, 1, 20.0f) * 100.0f, zcap = argf(a, 3, 1e9f) * 100.0f;
    const char *rows = a->n > 2 ? a->v[2] : "*";
    int negate = rows[0] == '!';
    int32_t n, i, any = strcmp(rows, "*") == 0, shown = 0;
    int ix, iy;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag();
    if (cell < 100.0f) {
        snprintf(err, ecap, "cell must be at least 1 m");
        return 0;
    }
    memset(cnt, 0, sizeof cnt);
    for (i = 0; i < n; i++) {
        uint8_t *act = actors[i];
        float o[3], e[3];
        char row[64];
        int x0, x1, y0, y1;
        if (!act || !is_editor_actor(act, tag) || is_gate(act)) {
            continue;
        }
        if (!any) {
            char pat[80];
            actor_row(act, row, sizeof row);
            snprintf(pat, sizeof pat, ",%s,", row);
            {
                char hay[1024];
                snprintf(hay, sizeof hay, ",%s,", rows + negate);
                if (negate ? (row[0] && strstr(hay, pat) != NULL) : (!row[0] || !strstr(hay, pat))) {
                    continue;
                }
            }
        }
        if (!actor_bounds(act, o, e) || o[2] - e[2] > zcap) {
            continue;
        }
        x0 = (int)floorf((o[0] - e[0]) / cell) + G / 2;
        x1 = (int)floorf((o[0] + e[0] - 1.0f) / cell) + G / 2;
        y0 = (int)floorf((o[1] - e[1]) / cell) + G / 2;
        y1 = (int)floorf((o[1] + e[1] - 1.0f) / cell) + G / 2;
        for (ix = x0 < 0 ? 0 : x0; ix <= x1 && ix < G; ix++) {
            for (iy = y0 < 0 ? 0 : y0; iy <= y1 && iy < G; iy++) {
                if (cnt[ix][iy] == 0 || o[2] + e[2] > top[ix][iy]) {
                    top[ix][iy] = o[2] + e[2];
                }
                if (cnt[ix][iy] < 65535) {
                    cnt[ix][iy]++;
                }
            }
        }
    }
    for (ix = 0; ix < G; ix++) {
        for (iy = 0; iy < G; iy++) {
            if (cnt[ix][iy]) {
                outf("cell %d %d %d %.1f", ix - G / 2, iy - G / 2, cnt[ix][iy], top[ix][iy] / 100.0f);
                shown++;
            }
        }
    }
    outf("cells %d", shown);
    return 1;
}

/* console <command ...> : KismetSystemLibrary::ExecuteConsoleCommand in the open world */
static int verb_console(const char *line, char *err, int ecap) {
    uint8_t parms[0x80];
    uint8_t *cdo = class_cdo("KismetSystemLibrary");
    uint8_t *fn = cdo ? object_function(cdo, "ExecuteConsoleCommand") : 0;
    const char *cmd = line;
    struct fstr s;
    int ow, oc;
    while (*cmd && !isspace((unsigned char)*cmd)) cmd++;
    while (*cmd && isspace((unsigned char)*cmd)) cmd++;
    if (!fn || !*cmd) {
        snprintf(err, ecap, fn ? "usage: console <command>" : "ExecuteConsoleCommand not found");
        return 0;
    }
    ow = param_offset(fn, "WorldContextObject");
    oc = param_offset(fn, "Command");
    if (ow < 0 || oc < 0 || !engine_fstring(cmd, &s)) {
        snprintf(err, ecap, "could not build the call");
        return 0;
    }
    memset(parms, 0, sizeof parms);
    *(uint8_t **)(parms + ow) = current_world();
    memcpy(parms + oc, &s, sizeof s);
    call(cdo, fn, parms);
    outf("console %s", cmd);
    return 1;
}

/* cvar <name> : float value through KismetSystemLibrary::GetConsoleVariableFloatValue (and Int) */
static int verb_cvar(const struct args *a, char *err, int ecap) {
    uint8_t parms[0x80];
    uint8_t *cdo = class_cdo("KismetSystemLibrary");
    const char *names[2] = {"GetConsoleVariableFloatValue", "GetConsoleVariableIntValue"};
    int k;
    if (a->n < 2) {
        snprintf(err, ecap, "usage: cvar <name>");
        return 0;
    }
    for (k = 0; k < 2; k++) {
        uint8_t *fn = cdo ? object_function(cdo, names[k]) : 0;
        struct fstr s;
        int on, orv;
        if (!fn) continue;
        on = param_offset(fn, "VariableName");
        orv = param_offset(fn, "ReturnValue");
        if (on < 0 || orv < 0 || !engine_fstring(a->v[1], &s)) continue;
        memset(parms, 0, sizeof parms);
        memcpy(parms + on, &s, sizeof s);
        call(cdo, fn, parms);
        if (k == 0) outf("cvar %s float=%g", a->v[1], *(float *)(parms + orv));
        else outf("cvar %s int=%d", a->v[1], *(int32_t *)(parms + orv));
    }
    return 1;
}

static HWND agent_game_window(void);
/* raise [off] : put the game window on top so screenshots of it are not covered */
static int verb_raise(const struct args *a, char *err, int ecap) {
    HWND w = agent_game_window();
    int off = a->n > 1 && ieq(a->v[1], "off");
    if (!w) {
        snprintf(err, ecap, "no game window");
        return 0;
    }
    SetWindowPos(w, off ? HWND_NOTOPMOST : HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_SHOWWINDOW);
    if (!off) {
        ShowWindow(w, SW_RESTORE);
        SetForegroundWindow(w);
    }
    outf("raise %s", off ? "off" : "on");
    return 1;
}

/* grid <cell_m> <x0_m> <y0_m> <x1_m> <y1_m> [max_bottom_m]
   Occupancy bitmap of placed objects (gates and flat ground tiles left out). One text row per cell row,
   from y0 upwards; '.' is empty, otherwise the top height of the tallest object in 2 m steps:
   0-9 then a-z (0 = under 2 m, a = 20 m, z = 70 m or more). Objects whose bottom is above
   max_bottom (default 3 m) are skipped, so balloons and bridges do not block the ground. */
static int verb_grid(const struct args *a, char *err, int ecap) {
    enum { GMAX = 400 };
    static float top[GMAX][GMAX];
    static char rowbuf[GMAX + 1];
    const char *code = "0123456789abcdefghijklmnopqrstuvwxyz";
    float cell, x0, y0, x1, y1, maxb;
    int nx, ny, ix, iy, used = 0;
    int32_t n, i;
    uint8_t **actors;
    uint64_t tag = editor_tag(), ground = 0;
    if (a->n < 6) {
        snprintf(err, ecap, "usage: grid <cell_m> <x0> <y0> <x1> <y1> [max_bottom_m]");
        return 0;
    }
    cell = argf(a, 1, 4) * 100.0f; x0 = argf(a, 2, 0) * 100.0f; y0 = argf(a, 3, 0) * 100.0f;
    x1 = argf(a, 4, 0) * 100.0f; y1 = argf(a, 5, 0) * 100.0f; maxb = argf(a, 6, 3) * 100.0f;
    if (cell < 50.0f || x1 <= x0 || y1 <= y0) {
        snprintf(err, ecap, "bad grid");
        return 0;
    }
    nx = (int)((x1 - x0) / cell + 0.999f); ny = (int)((y1 - y0) / cell + 0.999f);
    if (nx > GMAX || ny > GMAX) {
        snprintf(err, ecap, "grid too large (max %d x %d cells)", GMAX, GMAX);
        return 0;
    }
    make_fname("Ground", &ground);
    for (iy = 0; iy < ny; iy++) for (ix = 0; ix < nx; ix++) top[iy][ix] = -1.0f;
    actors = level_actors(&n);
    for (i = 0; i < n; i++) {
        uint8_t *act = actors[i];
        float o[3], e[3];
        int ax0, ax1, ay0, ay1;
        if (!act || !is_editor_actor(act, tag) || is_gate(act) || !actor_bounds(act, o, e)) continue;
        if (ground && actor_has_tag(act, ground) && e[2] < 60.0f) continue;      /* ground tiles (hill slopes stay) */
        if (e[2] < 25.0f && o[2] - e[2] < 40.0f) continue;                        /* other flat surface pieces */
        if (o[2] - e[2] > maxb) continue;
        if (e[0] > 40000.0f || e[1] > 40000.0f) continue;                          /* the horizon plane */
        ax0 = (int)floorf((o[0] - e[0] - x0) / cell); ax1 = (int)floorf((o[0] + e[0] - x0) / cell);
        ay0 = (int)floorf((o[1] - e[1] - y0) / cell); ay1 = (int)floorf((o[1] + e[1] - y0) / cell);
        if (ax1 < 0 || ay1 < 0 || ax0 >= nx || ay0 >= ny) continue;
        if (ax0 < 0) ax0 = 0; if (ay0 < 0) ay0 = 0; if (ax1 >= nx) ax1 = nx - 1; if (ay1 >= ny) ay1 = ny - 1;
        for (iy = ay0; iy <= ay1; iy++)
            for (ix = ax0; ix <= ax1; ix++)
                if (o[2] + e[2] > top[iy][ix]) top[iy][ix] = o[2] + e[2];
        used++;
    }
    outf("grid cell=%.1f x0=%.1f y0=%.1f nx=%d ny=%d objects=%d", cell / 100, x0 / 100, y0 / 100, nx, ny, used);
    for (iy = 0; iy < ny; iy++) {
        for (ix = 0; ix < nx; ix++) {
            float t = top[iy][ix];
            int k = t < 0 ? -1 : (int)(t / 200.0f);
            rowbuf[ix] = k < 0 ? '.' : code[k > 35 ? 35 : k];
        }
        rowbuf[nx] = 0;
        outf("%s", rowbuf);
    }
    return 1;
}

/* boxes <x0> <y0> <x1> <y1> [max_bottom_m] : every placed object whose bounds centre is inside (metres),
   one line each: row x0 y0 z0 x1 y1 z1 (metres, 1 decimal). Flat ground tiles are left out. */
static int verb_boxes(const struct args *a, char *err, int ecap) {
    float x0 = argf(a, 1, 0) * 100, y0 = argf(a, 2, 0) * 100, x1 = argf(a, 3, 0) * 100, y1 = argf(a, 4, 0) * 100;
    float maxb = argf(a, 5, 1e6f) * 100;
    int32_t n, i, count = 0;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag(), ground = 0;
    if (a->n < 5) {
        snprintf(err, ecap, "usage: boxes <x0> <y0> <x1> <y1> [max_bottom_m]");
        return 0;
    }
    make_fname("Ground", &ground);
    for (i = 0; i < n; i++) {
        uint8_t *act = actors[i];
        float o[3], e[3];
        char row[64];
        if (!act || !is_editor_actor(act, tag) || is_gate(act) || !actor_bounds(act, o, e)) continue;
        if (o[0] < x0 || o[0] > x1 || o[1] < y0 || o[1] > y1) continue;
        if (ground && actor_has_tag(act, ground) && e[2] < 60.0f) continue;
        if (e[2] < 25.0f && o[2] - e[2] < 40.0f) continue;
        if (o[2] - e[2] > maxb) continue;
        actor_row(act, row, sizeof row);
        outf("%s %.1f %.1f %.1f %.1f %.1f %.1f", row[0] ? row : "-", (o[0] - e[0]) / 100, (o[1] - e[1]) / 100,
             (o[2] - e[2]) / 100, (o[0] + e[0]) / 100, (o[1] + e[1]) / 100, (o[2] + e[2]) / 100);
        count++;
    }
    outf("boxes %d", count);
    return 1;
}

static int verb_clear(char *err, int ecap) {
    static uint8_t *doomed[100000];
    int32_t n, i, objects = 0, gates = 0;
    uint8_t **actors = level_actors(&n);
    uint64_t tag = editor_tag();
    if (!tag) {
        snprintf(err, ecap, "editor tag unknown (is the track editor open?)");
        return 0;
    }
    for (i = 0; i < n && objects + gates < (int)(sizeof doomed / sizeof doomed[0]); i++) {
        uint8_t *act = actors[i];
        if (!act || !is_editor_actor(act, tag)) {
            continue;
        }
        if (is_gate(act)) {
            gates++;
        } else {
            objects++;
        }
        doomed[objects + gates - 1] = act;
    }
    /* destroying removes actors from the level array, so only after the walk */
    for (i = 0; i < objects + gates; i++) {
        destroy_actor(doomed[i]);
    }
    if (gates) {
        refresh_gate_ui();
    }
    outf("cleared objects=%d gates=%d", objects, gates);
    return 1;
}

static int current_base(char *out, int cap) {
    uint8_t *pawn = editor_pawn();
    int off = field_offset(pawn, "BaseRowLevel");
    out[0] = 0;
    if (off <= 0) {
        return 0;
    }
    fname_str(rd64(pawn + off), out, cap);
    return 1;
}

static int verb_base(const struct args *a, char *err, int ecap) {
    uint8_t *pawn = editor_pawn(), *fn = object_function(pawn, "LoadLevel");
    uint8_t parms[0x40];
    struct row row;
    int o;
    if (a->n < 2) {
        snprintf(err, ecap, "usage: base <row>");
        return 0;
    }
    if (!fn) {
        snprintf(err, ecap, "editor function 'LoadLevel' not found");
        return 0;
    }
    if (!table_row("DT_LevelName", a->v[1], &row) && !make_fname(a->v[1], &row.fname)) {
        snprintf(err, ecap, "unknown base level %s", a->v[1]);
        return 0;
    }
    o = param_offset(fn, "RowName");
    memset(parms, 0, sizeof parms);
    memcpy(parms + (o >= 0 ? o : 0), &row.fname, 8);
    call(pawn, fn, parms);
    outf("base %s requested", a->v[1]);
    return 1;
}

static int map_directory(char *out, int cap) {
    uint8_t *gi = game_instance();
    int off = field_offset(gi, "LoadedMapDirectory");
    struct fstr s;
    out[0] = 0;
    if (off <= 0) {
        return 0;
    }
    memcpy(&s, gi + off, sizeof s);
    fstring_to_utf8(&s, out, cap);
    return 1;
}

static char *last_separator(char *path) {
    char *a = strrchr(path, '/'), *b = strrchr(path, '\\');
    return a > b ? a : b;
}

static void map_folder(char *out, int cap) {
    char dir[1024], *sep;
    map_directory(dir, sizeof dir);
    sep = last_separator(dir);
    snprintf(out, cap, "%s", sep ? sep + 1 : dir);
}

static int verb_setmap(const char *rest, char *err, int ecap) {
    uint8_t *gi = game_instance();
    int off = field_offset(gi, "LoadedMapDirectory");
    char dir[1024], fresh[1400], *sep;
    struct fstr s;
    rest += strspn(rest, " \t");
    if (!*rest || strlen(rest) >= 1024) {
        snprintf(err, ecap, "usage: setmap <folder|absolute directory>");
        return 0;
    }
    if (off <= 0) {
        snprintf(err, ecap, "LoadedMapDirectory not found");
        return 0;
    }
    if ((isalpha((unsigned char)rest[0]) && rest[1] == ':') || rest[0] == '/' || rest[0] == '\\') {
        /* A newly created, unsaved map has no directory to retarget. The host
           supplies its known absolute map directory after the first save. */
        snprintf(fresh, sizeof fresh, "%s", rest);
    } else {
        if (!map_directory(dir, sizeof dir) || !(sep = last_separator(dir))) {
            snprintf(err, ecap, "the editor has no current map directory; pass an absolute directory");
            return 0;
        }
        sep[1] = 0;
        snprintf(fresh, sizeof fresh, "%s%s", dir, rest);
    }
    if (!engine_fstring(fresh, &s)) {
        snprintf(err, ecap, "could not build the path");
        return 0;
    }
    /* the old string stays on the engine heap (tens of bytes) rather than being
       freed with an allocator this DLL does not own */
    memcpy(gi + off, &s, sizeof s);
    outf("map %s", fresh);
    return 1;
}

static int verb_laps(const struct args *a, char *err, int ecap) {
    uint8_t *gm = editor_game_mode();
    int off = field_offset(gm, "TrackLap");
    if (a->n < 2 || off <= 0) {
        snprintf(err, ecap, a->n < 2 ? "usage: laps <n>" : "TrackLap not found on the game mode");
        return 0;
    }
    *(int32_t *)(gm + off) = atoi(a->v[1]);
    refresh_gate_ui();
    outf("laps %d", atoi(a->v[1]));
    return 1;
}

static int verb_save(const char *rest, char *err, int ecap) {
    static uint8_t parms[0x400];
    uint8_t *pawn = editor_pawn(), *fn = object_function(pawn, "SaveMap");
    char folder[256] = {0}, title[512] = {0};
    struct fstr name;
    uint8_t text[24];
    int o_name, o_full;
    if (!fn) {
        snprintf(err, ecap, "editor function 'SaveMap' not found");
        return 0;
    }
    sscanf(rest, "%255s %511[^\n]", folder, title);
    if (!folder[0] || strcmp(folder, "-") == 0) {
        map_folder(folder, sizeof folder);
    }
    if (!folder[0]) {
        snprintf(err, ecap, "no map is open and no folder was given");
        return 0;
    }
    if (!title[0]) {
        snprintf(title, sizeof title, "%s", folder);
    }
    o_name = param_offset(fn, "SaveName");
    o_full = param_offset(fn, "FullName");
    if (o_name < 0 || o_full < 0 || rd16(fn + UFUNC_PARMSSIZE) > (int)sizeof parms) {
        snprintf(err, ecap, "SaveMap signature changed");
        return 0;
    }
    if (!engine_fstring(folder, &name) || !engine_ftext(title, text)) {
        snprintf(err, ecap, "could not build engine strings");
        return 0;
    }
    memset(parms, 0, sizeof parms); /* autosave? and UpdateWorkShop? false, workshopRef "" */
    memcpy(parms + o_name, &name, sizeof name);
    memcpy(parms + o_full, text, 24);
    call(pawn, fn, parms);
    outf("saved folder=%s title=%s", folder, title);
    return 1;
}

static int verb_mapinfo(void) {
    char dir[1024], folder[256], base[128], tag[128];
    int32_t n, i, objects = 0, gates = 0;
    uint8_t **actors = level_actors(&n);
    uint64_t t = editor_tag();
    for (i = 0; i < n; i++) {
        if (actors[i] && is_editor_actor(actors[i], t)) {
            if (is_gate(actors[i])) {
                gates++;
            } else {
                objects++;
            }
        }
    }
    map_directory(dir, sizeof dir);
    map_folder(folder, sizeof folder);
    current_base(base, sizeof base);
    fname_str(t, tag, sizeof tag);
    outf("folder %s", folder);
    outf("directory %s", dir);
    outf("base %s", base);
    outf("tag %s", t ? tag : "");
    outf("objects %d", objects);
    outf("gates %d", gates);
    outf("level_actors %d", n);
    return 1;
}

static int verb_resolve(void) {
    uint8_t *pawn = editor_pawn(), *ctrl = 0, *cam = 0, *mesh_cdo = class_cdo("StaticMeshComponent");
    int off;
    outf("names_rva=%llx", (unsigned long long)((uint8_t *)g_name_blocks - game_base));
    outf("objects_rva=%llx", (unsigned long long)((uint8_t *)g_objects - game_base));
    outf("gworld_rva=%llx", (unsigned long long)((uint8_t *)g_world_slot - game_base));
    outf("world_level_off=%x", g_world_level_off);
    outf("processevent_slot=%d", g_pe_slot);
    outf("processevent_rva=%llx", (unsigned long long)(rd64((uint8_t *)rd64(class_cdo("Object")) + g_pe_slot * 8) -
                                                       (uint64_t)game_base));
    if (pawn) {
        off = field_offset(pawn, "Controller");
        ctrl = off > 0 ? rdp(pawn + off) : 0;
        off = field_offset(pawn, "Camera");
        cam = off > 0 ? rdp(pawn + off) : 0;
        outf("pawn_camera_off=%x", off);
    }
    if (ctrl && valid_object(ctrl)) {
        outf("controller_vt_rva=%llx", (unsigned long long)(rd64(ctrl) - (uint64_t)game_base));
        outf("controller_pawn_off=%x", field_offset(ctrl, "Pawn"));
        off = field_offset(ctrl, "PlayerCameraManager");
        outf("controller_camera_manager_off=%x", off);
        if (off > 0 && valid_object(rdp(ctrl + off))) {
            /* APlayerCameraManager::ViewTarget (FTViewTarget); its POV follows the target pointer */
            outf("camera_view_target_off=%x", field_offset(rdp(ctrl + off), "ViewTarget"));
        }
    }
    if (cam && valid_object(cam)) {
        outf("camera_vt_rva=%llx", (unsigned long long)(rd64(cam) - (uint64_t)game_base));
    }
    if (mesh_cdo) {
        outf("mesh_vt_rva=%llx", (unsigned long long)(rd64(mesh_cdo) - (uint64_t)game_base));
    }
    return 1;
}

/* ---- debug verbs ---- */

static void prop_desc(uint8_t *p, char *buf, int cap) {
    char cname[64], inner[128], pname[256];
    uint64_t flags = rd64(p + FPROP_FLAGS);
    prop_type(p, cname, sizeof cname, inner, sizeof inner);
    fname_str(rd64(p + FFIELD_NAME), pname, sizeof pname);
    snprintf(buf, cap, "%s %s off=0x%x size=%d flags=0x%llx%s%s%s%s%s%s", cname, pname, rd32(p + FPROP_OFFSET),
             rd32(p + FPROP_ELEMSIZE), (unsigned long long)flags, (flags & CPF_PARM) ? " parm" : "",
             (flags & CPF_OUTPARM) ? " out" : "", (flags & CPF_REFPARM) ? " ref" : "", inner[0] ? " <" : "", inner,
             inner[0] ? ">" : "");
}

static void dump_props(uint8_t *s, int supers) {
    char nm[256];
    int depth = 0;
    while (s && readable(s, 0x60) && depth++ < 48) {
        uint8_t *p = rdp(s + USTRUCT_CHILDPROPS);
        int guard = 0;
        obj_name(s, nm, sizeof nm);
        outf("struct %llx %s size=0x%x", (unsigned long long)s, nm, rd32(s + USTRUCT_SIZE));
        while (p && readable(p, 0x80) && guard++ < 4000) {
            char d[600];
            prop_desc(p, d, sizeof d);
            outf("  %s", d);
            p = rdp(p + FFIELD_NEXT);
        }
        if (!supers) {
            return;
        }
        s = rdp(s + USTRUCT_SUPER);
    }
}

static void dump_function(uint8_t *f) {
    char nm[256], owner[1100];
    obj_name(f, nm, sizeof nm);
    obj_path(rdp(f + UOBJ_OUTER), owner, sizeof owner);
    outf("function %llx %s owner=%s flags=0x%x parms=%u size=0x%x", (unsigned long long)f, nm, owner,
         (unsigned)rd32(f + UFUNC_FLAGS), f[UFUNC_NUMPARMS], rd16(f + UFUNC_PARMSSIZE));
    dump_props(f, 0);
}

struct names_ctx {
    const char *sub;
    int limit, count;
};

static int names_cb(uint32_t index, const char *text, void *p) {
    struct names_ctx *c = (struct names_ctx *)p;
    if (icontains(text, c->sub)) {
        outf("0x%x %s", index, text);
        return ++c->count >= c->limit;
    }
    return 0;
}

static void hexdump(const uint8_t *p, int n) {
    int i, j;
    char line[200];
    for (i = 0; i < n; i += 16) {
        int used = snprintf(line, sizeof line, "%llx ", (unsigned long long)(p + i));
        for (j = i; j < i + 16 && j < n; j++) {
            used += snprintf(line + used, sizeof line - used, "%02x ", p[j]);
        }
        outf("%s", line);
    }
}

#include "camera.h"
#include "groups.h"
#include "anchors.h"
#include "zones.h"

static int parse_hex(const char *s, uint8_t *buf, int cap) {
    int n = 0;
    while (s[0] && s[1] && n < cap && isxdigit((unsigned char)s[0]) && isxdigit((unsigned char)s[1])) {
        char pair[3] = {s[0], s[1], 0};
        buf[n++] = (uint8_t)strtoul(pair, NULL, 16);
        s += 2;
    }
    return n;
}

/* returns 1 ok, 0 failed, -1 not a debug verb */
static int verb_debug(const char *verb, const struct args *a, char *err, int ecap) {
    int32_t n = object_count(), i, count = 0;
    const char *s1 = a->n > 1 ? a->v[1] : "";
    int limit = a->n > 2 ? atoi(a->v[2]) : 100;
    char nm[256], cn[256], path[1100];
    if (ieq(verb, "names")) {
        struct names_ctx c = {s1, limit, 0};
        walk_names(names_cb, &c);
        outf("count %d", c.count);
    } else if (ieq(verb, "objects")) {
        for (i = 0; i < n && count < limit; i++) {
            uint8_t *o = object_at(i);
            if (!o || !valid_object(o)) {
                continue;
            }
            obj_name(o, nm, sizeof nm);
            obj_class_name(o, cn, sizeof cn);
            if (icontains(nm, s1) || icontains(cn, s1)) {
                obj_path(o, path, sizeof path);
                outf("%llx idx=%d class=%s %s", (unsigned long long)o, i, cn, path);
                count++;
            }
        }
        outf("count %d of %d", count, n);
    } else if (ieq(verb, "object")) {
        uint8_t *o = (uint8_t *)strtoull(s1, NULL, 16);
        if (!o || !alive(o)) { snprintf(err, ecap, "object unavailable"); return 0; }
        obj_path(o, path, sizeof path); obj_class_name(o, cn, sizeof cn);
        outf("%llx class=%s %s", (unsigned long long)o, cn, path);
    } else if (ieq(verb, "funcs")) {
        for (i = 0; i < n && count < limit; i++) {
            uint8_t *o = object_at(i);
            if (!o || !valid_object(o)) {
                continue;
            }
            obj_name(o, nm, sizeof nm);
            if (icontains(nm, s1) && is_a(o, "Function")) {
                dump_function(o);
                count++;
            }
        }
        outf("count %d", count);
    } else if (ieq(verb, "func")) {
        uint8_t *f = (uint8_t *)strtoull(s1, NULL, 16);
        if (!valid_object(f) || !is_a(f, "Function")) {
            snprintf(err, ecap, "not a function");
            return 0;
        }
        dump_function(f);
    } else if (ieq(verb, "props")) {
        uint8_t *s = find_struct(s1);
        if (!s) {
            snprintf(err, ecap, "no struct %s", s1);
            return 0;
        }
        dump_props(s, 1);
    } else if (ieq(verb, "rows")) {
        const char *table = a->n > 3 ? a->v[3] : "DT_MapEditorAssets";
        uint8_t *dt = data_table(table), *elems;
        int32_t num;
        if (!dt) {
            snprintf(err, ecap, "no table %s", table);
            return 0;
        }
        elems = rdp(dt + DT_ROWMAP);
        num = rd32(dt + DT_ROWMAP + 8);
        for (i = 0; i < num && count < limit && readable(elems + (size_t)i * ROWMAP_ELEM, ROWMAP_ELEM); i++) {
            fname_str(rd64(elems + (size_t)i * ROWMAP_ELEM), nm, sizeof nm);
            if (icontains(nm, s1)) {
                outf("%s data=%llx", nm, (unsigned long long)rd64(elems + (size_t)i * ROWMAP_ELEM + 8));
                count++;
            }
        }
        outf("count %d of %d", count, num);
    } else if (ieq(verb, "fname")) {
        uint64_t f;
        if (!make_fname(s1, &f)) {
            snprintf(err, ecap, "not interned");
            return 0;
        }
        fname_str(f, nm, sizeof nm);
        outf("fname %016llx %s", (unsigned long long)f, nm);
    } else if (ieq(verb, "read")) {
        uint8_t *p = (uint8_t *)strtoull(s1, NULL, 16);
        int len = a->n > 2 ? atoi(a->v[2]) : 64;
        if (len < 1 || len > 65536 || !readable(p, len)) {
            snprintf(err, ecap, "unreadable");
            return 0;
        }
        hexdump(p, len);
    } else if (ieq(verb, "write")) {
        uint8_t *p = (uint8_t *)strtoull(s1, NULL, 16);
        uint8_t buf[1024];
        int len = parse_hex(a->n > 2 ? a->v[2] : "", buf, sizeof buf);
        if (!readable(p, len)) {
            snprintf(err, ecap, "unreadable");
            return 0;
        }
        memcpy(p, buf, len);
        outf("wrote %d", len);
    } else if (ieq(verb, "call")) {
        static uint8_t parms[4096];
        uint8_t *obj = (uint8_t *)strtoull(s1, NULL, 16);
        uint8_t *fn = (uint8_t *)strtoull(a->n > 2 ? a->v[2] : "0", NULL, 16);
        int len, size;
        if (!valid_object(obj) || !valid_object(fn) || !is_a(fn, "Function")) {
            snprintf(err, ecap, "bad object or function");
            return 0;
        }
        memset(parms, 0, sizeof parms);
        len = parse_hex(a->n > 3 ? a->v[3] : "", parms, sizeof parms);
        size = rd16(fn + UFUNC_PARMSSIZE);
        call(obj, fn, parms);
        hexdump(parms, len > size ? len : size);
    } else {
        return -1;
    }
    return 1;
}

/* ======================================================================
   dispatch, batch, jobs
   ====================================================================== */

static volatile int want_unload;

static int dispatch(const char *line, int allow_batch);

static int verb_batch(const char *path, int quiet) {
    static char line[4096];
    FILE *in = fopen(path, "r");
    int n = 0, ok = 0, failed = 0;
    if (!in) {
        outf("error cannot open %s", path);
        return 0;
    }
    while (fgets(line, sizeof line, in)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!line[0] || line[0] == '#') {
            continue;
        }
        snprintf(out_prefix, sizeof out_prefix, "#%d ", n);
        quiet_batch = quiet;
        if (dispatch(line, 0)) {
            ok++;
        } else {
            failed++;
        }
        quiet_batch = 0;
        n++;
    }
    out_prefix[0] = 0;
    fclose(in);
    outf("batch lines=%d ok=%d failed=%d", n, ok, failed);
    return 1;
}

/* line is "verb args..." without a sequence number. 1 ok, 0 failed. */
static int dispatch(const char *line, int allow_batch) {
    char copy[4096], err[300] = {0};
    struct args a;
    const char *verb, *rest;
    int r;
    snprintf(copy, sizeof copy, "%s", line);
    split_args(copy, &a);
    if (!a.n) {
        return 1;
    }
    verb = a.v[0];
    rest = line + strspn(line, " \t");
    rest += strcspn(rest, " \t");
    if (ieq(verb, "spawn")) {
        r = verb_spawn(&a, err, sizeof err);
    } else if (ieq(verb, "gate")) {
        r = verb_gate(&a, err, sizeof err);
    } else if (ieq(verb, "get")) {
        r = verb_get(&a, err, sizeof err);
    } else if (ieq(verb, "transform")) {
        r = verb_transform(&a, err, sizeof err);
    } else if (ieq(verb, "replace")) {
        r = verb_replace(&a, err, sizeof err);
    } else if (ieq(verb, "destroy")) {
        r = verb_destroy(&a, err, sizeof err);
    } else if (ieq(verb, "destroyat")) {
        r = verb_destroyat(&a, err, sizeof err);
    } else if (ieq(verb, "floating")) {
        r = verb_floating(&a, err, sizeof err);
    } else if (ieq(verb, "groups")) {
        r = verb_groups(&a, err, sizeof err);
    } else if (ieq(verb, "edit")) {
        r = verb_edit(&a, err, sizeof err);
    } else if (ieq(verb, "contacts")) {
        r = verb_contacts(&a, err, sizeof err);
    } else if (ieq(verb, "anchors")) {
        r = verb_anchors(&a, err, sizeof err);
    } else if (ieq(verb, "zones")) {
        r = verb_zones(&a, err, sizeof err);
    } else if (ieq(verb, "site")) {
        r = verb_site(&a, err, sizeof err);
    } else if (ieq(verb, "camera")) {
        r = verb_camera(&a, err, sizeof err);
    } else if (ieq(verb, "capture")) {
        r = verb_capture(&a, err, sizeof err);
    } else if (ieq(verb, "worldbounds")) {
        r = verb_worldbounds(&a, err, sizeof err);
    } else if (ieq(verb, "inventory")) {
        r = verb_inventory(&a, err, sizeof err);
    } else if (ieq(verb, "audit")) {
        r = verb_audit(&a, err, sizeof err);
    } else if (ieq(verb, "bounds")) {
        r = verb_bounds(&a, err, sizeof err);
    } else if (ieq(verb, "collide")) {
        r = verb_collide(&a, err, sizeof err);
    } else if (ieq(verb, "trace")) {
        r = verb_trace(&a, err, sizeof err);
    } else if (ieq(verb, "terrain")) {
        r = verb_terrain(&a, err, sizeof err);
    } else if (ieq(verb, "drop")) {
        r = verb_drop(&a, err, sizeof err);
    } else if (ieq(verb, "destroybox")) {
        r = verb_destroybox(&a, err, sizeof err);
    } else if (ieq(verb, "boxes")) {
        r = verb_boxes(&a, err, sizeof err);
    } else if (ieq(verb, "grid")) {
        r = verb_grid(&a, err, sizeof err);
    } else if (ieq(verb, "console")) {
        r = verb_console(line, err, sizeof err);
    } else if (ieq(verb, "cvar")) {
        r = verb_cvar(&a, err, sizeof err);
    } else if (ieq(verb, "raise")) {
        r = verb_raise(&a, err, sizeof err);
    } else if (ieq(verb, "cells")) {
        r = verb_cells(&a, err, sizeof err);
    } else if (ieq(verb, "gates")) {
        r = verb_gates();
    } else if (ieq(verb, "actors")) {
        r = verb_actors(&a);
    } else if (ieq(verb, "clear")) {
        r = verb_clear(err, sizeof err);
    } else if (ieq(verb, "base")) {
        r = verb_base(&a, err, sizeof err);
    } else if (ieq(verb, "setmap")) {
        r = verb_setmap(rest, err, sizeof err);
    } else if (ieq(verb, "laps")) {
        r = verb_laps(&a, err, sizeof err);
    } else if (ieq(verb, "save")) {
        r = verb_save(rest, err, sizeof err);
    } else if (ieq(verb, "mapinfo")) {
        r = verb_mapinfo();
    } else if (ieq(verb, "resolve")) {
        r = verb_resolve();
    } else if (ieq(verb, "batch") && allow_batch) {
        r = a.n > 1 ? verb_batch(a.v[1], a.n > 2 && ieq(a.v[2], "quiet"))
                    : (snprintf(err, sizeof err, "usage: batch <path> [quiet]"), 0);
    } else {
        r = verb_debug(verb, &a, err, sizeof err);
        if (r < 0) {
            snprintf(err, sizeof err, "unknown verb %s", verb);
            r = 0;
        }
    }
    if (!r) {
        outf("error %s", err[0] ? err : "failed");
    }
    return r;
}

static char job_line[4096];
static volatile int job_ready;

static uint64_t job_fn(void *arg) {
    const char *line = (const char *)arg, *rest;
    char verb[64] = {0};
    int seq = 0;
    if (sscanf(line, "%d %63s", &seq, verb) < 2) {
        outf("0 error bad command");
        return 0;
    }
    rest = strchr(line, ' ') + 1;
    if (ieq(verb, "ping")) {
        outf("%d ok pong build=%s %s resolved=%d", seq, __DATE__, __TIME__, resolved());
        return 1;
    }
    if (ieq(verb, "unload")) {
        if (!foliage_restore()) {
            outf("%d error %s; unload refused to preserve restoration data", seq, foliage_error);
            return 0;
        }
        capture_stop_native();
        camera_release_native();
        want_unload = 1;
        outf("%d ok unloading", seq);
        return 1;
    }
    if (!resolved()) {
        outf("%d error unresolved: %s", seq, g_scan_error[0] ? g_scan_error : "unknown");
        return 0;
    }
    if (ieq(verb, "scan")) {
        outf("%d ok scan", seq);
        return (uint64_t)verb_resolve();
    }
    outf("%d ok %s", seq, verb);
    return (uint64_t)dispatch(rest, 1);
}

static void run_job(void) {
    char line[4096], verb[64] = {0};
    int seq = 0;
    uint64_t ok;
    DWORD t0 = GetTickCount();
    snprintf(line, sizeof line, "%s", job_line);
    sscanf(line, "%d %63s", &seq, verb);
    flush_readable_cache();
    out_prefix[0] = 0;
    out_file = fopen(OUT_FILE ".tmp", "w");
    if (ieq(verb, "scan")) {
        reset_resolution();
    }
    if (!ieq(verb, "ping") && !ieq(verb, "unload")) {
        ensure_resolved();
    }
    ok = guarded((void *)job_fn, line);
    out_prefix[0] = 0;
    if (tripped) {
        outf("%d fault %lx rip=%llx rva=%llx", seq, (unsigned long)fault_code, (unsigned long long)fault_rip,
             in_game(fault_rip) ? (unsigned long long)(fault_rip - (uint64_t)game_base) : 0ull);
        logf_("job %d faulted %lx at %llx", seq, (unsigned long)fault_code, (unsigned long long)fault_rip);
    }
    outf("end %d %s", seq, ok && !tripped ? "ok" : "failed");
    logf_("job %d %s %lums", seq, verb, (unsigned long)(GetTickCount() - t0));
    if (out_file) {
        fclose(out_file);
        out_file = 0;
        DeleteFileA(OUT_FILE);
        MoveFileA(OUT_FILE ".tmp", OUT_FILE);
    }
}

/* ======================================================================
   threads and loading
   ====================================================================== */

static WNDPROC old_proc;
static HWND game_window;
static HINSTANCE self_module;
static PVOID veh_handle;
static int managed_faulted;

static HWND agent_game_window(void) { return game_window; }

static LRESULT CALLBACK wndproc(HWND hwnd, UINT msg, WPARAM wparam, LPARAM lparam) {
    /* UE calls can pump window messages (render readback / level IO). Never
       nest guarded jobs: the fault guard owns one private stack/resume context. */
    if (job_ready && !armed) {
        job_ready = 0;
        run_job();
    }
    if (!armed && !want_unload && !managed_faulted && resolved() && GetTickCount() - foliage_last_poll >= 1000) {
        foliage_last_poll = GetTickCount();
        if (!guarded((void *)managed_tick, 0) || tripped) {
            /* Stop automatic work after faults; explicit commands still work. */
            managed_faulted = 1;
            logf_("managed background work faulted; reload DLL before re-enabling");
        }
    }
    return CallWindowProcW(old_proc, hwnd, msg, wparam, lparam);
}

static BOOL CALLBACK find_window(HWND hwnd, LPARAM param) {
    wchar_t title[80];
    DWORD pid = 0;
    GetWindowThreadProcessId(hwnd, &pid);
    if (pid == GetCurrentProcessId() && GetWindowTextW(hwnd, title, 80) && wcsstr(title, L"Uncrashed")) {
        *(HWND *)param = hwnd;
        return FALSE;
    }
    return TRUE;
}

static int read_line(const char *path, char *buf, int cap) {
    FILE *in = fopen(path, "r");
    int ok = 0;
    if (in) {
        ok = fgets(buf, cap, in) != 0;
        fclose(in);
    }
    if (ok) {
        buf[strcspn(buf, "\r\n")] = 0;
    }
    return ok;
}

static DWORD WINAPI pump(void *unused) {
    int seen = -1;
    char line[4096];
    (void)unused;
    if (read_line(CMD_FILE, line, sizeof line)) {
        sscanf(line, "%d", &seen); /* a command left over from an earlier process */
    }
    logf_("agent ready build=%s %s window=%p", __DATE__, __TIME__, (void *)game_window);
    while (!want_unload) {
        int seq = 0;
        if (read_line(CMD_FILE, line, sizeof line) && sscanf(line, "%d", &seq) == 1 && seq != seen) {
            seen = seq;
            snprintf(job_line, sizeof job_line, "%s", line);
            job_ready = 1;
            SendMessageW(game_window, WM_NULL, 0, 0);
        }
        if (job_ready || (!managed_faulted && GetTickCount() - foliage_last_poll >= 1000))
            SendMessageW(game_window, WM_NULL, 0, 0);
        Sleep(50);
    }
    SetWindowLongPtrW(game_window, GWLP_WNDPROC, (LONG_PTR)old_proc);
    Sleep(300);
    if (veh_handle) {
        RemoveVectoredExceptionHandler(veh_handle);
    }
    logf_("agent unloaded");
    FreeLibraryAndExitThread(self_module, 0);
    return 0;
}

BOOL WINAPI DllMain(HINSTANCE module, DWORD reason, LPVOID reserved) {
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        IMAGE_DOS_HEADER *dos;
        IMAGE_NT_HEADERS64 *nt;
        uint8_t *stack;
        DWORD old;
        self_module = module;
        DisableThreadLibraryCalls(module);
        game_base = (uint8_t *)GetModuleHandleW(GAME_MODULE);
        if (!game_base) {
            return FALSE;
        }
        dos = (IMAGE_DOS_HEADER *)game_base;
        nt = (IMAGE_NT_HEADERS64 *)(game_base + dos->e_lfanew);
        game_end = (uint64_t)game_base + nt->OptionalHeader.SizeOfImage;
        /* private job stack; its low page is a guard against runaway recursion */
        stack = (uint8_t *)VirtualAlloc(NULL, JOB_STACK_BYTES + 0x1000u, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
        if (!stack || !VirtualProtect(stack, 0x1000u, PAGE_NOACCESS, &old)) {
            return FALSE;
        }
        job_hi = stack + 0x1000u + JOB_STACK_BYTES;
        EnumWindows(find_window, (LPARAM)&game_window);
        if (!game_window) {
            return FALSE; /* jobs must run on the game thread */
        }
        veh_handle = AddVectoredExceptionHandler(1, veh);
        old_proc = (WNDPROC)SetWindowLongPtrW(game_window, GWLP_WNDPROC, (LONG_PTR)wndproc);
        CreateThread(NULL, 0, pump, NULL, 0, NULL);
    }
    return TRUE;
}
