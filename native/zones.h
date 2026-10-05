/* Map-scoped foliage exclusion zones. Non-destructive instance masking, not
   RemoveInstance: original transforms retained until restoration. Base foliage
   only; placed objects/gates never touched. Zone file is portable geometry, not
   unstable component addresses. Automatic reload while DLL remains resident. */
#include "foliage_state.h"

#define ZONE_CAP 128
#define MASK_CAP 60000
struct foliage_zone { char name[97]; float c[3],h[3],yaw; };
struct foliage_mask { uint8_t *comp; int index,serial; uint8_t local[48]; float lo[3],hi[3]; };
static struct foliage_mask foliage_masks[MASK_CAP];
static int foliage_nmask,foliage_nzones,foliage_failures,foliage_truncated,foliage_unavailable;
static char foliage_error[200],foliage_seen_session[160];
static struct foliage_zone foliage_zones[ZONE_CAP];
static uint64_t foliage_seen_stamp,foliage_seen_levels;
static int foliage_seen_objects;
static DWORD foliage_last_poll;

static int foliage_transform(uint8_t *comp,int index,uint8_t transform[48],int write,int world) {
    static uint8_t *get,*set; static int gi,gx,gw,gr,gs,si,sx,sw,sr,sd,st,ss;
    uint8_t p[0x100];
    if(!get) {
        get=object_function(comp,"GetInstanceTransform"); set=object_function(comp,"UpdateInstanceTransform");
        if(!get || !set) { get=0; return 0; }
        gi=param_offset(get,"InstanceIndex"); gx=param_offset(get,"OutInstanceTransform"); gw=param_offset(get,"bWorldSpace"); gr=param_offset(get,"ReturnValue");
        si=param_offset(set,"InstanceIndex"); sx=param_offset(set,"NewInstanceTransform"); sw=param_offset(set,"bWorldSpace"); sr=param_offset(set,"ReturnValue");
        sd=param_offset(set,"bMarkRenderStateDirty"); st=param_offset(set,"bTeleport"); gs=rd16(get+UFUNC_PARMSSIZE); ss=rd16(set+UFUNC_PARMSSIZE);
        if(gs>(int)sizeof p || ss>(int)sizeof p || gi<0 || gx<0 || gw<0 || gr<0 || si<0 || sx<0 || sw<0 || sr<0 || sd<0 || st<0 ||
           gi+4>gs || gx+48>gs || gw>=gs || gr>=gs || si+4>ss || sx+48>ss || sw>=ss || sr>=ss || sd>=ss || st>=ss) { get=0; return 0; }
    }
    memset(p,0,sizeof p);
    if(write) {
        memcpy(p+si,&index,4); memcpy(p+sx,transform,48); p[sw]=(uint8_t)world; p[sd]=1; p[st]=1;
        call(comp,set,p); return p[sr]!=0;
    }
    memcpy(p+gi,&index,4); p[gw]=(uint8_t)world; call(comp,get,p);
    if(!p[gr]) return 0; memcpy(transform,p+gx,48); return 1;
}

static int foliage_actor_set(uint8_t *act,int hidden,int collision) {
    uint8_t p[0x20],h=(uint8_t)hidden,c=(uint8_t)collision;
    uint8_t *fh=object_function(act,"SetActorHiddenInGame"),*fc=object_function(act,"SetActorEnableCollision");
    memset(p,0,sizeof p);
    if(!np_put(fh,p,sizeof p,"bNewHidden",&h,1) || !fc || param_offset(fc,"bNewActorEnableCollision")<0) return 0;
    call(act,fh,p); memset(p,0,sizeof p);
    if(!np_put(fc,p,sizeof p,"bNewActorEnableCollision",&c,1)) return 0;
    call(act,fc,p); return 1;
}

static int foliage_mask_alive(struct foliage_mask *m) {
    uint8_t now[48],*item; int i,serial;
    if(!alive(m->comp) || (m->index==-2?!env_named_base_plant(m->comp):!env_foliage_component(m->comp))) return 0;
    if(m->index==-2) {
        int hidden,collision; float o[3],e[3];
        if(!np_bool(m->comp,"bHidden",-1,&hidden) || !np_bool(m->comp,"bActorEnableCollision",-1,&collision) || !hidden || collision || !actor_bounds(m->comp,o,e)) return 0;
        for(i=0;i<3;i++) if(fabsf(o[i]-e[i]-m->lo[i])>.2f || fabsf(o[i]+e[i]-m->hi[i])>.2f) return 0;
    }
    item=object_item(rd32(m->comp+UOBJ_INDEX)); serial=item?rd32(item+0x10):0;
    if(m->serial && serial!=m->serial) return 0;
    if(!m->serial) m->serial=serial;
    if(m->index==-2) return 1;
    if(!foliage_transform(m->comp,m->index,now,0,0)) return 0;
    return foliage_hidden_transform_matches(m->local,now);
}

static int foliage_restore(void) {
    int i,kept=0;
    for(i=0;i<foliage_nmask;i++) {
        struct foliage_mask *m=&foliage_masks[i];
        /* Dead components belong to an unloaded base; nothing to restore. */
        if(!alive(m->comp)) continue;
        if(!foliage_mask_alive(m)) {
            uint8_t now[48];
            /* A failed setter can leave original state unchanged. No restoration
               needed, but never overwrite an externally edited/moved instance. */
            if(m->index>=0 && foliage_transform(m->comp,m->index,now,0,0) && foliage_original_transform_matches(m->local,now)) continue;
            foliage_masks[kept++]=*m; continue;
        }
        if(!(m->index==-2?foliage_actor_set(m->comp,m->local[0],m->local[1]):foliage_transform(m->comp,m->index,m->local,1,0)))
            foliage_masks[kept++]=*m;
    }
    foliage_nmask=kept;
    if(kept) snprintf(foliage_error,sizeof foliage_error,"%d instance restoration conflicts; originals retained; inspect before unload/reload",kept);
    return kept==0;
}

/* SAT for yaw-oriented clearing prism vs visual instance AABB. Bounds can include
   empty crown volume: whole-instance conservative clearing, not branch trimming. */
static int foliage_overlap(const struct foliage_zone *z,const float lo[3],const float hi[3]) {
    float c=cosf(z->yaw*(float)M_PI/180),s=sinf(z->yaw*(float)M_PI/180);
    float dx=(lo[0]+hi[0])*.5f-z->c[0],dy=(lo[1]+hi[1])*.5f-z->c[1];
    float ex=(hi[0]-lo[0])*.5f,ey=(hi[1]-lo[1])*.5f;
    if(lo[2]>z->c[2]+z->h[2] || hi[2]<z->c[2]-z->h[2]) return 0;
    if(fabsf(dx)>ex+fabsf(c)*z->h[0]+fabsf(s)*z->h[1] || fabsf(dy)>ey+fabsf(s)*z->h[0]+fabsf(c)*z->h[1]) return 0;
    return fabsf(c*dx+s*dy)<=z->h[0]+fabsf(c)*ex+fabsf(s)*ey && fabsf(-s*dx+c*dy)<=z->h[1]+fabsf(s)*ex+fabsf(c)*ey;
}

static int foliage_apply(void) {
    int i,j,hit; foliage_failures=foliage_truncated=foliage_unavailable=0;
    if(!foliage_restore()) return 0;
    if(!foliage_nzones) return 1;
    for(i=0;i<3;i++) { env_tree_roi_lo[i]=1e30f; env_tree_roi_hi[i]=-1e30f; }
    for(j=0;j<foliage_nzones;j++) {
        struct foliage_zone *z=&foliage_zones[j]; float c=fabsf(cosf(z->yaw*(float)M_PI/180)),s=fabsf(sinf(z->yaw*(float)M_PI/180));
        float e[3]={c*z->h[0]+s*z->h[1],s*z->h[0]+c*z->h[1],z->h[2]};
        env_extend(env_tree_roi_lo,env_tree_roi_hi,z->c,e);
    }
    env_nplants=0; env_tree_roi=1; env_tree_min_height=0; env_collect_trees(); env_tree_min_height=200; env_tree_roi=0;
    foliage_truncated=env_tree_truncated; foliage_unavailable=env_tree_unavailable;
    if(env_tree_truncated) { snprintf(foliage_error,sizeof foliage_error,"foliage inventory incomplete; no new masks applied"); return 0; }
    for(i=0;i<env_ntrees;i++) {
        struct env_tree *tree=&env_trees[i]; struct foliage_mask *m; uint8_t hidden[48],*item;
        hit=0;
        for(j=0;j<foliage_nzones;j++) if(foliage_overlap(&foliage_zones[j],tree->lo,tree->hi)) { hit=1; break; }
        if(!hit) continue;
        if(foliage_nmask==MASK_CAP) { foliage_truncated=1; break; }
        m=&foliage_masks[foliage_nmask]; m->comp=tree->comp; m->index=tree->instance;
        if(m->index==-2) {
            int hidden_before,collision_before;
            if(!np_bool(m->comp,"bHidden",-1,&hidden_before) || !np_bool(m->comp,"bActorEnableCollision",-1,&collision_before)) { foliage_failures++; continue; }
            memset(m->local,0,48); m->local[0]=(uint8_t)hidden_before; m->local[1]=(uint8_t)collision_before;
        } else if(!foliage_transform(m->comp,m->index,m->local,0,0)) { foliage_failures++; continue; }
        memcpy(hidden,m->local,48); memset(hidden+32,0,12);
        item=object_item(rd32(m->comp+UOBJ_INDEX)); m->serial=item?rd32(item+0x10):0;
        memcpy(m->lo,tree->lo,12); memcpy(m->hi,tree->hi,12);
        /* Retain original BEFORE invoking setter, including a possible fault. */
        foliage_nmask++;
        if(!(m->index==-2?foliage_actor_set(m->comp,1,0):foliage_transform(m->comp,m->index,hidden,1,0))) foliage_failures++;
    }
    if(foliage_failures || foliage_truncated) {
        snprintf(foliage_error,sizeof foliage_error,"partial foliage apply: failures=%d truncated=%d",foliage_failures,foliage_truncated); return 0;
    }
    return 1;
}

static uint64_t foliage_levels_hash(void) {
    uint8_t *levels[ENV_LEVEL_CAP]; int truncated=0,n=env_levels(levels,ENV_LEVEL_CAP,&truncated),i;
    uint64_t h=1469598103934665603ull;
    for(i=0;i<n;i++) h=(h^(uint64_t)levels[i])*1099511628211ull;
    return h^(uint64_t)truncated;
}

static int foliage_refresh(int force) {
    char directory[1024],path[1200],base[128],savedbase[128],line[512],session[160];
    WIN32_FILE_ATTRIBUTE_DATA attrs; uint64_t stamp=0,levels; FILE *f; int n=0,bad=0,i;
    static struct foliage_zone pending[ZONE_CAP];
    if(!editor_pawn()) return foliage_restore();
    map_directory(directory,sizeof directory); current_base(base,sizeof base); managed_session(session,sizeof session);
    snprintf(path,sizeof path,"%s\\UncrashedMCP.zones",directory);
    if(directory[0] && GetFileAttributesExA(path,GetFileExInfoStandard,&attrs))
        stamp=((uint64_t)attrs.ftLastWriteTime.dwHighDateTime<<32)|attrs.ftLastWriteTime.dwLowDateTime;
    levels=foliage_levels_hash();
    if(!force && !strcmp(session,foliage_seen_session) && stamp==foliage_seen_stamp && levels==foliage_seen_levels &&
       (!foliage_nzones || foliage_seen_objects==object_count())) return 1;
    /* Never carry old-map exclusions into a different map, even when its new
       configuration is invalid. Same-map malformed edits keep existing masks. */
    if(strcmp(session,foliage_seen_session)) {
        if(!foliage_restore()) return 0;
        foliage_nzones=0;
    }
    /* Parse complete new config before touching same-map instances. */
    if(stamp) {
        f=fopen(path,"r");
        if(!f) { snprintf(foliage_error,sizeof foliage_error,"cannot read zone file"); return 0; }
        if(!fgets(line,sizeof line,f) || sscanf(line,"UNCRASHED_ZONES_V1 %127s",savedbase)!=1 || strcmp(savedbase,base)) bad=1;
        while(!bad && fgets(line,sizeof line,f)) {
            struct foliage_zone *z; char extra;
            if(n==ZONE_CAP) { bad=1; break; } z=&pending[n];
            if(sscanf(line,"Z %96s %f %f %f %f %f %f %f %c",z->name,&z->c[0],&z->c[1],&z->c[2],&z->h[0],&z->h[1],&z->h[2],&z->yaw,&extra)!=8 ||
               !group_valid(z->name) || !isfinite(z->yaw)) { bad=1; break; }
            for(i=0;i<3;i++) if(!isfinite(z->c[i]) || fabsf(z->c[i])>1e8f || !isfinite(z->h[i]) || z->h[i]<=0 || z->h[i]>1e6f) bad=1;
            n++;
        }
        if(ferror(f)) bad=1; fclose(f);
        if(bad) { snprintf(foliage_error,sizeof foliage_error,"invalid zones file or base mismatch; existing masks unchanged"); return 0; }
    }
    if(!foliage_restore()) return 0;
    memcpy(foliage_zones,pending,(size_t)n*sizeof pending[0]); foliage_nzones=n; foliage_error[0]=0;
    if(!foliage_apply()) return 0;
    snprintf(foliage_seen_session,sizeof foliage_seen_session,"%s",session); foliage_seen_stamp=stamp;
    foliage_seen_levels=levels; foliage_seen_objects=object_count(); return 1;
}

static int verb_zones(const struct args *a,char *err,int ecap) {
    if(a->n!=2 || (!ieq(a->v[1],"status") && !ieq(a->v[1],"refresh"))) { snprintf(err,ecap,"usage: zones status|refresh"); return 0; }
    if(ieq(a->v[1],"refresh") && !foliage_refresh(1)) { snprintf(err,ecap,"%s",foliage_error); return 0; }
    outf("zones count=%d masked=%d failed=%d truncated=%d unavailable_sources=%d",foliage_nzones,foliage_nmask,foliage_failures,foliage_truncated,foliage_unavailable);
    if(foliage_error[0]) outf("warning %s",foliage_error);
    return 1;
}

static int verb_site(const struct args *a,char *err,int ecap) {
    FILE *f; int i,j,count=0; float lo[3],hi[3],minheight=argf(a,8,200); char session[160];
    if(a->n!=9 || !isfinite(minheight) || minheight<0) { snprintf(err,ecap,"usage: site output x0 y0 z0 x1 y1 z1 minheight_cm"); return 0; }
    for(i=0;i<3;i++) {
        lo[i]=argf(a,2+i,0); hi[i]=argf(a,5+i,0);
        if(!isfinite(lo[i]) || !isfinite(hi[i]) || lo[i]>=hi[i]) { snprintf(err,ecap,"invalid site bounds"); return 0; }
    }
    f=fopen(a->v[1],"w"); if(!f) { snprintf(err,ecap,"cannot write site inventory"); return 0; }
    managed_session(session,sizeof session); fprintf(f,"session %s\n",session);
    memcpy(env_tree_roi_lo,lo,12); memcpy(env_tree_roi_hi,hi,12); env_tree_roi=1;
    env_nplants=0; env_tree_min_height=minheight; env_collect_trees(); env_tree_min_height=200; env_tree_roi=0;
    for(i=0;i<env_ntrees+foliage_nmask;i++) {
        float *tl,*th; uint8_t *comp; int index,masked=i>=env_ntrees;
        if(!masked) {
            struct env_tree *t=&env_trees[i]; int k,owned=0;
            if(t->instance==-2) for(k=0;k<foliage_nmask;k++) if(foliage_masks[k].comp==t->comp && foliage_masks[k].index==-2) { owned=1; break; }
            if(owned) continue;
            tl=t->lo; th=t->hi; comp=t->comp; index=t->instance;
        }
        else { struct foliage_mask *m=&foliage_masks[i-env_ntrees]; if(!foliage_mask_alive(m)) continue; tl=m->lo; th=m->hi; comp=m->comp; index=m->index; }
        if(th[2]-tl[2]<minheight) continue;
        for(j=0;j<3;j++) if(tl[j]>hi[j] || th[j]<lo[j]) break;
        if(j<3) continue;
        /* Masked zero-size instances are not active inventory entries. */
        if(!masked && th[0]-tl[0]<.001f && th[1]-tl[1]<.001f) continue;
        fprintf(f,"T %llx %d %d %.9g %.9g %.9g %.9g %.9g %.9g\n",(unsigned long long)comp,index,masked,tl[0],tl[1],tl[2],th[0],th[1],th[2]); count++;
    }
    fprintf(f,"end count=%d truncated=%d unavailable_sources=%d\n",count,env_tree_truncated,env_tree_unavailable);
    i=ferror(f); fclose(f); if(i) { snprintf(err,ecap,"site inventory write failed"); return 0; }
    outf("site foliage=%d truncated=%d",count,env_tree_truncated); return 1;
}

static uint64_t managed_tick(void *unused) {
    (void)unused;
    flush_readable_cache();
    if(capture_job.actor && GetTickCount()-capture_job.started>30000) capture_stop_native();
    if(camera_held_pawn && (!alive(camera_held_pawn) || camera_held_pawn!=editor_pawn() || GetTickCount()-camera_held_since>=120000)) camera_release_native();
    if(!foliage_refresh(0) && foliage_error[0]) logf_("zones: %s",foliage_error);
    return 1;
}
