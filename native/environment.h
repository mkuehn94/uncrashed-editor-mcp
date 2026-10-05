/* Read-only scene / environment audits. Included by agent.c after kismet helpers.
   All geometry queries run on the game thread. Native units: cm.
   Bounds-footprint support is a conservative diagnostic, not a structural solver. */

static int ltc_init(uint8_t *comp);
static int ltc(uint8_t *comp, const float s[3], const float e[3], float hit[3]);

#define ENV_LEVEL_CAP 256
#define ENV_TREE_CAP 120000
#define ENV_ACTOR_CAP 60000
#define ENV_DETAIL_CAP 12

static uint8_t *env_plants[ENV_ACTOR_CAP];
static int env_nplants;

static int env_mesh_plant(uint8_t *comp) {
    static int mesh_off=-2; uint8_t *mesh; char path[1100]; int i;
    static const char *const names[]={"foliage","/plants/","/trees/","/pines/","/grass/","treepack","sm_pine","sm_oak","sm_alder","sm_birch","sm_bush","sm_grass","sm_fern",0};
    if(!comp || !alive(comp) || !is_a(comp,"StaticMeshComponent")) return 0;
    if(mesh_off==-2) mesh_off=prop_offset(find_struct("StaticMeshComponent"),"StaticMesh");
    mesh=mesh_off>0?rdp(comp+mesh_off):0; if(!mesh || !alive(mesh)) return 0;
    obj_path(mesh,path,sizeof path);
    if(icontains(path,"rock") || icontains(path,"boulder") || icontains(path,"pebble")) return 0;
    for(i=0;names[i];i++) if(icontains(path,names[i])) return 1;
    return 0;
}

static int env_named_base_plant(uint8_t *a) {
    uint8_t *root;
    if(!a || !alive(a) || is_editor_actor(a,editor_tag())) return 0;
    root=root_component(a);
    return root && !is_a(root,"InstancedStaticMeshComponent") && env_mesh_plant(root);
}

static int env_foliage_component(uint8_t *comp) {
    uint8_t *owner;
    if(!comp || !alive(comp) || !is_a(comp,"InstancedStaticMeshComponent")) return 0;
    owner=rdp(comp+UOBJ_OUTER);
    return owner && alive(owner) && (is_a(owner,"InstancedFoliageActor") || is_a(owner,"LandscapeProxy")) && env_mesh_plant(comp);
}

static int env_base_foliage(uint8_t *a) {
    return a && valid_object(a) && (is_a(a,"InstancedFoliageActor") || env_named_base_plant(a));
}

static int env_plant(uint8_t *a) {
    int i;
    if (env_base_foliage(a)) return 1;
    for (i = 0; i < env_nplants; i++) if (env_plants[i] == a) return 1;
    return 0;
}

/* mode 1: structural support (not foliage); mode 2: landscape only.
   Preserve drop's ignore list. Exhaustion is unknown, never a fabricated hit. */
static int env_line(uint8_t *self, const float s[3], const float e[3], float hp[3], uint8_t **actor,
                    int mode, int *exhausted) {
    uint8_t *saved[64], *ha = 0;
    int saved_n = k_nign_extra, i, found = 0;
    memcpy(saved, k_ign_extra, sizeof saved);
    while (k_line(self, s, e, hp, &ha)) {
        if (ha && ((mode == 2 && is_a(ha, "LandscapeProxy")) || (mode == 1 && !env_plant(ha)) ||
                   (mode == 3 && !env_plant(ha) && !is_editor_actor(ha,editor_tag())))) {
            found = 1;
            break;
        }
        if (!ha || k_nign_extra >= 64) { if (exhausted) *exhausted = 1; break; }
        for (i = 0; i < k_nign_extra && k_ign_extra[i] != ha; i++) {}
        if (i < k_nign_extra) { if (exhausted) *exhausted = 1; break; }
        k_ign_extra[k_nign_extra++] = ha;
    }
    memcpy(k_ign_extra, saved, sizeof saved);
    k_nign_extra = saved_n;
    if (actor) *actor = found ? ha : 0;
    return found;
}

static int env_levels(uint8_t **out, int cap, int *truncated) {
    uint8_t *world = current_world(), **levels;
    int off = prop_offset(find_struct("World"), "Levels"), n = 0, count, i, j;
    uint8_t *persistent = current_level();
    if (persistent && cap) out[n++] = persistent;
    if (!world || off <= 0) { *truncated = 1; return n; }
    levels = (uint8_t **)rdp(world + off);
    count = rd32(world + off + 8);
    if (count < 0 || count > 4096 || (count && !readable(levels, (size_t)count * 8))) { *truncated = 1; return n; }
    for (i = 0; i < count; i++) {
        if (!levels[i] || !alive(levels[i])) continue;
        for (j = 0; j < n && out[j] != levels[i]; j++) {}
        if (j < n) continue;
        if (n == cap) { *truncated = 1; break; }
        out[n++] = levels[i];
    }
    return n;
}

static uint8_t **env_level_actors(uint8_t *level, int *count) {
    uint8_t **data = (uint8_t **)rdp(level + LEVEL_ACTORS);
    *count = rd32(level + LEVEL_ACTORS + 8);
    if (*count < 0 || *count > 500000 || !readable(data, (size_t)*count * 8)) { *count = 0; return 0; }
    return data;
}

static void env_extend(float lo[3], float hi[3], const float o[3], const float e[3]) {
    int j;
    for (j = 0; j < 3; j++) {
        if (o[j] - e[j] < lo[j]) lo[j] = o[j] - e[j];
        if (o[j] + e[j] > hi[j]) hi[j] = o[j] + e[j];
    }
}

static int env_terrain_bounds(float lo[3], float hi[3], int *truncated) {
    uint8_t *levels[ENV_LEVEL_CAP];
    int nl = env_levels(levels, ENV_LEVEL_CAP, truncated), i, j, n, found = 0;
    for (j = 0; j < 3; j++) { lo[j] = 1e30f; hi[j] = -1e30f; }
    for (i = 0; i < nl; i++) {
        uint8_t **actors = env_level_actors(levels[i], &n);
        for (j = 0; j < n; j++) {
            float o[3], e[3];
            if (!actors[j] || !alive(actors[j]) || !is_a(actors[j], "LandscapeProxy") || !actor_bounds(actors[j], o, e)) continue;
            if (e[0] <= 0 || e[1] <= 0) continue;
            env_extend(lo, hi, o, e); found++;
        }
    }
    return found;
}

static int verb_worldbounds(const struct args *a, char *err, int ecap) {
    const char *scope = a->n > 1 ? a->v[1] : "placed";
    uint8_t *levels[ENV_LEVEL_CAP];
    float lo[3] = {1e30f,1e30f,1e30f}, hi[3] = {-1e30f,-1e30f,-1e30f};
    int i, j, n, count = 0, truncated = 0, nl;
    uint64_t tag = editor_tag(), playable_tag = 0;
    if (!ieq(scope,"placed") && !ieq(scope,"terrain") && !ieq(scope,"playable")) {
        snprintf(err,ecap,"scope must be placed, terrain or playable"); return 0;
    }
    if (ieq(scope,"terrain")) count = env_terrain_bounds(lo,hi,&truncated);
    else {
        if (ieq(scope,"playable")) make_fname("PlayableBounds", &playable_tag);
        nl = env_levels(levels, ENV_LEVEL_CAP, &truncated);
        for (i = 0; i < nl; i++) {
            uint8_t **actors = env_level_actors(levels[i], &n);
            for (j = 0; j < n; j++) {
                float o[3], e[3]; uint8_t *act = actors[j];
                if (!act || !alive(act)) continue;
                if (ieq(scope,"placed") ? !is_editor_actor(act,tag) : !actor_has_tag(act,playable_tag)) continue;
                if (!actor_bounds(act,o,e) || e[0]+e[1]+e[2] <= 0) continue;
                env_extend(lo,hi,o,e); count++;
            }
        }
    }
    outf("worldbounds scope=%s available=%d actors=%d truncated=%d",scope,count>0,count,truncated);
    if (count) outf("aabb min=%.3f,%.3f,%.3f max=%.3f,%.3f,%.3f",lo[0],lo[1],lo[2],hi[0],hi[1],hi[2]);
    if (!count && ieq(scope,"playable")) outf("note no_explicit_PlayableBounds_tag;_terrain_extent_is_not_a_playable_boundary");
    return 1;
}

/* Small disk inventory avoids sending thousands of actors through MCP context. */
static int verb_inventory(const struct args *a, char *err, int ecap) {
    FILE *f; int n, i, j, count = 0, truncated = 0;
    uint8_t *levels[ENV_LEVEL_CAP]; uint64_t tag = editor_tag();
    int nl = env_levels(levels,ENV_LEVEL_CAP,&truncated);
    if (truncated) { snprintf(err,ecap,"cannot enumerate all loaded levels"); return 0; }
    if (a->n != 2 || !(f = fopen(a->v[1],"w"))) { snprintf(err,ecap,"usage: inventory <writable_file>"); return 0; }
    for (j = 0; j < nl; j++) {
        uint8_t **actors = env_level_actors(levels[j],&n);
        for (i = 0; i < n; i++) {
            float o[3], e[3]; char row[64]; uint8_t *act = actors[i];
            if (!act || !is_editor_actor(act,tag) || is_gate(act) || !actor_bounds(act,o,e)) continue;
            actor_row(act,row,sizeof row);
            fprintf(f,"O %llx %s %.3f %.3f %.3f %.3f %.3f %.3f\n",(unsigned long long)act,row[0]?row:"-",
                    o[0]-e[0],o[1]-e[1],o[2]-e[2],o[0]+e[0],o[1]+e[1],o[2]+e[2]); count++;
        }
    }
    if (ferror(f)) { fclose(f); snprintf(err,ecap,"inventory write failed"); return 0; }
    fclose(f); outf("inventory actors=%d",count); return 1;
}

struct env_tree { uint8_t *comp, *actor; int instance; float lo[3], hi[3]; };
static struct env_tree env_trees[ENV_TREE_CAP];
static int env_ntrees, env_tree_truncated, env_tree_unavailable;
static float env_tree_min_height=200;
static int env_tree_roi;
static float env_tree_roi_lo[3],env_tree_roi_hi[3];

/* World-space visual bounds for an instanced mesh. Includes crowns even when
   physics only models its trunk. No instance is removed or modified. */
static void env_instance_box(const uint8_t *xf, const float lo[3], const float hi[3], float outlo[3], float outhi[3]) {
    float q[4], t[3], s[3], m[3][3], c[3], h[3], wc[3], wh[3]; int i,j;
    memcpy(q,xf,16); memcpy(t,xf+16,12); memcpy(s,xf+32,12);
    m[0][0]=1-2*(q[1]*q[1]+q[2]*q[2]); m[0][1]=2*(q[0]*q[1]-q[2]*q[3]); m[0][2]=2*(q[0]*q[2]+q[1]*q[3]);
    m[1][0]=2*(q[0]*q[1]+q[2]*q[3]); m[1][1]=1-2*(q[0]*q[0]+q[2]*q[2]); m[1][2]=2*(q[1]*q[2]-q[0]*q[3]);
    m[2][0]=2*(q[0]*q[2]-q[1]*q[3]); m[2][1]=2*(q[1]*q[2]+q[0]*q[3]); m[2][2]=1-2*(q[0]*q[0]+q[1]*q[1]);
    for(i=0;i<3;i++) { c[i]=(lo[i]+hi[i])*.5f*s[i]; h[i]=(hi[i]-lo[i])*.5f*fabsf(s[i]); }
    for(i=0;i<3;i++) {
        wc[i]=t[i]; wh[i]=0;
        for(j=0;j<3;j++) { wc[i]+=m[i][j]*c[j]; wh[i]+=fabsf(m[i][j])*h[j]; }
        outlo[i]=wc[i]-wh[i]; outhi[i]=wc[i]+wh[i];
    }
}

static int env_component_box(uint8_t *comp,float o[3],float e[3]) {
    static uint8_t *lib,*fn; static int ci,co,ce,size;
    uint8_t p[0x80]; int i;
    if(!fn) {
        lib=class_cdo("KismetSystemLibrary"); fn=object_function(lib,"GetComponentBounds");
        ci=fn?param_offset(fn,"Component"):-1; co=fn?param_offset(fn,"Origin"):-1; ce=fn?param_offset(fn,"BoxExtent"):-1;
        size=fn?rd16(fn+UFUNC_PARMSSIZE):0;
        if(ci<0 || co<0 || ce<0 || size>(int)sizeof p || ci+8>size || co+12>size || ce+12>size) { fn=0; return 0; }
    }
    memset(p,0,sizeof p); memcpy(p+ci,&comp,8); call(lib,fn,p); memcpy(o,p+co,12); memcpy(e,p+ce,12);
    for(i=0;i<3;i++) if(!isfinite(o[i]) || !isfinite(e[i]) || e[i]<0) return 0;
    return 1;
}

static void env_collect_trees(void) {
    uint8_t *levels[ENV_LEVEL_CAP]; int truncated=0, nl=env_levels(levels,ENV_LEVEL_CAP,&truncated);
    int i,j,n=object_count(); env_ntrees=0; env_tree_truncated=truncated; env_tree_unavailable=0;
    for(i=0;i<n;i++) {
        uint8_t *comp=object_at(i), *outer, *mesh, *fnc, *fnt, *fnb;
        uint8_t pc[0x20], pt[0x100], pb[0x80];
        int mo,cr,ti,tx,tw,tr,br,count,k,loaded=0,ts,cs,bs;
        float lo[3],hi[3];
        if(!env_foliage_component(comp)) continue;
        outer=rdp(comp+UOBJ_OUTER);
        for(j=0;outer && j<32;j++,outer=rdp(outer+UOBJ_OUTER)) {
            for(k=0;k<nl;k++) if(outer==levels[k]) loaded=1;
            if(loaded) break;
        }
        if(!loaded) continue;
        if(env_tree_roi) {
            float center[3],extent[3];
            if(env_component_box(comp,center,extent)) {
                for(k=0;k<3;k++) if(center[k]-extent[k]>env_tree_roi_hi[k] || center[k]+extent[k]<env_tree_roi_lo[k]) break;
                if(k<3) continue;
            }
        }
        mo=field_offset(comp,"StaticMesh"); mesh=mo>0?rdp(comp+mo):0;
        fnc=object_function(comp,"GetInstanceCount"); fnt=object_function(comp,"GetInstanceTransform");
        fnb=mesh?object_function(mesh,"GetBoundingBox"):0;
        if(!fnc || !fnt || !fnb) { env_tree_truncated=1; continue; }
        cr=param_offset(fnc,"ReturnValue"); ti=param_offset(fnt,"InstanceIndex");
        tx=param_offset(fnt,"OutInstanceTransform"); tw=param_offset(fnt,"bWorldSpace"); tr=param_offset(fnt,"ReturnValue");
        br=param_offset(fnb,"ReturnValue"); ts=rd16(fnt+UFUNC_PARMSSIZE);
        cs=rd16(fnc+UFUNC_PARMSSIZE); bs=rd16(fnb+UFUNC_PARMSSIZE);
        if(cr<0 || cr+4>cs || cs>(int)sizeof pc || ti<0 || tx<0 || tw<0 || tr<0 || ts>(int)sizeof pt ||
           ti+4>ts || tx+48>ts || tw>=ts || tr>=ts || br<0 || br+25>bs || bs>(int)sizeof pb) {
            env_tree_truncated=1; continue;
        }
        memset(pc,0,sizeof pc); call(comp,fnc,pc); count=rd32(pc+cr);
        if(count<0 || count>200000) { env_tree_truncated=1; continue; }
        if(count==0) {
            /* Landscape grass can retain GPU-only instance data in shipping
               builds. Zero CPU count is not proof of an empty rendered source. */
            if(is_a(rdp(comp+UOBJ_OUTER),"LandscapeProxy")) env_tree_unavailable++;
            continue;
        }
        memset(pb,0,sizeof pb); call(mesh,fnb,pb); memcpy(lo,pb+br,12); memcpy(hi,pb+br+12,12);
        if(!pb[br+24]) { env_tree_truncated=1; continue; }
        for(j=0;j<count;j++) {
            struct env_tree candidate,*tree=&candidate;
            memset(pt,0,sizeof pt); memcpy(pt+ti,&j,4); pt[tw]=1; call(comp,fnt,pt);
            if(!pt[tr]) { env_tree_truncated=1; continue; }
            env_instance_box(pt+tx,lo,hi,tree->lo,tree->hi);
            for(k=0;k<3;k++) if(!isfinite(tree->lo[k]) || !isfinite(tree->hi[k]) || tree->lo[k]>tree->hi[k]) break;
            if(k<3) { env_tree_truncated=1; continue; }
            /* Ground cover does not get misleading 'canopy' warnings. */
            if(tree->hi[2]-tree->lo[2]<env_tree_min_height) continue;
            if(env_tree_roi) {
                for(k=0;k<3;k++) if(tree->lo[k]>env_tree_roi_hi[k] || tree->hi[k]<env_tree_roi_lo[k]) break;
                if(k<3) continue;
            }
            if(env_ntrees==ENV_TREE_CAP) { env_tree_truncated=1; return; }
            tree->comp=comp; tree->actor=rdp(comp+UOBJ_OUTER); tree->instance=j;
            env_trees[env_ntrees++]=*tree;
        }
    }
    /* Individually placed base flora (not editor props, never whole instanced
       forests). Classification uses recognised plant mesh paths, not actor names. */
    for(j=0;j<nl;j++) {
        int count,k; uint8_t **actors=env_level_actors(levels[j],&count);
        for(k=0;k<count;k++) {
            float o[3],e[3]; struct env_tree *tree;
            if(!env_named_base_plant(actors[k]) || !actor_bounds(actors[k],o,e) || e[2]*2<env_tree_min_height) continue;
            if(env_tree_roi) {
                for(i=0;i<3;i++) if(o[i]-e[i]>env_tree_roi_hi[i] || o[i]+e[i]<env_tree_roi_lo[i]) break;
                if(i<3) continue;
            }
            if(env_ntrees==ENV_TREE_CAP) { env_tree_truncated=1; return; }
            tree=&env_trees[env_ntrees++]; tree->comp=tree->actor=actors[k]; tree->instance=-2;
            for(i=0;i<3;i++) { tree->lo[i]=o[i]-e[i]; tree->hi[i]=o[i]+e[i]; }
        }
    }
    for(i=0;i<env_nplants;i++) {
        float o[3],e[3]; struct env_tree *tree;
        if(!alive(env_plants[i]) || !actor_bounds(env_plants[i],o,e) || 2*e[2]<200) continue;
        if(env_ntrees==ENV_TREE_CAP) { env_tree_truncated=1; break; }
        tree=&env_trees[env_ntrees++]; tree->comp=root_component(env_plants[i]); tree->actor=env_plants[i]; tree->instance=-1;
        for(j=0;j<3;j++) { tree->lo[j]=o[j]-e[j]; tree->hi[j]=o[j]+e[j]; }
    }
}

/* Actual environment collision hits against a conservative subject OBB.
   Keep component + instance index; collapsing foliage to its actor loses trees. */
static int env_physics(FILE *f,uint8_t *act,const float c[3],const float h[3],const float r[3],int *truncated) {
    static struct tarr hits;
    uint8_t parms[0x400], *ign[1]={act}, *seen_comp[ENV_DETAIL_CAP]; int seen_item[ENV_DETAIL_CAP];
    struct tarr types={k_types,k_ntypes,k_ntypes}, ignore={ign,1,1};
    float half[3],end[3]; int i,j,count=0;
    for(i=0;i<3;i++) { half[i]=fmaxf(h[i]-.1f*100,h[i]*.9f); end[i]=c[i]; } end[2]+=.5f;
    memset(parms,0,(size_t)K.b_size); memcpy(parms+K.b_wco,&act,8);
    memcpy(parms+K.b_s,c,12); memcpy(parms+K.b_e,end,12); memcpy(parms+K.b_half,half,12); memcpy(parms+K.b_rot,r,12);
    memcpy(parms+K.b_types,&types,16); memcpy(parms+K.b_ign,&ignore,16); parms[K.b_cx]=1;
    if(K.b_self>=0) parms[K.b_self]=1;
    hits.num=0; memcpy(parms+K.b_hits,&hits,16); call(K.ksl,K.fb,parms); memcpy(&hits,parms+K.b_hits,16);
    if(hits.num==0) return 0;
    if(hits.num<0 || hits.num>100000 || !readable(hits.data,(size_t)hits.num*K.h_size)) { *truncated=1; return 0; }
    if(K.h_component<0 || K.h_item<0) *truncated=1;
    for(i=0;i<hits.num;i++) {
        uint8_t *hr=(uint8_t *)hits.data+(size_t)i*K.h_size, *ha=weak_actor(hr+K.h_actor), *comp=0;
        int item=-1; float time=0;
        if(K.h_time>=0) memcpy(&time,hr+K.h_time,4);
        if(K.h_spen>=0 && !(hr[K.h_spen]&K.h_spen_mask) && time>.01f) continue;
        if(!ha || !env_plant(ha)) continue;
        if(K.h_component>=0) comp=weak_actor(hr+K.h_component);
        if(K.h_item>=0) item=rd32(hr+K.h_item);
        for(j=0;j<count;j++) if(seen_comp[j]==comp && seen_item[j]==item) break;
        if(j<count) continue;
        if(count==ENV_DETAIL_CAP) { *truncated=1; break; }
        seen_comp[count]=comp; seen_item[count]=item; count++;
        fprintf(f,"tree_collision actor=%llx component=%llx instance=%d source=%s\n",(unsigned long long)ha,(unsigned long long)comp,item,env_base_foliage(ha)?"base":"placed");
    }
    return count;
}

static int env_audit_one(FILE *f,uint8_t *act,int foundation,float spacing,float tolerance,float depth,int canopies,
                         int terrain_available,const float terrain_lo[3],const float terrain_hi[3],int *query_truncated) {
    float o[3],e[3],c[3],h[3],r[3]={0,0,0},ax[3][3],bot,top,gc[3],gh[2];
    int tilted,obb,i,j,nu,nv,total=0,supported=0,unsupported=0,penetrating=0,missing=0,corners=0;
    int terrain_samples=0,terrain_bad=0,collision_samples=0,details=0,tdetails=0,canopy_count=0,tree_hits=0;
    int have_gap=0,exhausted=0,underside=0; float maxgap=0,maxpen=0,maxterrain=0;
    uint8_t *root; char row[64];
    if(!alive(act) || !is_editor_actor(act,editor_tag()) || !actor_bounds(act,o,e)) return 0;
    actor_row(act,row,sizeof row); obb=actor_obb(act,c,h,r);
    if(!obb) { memcpy(c,o,12); memcpy(h,e,12); }
    if(h[0]<=0 || h[1]<=0 || !isfinite(h[0]+h[1]+h[2])) return 0;
    tilted=fabsf(r[0])>5 || fabsf(r[2])>5;
    memcpy(gc,c,12); gh[0]=h[0]; gh[1]=h[1]; rot_axes(r,ax);
    if(tilted) {
        float zero[3]={0,0,0}; memcpy(gc,o,12); gh[0]=e[0]; gh[1]=e[1]; rot_axes(zero,ax);
    }
    bot=o[2]-e[2]; top=o[2]+e[2]; root=root_component(act);
    underside=obb && root && ltc_init(root);
    nu=(int)ceilf(2*gh[0]/spacing); nv=(int)ceilf(2*gh[1]/spacing);
    if(nu<2) nu=2; if(nv<2) nv=2; if(nu>32) nu=32; if(nv>32) nv=32;
    /* Include both centre axes: corner-only rays miss small round / bevelled props. */
    if(nu%2) nu++; if(nv%2) nv++;
    fprintf(f,"actor id=%llx row=%s basis=%s footprint=%s min=%.3f,%.3f,%.3f max=%.3f,%.3f,%.3f rotation=%.3f,%.3f,%.3f\n",
            (unsigned long long)act,row[0]?row:"-",obb?"mesh_obb":"actor_aabb",obb&&!tilted?"oriented_bounds":"world_aabb_projection",
            o[0]-e[0],o[1]-e[1],bot,o[0]+e[0],o[1]+e[1],top,r[0],r[1],r[2]);
    for(i=0;i<=nu;i++) for(j=0;j<=nv;j++) {
        float u=(-gh[0]+2*gh[0]*i/nu)*.99f,v=(-gh[1]+2*gh[1]*j/nv)*.99f;
        float s[3],end[3],hp[3],bottom=bot,gap=0; uint8_t *ha=0;
        int ok=0;
        s[0]=end[0]=gc[0]+u*ax[0][0]+v*ax[1][0]; s[1]=end[1]=gc[1]+u*ax[0][1]+v*ax[1][1];
        if(foundation) {
            int bad=0,corner=-1;
            s[2]=bot+fmaxf(tolerance+5,50); end[2]=bot-depth; total++;
            ok=env_line(act,s,end,hp,&ha,1,&exhausted);
            if(ok) {
                gap=bot-hp[2]; have_gap=1;
                if(gap>maxgap) maxgap=gap;
                if(-gap>maxpen) maxpen=-gap;
                if(gap < -tolerance) { penetrating++; bad=1; }
                else if(gap <= tolerance) supported++;
                else { unsupported++; bad=1; }
            } else { missing++; unsupported++; bad=1; }
            if(i==0 && j==0) corner=0; if(i==nu && j==0) corner=1;
            if(i==0 && j==nv) corner=2; if(i==nu && j==nv) corner=3;
            if(bad && corner>=0) corners|=1<<corner;
            if(bad && details<ENV_DETAIL_CAP) {
                fprintf(f,"support_point x=%.3f y=%.3f bottom=%.3f gap=%.3f surface=%s\n",s[0],s[1],bot,ok?gap:-1e30f,
                        !ok?"unknown":is_a(ha,"LandscapeProxy")?"landscape":is_editor_actor(ha,editor_tag())?"placed":"base_geometry"); details++;
            }
        } else if(terrain_available) {
            /* Ray against the subject alone finds its collision underside. No
               collision at this XY is not evidence of terrain penetration. */
            if(underside) {
                s[2]=bot-50; end[2]=top+50;
                if(!ltc(root,s,end,hp)) continue;
                bottom=hp[2]; collision_samples++;
            }
            s[2]=terrain_hi[2]+1000; end[2]=terrain_lo[2]-1000;
            if(env_line(act,s,end,hp,&ha,2,&exhausted)) {
                float pen=hp[2]-bottom; terrain_samples++;
                if(pen>maxterrain) maxterrain=pen;
                if(pen>tolerance) {
                    terrain_bad++;
                    if(tdetails++<ENV_DETAIL_CAP) fprintf(f,"terrain_point x=%.3f y=%.3f bottom=%.3f terrain=%.3f depth=%.3f\n",s[0],s[1],bottom,hp[2],pen);
                }
            }
        }
    }
    if(foundation) {
        fprintf(f,"foundation samples=%d supported=%d unsupported=%d penetrating=%d missing=%d max_gap=%.3f max_penetration=%.3f corner_mask=%d spacing=%.3f,%.3f\n",
                total,supported,unsupported,penetrating,missing,have_gap?maxgap:-1,maxpen,corners,2*gh[0]/nu,2*gh[1]/nv);
    } else {
        tree_hits=env_physics(f,act,c,h,r,query_truncated);
        if(canopies) for(i=0;i<env_ntrees;i++) {
            struct env_tree *tree=&env_trees[i]; int overlap=1;
            if(tree->actor==act) continue;
            for(j=0;j<3;j++) if(tree->lo[j]>=o[j]+e[j]-5 || tree->hi[j]<=o[j]-e[j]+5) overlap=0;
            if(!overlap) continue;
            canopy_count++;
            if(canopy_count<=ENV_DETAIL_CAP) fprintf(f,"canopy actor=%llx component=%llx instance=%d source=%s min=%.3f,%.3f,%.3f max=%.3f,%.3f,%.3f\n",
                (unsigned long long)tree->actor,(unsigned long long)tree->comp,tree->instance,tree->instance==-1?"placed":"base",
                tree->lo[0],tree->lo[1],tree->lo[2],tree->hi[0],tree->hi[1],tree->hi[2]);
        }
        fprintf(f,"environment terrain_available=%d terrain_samples=%d collision_samples=%d terrain_basis=%s penetration_samples=%d max_depth=%.3f tree_hits=%d canopy_candidates=%d spacing=%.3f,%.3f\n",
                terrain_available,terrain_samples,collision_samples,underside?"collision_underside":"bounds_bottom",terrain_bad,maxterrain,tree_hits,canopy_count,2*gh[0]/nu,2*gh[1]/nv);
    }
    if(exhausted) *query_truncated=1;
    fprintf(f,"end %llx\n",(unsigned long long)act); return 1;
}

/* audit <request_file> <output_file> foundation|environment spacing gap depth canopies
   Request: P <plant_actor_hex> lines, then A <subject_actor_hex> lines.
   Fresh, read-only snapshot each call; no state leaks between audits. */
static int verb_audit(const struct args *a,char *err,int ecap) {
    static uint8_t *subjects[ENV_ACTOR_CAP];
    FILE *in,*out; int foundation,canopies,ns=0,i,done=0,failed=0,truncated=0,terrain_available;
    float spacing=argf(a,4,75),tolerance=argf(a,5,30),depth=argf(a,6,5000),lo[3],hi[3];
    char line[160],kind; unsigned long long addr;
    if(a->n<4 || (!ieq(a->v[3],"foundation") && !ieq(a->v[3],"environment")) ||
       !isfinite(spacing) || !isfinite(tolerance) || !isfinite(depth) || spacing<5 || tolerance<0 || depth<=tolerance) {
        snprintf(err,ecap,"usage: audit request output foundation|environment spacing_cm>=5 gap_cm>=0 depth_cm>gap canopies"); return 0;
    }
    foundation=ieq(a->v[3],"foundation"); canopies=argf(a,7,0)!=0;
    in=fopen(a->v[1],"r"); if(!in) { snprintf(err,ecap,"cannot read audit request"); return 0; }
    env_nplants=0;
    while(fgets(line,sizeof line,in)) if(sscanf(line," %c %llx",&kind,&addr)==2) {
        if(kind=='P' && env_nplants<ENV_ACTOR_CAP) env_plants[env_nplants++]=(uint8_t *)(uintptr_t)addr;
        else if(kind=='A' && ns<ENV_ACTOR_CAP) subjects[ns++]=(uint8_t *)(uintptr_t)addr;
        else { fclose(in); env_nplants=0; snprintf(err,ecap,"audit request invalid or too large"); return 0; }
    }
    fclose(in);
    out=fopen(a->v[2],"w"); if(!out) { env_nplants=0; snprintf(err,ecap,"cannot write audit report"); return 0; }
    if(!kismet_init(err,ecap)) { fclose(out); env_nplants=0; return 0; }
    terrain_available=env_terrain_bounds(lo,hi,&truncated)>0;
    env_ntrees=env_tree_truncated=env_tree_unavailable=0;
    if(!foundation && canopies) env_collect_trees();
    for(i=0;i<ns;i++) {
        if(env_audit_one(out,subjects[i],foundation,spacing,tolerance,depth,canopies,terrain_available,lo,hi,&truncated)) done++;
        else { fprintf(out,"error actor=%llx reason=unavailable_geometry\n",(unsigned long long)subjects[i]); failed++; }
    }
    fprintf(out,"scan checked=%d failed=%d terrain_available=%d foliage_instances=%d foliage_truncated=%d query_truncated=%d foliage_unavailable_sources=%d\n",
            done,failed,terrain_available,env_ntrees,env_tree_truncated,truncated,env_tree_unavailable);
    env_nplants=0;
    if(ferror(out)) { fclose(out); snprintf(err,ecap,"audit report write failed"); return 0; }
    fclose(out); outf("audit checked=%d failed=%d",done,failed); return 1;
}
