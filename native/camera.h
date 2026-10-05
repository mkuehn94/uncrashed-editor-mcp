/* Native camera + offscreen capture. Game-thread only; never activates a window.
   UPROPERTY/UFunction offsets are reflected. Render target is owned by a transient,
   untagged SceneCapture2D actor, never by the editor's placed-object collection. */

static int np_put(uint8_t *fn, uint8_t *p, int cap, const char *name, const void *v, int len) {
    int off=fn?param_offset(fn,name):-1, size=fn?rd16(fn+UFUNC_PARMSSIZE):0;
    if(off<0 || size>cap || off+len>size) return 0;
    memcpy(p+off,v,(size_t)len); return 1;
}

static uint8_t *np_object(uint8_t *o,const char *name) {
    int off=o&&alive(o)?field_offset(o,name):-1;
    uint8_t *v=off>0?rdp(o+off):0;
    return v&&alive(v)?v:0;
}

static int np_bool(uint8_t *o,const char *name,int value,int *previous) {
    uint8_t *p=o?find_property(rdp(o+UOBJ_CLASS),name):0;
    int off=p?rd32(p+FPROP_OFFSET):-1; char type[64],inner[128];
    uint8_t mask=p?p[FPROP_EXTRA+2]:0;
    if(!p) return 0;
    prop_type(p,type,sizeof type,inner,sizeof inner);
    if(!ieq(type,"BoolProperty")) return 0;
    off+=p[FPROP_EXTRA+1];
    if(off<=0 || !mask || !readable(o+off,1)) return 0;
    if(previous) *previous=(o[off]&mask)!=0;
    if(value>=0) { if(value) o[off]|=mask; else o[off]&=(uint8_t)~mask; }
    return 1;
}

static int np_pose(uint8_t *o,const char *method,const float loc[3],const float rot[3]) {
    uint8_t p[0x200],one=1,*fn=object_function(o,method);
    memset(p,0,sizeof p);
    if(!np_put(fn,p,sizeof p,"NewLocation",loc,12) || !np_put(fn,p,sizeof p,"NewRotation",rot,12)) return 0;
    np_put(fn,p,sizeof p,"bTeleport",&one,1); call(o,fn,p); return 1;
}

static uint8_t *camera_held_pawn;
static int camera_previous_movement;
static DWORD camera_held_since;

static int camera_release_native(void) {
    if(camera_held_pawn && alive(camera_held_pawn))
        np_bool(camera_held_pawn,"EnableCameraMovement",camera_previous_movement,0);
    camera_held_pawn=0; return 1;
}

static int camera_read_native(struct xf *t,float *fov) {
    uint8_t *cam=np_object(editor_pawn(),"Camera"); int off;
    if(!cam || !vec_getter(cam,"K2_GetComponentLocation",t->loc) || !vec_getter(cam,"K2_GetComponentRotation",t->rot)) return 0;
    t->scale[0]=t->scale[1]=t->scale[2]=1;
    off=field_offset(cam,"FieldOfView"); *fov=90;
    if(off>0) memcpy(fov,cam+off,4);
    return isfinite(*fov) && *fov>1 && *fov<179;
}

static int camera_set_native(const struct xf *t,int hold) {
    uint8_t *pawn=editor_pawn(),*cam=np_object(pawn,"Camera"),*arm=np_object(pawn,"SpringArm1");
    struct xf root=*t; int off;
    if(!cam || !arm) return 0;
    if(camera_held_pawn!=pawn) camera_release_native();
    if(hold && !camera_held_pawn) {
        if(!np_bool(pawn,"EnableCameraMovement",0,&camera_previous_movement)) return 0;
        camera_held_pawn=pawn;
        camera_held_since=GetTickCount();
    } else if(!hold) camera_release_native();
    /* Move actual root and camera world pose, not transient view-cache bytes.
       Leave SpringArm/socket state alone: its cached socket can lag while the
       editor's input/tick path is inactive. World setter computes valid relative
       camera transform against that socket and survives subsequent ticks. */
    if(!actor_get(pawn,&root)) return 0;
    memcpy(root.loc,t->loc,12);
    if(hold) camera_held_since=GetTickCount();
    if(!actor_set(pawn,&root) || !np_pose(cam,"K2_SetWorldLocationAndRotation",t->loc,t->rot)) return 0;
    off=field_offset(pawn,"CamLoc"); if(off>0) memcpy(pawn+off,t->loc,12);
    return 1;
}

static int verb_camera(const struct args *a,char *err,int ecap) {
    struct xf t; float fov; int i;
    if(a->n==2 && ieq(a->v[1],"release")) camera_release_native();
    else if(a->n>=7 && ieq(a->v[1],"set")) {
        for(i=0;i<3;i++) { t.loc[i]=argf(a,2+i,0); t.rot[i]=i<2?argf(a,5+i,0):0; t.scale[i]=1; }
        for(i=0;i<3;i++) if(!isfinite(t.loc[i]) || !isfinite(t.rot[i]) || fabsf(t.loc[i])>1e8f) {
            snprintf(err,ecap,"invalid camera pose"); return 0;
        }
        if(!camera_set_native(&t,argf(a,7,1)!=0)) { snprintf(err,ecap,"native camera move unavailable"); return 0; }
    } else if(a->n!=2 || !ieq(a->v[1],"get")) {
        snprintf(err,ecap,"usage: camera get|release|set x y z pitch yaw [hold]"); return 0;
    }
    if(!camera_read_native(&t,&fov)) { snprintf(err,ecap,"editor camera unavailable"); return 0; }
    outf("camera loc=%.5f,%.5f,%.5f rot=%.5f,%.5f,%.5f fov=%.4f held=%d",t.loc[0],t.loc[1],t.loc[2],t.rot[0],t.rot[1],t.rot[2],fov,
         camera_held_pawn==editor_pawn()); return 1;
}

/* UE4.27 UEnum::Names is TArray<TPair<FName,int64>> at 0x40. Validate names,
   count and value; never guess a version-dependent enum ordinal. */
static int np_enum(const char *type,const char *entry,uint8_t *value) {
    int i,n=object_count(); char name[256];
    for(i=0;i<n;i++) {
        uint8_t *o=object_at(i),*data; int j,num,max;
        if(!o || !alive(o) || !is_a(o,"Enum")) continue;
        obj_name(o,name,sizeof name); if(strcmp(name,type)) continue;
        data=rdp(o+0x40); num=rd32(o+0x48); max=rd32(o+0x4c);
        if(num<1 || num>512 || max<num || max>4096 || !readable(data,(size_t)num*16)) return 0;
        for(j=0;j<num;j++) {
            const char *tail; int64_t v;
            fname_str(rd64(data+j*16),name,sizeof name); tail=strrchr(name,':'); tail=tail?tail+1:name;
            if(strcmp(tail,entry)) continue;
            v=(int64_t)rd64(data+j*16+8); if(v<0 || v>255) return 0; *value=(uint8_t)v; return 1;
        }
        return 0;
    }
    return 0;
}

struct capture_job {
    uint8_t *actor,*component,*target,*world;
    unsigned long token; DWORD started;
};
static struct capture_job capture_job;
static unsigned long capture_serial;

static void capture_stop_native(void) {
    if(capture_job.actor && alive(capture_job.actor) && capture_job.world==current_world()) destroy_actor(capture_job.actor);
    memset(&capture_job,0,sizeof capture_job);
}

/* Capture-only exclusions: editor gizmos / water-volume wire boxes are not map
   geometry. Do not alter visibility/collision in the editor or saved map. */
static void capture_hide_editor_helpers(uint8_t *component) {
    uint8_t *levels[ENV_LEVEL_CAP],p[0x40],one=1;
    uint8_t *hide_actor=object_function(component,"HideActorComponents"),*hide_comp=object_function(component,"HideComponent");
    int truncated=0,nl=env_levels(levels,ENV_LEVEL_CAP,&truncated),i,j,n;
    const char *const batches[]={"LineBatcher","PersistentLineBatcher","ForegroundLineBatcher",0};
    for(i=0;batches[i];i++) {
        uint8_t *lines=np_object(current_world(),batches[i]);
        if(!lines || !is_a(lines,"LineBatchComponent")) continue;
        memset(p,0,sizeof p);
        if(np_put(hide_comp,p,sizeof p,"InComponent",&lines,8)) call(component,hide_comp,p);
    }
    for(i=0;i<nl;i++) {
        uint8_t **actors=env_level_actors(levels[i],&n);
        for(j=0;j<n;j++) {
            uint8_t *a=actors[j],*box;
            if(!a || !alive(a)) continue;
            if(a==editor_pawn() || is_a(a,"BP_CoordinateAxisWithUI_C") || is_a(a,"BP_MapEditor_RulerTool_C")) {
                memset(p,0,sizeof p);
                if(np_put(hide_actor,p,sizeof p,"InActor",&a,8) && np_put(hide_actor,p,sizeof p,"bIncludeFromChildActors",&one,1)) call(component,hide_actor,p);
            }
            if(is_a(a,"BP_meWater_C") && (box=np_object(a,"Box")) && is_a(box,"BoxComponent")) {
                memset(p,0,sizeof p);
                if(np_put(hide_comp,p,sizeof p,"InComponent",&box,8)) call(component,hide_comp,p);
            }
        }
    }
}

static int capture_begin_native(int width,int height,const struct xf *t,float fov,char *err,int ecap) {
    uint8_t p[0x200],one=1,fmt,source,*lib=class_cdo("KismetRenderingLibrary"),*game=class_cdo("GameplayStatics");
    uint8_t *fn,*actor=0,*component=0,*target=0,*cls=find_struct("SceneCapture2D"),*world=current_world();
    float transform[12],clear[4]={0,0,0,1}; int ret,off;
    if(capture_job.actor) { snprintf(err,ecap,"capture busy; finish or stop existing job"); return 0; }
    if(!world || !editor_pawn() || !lib || !game || !cls || !np_enum("ETextureRenderTargetFormat","RTF_RGBA8_SRGB",&fmt) ||
       !np_enum("ESceneCaptureSource","SCS_FinalColorLDR",&source)) goto unavailable;
    make_ftransform(t,transform); memset(p,0,sizeof p); fn=object_function(game,"BeginDeferredActorSpawnFromClass");
    if(!np_put(fn,p,sizeof p,"WorldContextObject",&world,8) || !np_put(fn,p,sizeof p,"ActorClass",&cls,8) ||
       !np_put(fn,p,sizeof p,"SpawnTransform",transform,48) || !np_put(fn,p,sizeof p,"CollisionHandlingOverride",&one,1) ||
       (ret=param_offset(fn,"ReturnValue"))<0 || ret+8>rd16(fn+UFUNC_PARMSSIZE)) goto unavailable;
    call(game,fn,p); actor=rdp(p+ret); if(!actor || !alive(actor)) goto unavailable;
    capture_job.actor=actor; capture_job.world=world; /* cleanup also covers partial construction */
    fn=object_function(game,"FinishSpawningActor"); memset(p,0,sizeof p);
    if(!np_put(fn,p,sizeof p,"Actor",&actor,8) || !np_put(fn,p,sizeof p,"SpawnTransform",transform,48)) goto unavailable;
    call(game,fn,p); component=np_object(actor,"CaptureComponent2D"); if(!component) goto unavailable;
    /* Prevent capture during setup, and allow only explicit one-shot renders. */
    if(!np_bool(component,"bCaptureEveryFrame",0,0) || !np_bool(component,"bCaptureOnMovement",0,0)) goto unavailable;
    np_bool(component,"bAlwaysPersistRenderingState",1,0);
    fn=object_function(lib,"CreateRenderTarget2D"); memset(p,0,sizeof p);
    if(!np_put(fn,p,sizeof p,"WorldContextObject",&world,8) || !np_put(fn,p,sizeof p,"Width",&width,4) ||
       !np_put(fn,p,sizeof p,"Height",&height,4) || !np_put(fn,p,sizeof p,"Format",&fmt,1) ||
       !np_put(fn,p,sizeof p,"ClearColor",clear,16) || (ret=param_offset(fn,"ReturnValue"))<0 || ret+8>rd16(fn+UFUNC_PARMSSIZE)) goto unavailable;
    call(lib,fn,p); target=rdp(p+ret); if(!target || !alive(target)) goto unavailable;
    off=field_offset(component,"TextureTarget"); if(off<=0) goto unavailable; memcpy(component+off,&target,8);
    off=field_offset(component,"CaptureSource"); if(off<=0) goto unavailable; component[off]=source;
    off=field_offset(component,"FOVAngle"); if(off<=0) goto unavailable; memcpy(component+off,&fov,4);
    fn=object_function(component,"CaptureScene"); if(!fn || !actor_set(actor,t)) goto unavailable;
    capture_hide_editor_helpers(component);
    np_bool(component,"bCaptureEveryFrame",1,0);
    memset(p,0,sizeof p); call(component,fn,p);
    capture_job.component=component; capture_job.target=target;
    capture_job.token=++capture_serial; capture_job.started=GetTickCount();
    outf("capture token=%lu width=%d height=%d backend=engine",capture_job.token,width,height); return 1;
unavailable:
    capture_stop_native(); snprintf(err,ecap,"native render-target capture unavailable (reflection or resource creation failed)"); return 0;
}

/* start w h x y z pitch yaw fov; export token directory filename; stop token.
   Export is a separate job so rendering can run between preparation/readback. */
static int verb_capture(const struct args *a,char *err,int ecap) {
    int i;
    if(a->n==11 && ieq(a->v[1],"start")) {
        struct xf t; int w=atoi(a->v[2]),h=atoi(a->v[3]); float fov=argf(a,10,90);
        if(w<64 || h<64 || w>4096 || h>4096 || w*(int64_t)h>4194304 || !isfinite(fov) || fov<=1 || fov>=179) {
            snprintf(err,ecap,"capture size/FOV out of range"); return 0;
        }
        for(i=0;i<3;i++) { t.loc[i]=argf(a,4+i,0); t.rot[i]=i<2?argf(a,7+i,0):0; t.scale[i]=1; }
        /* Slot 9 is roll. Kept explicit for saved/tilted camera poses. */
        t.rot[2]=argf(a,9,0);
        for(i=0;i<3;i++) if(!isfinite(t.loc[i]) || !isfinite(t.rot[i]) || fabsf(t.loc[i])>1e8f) {
            snprintf(err,ecap,"invalid capture pose"); return 0;
        }
        return capture_begin_native(w,h,&t,fov,err,ecap);
    }
    if(a->n<3 || !capture_job.actor || strtoul(a->v[2],0,10)!=capture_job.token || capture_job.world!=current_world() ||
       !alive(capture_job.actor) || !alive(capture_job.target)) { snprintf(err,ecap,"capture token expired or unknown"); return 0; }
    if(a->n==3 && ieq(a->v[1],"stop")) { capture_stop_native(); return 1; }
    if(a->n==3 && ieq(a->v[1],"render")) {
        uint8_t p[8]={0},*fn=object_function(capture_job.component,"CaptureScene");
        if(!fn) { snprintf(err,ecap,"CaptureScene unavailable"); return 0; }
        call(capture_job.component,fn,p); capture_job.started=GetTickCount(); return 1;
    }
    if(a->n==5 && ieq(a->v[1],"export")) {
        uint8_t p[0x100],*lib=class_cdo("KismetRenderingLibrary"),*fn=object_function(lib,"ExportRenderTarget");
        struct fstr dir,file; uint8_t *world=current_world();
        if(GetTickCount()-capture_job.started<100) { snprintf(err,ecap,"capture render pending; retry after 100ms"); return 0; }
        if(!engine_fstring(a->v[3],&dir) || !engine_fstring(a->v[4],&file)) { snprintf(err,ecap,"capture path too long"); return 0; }
        memset(p,0,sizeof p);
        if(!np_put(fn,p,sizeof p,"WorldContextObject",&world,8) || !np_put(fn,p,sizeof p,"TextureRenderTarget",&capture_job.target,8) ||
           !np_put(fn,p,sizeof p,"FilePath",&dir,16) || !np_put(fn,p,sizeof p,"Filename",&file,16)) {
            snprintf(err,ecap,"ExportRenderTarget signature unavailable"); return 0;
        }
        call(lib,fn,p); outf("capture exported token=%lu",capture_job.token); return 1;
    }
    snprintf(err,ecap,"usage: capture start w h x y z pitch yaw roll fov | export token directory filename | stop token"); return 0;
}
