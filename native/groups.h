/* Reversible editor operations. No gates/base actors accepted. All request
   preconditions checked before mutation; explicit failures get best-effort rollback.
   A caught native fault can still leave a partial edit: Python journals BEFORE state. */
#define EDIT_CAP 5000

static void managed_session(char *out,int cap) {
    char dir[1024],base[128]; uint64_t h=1469598103934665603ull; int i;
    map_directory(dir,sizeof dir); current_base(base,sizeof base);
    for(i=0;dir[i];i++) h=(h^(uint8_t)dir[i])*1099511628211ull;
    for(i=0;base[i];i++) h=(h^(uint8_t)base[i])*1099511628211ull;
    snprintf(out,cap,"%lx-%llx-%llx-%llx-%llx",(unsigned long)GetCurrentProcessId(),(unsigned long long)current_world(),
             (unsigned long long)current_level(),(unsigned long long)editor_pawn(),(unsigned long long)h);
}

static int group_valid(const char *s) {
    int i,n=(int)strlen(s);
    if(n<1 || n>96 || !strcmp(s,"None") || !strcmp(s,"Group")) return 0;
    for(i=0;i<n;i++) if(!isalnum((unsigned char)s[i]) && !strchr("_.-",s[i])) return 0;
    return 1;
}

static void actor_group(uint8_t *act,char *out,int cap) {
    uint64_t *tags; int n=actor_tag_list(act,&tags),i; char s[128];
    snprintf(out,cap,"-");
    for(i=0;i+1<n;i++) {
        fname_str(tags[i],s,sizeof s);
        if(ieq(s,"Group")) { fname_str(tags[i+1],out,cap); return; }
    }
}

static int group_set(uint8_t *act,const char *name) {
    uint8_t p[0x100],info[24],*lib=class_cdo("BPL_Functions_C"),*get,*set,*prop,*st=find_struct("S_MapEditorInfo");
    uint64_t group=0; int oi,go=-1,guard=0; char n[160],current[128];
    actor_group(act,current,sizeof current); if(ieq(current,name)) return 1;
    /* Canonical game tag setter manages UE arrays/allocator. Reject extra tags
       which that setter would discard instead of silently deleting user metadata. */
    {
        uint64_t *tags; int count=actor_tag_list(act,&tags),i,nonempty=0; char tagname[128],row[64];
        actor_row(act,row,sizeof row);
        for(i=0;i<count;i++) if(tags[i]) {
            nonempty++; fname_str(tags[i],tagname,sizeof tagname);
            if(tags[i]!=editor_tag() && !ieq(tagname,row) && !ieq(tagname,current) &&
               !ieq(tagname,"Group") && !ieq(tagname,"ME_StaticMesh") && !ieq(tagname,"ME_Blueprint")) return 0;
        }
        /* Spawn Asset retains a None placeholder; actor_tags intentionally hides it. */
        if(nonempty!=3 && nonempty!=5) return 0;
    }
    if(strcmp(name,"-") && (!group_valid(name) || !engine_fname(name,&group))) return 0;
    get=lib?object_function(lib,"GetMapEditorTags"):0; set=lib?object_function(lib,"SetMapEditorTags"):0;
    for(prop=st?rdp(st+USTRUCT_CHILDPROPS):0;prop && guard++<64;prop=rdp(prop+FFIELD_NEXT)) {
        fname_str(rd64(prop+FFIELD_NAME),n,sizeof n);
        if(!strncmp(n,"GroupId_",8)) { go=rd32(prop+FPROP_OFFSET); break; }
    }
    memset(p,0,sizeof p);
    if(go<0 || go+8>(int)sizeof info || !np_put(get,p,sizeof p,"Actor",&act,8) ||
       !np_put(get,p,sizeof p,"__WorldContext",&act,8) || (oi=param_offset(get,"Info"))<0 || oi+24>rd16(get+UFUNC_PARMSSIZE)) return 0;
    call(lib,get,p); memcpy(info,p+oi,24); memcpy(info+go,&group,8);
    memset(p,0,sizeof p);
    if(!np_put(set,p,sizeof p,"Actor",&act,8) || !np_put(set,p,sizeof p,"Info",info,24) ||
       !np_put(set,p,sizeof p,"__WorldContext",&act,8)) return 0;
    call(lib,set,p); actor_group(act,current,sizeof current); return ieq(current,name);
}

static void group_print_xf(FILE *f,const struct xf *t) {
    int i; for(i=0;i<3;i++) fprintf(f," %.9g",t->loc[i]);
    for(i=0;i<3;i++) fprintf(f," %.9g",t->rot[i]);
    for(i=0;i<3;i++) fprintf(f," %.9g",t->scale[i]);
}

static int verb_groups(const struct args *a,char *err,int ecap) {
    int i,j,n,nl,truncated=0,count=0,failed=0; uint8_t *levels[ENV_LEVEL_CAP]; uint64_t tag=editor_tag();
    FILE *f; char session[160];
    if(a->n!=2 || !(f=fopen(a->v[1],"w"))) { snprintf(err,ecap,"usage: groups writable_file"); return 0; }
    managed_session(session,sizeof session); fprintf(f,"session %s\n",session);
    nl=env_levels(levels,ENV_LEVEL_CAP,&truncated);
    for(j=0;j<nl;j++) {
        uint8_t **actors=env_level_actors(levels[j],&n);
        for(i=0;i<n;i++) {
            uint8_t *act=actors[i]; struct xf t; char row[64],group[128];
            if(!act || !is_editor_actor(act,tag) || is_gate(act)) continue;
            actor_row(act,row,sizeof row); actor_group(act,group,sizeof group);
            if(!row[0] || !actor_get(act,&t) || (strcmp(group,"-") && !group_valid(group))) { failed++; continue; }
            fprintf(f,"O %llx %s %s",(unsigned long long)act,row,group); group_print_xf(f,&t); fputc('\n',f); count++;
        }
    }
    fprintf(f,"end objects=%d failed=%d truncated=%d\n",count,failed,truncated);
    if(ferror(f)) { fclose(f); snprintf(err,ecap,"group inventory write failed"); return 0; }
    fclose(f); outf("groups objects=%d failed=%d truncated=%d",count,failed,truncated); return 1;
}

struct edit_item {
    char kind,row[64],oldgroup[128],newgroup[128];
    uint8_t *actor,*created; struct xf before,after; int applied,deleted;
};
static struct edit_item edit_items[EDIT_CAP];

static int edit_read_xf(const char *line,struct xf *t,int *used) {
    int n=0,i;
    if(sscanf(line," %f %f %f %f %f %f %f %f %f %n",&t->loc[0],&t->loc[1],&t->loc[2],&t->rot[0],&t->rot[1],&t->rot[2],
              &t->scale[0],&t->scale[1],&t->scale[2],&n)!=9 || !n) return 0;
    for(i=0;i<3;i++) if(!isfinite(t->loc[i]) || fabsf(t->loc[i])>1e8f || !isfinite(t->rot[i]) || !isfinite(t->scale[i]) ||
                        t->scale[i]<=0 || t->scale[i]>10000) return 0;
    *used=n; return 1;
}

static int edit_same(const struct xf *a,const struct xf *b) {
    int i; float qa[4],qb[4],dot=0;
    for(i=0;i<3;i++) if(fabsf(a->loc[i]-b->loc[i])>.2f || fabsf(a->scale[i]-b->scale[i])>.00002f) return 0;
    rotator_quat(a->rot,qa); rotator_quat(b->rot,qb);
    for(i=0;i<4;i++) dot+=qa[i]*qb[i];
    return fabsf(dot)>.9999998f;
}

static int edit_rollback(int n,FILE *out) {
    int i,ok=1; char error[160];
    for(i=n-1;i>=0;i--) {
        struct edit_item *e=&edit_items[i];
        if(e->deleted) {
            uint64_t group=0;
            if(strcmp(e->oldgroup,"-") && !engine_fname(e->oldgroup,&group)) { ok=0; continue; }
            e->actor=editor_spawn(e->row,&e->before,group,error,sizeof error);
            if(!e->actor) ok=0;
        } else if(e->kind=='T' && e->applied) {
            if(!alive(e->actor) || !actor_set(e->actor,&e->before) || !group_set(e->actor,e->oldgroup)) ok=0;
        }
        if(e->created && alive(e->created) && !destroy_actor(e->created)) ok=0;
    }
    fprintf(out,"rollback complete=%d\n",ok); fflush(out); return ok;
}

/* T actor row oldgroup <old xf> newgroup <new xf>; D actor row oldgroup <old xf>;
   C row newgroup <new xf>. First line: session <token>. All cm / degrees. */
static int verb_edit(const struct args *a,char *err,int ecap) {
    FILE *in,*out; char line[1800],token[160],actual[160]; int n=0,i,j,bad=0,used,skip;
    if(a->n!=3 || !(in=fopen(a->v[1],"r"))) { snprintf(err,ecap,"usage: edit request_file output_file"); return 0; }
    managed_session(actual,sizeof actual);
    if(!fgets(line,sizeof line,in) || sscanf(line,"session %159s",token)!=1 || strcmp(actual,token)) bad=1;
    memset(edit_items,0,sizeof edit_items);
    while(!bad && fgets(line,sizeof line,in)) {
        struct edit_item *e; unsigned long long addr=0;
        if(n==EDIT_CAP) { bad=1; break; } e=&edit_items[n]; e->kind=line[0]; skip=0;
        if(e->kind=='C') {
            if(sscanf(line,"C %63s %127s %n",e->row,e->newgroup,&skip)!=2 || !skip || !edit_read_xf(line+skip,&e->after,&used)) { bad=1; break; }
            skip+=used;
        } else if(e->kind=='T' || e->kind=='D') {
            if(sscanf(line,"%*c %llx %63s %127s %n",&addr,e->row,e->oldgroup,&skip)!=3 || !skip ||
               !edit_read_xf(line+skip,&e->before,&used)) { bad=1; break; }
            skip+=used; e->actor=(uint8_t *)(uintptr_t)addr;
            if(e->kind=='T') {
                int k=0;
                if(sscanf(line+skip," %127s %n",e->newgroup,&k)!=1 || !k || !edit_read_xf(line+skip+k,&e->after,&used)) { bad=1; break; }
                skip+=k+used;
            }
        } else { bad=1; break; }
        if(line[skip] || (e->newgroup[0] && strcmp(e->newgroup,"-") && !group_valid(e->newgroup))) { bad=1; break; }
        n++;
    }
    if(ferror(in)) bad=1;
    fclose(in);
    if(bad || !n) { snprintf(err,ecap,"invalid, stale-session or oversized edit request (max %d)",EDIT_CAP); return 0; }
    /* Validate every target and expected state, including duplicate handles. */
    for(i=0;i<n;i++) {
        struct edit_item *e=&edit_items[i]; struct row row;
        if(e->kind=='C') { if(!table_row("DT_MapEditorAssets",e->row,&row)) bad=1; }
        else {
            struct xf t; char r[64],g[128];
            if(!e->actor || !is_editor_actor(e->actor,editor_tag()) || is_gate(e->actor) || !is_level_actor(e->actor) || !actor_get(e->actor,&t)) bad=1;
            else {
                actor_row(e->actor,r,sizeof r); actor_group(e->actor,g,sizeof g);
                if(!ieq(r,e->row) || !ieq(g,e->oldgroup) || !edit_same(&t,&e->before)) bad=1;
            }
            for(j=0;j<i;j++) if(edit_items[j].actor==e->actor) bad=1;
        }
        if(bad) { snprintf(err,ecap,"edit precondition failed at item %d; nothing changed",i); return 0; }
    }
    out=fopen(a->v[2],"w"); if(!out) { snprintf(err,ecap,"cannot write edit result; nothing changed"); return 0; }
    /* Creation first, deletion last. Originals stay alive on spawn failures. */
    for(i=0;i<n;i++) if(edit_items[i].kind=='C') {
        struct edit_item *e=&edit_items[i]; uint64_t group=0;
        if(strcmp(e->newgroup,"-") && !engine_fname(e->newgroup,&group)) goto failed;
        e->created=editor_spawn(e->row,&e->after,group,err,ecap); if(!e->created) goto failed;
        fprintf(out,"C %d %llx\n",i,(unsigned long long)e->created); fflush(out);
    }
    for(i=0;i<n;i++) if(edit_items[i].kind=='T') {
        struct edit_item *e=&edit_items[i]; struct xf t;
        e->applied=1;
        if(!actor_set(e->actor,&e->after) || !group_set(e->actor,e->newgroup) || !actor_get(e->actor,&t) || !edit_same(&t,&e->after)) goto failed;
        fprintf(out,"T %d %llx\n",i,(unsigned long long)e->actor); fflush(out);
    }
    for(i=0;i<n;i++) if(edit_items[i].kind=='D') {
        struct edit_item *e=&edit_items[i];
        if(!destroy_actor(e->actor)) goto failed;
        e->deleted=1; fprintf(out,"D %d %llx\n",i,(unsigned long long)e->actor); fflush(out);
    }
    fprintf(out,"end changed=%d complete=1\n",n); fclose(out); outf("edit changed=%d complete=1",n); return 1;
failed:
    bad=edit_rollback(n,out); fclose(out);
    snprintf(err,ecap,"edit failed at item %d; rollback_complete=%d; inspect journal before retrying",i,bad); return 0;
}

/* Assembly graph facts: one oriented bounds-vs-real-collision query per member.
   Python requires reciprocal hits; no claim of exact mesh-mesh contact or loads. */
static int verb_contacts(const struct args *a,char *err,int ecap) {
    FILE *in,*out; char line[160]; int count=0,i,nh,j; float gap=argf(a,3,5);
    if(a->n!=4 || !isfinite(gap) || gap<0 || gap>100 || !(in=fopen(a->v[1],"r"))) {
        snprintf(err,ecap,"usage: contacts request output tolerance_cm (0..100)"); return 0;
    }
    out=fopen(a->v[2],"w"); if(!out) { fclose(in); snprintf(err,ecap,"cannot write contacts"); return 0; }
    if(!kismet_init(err,ecap)) { fclose(in); fclose(out); return 0; }
    while(fgets(line,sizeof line,in)) {
        unsigned long long address; uint8_t *act,*hit[512]; float c[3],h[3],rot[3]={0,0,0}; int obb;
        if(sscanf(line,"A %llx",&address)!=1 || ++count>EDIT_CAP) { fclose(in); fclose(out); snprintf(err,ecap,"invalid contacts request"); return 0; }
        act=(uint8_t *)(uintptr_t)address;
        if(!act || !is_editor_actor(act,editor_tag()) || is_gate(act)) { fprintf(out,"error actor=%llx\n",address); continue; }
        obb=actor_obb(act,c,h,rot); if(!obb && !actor_bounds(act,c,h)) { fprintf(out,"error actor=%llx\n",address); continue; }
        for(i=0;i<3;i++) h[i]+=gap;
        nh=k_box(act,c,h,rot,hit,512);
        fprintf(out,"A %llx %d",address,nh==512);
        for(j=0;j<nh;j++) if(hit[j] && is_editor_actor(hit[j],editor_tag()) && !is_gate(hit[j])) fprintf(out," %llx",(unsigned long long)hit[j]);
        fputc('\n',out);
    }
    fprintf(out,"end count=%d\n",count);
    i=ferror(out)||ferror(in); fclose(in); fclose(out);
    if(i) { snprintf(err,ecap,"contact query IO failed"); return 0; }
    return 1;
}
