/* Assembly foundation anchors: bare landscape/base geometry, never other placed
   members. Vertical collision spans when available; bounds fallback is explicit. */
static int verb_anchors(const struct args *a,char *err,int ecap) {
    FILE *in,*out; char line[160]; float tol=argf(a,3,30),tlo[3],thi[3]; int trunc=0,terrain,count=0,bad=0;
    if(a->n!=4 || !isfinite(tol) || tol<0 || tol>100 || !(in=fopen(a->v[1],"r"))) {
        snprintf(err,ecap,"usage: anchors request output tolerance_cm (0..100)"); return 0;
    }
    out=fopen(a->v[2],"w"); if(!out) { fclose(in); snprintf(err,ecap,"cannot write anchors"); return 0; }
    if(!kismet_init(err,ecap)) { fclose(in); fclose(out); return 0; }
    env_nplants=0; terrain=env_terrain_bounds(tlo,thi,&trunc)>0;
    while(fgets(line,sizeof line,in)) {
        unsigned long long address; uint8_t *act,*root; float o[3],e[3],c[3],h[3],rot[3]={0,0,0},ax[3][3];
        int i,j,samples=0,anchored=0,missing=0,buried=0,collision,obb;
        if(sscanf(line,"A %llx",&address)!=1 || ++count>EDIT_CAP) { bad=1; break; }
        act=(uint8_t *)(uintptr_t)address;
        if(!act || !is_editor_actor(act,editor_tag()) || is_gate(act) || !actor_bounds(act,o,e)) {
            fprintf(out,"error actor=%llx\n",address); continue;
        }
        obb=actor_obb(act,c,h,rot);
        if(!obb || fabsf(rot[0])>5 || fabsf(rot[2])>5) { memcpy(c,o,12); memcpy(h,e,12); memset(rot,0,12); }
        rot_axes(rot,ax); root=root_component(act); collision=obb && root && ltc_init(root);
        for(i=0;i<5;i++) for(j=0;j<5;j++) {
            float s[3],end[3],hp[3],top=o[2]+e[2],bottom=o[2]-e[2]; uint8_t *surface=0; int exhausted=0;
            float u=h[0]*(i*.5f-1)*.95f,v=h[1]*(j*.5f-1)*.95f;
            s[0]=end[0]=c[0]+u*ax[0][0]+v*ax[1][0]; s[1]=end[1]=c[1]+u*ax[0][1]+v*ax[1][1];
            if(collision) {
                s[2]=top+50; end[2]=bottom-50;
                if(!ltc(root,s,end,hp)) continue;
                top=hp[2]; s[2]=bottom-50; end[2]=o[2]+e[2]+50;
                if(!ltc(root,s,end,hp)) { missing++; continue; }
                bottom=hp[2];
            }
            samples++;
            s[2]=terrain?thi[2]+1000:top+tol+50; end[2]=terrain?tlo[2]-1000:bottom-1000;
            if(!env_line(act,s,end,hp,&surface,terrain?2:3,&exhausted)) { missing++; if(exhausted) trunc=1; continue; }
            if(hp[2]>top+tol) buried++;
            else if(hp[2]>=bottom-tol) anchored++;
        }
        fprintf(out,"F %llx samples=%d anchored=%d missing=%d buried=%d basis=%s\n",address,samples,anchored,missing,buried,
                collision?"collision_vertical_span":"bounds_span");
    }
    fprintf(out,"end count=%d truncated=%d\n",count,trunc);
    bad=bad||ferror(out)||ferror(in); fclose(in); fclose(out);
    if(bad) { snprintf(err,ecap,"invalid or incomplete anchor query"); return 0; }
    return 1;
}
