/* Pure UE4 FTransform matching for reversible instance masks. No engine calls.
   Instanced meshes store FMatrix: scale zero destroys its rotation basis. Reading
   it back can yield identity/arbitrary quaternion. ONLY translation + zero scale
   identify a masked slot; original quaternion must come from retained snapshot. */
#include <math.h>
#include <stdint.h>
#include <string.h>

static int foliage_hidden_transform_matches(const uint8_t original[48],const uint8_t current[48]) {
    int i; float a,b;
    for(i=4;i<=6;i++) {
        memcpy(&a,original+i*4,4); memcpy(&b,current+i*4,4);
        if(!isfinite(a) || !isfinite(b) || fabsf(a-b)>.1f) return 0;
    }
    for(i=8;i<=10;i++) {
        memcpy(&b,current+i*4,4);
        if(!isfinite(b) || fabsf(b)>.000001f) return 0;
    }
    return 1;
}

static int foliage_original_transform_matches(const uint8_t original[48],const uint8_t current[48]) {
    int i; float a,b; double dot=0,na=0,nb=0;
    for(i=0;i<12;i++) {
        if(i==7 || i==11) continue;
        memcpy(&a,original+i*4,4); memcpy(&b,current+i*4,4);
        if(!isfinite(a) || !isfinite(b)) return 0;
        if(i<4) { dot+=(double)a*b; na+=(double)a*a; nb+=(double)b*b; }
        else if(fabsf(a-b)>(i<7?.1f:.00001f)) return 0;
    }
    return na>.0001 && nb>.0001 && dot*dot/(na*nb)>.999999;
}
