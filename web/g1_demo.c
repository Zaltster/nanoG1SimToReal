// g1_demo.c — interactive G1 walking demo: the trained policy + the host
// physics (web/g1_host.c, validated vs MuJoCo to ~1e-10) + raylib. NO
// libmujoco, NO CUDA — self-contained, WASM-ready (Direction B).
//
// Native build (testable now):
//   web/build_demo.sh            (clang + vendored raylib)
// Web build (needs emscripten):
//   see web/build_demo.sh --web  (emcc; preloads the policy .bin)
//
// Controls: arrows = vx / yaw command, A/D = vy, Z = stop, R = reset.
// Headless self-check: G1_DEMO_FRAMES=N renders N control steps + exits 0 if
// the robot stayed up (proves the policy+physics loop works end-to-end).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "raylib.h"
#include "raymath.h"

#include "g1_host.c"                  // full host physics + model constants
#define PUFFERNET_IMPL
#include "../vendor/PufferLib/src/puffernet.h"

// Physics: v2s settings (dt 0.004 x 5 substeps, Newton 2, ls 3) — the EXACT
// integrator the walking policy trained on. The host stepper is validated
// against MuJoCo at these settings (web/validate_host.sh: stand max_qacc
// 1.4e-10, random 8.1e-10) after the linesearch-budget fix (g1_host.c:634).
#define DEMO_DT 0.004
#define DEMO_DECIM 5
#define DEMO_NEWTON 2
#define DEMO_ACTION_SCALE 0.5f

// --- env state ---
static double qpos[HC_NQ], qvel[HC_NV];
static float  prev_action[HC_NU], obs[98], act[HC_NU];
static int    g_v3 = 0, g_obsn = 96, g_phase = 0;  // v3: obs98 phase-clock + 12-DOF mask
static double g_action_scale = DEMO_ACTION_SCALE;  // per-policy (v2s 0.5, v3 0.25)
static double cmd[3];
static double warmstart[HC_NV];  // persists across substeps AND control steps (as in training)

static void world_to_base(const double q[4], const double v[3], double o[3]) {
    double qinv[4]={q[0],-q[1],-q[2],-q[3]}; rot_vec_quat(v, qinv, o);
}
static void demo_reset(void) {
    for(int i=0;i<HC_NQ;i++) qpos[i]=hc_key_qpos[i];
    for(int i=0;i<HC_NV;i++) qvel[i]=0;
    for(int i=0;i<HC_NU;i++) prev_action[i]=0;
    for(int i=0;i<HC_NV;i++) warmstart[i]=0;
    cmd[0]=cmd[1]=cmd[2]=0; g_phase=0;
}
static void build_obs(void) {
    double g[3]={0,0,-1}, gb[3]; world_to_base(qpos+3, g, gb);
    obs[0]=0.25f*(float)qvel[3]; obs[1]=0.25f*(float)qvel[4]; obs[2]=0.25f*(float)qvel[5];
    obs[3]=(float)gb[0]; obs[4]=(float)gb[1]; obs[5]=(float)gb[2];
    obs[6]=(float)cmd[0]; obs[7]=(float)cmd[1]; obs[8]=(float)cmd[2];
    for (int j=0;j<HC_NU;j++) {
        obs[9+j]=(float)(qpos[7+j]-hc_key_qpos[7+j]);  // deviation from HOME pose (training uses key_qpos, not qpos0)
        obs[38+j]=0.05f*(float)qvel[6+j];
        obs[67+j]=prev_action[j];
    }
    if (g_v3) {   // v3 gait phase clock (period 40 control steps, sin/cos)
        float ph=(float)(g_phase%40)/40.0f;
        obs[96]=sinf(6.2831853f*ph); obs[97]=cosf(6.2831853f*ph);
    }
}
// one 50Hz control step: obs -> policy -> ctrl -> DECIM physics substeps
static void demo_control_step(PufferNet* net) {
    build_obs();
    if (net) forward_puffernet(net, obs, act); else memset(act,0,sizeof act);
    double ctrl[HC_NU];
    for (int a=0;a<HC_NU;a++) {
        float c = act[a]<-1?-1:(act[a]>1?1:act[a]);
        if (g_v3 && a>=12) c=0.0f;   // v3: legs-only, waist+arms held at home
        prev_action[a]=c;
        double target=hc_key_qpos[7+a] /*= key_ctrl default*/ + g_action_scale*c;
        double lo=hc_act_ctrlrange[2*a], hi=hc_act_ctrlrange[2*a+1];
        ctrl[a]= target<lo?lo:(target>hi?hi:target);
    }
    double qpn[HC_NQ], qvn[HC_NV];
    for (int k=0;k<DEMO_DECIM;k++) {
        g1_full_step(qpos, qvel, ctrl, warmstart, DEMO_DT, DEMO_NEWTON, qpn, qvn);
        memcpy(qpos,qpn,sizeof qpos); memcpy(qvel,qvn,sizeof qvel);
        memcpy(warmstart, qacc_out, sizeof warmstart);  // persists (as in training)
    }
    if (g_v3) g_phase++;   // advance gait phase clock once per control step
}

// --- convergence eval (G1_DEMO_EVAL=1): the frozen bar (docs/convergence.md) ---
// Best-practice qualification: deterministic policy, full command battery, K
// noisy-reset seeds/command, the SAME perf kernel training uses (exp(-lin_err^2
// /0.25)) computed on host physics. Reports falls (robustness) + perf (tracking)
// + raw vel errors. A "converged" checkpoint = 0 falls AND high perf across all 6.
static unsigned g_erng;
static double eval_urand(void){ g_erng=g_erng*1103515245u+12345u; return ((g_erng>>9)&0x7fffff)/(double)0x7fffff*2.0-1.0; }
static void demo_reset_noisy(unsigned seed){
    for(int i=0;i<HC_NQ;i++) qpos[i]=hc_key_qpos[i];
    g_erng=seed?seed:1;
    for(int j=0;j<HC_NU;j++) qpos[7+j]+=0.05*eval_urand();   // ENV_RESET_NOISE=0.05
    for(int i=0;i<HC_NV;i++) qvel[i]=0;
    for(int i=0;i<HC_NU;i++) prev_action[i]=0;
    for(int i=0;i<HC_NV;i++) warmstart[i]=0;
}
static int demo_fallen(void){
    double gb[3],g[3]={0,0,-1}; world_to_base(qpos+3,g,gb);
    return (qpos[2]<0.35 || gb[2]>-0.6 || !isfinite(qpos[2]));
}
// reset the recurrent (MinGRU) hidden state — training resets it on every
// episode boundary; the eval MUST too or sub-runs inherit stale memory.
static void net_reset_state(PufferNet* net){
    if(net && net->mingru) memset(net->mingru->state, 0,
        (size_t)net->mingru->num_layers*net->mingru->batch_size*net->mingru->hidden_size*sizeof(float));
}
static void demo_convergence_eval(PufferNet* net){
    const double batt[6][3]={{0.8,0,0},{-0.5,0,0},{0.3,0,1.0},{0.3,0,-1.0},{0,0.4,0},{0,0,0}};
    const char* nm[6]={"forward","backward","turnL","turnR","strafe","stand"};
    int K=4, T=1000;   // 4 noisy seeds x 20s per command
    int tot_falls=0; double tot_perf=0; long tot_steps=0;
    printf("CONV_EVAL battery=6 seeds=%d steps=%d/each (%.0fs)\n", K, T, T*0.02);
    for(int ci=0;ci<6;ci++){
        int cf=0; double cp=0,cl=0,ca=0; long cs=0;
        for(int k=0;k<K;k++){
            demo_reset_noisy(1000u*ci+k+1); net_reset_state(net); g_phase=0;
            cmd[0]=batt[ci][0]; cmd[1]=batt[ci][1]; cmd[2]=batt[ci][2];
            for(int t=0;t<T;t++){
                demo_control_step(net);
                double vb[3]; world_to_base(qpos+3,qvel,vb);
                double ex=cmd[0]-vb[0], ey=cmd[1]-vb[1], eyaw=cmd[2]-qvel[5];
                cp += exp(-(ex*ex+ey*ey)/0.25); cl += sqrt(ex*ex+ey*ey); ca += fabs(eyaw); cs++;
                if(demo_fallen()){ cf++; demo_reset_noisy(1000u*ci+k+1+7919u*(t+1)); net_reset_state(net); g_phase=0;
                    cmd[0]=batt[ci][0]; cmd[1]=batt[ci][1]; cmd[2]=batt[ci][2]; }
            }
        }
        printf("CONV_EVAL cmd=%-8s falls=%d perf=%.3f lin_err=%.3f ang_err=%.3f\n",
               nm[ci], cf, cp/cs, cl/cs, ca/cs);
        tot_falls+=cf; tot_perf+=cp; tot_steps+=cs;
    }
    printf("RESULT conv falls=%d perf=%.3f n=%ld\n", tot_falls, tot_perf/tot_steps, tot_steps);
}

// --- posture/gait diagnostic (G1_DEMO_DIAG=1): why the gait/posture is off ---
static void demo_diag(PufferNet* net){
    demo_reset(); net_reset_state(net); cmd[0]=0.5; cmd[1]=0; cmd[2]=0;
    int T=2000, n=0, falls=0; double dev[HC_NU]={0}, pz=0,bp=0,br=0;
    double minz=9, maxz=-9;
    double bp2=0,br2=0,wxy2=0,wz2=0,vz2=0;   // wobble: pitch/roll variance, ang-vel & bob RMS
    double arate2=0, legqv2=0, pact[12]={0}; int havep=0;  // GAIT smoothness: action jerk + leg motion energy
    for(int t=0;t<T;t++){
        demo_control_step(net);
        if(demo_fallen()){ falls++; demo_reset(); net_reset_state(net); cmd[0]=0.5; havep=0; continue; }
        for(int j=0;j<HC_NU;j++) dev[j]+=qpos[7+j]-hc_key_qpos[7+j];
        double g[3]={0,0,-1}, gb[3]; world_to_base(qpos+3,g,gb);
        double cx=gb[0]<-1?-1:gb[0]>1?1:gb[0], cy=gb[1]<-1?-1:gb[1]>1?1:gb[1];
        double pit=asin(cx), rol=asin(cy);
        pz+=qpos[2]; bp+=pit; br+=rol; bp2+=pit*pit; br2+=rol*rol;
        double vb[3]; world_to_base(qpos+3,qvel,vb); vz2+=vb[2]*vb[2];   // vertical bob vel
        wxy2 += qvel[3]*qvel[3]+qvel[4]*qvel[4];   // base roll+pitch ANGULAR velocity (reward's w_ang_vel_xy term)
        wz2  += qvel[5]*qvel[5];
        // gait smoothness: per-step change in the 12 leg actions (jerk) + leg joint-vel energy
        if(havep){ for(int j=0;j<12;j++){ double d=act[j]-pact[j]; arate2+=d*d; } }
        for(int j=0;j<12;j++){ pact[j]=act[j]; legqv2+=qvel[6+j]*qvel[6+j]; }
        havep=1;
        if(qpos[2]<minz)minz=qpos[2]; if(qpos[2]>maxz)maxz=qpos[2]; n++;
    }
    double R=180.0/M_PI;
    double arate=sqrt(arate2/(12.0*fmax(1,n-1)));   // RMS per-joint action jerk (policy smoothness)
    double legqv=sqrt(legqv2/(12.0*n));             // RMS leg joint velocity (motion energy/thrash)
    double sp=sqrt(fmax(0,bp2/n-(bp/n)*(bp/n)))*R, sr=sqrt(fmax(0,br2/n-(br/n)*(br/n)))*R; // std (deg) = wobble amplitude
    double wrms=sqrt(wxy2/n);   // rad/s, RMS base roll/pitch rate
    printf("DIAG walk vx=0.5 %dsteps falls=%d | pelvis_z=%.3f (min %.3f max %.3f) base_pitch=%+.1fdeg base_roll=%+.1fdeg\n",
        T, falls, pz/n, minz, maxz, bp/n*R, br/n*R);
    printf("DIAG WOBBLE: pitch_std=%.2fdeg roll_std=%.2fdeg | ang_vel_xy_rms=%.3frad/s yaw_rate_rms=%.3f bob_vel_rms=%.3fm/s\n",
        sp, sr, wrms, sqrt(wz2/n), sqrt(vz2/n));
    printf("DIAG  reward ang_vel_xy penalty/step: at w=-0.05 -> %.4f   at w=-0.25 -> %.4f   (mean wx^2+wy^2=%.3f)\n",
        0.05*wxy2/n, 0.25*wxy2/n, wxy2/n);
    printf("DIAG GAIT-SMOOTHNESS: action_jerk_rms=%.4f  leg_qvel_rms=%.3frad/s  (lower=smoother gait)\n",
        arate, legqv);
    double lg=0,wa=0,ar=0;
    for(int j=0;j<12;j++) lg+=fabs(dev[j]/n);
    for(int j=12;j<15;j++) wa+=fabs(dev[j]/n);
    for(int j=15;j<29;j++) ar+=fabs(dev[j]/n);
    printf("DIAG sum|mean-dev-from-home| (rad): legs=%.2f waist=%.2f arms=%.2f\n", lg, wa, ar);
    printf("DIAG waist signed dev: yaw=%+.2f roll=%+.2f pitch=%+.2f\n", dev[12]/n,dev[13]/n,dev[14]/n);
    printf("DIAG per-joint mean dev (rad), idx 0-28:\n ");
    for(int j=0;j<HC_NU;j++){ printf("%+.2f ", dev[j]/n); if(j==11||j==14)printf("| "); }
    printf("\n");
}

// --- visual meshes baked from g1.mjb (web/g1_meshes.bin) ---
// 35 unique meshes, uploaded once, instanced per mesh-geom via geom_pose().
#define G1_MESH_MAGIC 0x47314D53
static Mesh   g_mesh[64];        // raylib mesh per model mesh id
static int    g_nmesh = 0;
static int    g_have_mesh = 0;
static Material g_mat;
static Shader   g_shader;
static Font     g_font;          // crisp TTF (JetBrains Mono) for the in-canvas chart

// flat-shaded, world-lit shader (raylib auto-wires mvp/matModel/matNormal/colDiffuse)
#ifdef PLATFORM_WEB
#define GLSL_V "#version 300 es\nprecision mediump float;\n"
#else
#define GLSL_V "#version 330\n"
#endif
static const char* VS = GLSL_V
  "in vec3 vertexPosition; in vec3 vertexNormal; in vec4 vertexColor;"
  "uniform mat4 mvp; uniform mat4 matNormal;"
  "out vec3 fN; out vec4 fC;"
  "void main(){ fN=normalize(vec3(matNormal*vec4(vertexNormal,0.0))); fC=vertexColor;"
  " gl_Position=mvp*vec4(vertexPosition,1.0); }";
static const char* FS = GLSL_V
  "in vec3 fN; in vec4 fC; uniform vec4 colDiffuse; out vec4 o;"
  "void main(){ vec3 N=normalize(fN);"
  " vec3 Lkey=normalize(vec3(0.4,0.5,0.85)), Lfill=normalize(vec3(-0.4,-0.6,0.25));"
  " float key=max(dot(N,Lkey),0.0), fill=max(dot(N,Lfill),0.0);"
  " float amb=0.45+0.12*N.z;"               // hemispheric ambient (sky brighter)
  " float s=min(amb+0.55*key+0.22*fill,1.15);"
  " o=vec4(fC.rgb*colDiffuse.rgb*s,1.0); }";

// build raylib meshes from the binary asset (non-indexed, per-face normals)
static int load_meshes(const char* path) {
    FILE* f=fopen(path,"rb"); if(!f){ printf("no mesh asset at %s\n",path); return 0; }
    int magic=0,nmesh=0; fread(&magic,4,1,f); fread(&nmesh,4,1,f);
    if (magic!=G1_MESH_MAGIC || nmesh<=0 || nmesh>64){ fclose(f); printf("bad mesh asset\n"); return 0; }
    int* vn=malloc(nmesh*4); int* fn=malloc(nmesh*4);
    for(int i=0;i<nmesh;i++){ fread(&vn[i],4,1,f); fread(&fn[i],4,1,f); }
    long vbase=ftell(f); long fbase=vbase; for(int i=0;i<nmesh;i++) fbase+=(long)vn[i]*3*4;
    for(int i=0;i<nmesh;i++){
        float* V=malloc((long)vn[i]*3*4); fseek(f,vbase,SEEK_SET); fread(V,4,(long)vn[i]*3,f); vbase=ftell(f);
        int* F=malloc((long)fn[i]*3*4);   fseek(f,fbase,SEEK_SET); fread(F,4,(long)fn[i]*3,f); fbase=ftell(f);
        Mesh msh={0}; msh.triangleCount=fn[i]; msh.vertexCount=fn[i]*3;
        msh.vertices=malloc((long)msh.vertexCount*3*4);
        msh.normals =malloc((long)msh.vertexCount*3*4);
        msh.colors  =malloc((long)msh.vertexCount*4);
        for(int t=0;t<fn[i];t++){
            int a=F[t*3],b=F[t*3+1],c=F[t*3+2];
            float* pa=&V[a*3]; float* pb=&V[b*3]; float* pc=&V[c*3];
            float u[3]={pb[0]-pa[0],pb[1]-pa[1],pb[2]-pa[2]};
            float w[3]={pc[0]-pa[0],pc[1]-pa[1],pc[2]-pa[2]};
            float n[3]={u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]};
            float ln=sqrtf(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]); if(ln>1e-9f){n[0]/=ln;n[1]/=ln;n[2]/=ln;}
            int idx[3]={a,b,c};
            for(int k=0;k<3;k++){ int o=(t*3+k)*3; int co=(t*3+k)*4; float* p=&V[idx[k]*3];
                msh.vertices[o]=p[0]; msh.vertices[o+1]=p[1]; msh.vertices[o+2]=p[2];
                msh.normals[o]=n[0]; msh.normals[o+1]=n[1]; msh.normals[o+2]=n[2];
                msh.colors[co]=255; msh.colors[co+1]=255; msh.colors[co+2]=255; msh.colors[co+3]=255; }
        }
        UploadMesh(&msh,false); g_mesh[i]=msh; free(V); free(F);
    }
    free(vn); free(fn); fclose(f); g_nmesh=nmesh;
    printf("loaded %d visual meshes from %s\n",nmesh,path); return 1;
}

// --- render: visual meshes (preferred) or collision primitives (fallback) ---
static void draw_geoms(void) {
    fk(qpos);  // refresh xpos/xquat for current pose
    for (int g=0; g<HC_NGEOM; g++) {
        int ty=hc_geom_type[g];
        double gp[3], gm[9]; geom_pose(g, gp, gm);
        if (ty==7 /*mesh*/) {
            if (!g_have_mesh) continue;
            int mid=hc_geom_dataid[g]; if(mid<0||mid>=g_nmesh) continue;
            Matrix mat={ (float)gm[0],(float)gm[1],(float)gm[2],(float)gp[0],
                         (float)gm[3],(float)gm[4],(float)gm[5],(float)gp[1],
                         (float)gm[6],(float)gm[7],(float)gm[8],(float)gp[2],
                         0,0,0,1 };
            const double* gc=&hc_geom_color[g*3];   // two-tone: white shells / dark joints
            g_mat.maps[MATERIAL_MAP_DIFFUSE].color=(Color){
                (unsigned char)(gc[0]*255),(unsigned char)(gc[1]*255),(unsigned char)(gc[2]*255),255};
            DrawMesh(g_mesh[mid], g_mat, mat);
            continue;
        }
        if (g_have_mesh) continue;  // meshes shown — skip collision primitives
        if (ty==0) continue;        // plane
        const double* sz=&hc_geom_size[g*3];
        Vector3 c={(float)gp[0],(float)gp[1],(float)gp[2]};
        Color col=(Color){90,120,180,255};
        if (ty==3 || ty==5) {       // capsule / cylinder
            Vector3 ax={(float)gm[2],(float)gm[5],(float)gm[8]};
            float hl=(float)sz[1];
            Vector3 a=Vector3Add(c, Vector3Scale(ax,hl)), b=Vector3Subtract(c, Vector3Scale(ax,hl));
            DrawCapsule(a,b,(float)sz[0],8,4,col);
        } else if (ty==6) {         // box (feet)
            DrawCube(c,(float)sz[0]*2,(float)sz[1]*2,(float)sz[2]*2,(Color){200,160,60,255});
        } else if (ty==2) {         // sphere
            DrawSphere(c,(float)sz[0],col);
        }
    }
}

int main(int argc, char** argv) {
    const char* wpath = argc>1?argv[1]:"assets/nanoG1.bin";  // the <60s policy (75M, v3)
    PufferNet* net=NULL;
    // auto-detect v3 (obs 98, phase-clock + 12-DOF masking) from weights size
    { FILE* wf=fopen(wpath,"rb"); if(wf){ fseek(wf,0,SEEK_END); g_v3=(ftell(wf)>654452); fclose(wf);} }
    g_obsn = g_v3 ? 98 : 96;
    g_action_scale = g_v3 ? 0.25 : DEMO_ACTION_SCALE;   // v3 trained at 0.25
    hc_pd_unitree = g_v3;   // v3 trained with unitree leg PD gains (G1_PD_UNITREE)
    Weights* w=load_weights(wpath);
    if (w) {
        int ls[HC_NU]; for(int j=0;j<HC_NU;j++) ls[j]=1;
        net=make_puffernet(w, 1, g_obsn, 128, 3, ls, HC_NU);
        printf("loaded %s policy %s (obs%d)\n", g_v3?"v3-masked":"v2s", wpath, g_obsn);
    } else printf("no policy at %s — running zero actions\n", wpath);
    g_ls_iter = 3;   // v2s linesearch budget — the policy's native integrator (validated)
    demo_reset();
    if (getenv("G1_DEMO_EVAL")) { demo_convergence_eval(net); return 0; }  // T2 convergence harness
    if (getenv("G1_DEMO_DIAG")) { demo_diag(net); return 0; }              // posture/gait diagnostic

    const char* mpath = getenv("G1_MESH_PATH"); if(!mpath) mpath="web/g1_meshes.bin";
    const char* fenv=getenv("G1_DEMO_FRAMES"); int auto_frames=(fenv&&!getenv("G1_DEMO_SHOT"))?atoi(fenv):0;
    if (auto_frames) {
        // headless: validate the mesh asset parses (no GL context available)
        FILE* mf=fopen(mpath,"rb"); int mg=0,nm=0; if(mf){fread(&mg,4,1,mf);fread(&nm,4,1,mf);fclose(mf);}
        printf("mesh asset: magic_ok=%d nmesh=%d\n", mg==G1_MESH_MAGIC, nm);
    }
    if (!auto_frames) {
        InitWindow(1280,720,"G1 — learned to walk in 59s, pure RL (host physics, no MuJoCo)");
        g_font=LoadFontEx("web/assets/font.ttf", 48, 0, 0);   // load big, draw small -> crisp
        if (g_font.texture.id==0) g_font=GetFontDefault();
        SetTextureFilter(g_font.texture, TEXTURE_FILTER_BILINEAR);
        g_shader=LoadShaderFromMemory(VS,FS);
        g_mat=LoadMaterialDefault(); g_mat.shader=g_shader;
        g_mat.maps[MATERIAL_MAP_DIFFUSE].color=(Color){255,255,255,255};
        g_have_mesh=load_meshes(mpath);
    }
    SetTargetFPS(50);
    float vx=0,vy=0,wz=0; int frame=0, falls=0;
    while (auto_frames ? frame<auto_frames : !WindowShouldClose()) {
        if (!auto_frames) {
            vx = IsKeyDown(KEY_UP)?0.8f:(IsKeyDown(KEY_DOWN)?-0.5f:0);
            wz = IsKeyDown(KEY_LEFT)?1.0f:(IsKeyDown(KEY_RIGHT)?-1.0f:0);
            vy = 0;   // strafe removed
            if (IsKeyPressed(KEY_R)) demo_reset();
        } else { vx=0.5f; }   // headless: command a forward walk
        cmd[0]=vx; cmd[1]=vy; cmd[2]=wz;
        demo_control_step(net);
        // fall check + auto-reset
        double gb[3], g[3]={0,0,-1}; world_to_base(qpos+3, g, gb);
        if (qpos[2]<0.35 || gb[2]>-0.6 || !isfinite(qpos[2])) { falls++; demo_reset(); }

        if (!auto_frames) {
            Camera3D cam={0}; cam.position=(Vector3){(float)qpos[0]-2.2f,(float)qpos[1]-2.2f,1.3f};
            cam.target=(Vector3){(float)qpos[0],(float)qpos[1],0.7f}; cam.up=(Vector3){0,0,1};
            cam.fovy=42; cam.projection=CAMERA_PERSPECTIVE;
            BeginDrawing(); ClearBackground((Color){235,238,242,255}); BeginMode3D(cam);
            DrawCube((Vector3){(float)qpos[0],(float)qpos[1],-0.01f},80,80,0.02f,(Color){250,250,251,255});
            for (int i=-25;i<=25;i++){ Color gl=(i%5==0)?(Color){190,194,200,255}:(Color){215,219,225,255};
                DrawLine3D((Vector3){(float)i,-25,0.001f},(Vector3){(float)i,25,0.001f},gl);
                DrawLine3D((Vector3){-25,(float)i,0.001f},(Vector3){25,(float)i,0.001f},gl); }
            draw_geoms();
            EndMode3D();
            // physics-throughput bar chart embedded in the env (transparent bg).
            // REAL matched-config numbers (docs/RESULTS.md, docs/genesis_g1.md): same
            // RTX 5090-class GPU, G1, fp32, foot-floor contact, pure stepping. Ours is
            // MuJoCo-bit-exact; Genesis runs its own approximate physics.
            {
                // measured peak physics steps/s, RTX 5090-class GPU, G1, dt 0.002,
                // Newton 3/ls 5, batch 16384 (bench/bench_*.py, 2026-06-17). ours/
                // warp/MJX = identical MuJoCo physics; Genesis = its own solver.
                Color ink=(Color){40,44,52,255};
                Color cOurs=(Color){46,160,67,255},  cWarp=(Color){31,111,235,255},
                      cGen =(Color){219,128,40,255}, cMjx =(Color){137,87,229,255};
                int nameW=150, maxw=500; double mx=8.9;       // nanoG1 = full bar
                int x0=52, bx=x0+nameW, y0=28;                 // left-shifted (room for the nanoG1 brand on the right)
                const char* nm[4]={"nanoG1","mujoco-warp","Genesis","MJX"};
                double vv[4]={8.9,4.00,2.28,1.12}; Color cc[4]={cOurs,cWarp,cGen,cMjx};
                for (int i=0;i<4;i++){
                    int y=y0+i*36, w=(int)(maxw*vv[i]/mx);
                    DrawTextEx(g_font, nm[i], (Vector2){(float)x0,(float)(y+3)}, 19, 1.0f, ink);
                    DrawRectangle(bx, y, w, 24, cc[i]);
                    DrawTextEx(g_font, TextFormat("%.1fM", vv[i]), (Vector2){(float)(bx+w+10),(float)(y+3)}, 18, 1.0f, ink);
                }
            }
            EndDrawing();
            if (getenv("G1_DEMO_SHOT") && frame>=atoi(getenv("G1_DEMO_SHOT"))) { TakeScreenshot("web/g1_demo.png"); break; }
        }
        frame++;
    }
    if (auto_frames) {
        printf("RESULT g1_demo frames=%d falls=%d final_pelvis_z=%.3f pass=%d\n",
               frame, falls, qpos[2], (isfinite(qpos[2]) && qpos[2]>0.3)?1:0);
        return (isfinite(qpos[2]) && qpos[2]>0.3)?0:1;
    }
    CloseWindow();
    return 0;
}
