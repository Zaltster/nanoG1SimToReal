// g1_host.c — single-env G1 SMOOTH dynamics on the host CPU (no libmujoco,
// no CUDA). Serial transcription of scripts/oracle_smooth.py, double-precision.
// This is the foundation of the browser/WASM demo (Direction B): the
// specialized physics with zero dependencies. Contact/constraint is layered
// on in a follow-up; this file is the smooth core, validated against the
// recorded MuJoCo "air" trajectory (constraints disabled).
//
// Build (validator):  clang -O2 web/g1_host.c -lm -o /tmp/g1host_test
// Run:                /tmp/g1host_test traj/air.bin
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include "g1_model_const.h"

// ---- small vector/quat helpers (mirror oracle_smooth.py top) -------------
static void quat_mul(const double a[4], const double b[4], double o[4]) {
    double r[4] = {
        a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3],
        a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2],
        a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1],
        a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0]};
    memcpy(o, r, sizeof r);
}
static void quat_norm(double q[4]) {
    double n = sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    if (n > 1e-15) { q[0]/=n; q[1]/=n; q[2]/=n; q[3]/=n; }
}
static void cross3(const double a[3], const double b[3], double o[3]) {
    o[0]=a[1]*b[2]-a[2]*b[1]; o[1]=a[2]*b[0]-a[0]*b[2]; o[2]=a[0]*b[1]-a[1]*b[0];
}
static void rot_vec_quat(const double v[3], const double q[4], double o[3]) {
    double u[3]={q[1],q[2],q[3]}, w=q[0], c1[3], c2[3];
    cross3(u,v,c1); cross3(u,c1,c2);
    for (int k=0;k<3;k++) o[k]=v[k]+2.0*(w*c1[k]+c2[k]);
}
static void axis_angle_quat(const double ax[3], double angle, double o[4]) {
    double h=0.5*angle, s=sin(h);
    o[0]=cos(h); o[1]=ax[0]*s; o[2]=ax[1]*s; o[3]=ax[2]*s;
}
static void quat2mat(const double q[4], double M[9]) {
    double w=q[0],x=q[1],y=q[2],z=q[3];
    M[0]=1-2*(y*y+z*z); M[1]=2*(x*y-w*z);  M[2]=2*(x*z+w*y);
    M[3]=2*(x*y+w*z);   M[4]=1-2*(x*x+z*z);M[5]=2*(y*z-w*x);
    M[6]=2*(x*z-w*y);   M[7]=2*(y*z+w*x);  M[8]=1-2*(x*x+y*y);
}
static void cross_motion(const double vel[6], const double v[6], double r[6]) {
    r[0]=-vel[2]*v[1]+vel[1]*v[2];
    r[1]= vel[2]*v[0]-vel[0]*v[2];
    r[2]=-vel[1]*v[0]+vel[0]*v[1];
    r[3]=-vel[2]*v[4]+vel[1]*v[5]-vel[5]*v[1]+vel[4]*v[2];
    r[4]= vel[2]*v[3]-vel[0]*v[5]+vel[5]*v[0]-vel[3]*v[2];
    r[5]=-vel[1]*v[3]+vel[0]*v[4]-vel[4]*v[0]+vel[3]*v[1];
}
static void cross_force(const double vel[6], const double f[6], double r[6]) {
    r[0]=-vel[2]*f[1]+vel[1]*f[2]-vel[5]*f[4]+vel[4]*f[5];
    r[1]= vel[2]*f[0]-vel[0]*f[2]+vel[5]*f[3]-vel[3]*f[5];
    r[2]=-vel[1]*f[0]+vel[0]*f[1]-vel[4]*f[3]+vel[3]*f[4];
    r[3]=-vel[2]*f[4]+vel[1]*f[5];
    r[4]= vel[2]*f[3]-vel[0]*f[5];
    r[5]=-vel[1]*f[3]+vel[0]*f[4];
}
static void mul_inert_vec(const double i[10], const double v[6], double r[6]) {
    r[0]=i[0]*v[0]+i[3]*v[1]+i[4]*v[2]-i[8]*v[4]+i[7]*v[5];
    r[1]=i[3]*v[0]+i[1]*v[1]+i[5]*v[2]+i[8]*v[3]-i[6]*v[5];
    r[2]=i[4]*v[0]+i[5]*v[1]+i[2]*v[2]-i[7]*v[3]+i[6]*v[4];
    r[3]=i[8]*v[1]-i[7]*v[2]+i[9]*v[3];
    r[4]=i[6]*v[2]-i[8]*v[0]+i[9]*v[4];
    r[5]=i[7]*v[0]-i[6]*v[1]+i[9]*v[5];
}
static void quat_integrate(double quat[4], const double vel[3], double scale) {
    double tmp[3]={vel[0],vel[1],vel[2]};
    double n=sqrt(tmp[0]*tmp[0]+tmp[1]*tmp[1]+tmp[2]*tmp[2]);
    if (n>1e-15) { tmp[0]/=n; tmp[1]/=n; tmp[2]/=n; }
    double qrot[4]; axis_angle_quat(tmp, scale*n, qrot);
    double qn[4]={quat[0],quat[1],quat[2],quat[3]}; quat_norm(qn);
    quat_mul(qn, qrot, quat);
}

// ---- scratch state (single env) ------------------------------------------
static double xpos[HC_NBODY][3], xquat[HC_NBODY][4];
static double xanchor[HC_NJNT][3], xaxis[HC_NJNT][3];
static double subtree_com[HC_NBODY][3], cinert[HC_NBODY][10], cdof[HC_NV][6];
static double crb[HC_NBODY][10], qM[HC_NM], qLD[HC_NM], qLDiagInv[HC_NV];
static double cvel[HC_NBODY][6], cdof_dot[HC_NV][6];
static double qfrc_bias[HC_NV], qfrc_smooth[HC_NV], qacc_smooth[HC_NV];

static void fk(const double* qpos) {
    memset(xpos[0],0,3*sizeof(double));
    xquat[0][0]=1; xquat[0][1]=xquat[0][2]=xquat[0][3]=0;
    for (int i=1;i<HC_NBODY;i++) {
        int jntadr=hc_body_jntadr[i], jntnum=hc_body_jntnum[i];
        double xp[3], xq[4];
        if (jntnum==1 && hc_jnt_type[jntadr]==HC_JNT_FREE) {
            int qadr=hc_jnt_qposadr[jntadr];
            for(int k=0;k<3;k++) xp[k]=qpos[qadr+k];
            for(int k=0;k<4;k++) xq[k]=qpos[qadr+3+k];
            quat_norm(xq);
            for(int k=0;k<3;k++){ xanchor[jntadr][k]=xp[k]; xaxis[jntadr][k]=hc_jnt_axis[jntadr*3+k]; }
        } else {
            int pid=hc_body_parentid[i];
            if (pid) {
                double t[3]; rot_vec_quat(&hc_body_pos[i*3], xquat[pid], t);
                for(int k=0;k<3;k++) xp[k]=t[k]+xpos[pid][k];
                quat_mul(xquat[pid], &hc_body_quat[i*4], xq);
            } else {
                for(int k=0;k<3;k++) xp[k]=hc_body_pos[i*3+k];
                for(int k=0;k<4;k++) xq[k]=hc_body_quat[i*4+k];
            }
            for (int j=jntadr;j<jntadr+jntnum;j++) {
                int qadr=hc_jnt_qposadr[j];
                double anchor[3], jp[3]={hc_jnt_pos[j*3],hc_jnt_pos[j*3+1],hc_jnt_pos[j*3+2]};
                double r[3]; rot_vec_quat(jp, xq, r);
                for(int k=0;k<3;k++) anchor[k]=r[k]+xp[k];
                double ax[3]={hc_jnt_axis[j*3],hc_jnt_axis[j*3+1],hc_jnt_axis[j*3+2]};
                double qloc[4]; axis_angle_quat(ax, qpos[qadr]-hc_qpos0[qadr], qloc);
                double xq2[4]; quat_mul(xq, qloc, xq2); memcpy(xq,xq2,sizeof xq);
                double r2[3]; rot_vec_quat(jp, xq, r2);
                for(int k=0;k<3;k++) xp[k]=anchor[k]-r2[k];
                for(int k=0;k<3;k++) xanchor[j][k]=anchor[k];
                rot_vec_quat(ax, xq, xaxis[j]);
            }
            quat_norm(xq);
        }
        for(int k=0;k<3;k++) xpos[i][k]=xp[k];
        memcpy(xquat[i],xq,sizeof xq); quat_norm(xquat[i]);
    }
}

static void com_pos(const double* qpos) {
    double xipos[HC_NBODY][3], ximat[HC_NBODY][9];
    memset(ximat[0],0,9*sizeof(double)); ximat[0][0]=ximat[0][4]=ximat[0][8]=1;
    memset(xipos[0],0,3*sizeof(double));
    for (int i=1;i<HC_NBODY;i++) {
        double t[3]; rot_vec_quat(&hc_body_ipos[i*3], xquat[i], t);
        for(int k=0;k<3;k++) xipos[i][k]=t[k]+xpos[i][k];
        double q[4]; quat_mul(xquat[i], &hc_body_iquat[i*4], q);
        quat2mat(q, ximat[i]);
    }
    // subtree_com: mass-weighted moments, backward accumulate, normalize
    for (int i=0;i<HC_NBODY;i++)
        for(int k=0;k<3;k++) subtree_com[i][k]=hc_body_mass[i]*xipos[i][k];
    for (int i=HC_NBODY-1;i>0;i--)
        for(int k=0;k<3;k++) subtree_com[hc_body_parentid[i]][k]+=subtree_com[i][k];
    for (int i=0;i<HC_NBODY;i++) {
        if (hc_body_subtreemass[i]>1e-15)
            for(int k=0;k<3;k++) subtree_com[i][k]/=hc_body_subtreemass[i];
        else for(int k=0;k<3;k++) subtree_com[i][k]=xipos[i][k];
    }
    // cinert
    memset(cinert[0],0,10*sizeof(double));
    for (int i=1;i<HC_NBODY;i++) {
        int root=hc_body_rootid[i];
        double dif[3]; for(int k=0;k<3;k++) dif[k]=xipos[i][k]-subtree_com[root][k];
        const double* I=&hc_body_inertia[i*3]; const double* M=ximat[i]; double mass=hc_body_mass[i];
        double full[3][3];
        for(int r=0;r<3;r++)for(int c=0;c<3;c++){ double s=0; for(int kk=0;kk<3;kk++) s+=M[r*3+kk]*I[kk]*M[c*3+kk]; full[r][c]=s; }
        double* ci=cinert[i];
        ci[0]=full[0][0]+mass*(dif[1]*dif[1]+dif[2]*dif[2]);
        ci[1]=full[1][1]+mass*(dif[0]*dif[0]+dif[2]*dif[2]);
        ci[2]=full[2][2]+mass*(dif[0]*dif[0]+dif[1]*dif[1]);
        ci[3]=full[0][1]-mass*dif[0]*dif[1];
        ci[4]=full[0][2]-mass*dif[0]*dif[2];
        ci[5]=full[1][2]-mass*dif[1]*dif[2];
        ci[6]=mass*dif[0]; ci[7]=mass*dif[1]; ci[8]=mass*dif[2]; ci[9]=mass;
    }
    // cdof
    memset(cdof, 0, sizeof cdof);
    double xmat1[9]; quat2mat(xquat[1], xmat1);
    for (int j=0;j<HC_NJNT;j++) {
        int da=hc_jnt_dofadr[j], bodyid=hc_jnt_bodyid[j], root=hc_body_rootid[bodyid];
        double off[3]; for(int k=0;k<3;k++) off[k]=subtree_com[root][k]-xanchor[j][k];
        if (hc_jnt_type[j]==HC_JNT_FREE) {
            for(int k=0;k<3;k++) cdof[da+k][3+k]=1.0;
            for(int k=0;k<3;k++){
                double axis[3]={xmat1[0*3+k],xmat1[1*3+k],xmat1[2*3+k]}; // column k
                double c[3]; cross3(axis, off, c);
                for(int t=0;t<3;t++){ cdof[da+3+k][t]=axis[t]; cdof[da+3+k][3+t]=c[t]; }
            }
        } else {
            double axis[3]={xaxis[j][0],xaxis[j][1],xaxis[j][2]};
            double c[3]; cross3(axis, off, c);
            for(int t=0;t<3;t++){ cdof[da][t]=axis[t]; cdof[da][3+t]=c[t]; }
        }
    }
}

static void crb_qM(void) {
    for (int i=0;i<HC_NBODY;i++) memcpy(crb[i],cinert[i],10*sizeof(double));
    for (int i=HC_NBODY-1;i>0;i--)
        if (hc_body_parentid[i]>0)
            for(int k=0;k<10;k++) crb[hc_body_parentid[i]][k]+=crb[i][k];
    for (int i=0;i<HC_NV;i++) {
        double buf[6]; mul_inert_vec(crb[hc_dof_bodyid[i]], cdof[i], buf);
        int adr=hc_dof_Madr[i];
        double d=0; for(int k=0;k<6;k++) d+=cdof[i][k]*buf[k];
        qM[adr]=hc_dof_armature[i]+d; adr++;
        int j=hc_dof_parentid[i];
        while (j>=0){ double s=0; for(int k=0;k<6;k++) s+=cdof[j][k]*buf[k]; qM[adr]=s; adr++; j=hc_dof_parentid[j]; }
    }
}

static void factor(void) {
    memcpy(qLD,qM,HC_NM*sizeof(double));
    for (int k=HC_NV-1;k>=0;k--) {
        int Madr_kk=hc_dof_Madr[k], Madr_ki=Madr_kk+1, i=hc_dof_parentid[k];
        while (i>=0) {
            double tmp=qLD[Madr_ki]/qLD[Madr_kk];
            int cnt=hc_dof_Madr[i+1]-hc_dof_Madr[i];
            for (int c=0;c<cnt;c++) qLD[hc_dof_Madr[i]+c]-=qLD[Madr_ki+c]*tmp;
            qLD[Madr_ki]=tmp; i=hc_dof_parentid[i]; Madr_ki++;
        }
    }
    for (int i=0;i<HC_NV;i++) qLDiagInv[i]=1.0/qLD[hc_dof_Madr[i]];
}

static void solve_LD(double* x) {
    for (int i=HC_NV-1;i>=0;i--) if (x[i]!=0.0) {
        int adr=hc_dof_Madr[i]+1, j=hc_dof_parentid[i];
        while (j>=0){ x[j]-=qLD[adr]*x[i]; adr++; j=hc_dof_parentid[j]; }
    }
    for (int i=0;i<HC_NV;i++) x[i]*=qLDiagInv[i];
    for (int i=0;i<HC_NV;i++) {
        int adr=hc_dof_Madr[i]+1, j=hc_dof_parentid[i];
        while (j>=0){ x[i]-=qLD[adr]*x[j]; adr++; j=hc_dof_parentid[j]; }
    }
}

static void com_vel(const double* qvel) {
    memset(cvel, 0, sizeof cvel); memset(cdof_dot, 0, sizeof cdof_dot);
    for (int i=1;i<HC_NBODY;i++) {
        double v[6]; memcpy(v, cvel[hc_body_parentid[i]], sizeof v);
        int bda=hc_body_dofadr[i], dn=hc_body_dofnum[i];
        if (dn==6) {
            for(int k=0;k<3;k++) for(int t=0;t<6;t++) v[t]+=cdof[bda+k][t]*qvel[bda+k];
            for(int k=0;k<3;k++) cross_motion(v, cdof[bda+3+k], cdof_dot[bda+3+k]);
            for(int k=0;k<3;k++) for(int t=0;t<6;t++) v[t]+=cdof[bda+3+k][t]*qvel[bda+3+k];
        } else {
            for(int k=0;k<dn;k++){ cross_motion(v, cdof[bda+k], cdof_dot[bda+k]);
                for(int t=0;t<6;t++) v[t]+=cdof[bda+k][t]*qvel[bda+k]; }
        }
        memcpy(cvel[i], v, sizeof v);
    }
}

static void rne(const double* qvel) {
    double cacc[HC_NBODY][6], cfrc[HC_NBODY][6];
    memset(cacc[0],0,6*sizeof(double));
    for(int k=0;k<3;k++) cacc[0][3+k]=-hc_gravity[k];
    for (int i=1;i<HC_NBODY;i++) {
        int bda=hc_body_dofadr[i], dn=hc_body_dofnum[i];
        double tmp[6]={0,0,0,0,0,0};
        for(int k=0;k<dn;k++) for(int t=0;t<6;t++) tmp[t]+=cdof_dot[bda+k][t]*qvel[bda+k];
        for(int t=0;t<6;t++) cacc[i][t]=cacc[hc_body_parentid[i]][t]+tmp[t];
        double a[6], b[6]; mul_inert_vec(cinert[i], cacc[i], a);
        double iv[6]; mul_inert_vec(cinert[i], cvel[i], iv);
        cross_force(cvel[i], iv, b);
        for(int t=0;t<6;t++) cfrc[i][t]=a[t]+b[t];
    }
    for (int i=HC_NBODY-1;i>0;i--)
        if (hc_body_parentid[i])
            for(int t=0;t<6;t++) cfrc[hc_body_parentid[i]][t]+=cfrc[i][t];
    for (int i=0;i<HC_NV;i++) {
        double s=0; const double* f=cfrc[hc_dof_bodyid[i]];
        for(int k=0;k<6;k++) s+=cdof[i][k]*f[k];
        qfrc_bias[i]=s;
    }
}

static double M_dense[HC_NV][HC_NV];
static void build_M_dense(void) {
    memset(M_dense, 0, sizeof M_dense);
    for (int i=0;i<HC_NV;i++) {
        int adr=hc_dof_Madr[i]; M_dense[i][i]=qM[adr]; int j=hc_dof_parentid[i]; adr++;
        while (j>=0){ M_dense[i][j]=M_dense[j][i]=qM[adr]; adr++; j=hc_dof_parentid[j]; }
    }
}

int hc_pd_unitree = 0;   // v3: use unitree leg PD gains (set by g1_demo.c)
// smooth forces only (no euler): fills qfrc_smooth, qacc_smooth, M_dense
static void smooth_forces(const double* qpos, const double* qvel, const double* ctrl) {
    fk(qpos); com_pos(qpos); crb_qM(); factor(); com_vel(qvel); rne(qvel);
    build_M_dense();
    for (int i=0;i<HC_NV;i++) qfrc_smooth[i]=-hc_dof_damping[i]*qvel[i]-qfrc_bias[i];
    for (int a=0;a<HC_NU;a++) {
        int jnt=hc_act_trnid[a], padr=hc_jnt_qposadr[jnt], dadr=hc_jnt_dofadr[jnt];
        double c=ctrl[a], lo=hc_act_ctrlrange[2*a], hi=hc_act_ctrlrange[2*a+1];
        c = c<lo?lo:(c>hi?hi:c);
        double force;
        if (hc_pd_unitree && a < 12) {   // v3: unitree leg PD (g1_staged_kernels.cuh:325)
            const double KP[6]={100,100,100,150,40,40}, KV[6]={2,2,2,4,2,2};
            int g=a%6;
            force = KP[g]*c - KP[g]*qpos[padr] - KV[g]*qvel[dadr];
        } else
        force=hc_act_gain0[a]*c + hc_act_bias1[a]*qpos[padr] + hc_act_bias2[a]*qvel[dadr];
        if (hc_jnt_actfrclimited[jnt]) {
            double flo=hc_jnt_actfrcrange[2*jnt], fhi=hc_jnt_actfrcrange[2*jnt+1];
            force = force<flo?flo:(force>fhi?fhi:force);
        }
        qfrc_smooth[dadr]+=force;
    }
    memcpy(qacc_smooth, qfrc_smooth, HC_NV*sizeof(double));
    solve_LD(qacc_smooth);
}

static void euler(const double* qpos, const double* qvel, const double* qacc,
                  double dt, double* qpos_next, double* qvel_next) {
    for (int i=0;i<HC_NV;i++) qvel_next[i]=qvel[i]+dt*qacc[i];
    memcpy(qpos_next, qpos, HC_NQ*sizeof(double));
    for (int j=0;j<HC_NJNT;j++) {
        int padr=hc_jnt_qposadr[j], vadr=hc_jnt_dofadr[j];
        if (hc_jnt_type[j]==HC_JNT_FREE) {
            for(int k=0;k<3;k++) qpos_next[padr+k]+=dt*qvel_next[vadr+k];
            quat_integrate(&qpos_next[padr+3], &qvel_next[vadr+3], dt);
        } else qpos_next[padr]+=dt*qvel_next[vadr];
    }
}

// smooth-only step (constraints/contact disabled) — for the 'air' validator
void g1_smooth_step(const double* qpos, const double* qvel, const double* ctrl,
                    double dt, double* qpos_next, double* qvel_next) {
    smooth_forces(qpos, qvel, ctrl);
    euler(qpos, qvel, qacc_smooth, dt, qpos_next, qvel_next);
}

// =========================================================================
// CONTACT + CONSTRAINT (transcribes oracle_contact.py + oracle_constraint.py)
// =========================================================================
#define HC_MINVAL 1e-15
static int g_ls_iter = HC_LS_ITER_DEFAULT;  // settable (v2s=3)
#define HC_NCON_MAX 16
#define HC_NEFC_MAX 160
enum { ST_QUAD=0, ST_LINNEG, ST_LINPOS, ST_SAT };

// narrowphase output
static int   ncon, con_g1[HC_NCON_MAX], con_g2[HC_NCON_MAX], con_ipair[HC_NCON_MAX];
static double con_dist[HC_NCON_MAX], con_pos[HC_NCON_MAX][3], con_frame[HC_NCON_MAX][9];
// efc
static int    efc_nf, efc_nefc, efc_ptype[HC_NEFC_MAX];
static double efc_J[HC_NEFC_MAX][HC_NV], efc_pos[HC_NEFC_MAX], efc_margin[HC_NEFC_MAX];
static double efc_floss[HC_NEFC_MAX], efc_D[HC_NEFC_MAX], efc_R[HC_NEFC_MAX], efc_aref[HC_NEFC_MAX];
static double qfrc_constraint[HC_NV], qacc_out[HC_NV];

static void geom_pose(int g, double gpos[3], double gmat[9]) {
    int b=hc_geom_bodyid[g];
    double t[3]; rot_vec_quat(&hc_geom_pos[g*3], xquat[b], t);
    for(int k=0;k<3;k++) gpos[k]=t[k]+xpos[b][k];
    double q[4]; quat_mul(xquat[b], &hc_geom_quat[g*4], q); quat2mat(q, gmat);
}
static void make_frame(const double n[3], double fr[9]) {
    double nn=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    double x[3]={n[0]/nn,n[1]/nn,n[2]/nn}, y[3]={0,0,0};
    if (-0.5<x[1] && x[1]<0.5) y[1]=1.0; else y[2]=1.0;
    double d=x[0]*y[0]+x[1]*y[1]+x[2]*y[2];
    for(int k=0;k<3;k++) y[k]-=x[k]*d;
    double yn=sqrt(y[0]*y[0]+y[1]*y[1]+y[2]*y[2]);
    for(int k=0;k<3;k++) y[k]/=yn;
    double z[3]; cross3(x,y,z);
    for(int k=0;k<3;k++){ fr[k]=x[k]; fr[3+k]=y[k]; fr[6+k]=z[k]; }
}
// column k of 3x3 mat (stored row-major) into v
static void matcol(const double M[9], int k, double v[3]){ v[0]=M[k]; v[1]=M[3+k]; v[2]=M[6+k]; }

// append a contact to the narrowphase list
static void emit_con(int g1,int g2,int ip,double dist,const double p[3],const double nml[3]){
    if (ncon>=HC_NCON_MAX) return;
    con_g1[ncon]=g1; con_g2[ncon]=g2; con_ipair[ncon]=ip; con_dist[ncon]=dist;
    for(int k=0;k<3;k++) con_pos[ncon][k]=p[k];
    make_frame(nml, con_frame[ncon]); ncon++;
}
static int sphere_sphere(const double p1[3],double r1,const double p2[3],double r2,
                         const double m1[9],const double m2[9],double margin,
                         double* odist,double opos[3],double onorm[3]){
    double dif[3]={p1[0]-p2[0],p1[1]-p2[1],p1[2]-p2[2]};
    double cd2=dif[0]*dif[0]+dif[1]*dif[1]+dif[2]*dif[2], md=margin+r1+r2;
    if (cd2>md*md) return 0;
    double dist=sqrt(cd2)-r1-r2, normal[3]={p2[0]-p1[0],p2[1]-p1[1],p2[2]-p1[2]};
    double nl=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
    if (nl>=1e-15){ for(int k=0;k<3;k++) normal[k]/=nl; }
    else { double a[3],b[3]; matcol(m1,2,a); matcol(m2,2,b); cross3(a,b,normal);
           double n=sqrt(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
           for(int k=0;k<3;k++) normal[k]/=n; }
    *odist=dist; for(int k=0;k<3;k++){ opos[k]=p1[k]+normal[k]*(r1+dist/2); onorm[k]=normal[k]; }
    return 1;
}
static double clampd(double x,double lo,double hi){ return x<lo?lo:(x>hi?hi:x); }

static void narrowphase(void) {
    ncon=0;
    for (int p=0;p<HC_NPAIR;p++) {
        int g1=hc_pair_geom1[p], g2=hc_pair_geom2[p];
        int t1=hc_geom_type[g1], t2=hc_geom_type[g2];
        double margin=hc_pair_margin[p], inclm=hc_pair_margin[p]-hc_pair_gap[p];
        double pos1[3],mat1[9],pos2[3],mat2[9];
        geom_pose(g1,pos1,mat1); geom_pose(g2,pos2,mat2);
        if (t1==HC_GEOM_PLANE && t2==HC_GEOM_BOX) {
            double norm[3]; matcol(mat1,2,norm);
            double dist=(pos2[0]-pos1[0])*norm[0]+(pos2[1]-pos1[1])*norm[1]+(pos2[2]-pos1[2])*norm[2];
            const double* sz=&hc_geom_size[g2*3]; int got=0;
            for (int i=0;i<8 && got<4;i++) {
                double vec[3]={(i&1?sz[0]:-sz[0]),(i&2?sz[1]:-sz[1]),(i&4?sz[2]:-sz[2])};
                double corner[3]; for(int r=0;r<3;r++) corner[r]=mat2[r*3+0]*vec[0]+mat2[r*3+1]*vec[1]+mat2[r*3+2]*vec[2];
                double ldist=norm[0]*corner[0]+norm[1]*corner[1]+norm[2]*corner[2];
                if (dist+ldist>margin || ldist>0) continue;
                double cdist=dist+ldist, cpos[3];
                for(int k=0;k<3;k++) cpos[k]=corner[k]+pos2[k]+norm[k]*(-cdist/2);
                if (cdist<inclm) emit_con(g1,g2,p,cdist,cpos,norm);
                got++;
            }
        } else if (t1==HC_GEOM_CAPSULE && t2==HC_GEOM_CAPSULE) {
            const double* s1=&hc_geom_size[g1*3]; const double* s2=&hc_geom_size[g2*3];
            double a1c[3],a2c[3]; matcol(mat1,2,a1c); matcol(mat2,2,a2c);
            double axis1[3],axis2[3]; for(int k=0;k<3;k++){ axis1[k]=a1c[k]*s1[1]; axis2[k]=a2c[k]*s2[1]; }
            double dif[3]={pos1[0]-pos2[0],pos1[1]-pos2[1],pos1[2]-pos2[2]};
            double ma=axis1[0]*axis1[0]+axis1[1]*axis1[1]+axis1[2]*axis1[2];
            double mb=-(axis1[0]*axis2[0]+axis1[1]*axis2[1]+axis1[2]*axis2[2]);
            double mc=axis2[0]*axis2[0]+axis2[1]*axis2[1]+axis2[2]*axis2[2];
            double u=-(axis1[0]*dif[0]+axis1[1]*dif[1]+axis1[2]*dif[2]);
            double v= axis2[0]*dif[0]+axis2[1]*dif[1]+axis2[2]*dif[2];
            double det=ma*mc-mb*mb;
            double od; double opos[3],onorm[3];
            if (fabs(det)>=1e-15) {
                double x1=(mc*u-mb*v)/det, x2=(ma*v-mb*u)/det;
                if (x1>1){ x1=1; x2=(v-mb)/mc; } else if (x1<-1){ x1=-1; x2=(v+mb)/mc; }
                if (x2>1){ x2=1; x1=clampd((u-mb)/ma,-1,1); } else if (x2<-1){ x2=-1; x1=clampd((u+mb)/ma,-1,1); }
                double c1[3],c2[3]; for(int k=0;k<3;k++){ c1[k]=pos1[k]+axis1[k]*x1; c2[k]=pos2[k]+axis2[k]*x2; }
                if (sphere_sphere(c1,s1[0],c2,s2[0],mat1,mat2,margin,&od,opos,onorm) && od<inclm)
                    emit_con(g1,g2,p,od,opos,onorm);
            } else {
                int got=0;
                for (int si=0; si<2 && got<2; si++) {
                    double x1s=si?-1.0:1.0, x2=clampd((v-mb*x1s)/mc,-1,1);
                    double c1[3],c2[3]; for(int k=0;k<3;k++){ c1[k]=pos1[k]+axis1[k]*x1s; c2[k]=pos2[k]+axis2[k]*x2; }
                    if (sphere_sphere(c1,s1[0],c2,s2[0],mat1,mat2,margin,&od,opos,onorm)){ if(od<inclm) emit_con(g1,g2,p,od,opos,onorm); got++; }
                }
            }
        }
    }
}

// jac_point translation part: jacp[3][nv] for a world point on `body`
static void jac_point(const double point[3], int body, double jacp[3][HC_NV]) {
    for(int r=0;r<3;r++) for(int c=0;c<HC_NV;c++) jacp[r][c]=0;
    double off[3]; for(int k=0;k<3;k++) off[k]=point[k]-subtree_com[1][k];  // root=pelvis(1)
    int i=hc_body_dofadr[body]+hc_body_dofnum[body]-1;
    while (i>=0) {
        double rot[3]={cdof[i][0],cdof[i][1],cdof[i][2]}, tr[3]={cdof[i][3],cdof[i][4],cdof[i][5]};
        double cr[3]; cross3(rot,off,cr);
        for(int r=0;r<3;r++) jacp[r][i]=tr[r]+cr[r];
        i=hc_dof_parentid[i];
    }
}

static double get_impedance(const double s[5], double pos, double margin) {
    if (s[0]==s[1] || s[2]<=HC_MINVAL) return 0.5*(s[0]+s[1]);
    double x=fabs((pos-margin)/s[2]);
    if (x>=1) return s[1]; if (x<=0) return s[0];
    double p=s[4], y;
    if (p==1) y=x;
    else if (x<=s[3]) { double a=1.0/pow(s[3],p-1); y=a*pow(x,p); }
    else { double b=1.0/pow(1-s[3],p-1); y=1-b*pow(1-x,p); }
    return s[0]+y*(s[1]-s[0]);
}

static void assemble(const double* qpos, const double* qvel, double dt) {
    int n=0;
    double rows_dA[HC_NEFC_MAX], rows_solref[HC_NEFC_MAX][2], rows_solimp[HC_NEFC_MAX][5];
    // friction dofs
    efc_nf=0;
    for (int i=0;i<HC_NV;i++) if (hc_dof_frictionloss[i]) {
        for(int c=0;c<HC_NV;c++) efc_J[n][c]=0; efc_J[n][i]=1.0;
        efc_pos[n]=0; efc_margin[n]=0; efc_floss[n]=hc_dof_frictionloss[i];
        rows_solref[n][0]=hc_dof_solref[i*2]; rows_solref[n][1]=hc_dof_solref[i*2+1];
        for(int k=0;k<5;k++) rows_solimp[n][k]=hc_dof_solimp[i*5+k];
        rows_dA[n]=hc_dof_invweight0[i]; efc_ptype[n]=0; n++; efc_nf++;
    }
    // joint limits
    for (int j=0;j<HC_NJNT;j++) if (hc_jnt_limited[j]) {
        double value=qpos[hc_jnt_qposadr[j]];
        for (int s=-1;s<=1;s+=2) {
            double dist=s*(hc_jnt_range[j*2+(s+1)/2]-value);
            if (dist<hc_jnt_margin[j]) {
                for(int c=0;c<HC_NV;c++) efc_J[n][c]=0; efc_J[n][hc_jnt_dofadr[j]]=-(double)s;
                efc_pos[n]=dist; efc_margin[n]=hc_jnt_margin[j]; efc_floss[n]=0;
                rows_solref[n][0]=hc_jnt_solref[j*2]; rows_solref[n][1]=hc_jnt_solref[j*2+1];
                for(int k=0;k<5;k++) rows_solimp[n][k]=hc_jnt_solimp[j*5+k];
                rows_dA[n]=hc_dof_invweight0[hc_jnt_dofadr[j]]; efc_ptype[n]=1; n++;
            }
        }
    }
    // contacts
    int con_start[HC_NCON_MAX], con_dim[HC_NCON_MAX]; double con_fri0[HC_NCON_MAX];
    for (int ci=0; ci<ncon; ci++) {
        int p=con_ipair[ci];
        double inclm=hc_pair_margin[p]-hc_pair_gap[p];
        if (con_dist[ci]>=inclm) { con_dim[ci]=0; continue; }
        int condim=hc_pair_dim[p];
        double fri5[5]={hc_pair_friction[p*5+0],hc_pair_friction[p*5+0],
                        hc_pair_friction[p*5+1],hc_pair_friction[p*5+2],hc_pair_friction[p*5+2]};
        int b1=hc_geom_bodyid[con_g1[ci]], b2=hc_geom_bodyid[con_g2[ci]];
        double j1[3][HC_NV], j2[3][HC_NV], jd[3][HC_NV];
        jac_point(con_pos[ci], b2, j2); jac_point(con_pos[ci], b1, j1);
        for(int r=0;r<3;r++) for(int c=0;c<HC_NV;c++) jd[r][c]=j2[r][c]-j1[r][c];
        const double* fr=con_frame[ci];
        double tran=hc_body_invweight0[b1*2]+hc_body_invweight0[b2*2];
        con_start[ci]=n; con_dim[ci]=condim; con_fri0[ci]=fri5[0];
        // frame rows in contact frame: jrows[r][c] = sum_k fr[r*3+k]*jd[k][c]
        double jrows[3][HC_NV];
        for(int r=0;r<3;r++) for(int c=0;c<HC_NV;c++) jrows[r][c]=fr[r*3+0]*jd[0][c]+fr[r*3+1]*jd[1][c]+fr[r*3+2]*jd[2][c];
        if (condim==1) {
            for(int c=0;c<HC_NV;c++) efc_J[n][c]=jrows[0][c];
            efc_pos[n]=con_dist[ci]; efc_margin[n]=inclm; efc_floss[n]=0;
            rows_solref[n][0]=hc_pair_solref[p*2]; rows_solref[n][1]=hc_pair_solref[p*2+1];
            for(int k=0;k<5;k++) rows_solimp[n][k]=hc_pair_solimp[p*5+k];
            rows_dA[n]=tran; efc_ptype[n]=2; n++;
        } else {
            for (int k=1;k<condim;k++) {
                double mu_k=fri5[k-1];
                for (int sg=0;sg<2;sg++) {
                    double sgn=sg?-1.0:1.0;
                    for(int c=0;c<HC_NV;c++) efc_J[n][c]=jrows[0][c]+sgn*mu_k*jrows[k][c];
                    efc_pos[n]=con_dist[ci]; efc_margin[n]=inclm; efc_floss[n]=0;
                    rows_solref[n][0]=hc_pair_solref[p*2]; rows_solref[n][1]=hc_pair_solref[p*2+1];
                    for(int q=0;q<5;q++) rows_solimp[n][q]=hc_pair_solimp[p*5+q];
                    rows_dA[n]=tran+mu_k*mu_k*tran; efc_ptype[n]=3; n++;
                }
            }
        }
    }
    efc_nefc=n;
    // impedance -> R,K,B,I ; aref ; pyramidal R adjustment
    double dt2=2*dt, K[HC_NEFC_MAX], B[HC_NEFC_MAX], Im[HC_NEFC_MAX];
    for (int i=0;i<n;i++) {
        double sr[2]={rows_solref[i][0],rows_solref[i][1]}, si[5];
        for(int k=0;k<5;k++) si[k]=rows_solimp[i][k];
        if (sr[0]>0) sr[0]=sr[0]>dt2?sr[0]:dt2;
        si[0]=clampd(si[0],0.0001,0.9999); si[1]=clampd(si[1],0.0001,0.9999);
        si[2]=si[2]<0?0:si[2]; si[3]=clampd(si[3],0.0001,0.9999); si[4]=si[4]<1?1:si[4];
        double imp=get_impedance(si, efc_pos[i], efc_margin[i]);
        efc_R[i]=fmax(HC_MINVAL,(1-imp)*rows_dA[i]/imp);
        if (efc_ptype[i]==0) K[i]=0;
        else if (sr[0]>0) K[i]=1.0/fmax(HC_MINVAL, si[1]*si[1]*sr[0]*sr[0]*sr[1]*sr[1]);
        else K[i]=-sr[0]/fmax(HC_MINVAL, si[1]*si[1]);
        if (sr[1]>0) B[i]=2.0/fmax(HC_MINVAL, si[1]*sr[0]);
        else B[i]=-sr[1]/fmax(HC_MINVAL, si[1]);
        Im[i]=imp;
    }
    for (int ci=0; ci<ncon; ci++) if (con_dim[ci]>1) {
        int st=con_start[ci]; double R1=efc_R[st]/fmax(HC_MINVAL,HC_IMPRATIO);
        double mu=con_fri0[ci]*sqrt(R1/efc_R[st]); double Rpy=2*mu*mu*efc_R[st];
        for (int k=0;k<2*(con_dim[ci]-1);k++) efc_R[st+k]=Rpy;
    }
    for (int i=0;i<n;i++) efc_D[i]=1.0/efc_R[i];
    for (int i=0;i<n;i++) {
        double vel=0; for(int c=0;c<HC_NV;c++) vel+=efc_J[i][c]*qvel[c];
        efc_aref[i]=-B[i]*vel - K[i]*Im[i]*(efc_pos[i]-efc_margin[i]);
    }
}

// constraint_update: force[], state[], returns cost
static double constraint_update(const double* jar, double* force, int* state) {
    double cost=0;
    for (int i=0;i<efc_nefc;i++) {
        force[i]=-efc_D[i]*jar[i]; state[i]=ST_QUAD;
        if (i<efc_nf) {
            double Rf=efc_R[i]*efc_floss[i];
            if (jar[i]<=-Rf){ cost+=-0.5*efc_R[i]*efc_floss[i]*efc_floss[i]-efc_floss[i]*jar[i]; force[i]=efc_floss[i]; state[i]=ST_LINNEG; }
            else if (jar[i]>=Rf){ cost+=-0.5*efc_R[i]*efc_floss[i]*efc_floss[i]+efc_floss[i]*jar[i]; force[i]=-efc_floss[i]; state[i]=ST_LINPOS; }
            else cost+=0.5*efc_D[i]*jar[i]*jar[i];
        } else {
            if (jar[i]>=0){ force[i]=0; state[i]=ST_SAT; }
            else cost+=0.5*efc_D[i]*jar[i]*jar[i];
        }
    }
    return cost;
}

// dense SPD Cholesky solve  H x = b  (H nv*nv, overwrites a working copy)
static void chol_solve(double H[HC_NV][HC_NV], const double* b, double* x) {
    static double Lc[HC_NV][HC_NV];
    for (int i=0;i<HC_NV;i++) for(int j=0;j<=i;j++) {
        double s=H[i][j]; for(int k=0;k<j;k++) s-=Lc[i][k]*Lc[j][k];
        if (i==j) Lc[i][j]=sqrt(s); else Lc[i][j]=s/Lc[j][j];
    }
    double y[HC_NV];
    for (int i=0;i<HC_NV;i++){ double s=b[i]; for(int k=0;k<i;k++) s-=Lc[i][k]*y[k]; y[i]=s/Lc[i][i]; }
    for (int i=HC_NV-1;i>=0;i--){ double s=y[i]; for(int k=i+1;k<HC_NV;k++) s-=Lc[k][i]*x[k]; x[i]=s/Lc[i][i]; }
}

// PrimalEval: cost + derivatives at alpha along search
static void primal_eval(const double qg[3], double (*quads)[3], const double* jaref,
                        const double* jv, double alpha, double* cost,double* d0,double* d1){
    double t[3]={qg[0],qg[1],qg[2]};
    for (int i=0;i<efc_nefc;i++) {
        if (i<efc_nf) {
            double x=jaref[i]+alpha*jv[i], f=efc_floss[i], Rf=efc_R[i]*efc_floss[i];
            if (-Rf<x && x<Rf){ for(int k=0;k<3;k++) t[k]+=quads[i][k]; }
            else if (x<=-Rf){ t[0]+=f*(-0.5*Rf-jaref[i]); t[1]+=-f*jv[i]; }
            else { t[0]+=f*(-0.5*Rf+jaref[i]); t[1]+=f*jv[i]; }
        } else if (jaref[i]+alpha*jv[i]<0) { for(int k=0;k<3;k++) t[k]+=quads[i][k]; }
    }
    *cost=alpha*alpha*t[2]+alpha*t[1]+t[0]; *d0=2*alpha*t[2]+t[1];
    *d1=fmax(2*t[2],HC_MINVAL);
}

typedef struct { double alpha,cost,d0,d1; } Pt;
static const double *ps_qg_jaref_jv; // unused placeholder to silence -Wunused
static double g_qg[3]; static double g_quads[HC_NEFC_MAX][3];
static double g_jaref[HC_NEFC_MAX], g_jv[HC_NEFC_MAX]; static int g_lsiter;
static Pt mkpt(double a){ Pt p; p.alpha=a; g_lsiter++;
    primal_eval(g_qg,g_quads,g_jaref,g_jv,a,&p.cost,&p.d0,&p.d1); return p; }

// PrimalSearch — fills Mv,jv; returns alpha (or -1 to signal "no step")
static double Mv_out[HC_NV];
static double primal_search(const double* search, const double* Ma,
                            const double* qfrc_sm, double gauss0, const double* jaref_in) {
    double snorm=0; for(int i=0;i<HC_NV;i++) snorm+=search[i]*search[i]; snorm=sqrt(snorm);
    if (snorm<HC_MINVAL) return -1;
    double scale=1.0/(HC_MEANINERTIA*(HC_NV>1?HC_NV:1));
    double gtol=HC_TOL*HC_LSTOL*snorm/scale;
    for (int i=0;i<HC_NV;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=M_dense[i][c]*search[c]; Mv_out[i]=s; }
    for (int i=0;i<efc_nefc;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=efc_J[i][c]*search[c]; g_jv[i]=s; }
    double sMa=0,sqf=0,sMv=0;
    for(int i=0;i<HC_NV;i++){ sMa+=search[i]*Ma[i]; sqf+=qfrc_sm[i]*search[i]; sMv+=search[i]*Mv_out[i]; }
    g_qg[0]=gauss0; g_qg[1]=sMa-sqf; g_qg[2]=0.5*sMv;
    for(int i=0;i<efc_nefc;i++){ g_jaref[i]=jaref_in[i];
        g_quads[i][0]=0.5*jaref_in[i]*efc_D[i]*jaref_in[i];
        g_quads[i][1]=g_jv[i]*efc_D[i]*jaref_in[i];
        g_quads[i][2]=0.5*g_jv[i]*efc_D[i]*g_jv[i]; }
    g_lsiter=0;
    Pt p0=mkpt(0.0);
    Pt p1=mkpt(p0.alpha - p0.d0/p0.d1);
    if (fabs(p1.d0)<gtol) return p1.alpha;
    int dir = p1.d0<0 ? 1 : -1;
    Pt p2=p0;
    while (p1.d0*dir<=-gtol && g_lsiter<g_ls_iter) {
        p2=p1; p1=mkpt(p1.alpha - p1.d0/p1.d1);
        if (fabs(p1.d0)<gtol) return p1.alpha;
    }
    if (g_lsiter>=g_ls_iter) return p1.alpha;  // runtime ls budget (oracle uses self.ls_iterations); NOT the hardcoded default — bit-exact at v1 where they're equal, but the constant breaks v2s (ls=3)
    Pt p2next=p1, p1next=mkpt(p1.alpha - p1.d0/p1.d1);
    while (g_lsiter<g_ls_iter) {
        Pt pmid=mkpt(0.5*(p1.alpha+p2.alpha));
        Pt cand[3]={p1next,p2next,pmid}; int have_best=0; Pt best;
        for(int c=0;c<3;c++) if (fabs(cand[c].d0)<gtol && (!have_best||cand[c].cost<best.cost)){ best=cand[c]; have_best=1; }
        if (have_best) return best.alpha;
        // update_bracket for p1 and p2
        int b1=0,b2=0;
        for(int c=0;c<3;c++){
            if (p1.d0<0 && cand[c].d0<0 && p1.d0<cand[c].d0){ p1=cand[c]; b1=1; }
            else if (p1.d0>0 && cand[c].d0>0 && p1.d0>cand[c].d0){ p1=cand[c]; b1=2; }
        }
        if (b1) p1next=mkpt(p1.alpha - p1.d0/p1.d1);
        for(int c=0;c<3;c++){
            if (p2.d0<0 && cand[c].d0<0 && p2.d0<cand[c].d0){ p2=cand[c]; b2=1; }
            else if (p2.d0>0 && cand[c].d0>0 && p2.d0>cand[c].d0){ p2=cand[c]; b2=2; }
        }
        if (b2) p2next=mkpt(p2.alpha - p2.d0/p2.d1);
        if (!b1 && !b2) return pmid.alpha;
    }
    if (p1.cost<=p2.cost && p1.cost<p0.cost) return p1.alpha;
    if (p2.cost<=p1.cost && p2.cost<p0.cost) return p2.alpha;
    return 0.0;
}

// full constraint solve -> qacc_out, qfrc_constraint
static void constraint_solve(const double* qacc_ws, int maxiter) {
    if (efc_nefc==0){ memcpy(qacc_out, qacc_smooth, HC_NV*sizeof(double)); memset(qfrc_constraint,0,sizeof qfrc_constraint); return; }
    static double force[HC_NEFC_MAX]; int state[HC_NEFC_MAX];
    double jar[HC_NEFC_MAX], Ma[HC_NV], qacc[HC_NV];
    // warmstart selection
    for(int i=0;i<efc_nefc;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=efc_J[i][c]*qacc_ws[c]; jar[i]=s-efc_aref[i]; }
    double cost_ws=constraint_update(jar, force, state);
    for(int i=0;i<HC_NV;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=M_dense[i][c]*qacc_ws[c]; Ma[i]=s; }
    double g=0; for(int i=0;i<HC_NV;i++) g+=0.5*(Ma[i]-qfrc_smooth[i])*(qacc_ws[i]-qacc_smooth[i]);
    cost_ws+=g;
    for(int i=0;i<efc_nefc;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=efc_J[i][c]*qacc_smooth[c]; jar[i]=s-efc_aref[i]; }
    double cost_sm=constraint_update(jar, force, state);
    if (cost_ws<=cost_sm) memcpy(qacc,qacc_ws,sizeof qacc); else memcpy(qacc,qacc_smooth,sizeof qacc);
    // Newton
    for(int i=0;i<HC_NV;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=M_dense[i][c]*qacc[c]; Ma[i]=s; }
    for(int i=0;i<efc_nefc;i++){ double s=0; for(int c=0;c<HC_NV;c++) s+=efc_J[i][c]*qacc[c]; jar[i]=s-efc_aref[i]; }
    double cost=constraint_update(jar, force, state);
    for(int c=0;c<HC_NV;c++){ double s=0; for(int i=0;i<efc_nefc;i++) s+=efc_J[i][c]*force[i]; qfrc_constraint[c]=s; }
    double gauss=0; for(int i=0;i<HC_NV;i++) gauss+=0.5*(Ma[i]-qfrc_smooth[i])*(qacc[i]-qacc_smooth[i]);
    cost+=gauss;
    double scale=1.0/(HC_MEANINERTIA*(HC_NV>1?HC_NV:1));
    static double H[HC_NV][HC_NV]; double grad[HC_NV], Mgrad[HC_NV], search[HC_NV];
    // initial search BEFORE the loop (oracle: H,grad,Mgrad,search precomputed)
    memcpy(H, M_dense, sizeof H);
    for(int i=0;i<efc_nefc;i++) if (state[i]==ST_QUAD)
        for(int r=0;r<HC_NV;r++) for(int c=0;c<HC_NV;c++) H[r][c]+=efc_D[i]*efc_J[i][r]*efc_J[i][c];
    for(int i=0;i<HC_NV;i++) grad[i]=Ma[i]-qfrc_smooth[i]-qfrc_constraint[i];
    chol_solve(H, grad, Mgrad);
    for(int i=0;i<HC_NV;i++) search[i]=-Mgrad[i];
    for (int it=0; it<maxiter; it++) {
        double alpha=primal_search(search, Ma, qfrc_smooth, gauss, jar);
        if (alpha==0.0 || alpha==-1) break;
        for(int i=0;i<HC_NV;i++){ qacc[i]+=alpha*search[i]; Ma[i]+=alpha*Mv_out[i]; }
        for(int i=0;i<efc_nefc;i++) jar[i]+=alpha*g_jv[i];
        double oldcost=cost;
        cost=constraint_update(jar, force, state);
        for(int c=0;c<HC_NV;c++){ double s=0; for(int i=0;i<efc_nefc;i++) s+=efc_J[i][c]*force[i]; qfrc_constraint[c]=s; }
        gauss=0; for(int i=0;i<HC_NV;i++) gauss+=0.5*(Ma[i]-qfrc_smooth[i])*(qacc[i]-qacc_smooth[i]);
        cost+=gauss;
        // recompute H, grad, Mgrad (the NEXT search), then break-check on NEW grad
        memcpy(H, M_dense, sizeof H);
        for(int i=0;i<efc_nefc;i++) if (state[i]==ST_QUAD)
            for(int r=0;r<HC_NV;r++) for(int c=0;c<HC_NV;c++) H[r][c]+=efc_D[i]*efc_J[i][r]*efc_J[i][c];
        for(int i=0;i<HC_NV;i++) grad[i]=Ma[i]-qfrc_smooth[i]-qfrc_constraint[i];
        chol_solve(H, grad, Mgrad);
        double improvement=scale*(oldcost-cost), gn=0;
        for(int i=0;i<HC_NV;i++) gn+=grad[i]*grad[i];
        double gradient=scale*sqrt(gn);
        if (improvement<HC_TOL || gradient<HC_TOL) break;
        for(int i=0;i<HC_NV;i++) search[i]=-Mgrad[i];
    }
    memcpy(qacc_out, qacc, HC_NV*sizeof(double));
}

// FULL single-env step (smooth + contact/constraint + euler)
void g1_full_step(const double* qpos, const double* qvel, const double* ctrl,
                  const double* qacc_ws, double dt, int maxiter,
                  double* qpos_next, double* qvel_next) {
    smooth_forces(qpos, qvel, ctrl);
    narrowphase();
    assemble(qpos, qvel, dt);
    constraint_solve(qacc_ws, maxiter);
    euler(qpos, qvel, qacc_out, dt, qpos_next, qvel_next);
}

// ------------------------- validator main ---------------------------------
#ifdef HC_VALIDATE
#include "../tests/traj_format.h"
int main(int argc, char** argv) {
    const char* path = argc>1?argv[1]:"traj/air.bin";
    FILE* f=fopen(path,"rb"); if(!f){perror("fopen");return 1;}
    TrajHeader h; if(fread(&h,sizeof h,1,f)!=1||h.magic!=TRAJ_MAGIC){printf("bad header\n");return 1;}
    printf("traj %s: %d steps, dt=%.4f\n", path, h.nsteps, h.dt);
    double max_qacc=0, max_qpos=0, max_qvel=0, max_qfrc=0; int max_ncon_err=0;
    for (int t=0;t<h.nsteps;t++) {
        TrajStep s; if(fread(&s,sizeof s,1,f)!=1){printf("short read\n");break;}
        double qpn[HC_NQ], qvn[HC_NV];
        // air (scenario 0) = smooth-only drop test (constraints disabled in the
        // reference); stand/random = full contact step.
        double* ref_qacc;
        int newton=HC_ITER_DEFAULT; const char* en=getenv("G1_TEST_NEWTON"); if(en) newton=atoi(en);
        const char* el=getenv("G1_TEST_LS"); if(el) g_ls_iter=atoi(el);
        if (h.scenario==TRAJ_SCEN_AIR) {
            g1_smooth_step(s.qpos, s.qvel, s.ctrl, h.dt, qpn, qvn);
            memcpy(qacc_out, qacc_smooth, HC_NV*sizeof(double));
            memset(qfrc_constraint, 0, sizeof qfrc_constraint); ncon=s.ncon;
            ref_qacc=s.qacc_smooth;
        } else {
            g1_full_step(s.qpos, s.qvel, s.ctrl, s.qacc_warmstart, h.dt, newton, qpn, qvn);
            ref_qacc=s.qacc;
        }
        if (ncon != s.ncon) max_ncon_err++;
        for(int i=0;i<HC_NV;i++){
            double e=fabs(qacc_out[i]-ref_qacc[i]); if(e>max_qacc)max_qacc=e;
            double ev=fabs(qvn[i]-s.qvel_next[i]); if(ev>max_qvel)max_qvel=ev;
            double ef=fabs(qfrc_constraint[i]-s.qfrc_constraint[i]); if(ef>max_qfrc)max_qfrc=ef;
        }
        for(int i=0;i<HC_NQ;i++){ double e=fabs(qpn[i]-s.qpos_next[i]); if(e>max_qpos)max_qpos=e; }
    }
    fclose(f);
    int pass = max_qacc<1e-6 && max_qpos<1e-8 && max_qvel<1e-8 && max_qfrc<1e-6 && max_ncon_err==0;
    printf("RESULT host_full max_qacc=%.3e max_qfrc_con=%.3e max_qpos_next=%.3e max_qvel_next=%.3e ncon_mismatch=%d pass=%d\n",
           max_qacc, max_qfrc, max_qpos, max_qvel, max_ncon_err, pass);
    return pass?0:1;
}
#endif
