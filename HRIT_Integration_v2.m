% HRIT_Integration_v2.m
% Full HRIT Computational Architecture - Fully Simulated Clinical States
%
% All component values computed from simulation — no manual specification.
%
% Flow state parameters grounded in literature:
%   Learning Flow: transient hypofrontality (Dietrich 2004)
%     I_Sim HIGH (perceptual inference maximal)
%     I_Id  SLIGHTLY BELOW reference (self-referential processing suppressed)
%     I_Prec SLIGHTLY BELOW reference (metacognitive monitoring suppressed)
%   Expert Flow: medial PFC dominance (Limb & Braun 2008, de Manzano 2010)
%     I_Sim MODERATE-HIGH (efficient, mastered task)
%     I_Id  HIGH (autobiographical self-expression dominant)
%     I_Prec MODERATE (lateral PFC partially deactivated)
%
% Ordering prediction: Expert Flow CII > Learning Flow CII > Normal Waking
% Distinct component signatures generate distinct EEG/MEG predictions.
%
% Equations (locked):
%   CII      = [0.40*I_Sim^p + 0.35*I_Id^p + 0.25*I_Prec^p]^(1/p)
%   p        = PE_residual_dominant / (MC_dominant + epsilon)
%   dominant = argmax(MC_Sim, MC_SelfRef)
%   PDS_expressed = PDS * sigmoid(k*(CII - MPT))
%   CSV      = (CII, PDS_expressed)
%
% Author: Hamza S. Almahi

clear; close all; clc;

%% ── SHARED PARAMETERS ────────────────────────────────────────────────────

w_Sim  = 0.40;
w_Id   = 0.35;
w_Prec = 0.25;
epsilon = 1e-6;
k      = 5.0;
alpha  = 0.15;
sigmoid = @(x) 1./(1 + exp(-x));

% I_Intero
dt        = 0.01;
T_intero  = 12000;
sigma_p   = 1.0;
kappa     = 0.5;
IBT       = 0.0;
w_pds     = 2.0;
PSC       = 0.5;
beta_CAL  = 0.15;
ICP_ref   = 0.25;
CAL_decay = 0.003;
t_perturb = 1000;

% I_Sim
n_trials_sim = 6;
T1       = 4;
n_seq    = 4;
obs_HHHL = [1 1 1 2];

% I_Id
n_trials_id = 10;

% I_Prec
n_trials_prec = 10;

%% ── CLINICAL STATE DEFINITIONS ───────────────────────────────────────────
%
% Parameters per state (17 fields):
%  1  label
%  2  sigma_o   (I_Intero: observation variance)
%  3  push      (I_Intero: sustained drive)
%  4  CAL_rec   (I_Intero: allostatic load recovery)
%  5  F_thr     (I_Intero: free energy threshold for CAL accumulation)
%  6  A1_prec   (I_Sim: level-1 A matrix precision)
%  7  B1_stab   (I_Sim: level-1 B matrix stability)
%  8  A2_prec   (I_Sim: level-2 A matrix precision)
%  9  B2_stab   (I_Sim: level-2 B matrix stability)
%  10 D2_sim    (I_Sim: level-2 D prior)
%  11 B_id      (I_Id: B matrix stability)
%  12 D_id      (I_Id: D prior)
%  13 T_id      (I_Id: trial length)
%  14 A2_prec_p (I_Prec: A matrix precision)
%  15 B_prec    (I_Prec: B matrix stability)
%  16 D_prec    (I_Prec: D prior)
%  17 T_prec    (I_Prec: trial length)
%
% Flow state rationale:
%   Learning Flow:
%     I_Sim: A1=0.95,B1=0.70,D2 concentrated -> I_Sim > 1 (high perceptual)
%     I_Id:  B=0.85, D=flat -> I_Id slightly below ref (hypofrontality)
%     I_Prec:A2=0.75,B=0.75 -> I_Prec slightly below ref (hypofrontality)
%   Expert Flow:
%     I_Sim: A1=0.95,B1=0.90,D2 concentrated -> I_Sim ~ 1 (efficient)
%     I_Id:  B=0.999,D concentrated -> I_Id HIGH (autobio self-expression)
%     I_Prec:A2=0.98,B=0.85 -> I_Prec HIGH (sharp metacognitive precision)

states = {
'Normal Waking', ...
    1.0,    0.0,    true,   2.0, ...
    0.95,   0.90,   0.90,   0.90,   [1;1;1;1]/4, ...
    0.990,  [0.90;0.10],            8, ...
    0.90,   0.85,   [0.85;0.15],    6;

'DPDR (Depth Collapse)', ...
    20.0,  -0.075,  true,   5.0, ...
    0.95,   0.90,   0.90,   0.90,   [1;1;1;1]/4, ...
    0.990,  [0.90;0.10],            8, ...
    0.55,   0.60,   [0.50;0.50],    6;

'Cotard (Depth Inversion)', ...
    5.0,   -0.50,   false,  0.0, ...
    0.95,   0.90,   0.90,   0.90,   [1;1;1;1]/4, ...
    0.500,  [0.50;0.50],            8, ...
    0.55,   0.60,   [0.50;0.50],    6;

'Propofol Anaesthesia', ...
    1.0,    0.0,    true,   2.0, ...
    0.20,   0.20,   0.20,   0.20,   [1;1;1;1]/4, ...
    0.500,  [0.50;0.50],            8, ...
    0.55,   0.60,   [0.50;0.50],    6;

'Expert Flow', ...
    1.0,    0.0,    true,   2.0, ...
    0.95,   0.90,   0.90,   0.90,   [0.90;0.04;0.03;0.03], ...
    0.999,  [0.99;0.01],            8, ...
    0.98,   0.85,   [0.95;0.05],    6;

'Learning Flow', ...
    1.0,    0.0,    true,   2.0, ...
    0.95,   0.70,   0.90,   0.70,   [0.90;0.04;0.03;0.03], ...
    0.980,  [0.88;0.12],            8, ...
    0.92,   0.87,   [0.80;0.20],    6;

'Anxiety/Fragmented', ...
    1.0,   -0.05,   true,   2.0, ...
    0.95,   0.50,   0.50,   0.50,   [1;1;1;1]/4, ...
    0.700,  [0.50;0.50],            8, ...
    0.55,   0.60,   [0.50;0.50],    6;
};

n_states = size(states,1);

colors_map = {
    [0    0.65 0   ];
    [0    0.4  0.8 ];
    [0.6  0    0.8 ];
    [0.85 0    0   ];
    [0.9  0.6  0   ];
    [0.2  0.7  0.9 ];
    [0.5  0.5  0.5 ];
};

%% ── STORAGE ──────────────────────────────────────────────────────────────

I_Sim_all   = zeros(1,n_states);
I_Id_all    = zeros(1,n_states);
I_Prec_all  = zeros(1,n_states);
PDS_all     = zeros(1,n_states);
ICP_all     = zeros(1,n_states);
PE_Sim_all  = zeros(1,n_states);
MC_Sim_all  = zeros(1,n_states);
PE_Id_all   = zeros(1,n_states);
MC_Id_all   = zeros(1,n_states);
PE_Prec_all = zeros(1,n_states);
MC_Prec_all = zeros(1,n_states);

acc_Sim_base  = NaN;
acc_Id_base   = NaN;
acc_Prec_base = NaN;

%% ── SIMULATION LOOP ──────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════════\n')
fprintf('HRIT Integration v2: Simulating all clinical states\n')
fprintf('═══════════════════════════════════════════════════════\n\n')

for s = 1:n_states

    label      = states{s,1};
    fprintf('── %s ──\n', label)

    sigma_o    = states{s,2};
    push       = states{s,3};
    CAL_rec    = states{s,4};
    F_thr      = states{s,5};
    A1_prec    = states{s,6};
    B1_stab    = states{s,7};
    A2_prec    = states{s,8};
    B2_stab    = states{s,9};
    D2_sim     = states{s,10};
    B_id       = states{s,11};
    D_id       = states{s,12};
    T_id_s     = states{s,13};
    A2_prec_p  = states{s,14};
    B_prec     = states{s,15};
    D_prec     = states{s,16};
    T_prec_s   = states{s,17};

    %% ── I_INTERO ─────────────────────────────────────────────────────
    fprintf('  I_Intero...\n')
    mu_i=0; y_i=0; a_i=0; CAL=0; F_prev=0;
    PDS_t=zeros(1,T_intero);
    ICP_t=zeros(1,T_intero);

    for i = 2:T_intero
        y_i = y_i + push*dt;
        if i==t_perturb; y_i=y_i+4.0; end
        y_i = y_i + a_i*dt;
        eps_o = y_i - mu_i;
        eps_p = IBT - mu_i;
        mu_i  = mu_i + dt*(eps_o/sigma_o + eps_p/sigma_p);
        a_i   = -kappa*eps_o/sigma_o;
        F_i   = 0.5*(eps_o^2/sigma_o + eps_p^2/sigma_p);
        ICP_i = 1/(1+abs(eps_o));
        ICP_t(i) = ICP_i;
        F_excess = max(0, F_i-F_thr);
        CAL = CAL + dt*F_excess;
        if CAL_rec && F_i<0.3
            CAL = max(0, CAL-CAL_decay*dt);
        end
        PDS_t(i) = tanh((w_pds*(ICP_i-ICP_ref)-beta_CAL*CAL)/PSC);
        F_prev = F_i;
    end
    PDS_all(s) = mean(PDS_t(end-2000:end));
    ICP_all(s) = mean(ICP_t(end-2000:end));

    %% ── I_SIM ────────────────────────────────────────────────────────
    fprintf('  I_Sim...\n')
    A1 = [A1_prec,1-A1_prec; 1-A1_prec,A1_prec];
    A1 = A1./sum(A1,1);
    A2 = [A2_prec,1-A2_prec,0.98,0.02;
          1-A2_prec,A2_prec,0.02,0.98];
    A2 = A2./sum(A2,1);
    B1 = [B1_stab,1-B1_stab; 1-B1_stab,B1_stab];
    B1 = B1./sum(B1,1);
    B2 = B2_stab*eye(n_seq)+(1-B2_stab)*ones(n_seq)/n_seq;
    B2 = B2./sum(B2,1);

    mdp1.A={A1}; mdp1.B={B1};
    mdp1.C={zeros(2,T1)}; mdp1.D={[0.5;0.5]};
    mdp1.V=ones(T1-1,1,1); mdp1.T=T1;
    mdp1.o=obs_HHHL(:)';

    mdp2.A={A2}; mdp2.B={B2};
    mdp2.C={zeros(2,n_trials_sim)}; mdp2.D={D2_sim};
    mdp2.V=ones(n_trials_sim-1,1,1); mdp2.T=n_trials_sim;
    mdp2.MDP=mdp1; mdp2.link=1;

    MDP2 = spm_MDP_VB_X(mdp2);

    xn2=MDP2.xn{1};
    acc_L2=0;
    for tau=1:n_trials_sim
        t_idx=min(tau,size(xn2,4));
        q_s2=squeeze(xn2(end,:,tau,t_idx));
        q_s2=q_s2(:)/sum(q_s2(:));
        llik2=log(A2(1,:)'+1e-16);
        acc_L2=acc_L2+q_s2'*llik2;
    end
    acc_L2=acc_L2/n_trials_sim;

    acc_L1=0; PE_L1=0;
    if isfield(MDP2,'mdp') && ~isempty(MDP2.mdp)
        n_emb=numel(MDP2.mdp);
        for tr=1:n_emb
            mdp1_tr=MDP2.mdp(tr);
            if isfield(mdp1_tr,'xn') && ~isempty(mdp1_tr.xn)
                xn1=mdp1_tr.xn{1};
                acc_tr=0;
                for tau=1:T1
                    t_idx=min(tau,size(xn1,4));
                    q_s1=squeeze(xn1(end,:,tau,t_idx));
                    q_s1=q_s1(:)/sum(q_s1(:));
                    o1_t=obs_HHHL(min(tau,length(obs_HHHL)));
                    llik1=log(A1(o1_t,:)'+1e-16);
                    acc_tr=acc_tr+q_s1'*llik1;
                end
                acc_L1=acc_L1+acc_tr/T1;
            end
            if isfield(mdp1_tr,'vn') && ~isempty(mdp1_tr.vn)
                vn1=mdp1_tr.vn{1};
                PE_L1=PE_L1+norm(vn1(end,:));
            end
        end
        acc_L1=acc_L1/n_emb;
        PE_L1=PE_L1/n_emb;
    else
        acc_L1=acc_L2; PE_L1=0;
    end

    acc_Sim=0.5*acc_L1+0.5*acc_L2;
    vn2=MDP2.vn{1};
    PE_L2=norm(vn2(end,:));
    PE_Sim_all(s)=0.5*PE_L1+0.5*PE_L2;
    MC_Sim_all(s)=max(0,mean(-MDP2.F)-abs(acc_L2));

    if s==1; acc_Sim_base=acc_Sim; end
    I_Sim_all(s)=abs(acc_Sim_base)/abs(acc_Sim);

    %% ── I_ID ─────────────────────────────────────────────────────────
    fprintf('  I_Id...\n')
    A_id=[0.90,0.10;0.10,0.90];
    A_id=A_id./sum(A_id,1);
    B_id_mat=[B_id,1-B_id;1-B_id,B_id];
    B_id_mat=B_id_mat./sum(B_id_mat,1);
    obs_id=ones(1,T_id_s);

    acc_id_tr=zeros(1,n_trials_id);
    PE_id_tr=zeros(1,n_trials_id);
    F_id_tr=zeros(1,n_trials_id);

    for tr=1:n_trials_id
        mdp_id.A={A_id}; mdp_id.B={B_id_mat};
        mdp_id.C={zeros(2,T_id_s)}; mdp_id.D={D_id};
        mdp_id.V=ones(T_id_s-1,1,1); mdp_id.T=T_id_s;
        mdp_id.o=obs_id;
        MDP_id=spm_MDP_VB_X(mdp_id);
        xn_id=MDP_id.xn{1};
        acc=0;
        for tau=1:T_id_s
            t_idx=min(tau,size(xn_id,4));
            q_s=squeeze(xn_id(end,:,tau,t_idx));
            q_s=q_s(:)/sum(q_s(:));
            llik=log(A_id(obs_id(tau),:)'+1e-16);
            acc=acc+q_s'*llik;
        end
        acc_id_tr(tr)=acc/T_id_s;
        F_id_tr(tr)=mean(-MDP_id.F);
        vn_id=MDP_id.vn{1};
        PE_id_tr(tr)=norm(vn_id(end,:));
    end

    acc_id=mean(acc_id_tr);
    if s==1; acc_Id_base=acc_id; end
    I_Id_all(s)=abs(acc_Id_base)/abs(acc_id);
    PE_Id_all(s)=mean(PE_id_tr);
    MC_Id_all(s)=max(0,mean(F_id_tr)-abs(acc_id));

    %% ── I_PREC ───────────────────────────────────────────────────────
    fprintf('  I_Prec...\n')
    A_prec_mat=[A2_prec_p,1-A2_prec_p;1-A2_prec_p,A2_prec_p];
    A_prec_mat=A_prec_mat./sum(A_prec_mat,1);
    B_prec_mat=[B_prec,1-B_prec;1-B_prec,B_prec];
    B_prec_mat=B_prec_mat./sum(B_prec_mat,1);
    obs_prec=ones(1,T_prec_s);

    acc_prec_tr=zeros(1,n_trials_prec);
    PE_prec_tr=zeros(1,n_trials_prec);
    F_prec_tr=zeros(1,n_trials_prec);

    for tr=1:n_trials_prec
        mdp_prec.A={A_prec_mat}; mdp_prec.B={B_prec_mat};
        mdp_prec.C={zeros(2,T_prec_s)}; mdp_prec.D={D_prec};
        mdp_prec.V=ones(T_prec_s-1,1,1); mdp_prec.T=T_prec_s;
        mdp_prec.o=obs_prec;
        MDP_prec=spm_MDP_VB_X(mdp_prec);
        xn_prec=MDP_prec.xn{1};
        acc=0;
        for tau=1:T_prec_s
            t_idx=min(tau,size(xn_prec,4));
            q_s=squeeze(xn_prec(end,:,tau,t_idx));
            q_s=q_s(:)/sum(q_s(:));
            llik=log(A_prec_mat(obs_prec(tau),:)'+1e-16);
            acc=acc+q_s'*llik;
        end
        acc_prec_tr(tr)=acc/T_prec_s;
        F_prec_tr(tr)=mean(-MDP_prec.F);
        vn_prec=MDP_prec.vn{1};
        PE_prec_tr(tr)=norm(vn_prec(end,:));
    end

    acc_prec=mean(acc_prec_tr);
    if s==1; acc_Prec_base=acc_prec; end
    I_Prec_all(s)=abs(acc_Prec_base)/abs(acc_prec);
    PE_Prec_all(s)=mean(PE_prec_tr);
    MC_Prec_all(s)=max(0,mean(F_prec_tr)-abs(acc_prec));

    fprintf('  Done: I_Sim=%.3f I_Id=%.3f I_Prec=%.3f PDS=%.3f\n', ...
            I_Sim_all(s),I_Id_all(s),I_Prec_all(s),PDS_all(s))

end

%% ── INTEGRATION ──────────────────────────────────────────────────────────

fprintf('\nComputing CII, p(Psi), CSV...\n')

I_Sim_max  = max(I_Sim_all);
I_Id_max   = max(I_Id_all);
I_Prec_max = max(I_Prec_all);
CII_max    = w_Sim*I_Sim_max + w_Id*I_Id_max + w_Prec*I_Prec_max;
MPT        = alpha*CII_max;

p_Psi         = zeros(1,n_states);
CII           = zeros(1,n_states);
PDS_expressed = zeros(1,n_states);
MC_Self       = MC_Id_all + MC_Prec_all;
PE_Self       = PE_Id_all + PE_Prec_all;

for s = 1:n_states
    if MC_Sim_all(s) >= MC_Self(s)
        p_Psi(s) = PE_Sim_all(s)/(MC_Sim_all(s)+epsilon);
    else
        p_Psi(s) = PE_Self(s)/(MC_Self(s)+epsilon);
    end

    p = p_Psi(s);
    if p < 1e-6
        CII(s) = I_Sim_all(s)^w_Sim * I_Id_all(s)^w_Id * I_Prec_all(s)^w_Prec;
    else
        CII(s) = (w_Sim*I_Sim_all(s)^p + w_Id*I_Id_all(s)^p + w_Prec*I_Prec_all(s)^p)^(1/p);
    end

    PDS_expressed(s) = PDS_all(s).*sigmoid(k*(CII(s)-MPT));
end

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

labels = states(:,1);

fprintf('\n')
fprintf('══════════════════════════════════════════════════════════════════════════════════\n')
fprintf('HRIT Integration v2: Full CSV State Space (Fully Simulated)\n')
fprintf('══════════════════════════════════════════════════════════════════════════════════\n')
fprintf('%-25s %6s %6s %6s %7s %7s %8s\n','State','I_Sim','I_Id','I_Prec','p','CII','PDS_exp')
fprintf('%-25s %6s %6s %6s %7s %7s %8s\n','-----','-----','----','------','---','---','-------')
for s = 1:n_states
    fprintf('%-25s %6.3f %6.3f %6.3f %7.3f %7.3f %8.3f\n', ...
            labels{s},I_Sim_all(s),I_Id_all(s),I_Prec_all(s), ...
            p_Psi(s),CII(s),PDS_expressed(s))
end
fprintf('══════════════════════════════════════════════════════════════════════════════════\n')
fprintf('CII_max=%.3f | MPT=%.3f | alpha=%.2f\n\n',CII_max,MPT,alpha)

fprintf('Expected ordering: Expert Flow CII > Learning Flow CII > Normal Waking CII\n')
fprintf('Expert Flow  : CII = %.3f\n', CII(5))
fprintf('Learning Flow: CII = %.3f\n', CII(6))
fprintf('Normal Waking: CII = %.3f\n\n', CII(1))

%% ── FIGURE 1: CSV State Space ────────────────────────────────────────────

figure('Name','HRIT: CSV State Space','Position',[50 50 950 680]);
hold on
xline(1.0,'k--','LineWidth',1.2)
yline(0,'k--','LineWidth',1.2)
yline(-1,'r:','LineWidth',1.0)
yline(1,'g:','LineWidth',1.0)
xline(MPT,'Color',[0.5 0.5 0.5],'LineStyle',':','LineWidth',1.5)

for s = 1:n_states
    scatter(CII(s),PDS_expressed(s),200,colors_map{s},...
            'filled','MarkerEdgeColor','k','LineWidth',1.2)
    text(CII(s)+0.04,PDS_expressed(s)+0.03,labels{s},...
         'FontSize',9,'Color',colors_map{s},'FontWeight','bold')
end

xlabel('CII (Consciousness Integration Index)','FontSize',12)
ylabel('PDS_{expressed} (Phenomenal Depth Signal)','FontSize',12)
title('HRIT: CSV State Space — Fully Simulated','FontSize',13)
xlim([0 max(CII)*1.25+0.2]); ylim([-1.15 1.15])
text(MPT+0.03,-1.05,'MPT','FontSize',9,'Color',[0.5 0.5 0.5],'FontWeight','bold')
grid on; box on

%% ── FIGURE 2: Component profiles ────────────────────────────────────────

figure('Name','HRIT: Component profiles','Position',[50 50 1200 500]);

comp_data   = [I_Sim_all; I_Id_all; I_Prec_all];
comp_labels = {'I\_Sim','I\_Id','I\_Prec'};

for comp = 1:3
    subplot(1,4,comp)
    b = bar(comp_data(comp,:),'FaceColor','flat');
    b.CData = cell2mat(colors_map);
    hold on; yline(1.0,'k--','LineWidth',1.5)
    set(gca,'XTick',1:n_states,'XTickLabel',labels,'XTickLabelRotation',35,'FontSize',8)
    ylabel(comp_labels{comp}); title(comp_labels{comp}); grid on
end

subplot(1,4,4)
b4 = bar(CII,'FaceColor','flat');
b4.CData = cell2mat(colors_map);
hold on
yline(1.0,'k--','LineWidth',1.5)
yline(MPT,'Color',[0.5 0.5 0.5],'LineStyle',':','LineWidth',1.5)
set(gca,'XTick',1:n_states,'XTickLabel',labels,'XTickLabelRotation',35,'FontSize',8)
ylabel('CII'); title('CII'); grid on

sgtitle('HRIT Integration v2: Component profiles and CII','FontSize',13)

%% ── FIGURE 3: p(Psi) regime map ─────────────────────────────────────────

figure('Name','HRIT: p(Psi) regime','Position',[50 50 700 550]);

PE_range=linspace(0,2,200);
MC_range=linspace(0.1,3,200);
[PE_grid,MC_grid]=meshgrid(PE_range,MC_range);
p_grid=PE_grid./(MC_grid+epsilon);

imagesc(PE_range,MC_range,p_grid)
colormap(flipud(hot)); colorbar
hold on

for s = 1:n_states
    if MC_Sim_all(s)>=MC_Self(s)
        pe_plot=PE_Sim_all(s); mc_plot=MC_Sim_all(s);
    else
        pe_plot=PE_Self(s); mc_plot=MC_Self(s);
    end
    scatter(pe_plot,mc_plot,150,colors_map{s},...
            'filled','MarkerEdgeColor','w','LineWidth',1.5)
    text(pe_plot+0.03,mc_plot+0.05,labels{s},...
         'FontSize',8,'Color','w','FontWeight','bold')
end

contour(PE_range,MC_range,p_grid,[1 1],'w--','LineWidth',2)
text(1.5,1.6,'p=1','Color','w','FontSize',9)
xlabel('PE_{residual} dominant','FontSize',11)
ylabel('MC dominant','FontSize',11)
title('p(\Psi) regime map','FontSize',12)
set(gca,'YDir','normal')