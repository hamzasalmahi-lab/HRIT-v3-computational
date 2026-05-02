% HRIT_I_Sim_v2.m
% I_Sim: Hierarchical Perceptual Inference (True Two-level POMDP)
% HRIT Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Simulation
%   Pre-computational: Hierarchical Perceptual Inference
%   Computational    : Two-level POMDP with precision-weighted
%                      hierarchical message passing via MDP.link
%
% Architecture (Smith/Friston/Whyte 2022 Section 6):
%   Level 1 (fast): tone identity inference, T1 timesteps per trial
%                   hidden states: {high=1, low=2} tone state
%                   observations:  {high=1, low=2} tone observation
%   Level 2 (slow): sequence pattern inference across n_trials
%                   hidden states: {HHHL=1, LLLH=2, HHHH=3, LLLL=4}
%                   observations:  {high-dominated=1, low-dominated=2}
%                                  received from level-1 via MDP.link
%
% Link format (spm_MDP_check confirmed):
%   MDP.link = 1  (scalar 1x1: factor 1 level-1 -> modality 1 level-2)
%   SPM25 converts via spm_speye(2,2,0) = 2x2 identity
%
% Three conditions:
%   Normal waking       : flat D2, standard HHHL sequence
%   Integration collapse: flat D2, degraded A1/B1/A2/B2
%   Learning flow       : concentrated D2 (expert prior on HHHL)
%                         -> I_Sim > 1 because prior matches observations
%
% Author: Hamza S. Almahi

clear; close all; clc;

%% ── PARAMETERS ───────────────────────────────────────────────────────────

n_trials = 6;    % level-1 trials = level-2 timesteps
T1       = 4;    % timesteps per level-1 trial
n_seq    = 4;    % number of sequence types at level 2

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────

obs_HHHL = [1 1 1 2];   % standard with local deviant (high high high low)

conditions = {
%   label                   A1_prec B1_stab A2_prec B2_stab L1_obs
    'Normal Waking',        0.95,   0.90,   0.90,   0.90,   obs_HHHL;
    'Integration Collapse', 0.20,   0.20,   0.20,   0.20,   obs_HHHL;
    'Learning Flow',        0.95,   0.70,   0.90,   0.70,   obs_HHHL;
};

% D2 priors per condition
% Normal/Collapse: flat (no prior on sequence type)
% Learning flow: concentrated on HHHL=1 (expert agent knows the sequence)
D2_priors = {
    ones(n_seq,1)/n_seq;          % Normal: flat
    ones(n_seq,1)/n_seq;          % Collapse: flat
    [0.90; 0.04; 0.03; 0.03];    % Learning flow: concentrated on HHHL
};

% Level-2 reference observation per condition
% Normal/Collapse: high-dominated (1) consistent with HHHL
% Learning flow: same — agent correctly predicts high-dominated
L2_obs_ref = {1, 1, 1};

n_cond = size(conditions,1);
colors = {[0 0.65 0], [0.85 0 0], [0.2 0.4 0.9]};

%% ── STORAGE ──────────────────────────────────────────────────────────────

acc_L2_all = zeros(1, n_cond);
acc_L1_all = zeros(1, n_cond);
PE_L2_all  = zeros(1, n_cond);
PE_L1_all  = zeros(1, n_cond);
F_L2_all   = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════\n')
fprintf('HRIT I_Sim v2: True Hierarchical Two-level POMDP\n')
fprintf('═══════════════════════════════════════════════════\n\n')

for c = 1:n_cond

    label   = conditions{c,1};
    A1_prec = conditions{c,2};
    B1_stab = conditions{c,3};
    A2_prec = conditions{c,4};
    B2_stab = conditions{c,5};
    L1_obs  = conditions{c,6};
    D2      = D2_priors{c};
    o2_ref  = L2_obs_ref{c};

    fprintf('Running: %s\n', label)

    %% ── BUILD LEVEL-1 MDP ────────────────────────────────────────────
    A1 = [A1_prec,   1-A1_prec;
          1-A1_prec, A1_prec  ];
    A1 = A1 ./ sum(A1,1);

    B1 = [B1_stab,   1-B1_stab;
          1-B1_stab, B1_stab  ];
    B1 = B1 ./ sum(B1,1);

    D1 = [0.5; 0.5];
    C1 = zeros(2, T1);
    V1 = ones(T1-1, 1, 1);

    mdp1.A = {A1};
    mdp1.B = {B1};
    mdp1.C = {C1};
    mdp1.D = {D1};
    mdp1.V = V1;
    mdp1.T = T1;
    mdp1.o = L1_obs(:)';

    %% ── BUILD LEVEL-2 MDP ────────────────────────────────────────────
    % A2: maps sequence state to expected level-1 outcome
    % Rows = level-2 outcomes (1=high-dom, 2=low-dom)
    % Cols = sequence states (HHHL, LLLH, HHHH, LLLL)
    A2 = [A2_prec,    1-A2_prec,  0.98,  0.02;
          1-A2_prec,  A2_prec,    0.02,  0.98];
    A2 = A2 ./ sum(A2,1);

    B2_eye  = eye(n_seq);
    B2_flat = ones(n_seq)/n_seq;
    B2 = B2_stab * B2_eye + (1-B2_stab) * B2_flat;
    B2 = B2 ./ sum(B2,1);

    C2 = zeros(2, n_trials);
    V2 = ones(n_trials-1, 1, 1);

    mdp2.A    = {A2};
    mdp2.B    = {B2};
    mdp2.C    = {C2};
    mdp2.D    = {D2};
    mdp2.V    = V2;
    mdp2.T    = n_trials;
    mdp2.MDP  = mdp1;   % embed level-1
    mdp2.link = 1;      % [nf x Ng] = [1 x 1]: factor 1 -> modality 1

    %% ── RUN ──────────────────────────────────────────────────────────
    MDP2 = spm_MDP_VB_X(mdp2);

    %% ── EXTRACT LEVEL-2 QUANTITIES ───────────────────────────────────
    xn2 = MDP2.xn{1};

    acc_L2 = 0;
    for tau = 1:n_trials
        t_idx = min(tau, size(xn2,4));
        q_s2  = squeeze(xn2(end, :, tau, t_idx));
        q_s2  = q_s2(:) / sum(q_s2(:));
        llik2 = log(A2(o2_ref,:)' + 1e-16);
        acc_L2 = acc_L2 + q_s2' * llik2;
    end
    acc_L2_all(c) = acc_L2 / n_trials;

    F_L2_all(c)  = mean(-MDP2.F);
    vn2 = MDP2.vn{1};
    PE_L2_all(c) = norm(vn2(end,:));

    %% ── EXTRACT LEVEL-1 QUANTITIES ───────────────────────────────────
    acc_L1 = 0;
    PE_L1  = 0;

    if isfield(MDP2,'mdp') && ~isempty(MDP2.mdp)
        n_emb = numel(MDP2.mdp);

        for tr = 1:n_emb
            mdp1_tr = MDP2.mdp(tr);

            if isfield(mdp1_tr,'xn') && ~isempty(mdp1_tr.xn)
                xn1    = mdp1_tr.xn{1};
                acc_tr = 0;
                for tau = 1:T1
                    t_idx = min(tau, size(xn1,4));
                    q_s1  = squeeze(xn1(end,:,tau,t_idx));
                    q_s1  = q_s1(:) / sum(q_s1(:));
                    o1_t  = L1_obs(min(tau,length(L1_obs)));
                    llik1 = log(A1(o1_t,:)' + 1e-16);
                    acc_tr = acc_tr + q_s1' * llik1;
                end
                acc_L1 = acc_L1 + acc_tr / T1;
            end

            if isfield(mdp1_tr,'vn') && ~isempty(mdp1_tr.vn)
                vn1   = mdp1_tr.vn{1};
                PE_L1 = PE_L1 + norm(vn1(end,:));
            end
        end

        acc_L1_all(c) = acc_L1 / n_emb;
        PE_L1_all(c)  = PE_L1 / n_emb;
    else
        acc_L1_all(c) = acc_L2_all(c);
        PE_L1_all(c)  = PE_L2_all(c);
        fprintf('  Warning: level-1 mdp not accessible\n')
    end

end

%% ── NORMALISE I_Sim ──────────────────────────────────────────────────────
acc_combined = 0.5 * acc_L1_all + 0.5 * acc_L2_all;
acc_baseline = acc_combined(1);

I_Sim  = abs(acc_baseline) ./ abs(acc_combined);
PE_Sim = 0.5 * PE_L1_all + 0.5 * PE_L2_all;
MC_Sim = max(0, F_L2_all - abs(acc_L2_all));
MMN    = PE_L1_all;
P3b    = PE_L2_all;

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Sim v2: Results\n')
fprintf('═══════════════════════════════════════════════════════════════════\n')
fprintf('%-25s %7s %7s %7s %7s %7s\n', ...
        'Condition','I_Sim','PE_Sim','MC_Sim','MMN','P3b')
fprintf('%-25s %7s %7s %7s %7s %7s\n', ...
        '---------','-----','------','------','---','---')
for c = 1:n_cond
    fprintf('%-25s %7.3f %7.5f %7.5f %7.5f %7.5f\n', ...
            conditions{c,1}, I_Sim(c), PE_Sim(c), ...
            MC_Sim(c), MMN(c), P3b(c))
end
fprintf('═══════════════════════════════════════════════════════════════════\n')
fprintf('\nTargets:\n')
fprintf('  Normal Waking       : I_Sim = 1.000\n')
fprintf('  Integration Collapse: I_Sim < 0.50\n')
fprintf('  Learning Flow       : I_Sim > 1.000\n')
fprintf('  MMN: Normal > Collapse\n')
fprintf('  P3b: Non-zero at both levels\n\n')

%% ── FIGURE 1: Main results ───────────────────────────────────────────────

figure('Name','HRIT I_Sim v2: Hierarchical Results',...
       'Position',[50 50 1100 450]);

subplot(1,3,1)
b1 = bar(I_Sim,'FaceColor','flat');
b1.CData = cell2mat(colors');
hold on; yline(1.0,'k--','LineWidth',1.5)
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',15)
ylabel('I\_Sim'); title('I\_Sim (normalised accuracy)'); grid on

subplot(1,3,2)
b2 = bar(MMN,'FaceColor','flat');
b2.CData = cell2mat(colors');
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',15)
ylabel('MMN (PE_{L1})')
title('MMN equivalent: Level-1 PE'); grid on

subplot(1,3,3)
b3 = bar(P3b,'FaceColor','flat');
b3.CData = cell2mat(colors');
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',15)
ylabel('P3b (PE_{L2})')
title('P3b equivalent: Level-2 PE'); grid on

sgtitle('HRIT I\_Sim v2: Hierarchical Perceptual Inference','FontSize',13)

%% ── FIGURE 2: PE and MC ──────────────────────────────────────────────────

figure('Name','HRIT I_Sim v2: PE and MC',...
       'Position',[50 50 800 400]);

subplot(1,2,1)
b4 = bar(PE_Sim,'FaceColor','flat');
b4.CData = cell2mat(colors');
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',15)
ylabel('PE_{residual}'); title('PE_{residual} (input to p(\Psi))'); grid on

subplot(1,2,2)
b5 = bar(MC_Sim,'FaceColor','flat');
b5.CData = cell2mat(colors');
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',15)
ylabel('MC_{Sim}'); title('MC_{Sim} (Metabolic Integration Cost)'); grid on

sgtitle('HRIT I\_Sim v2: PE_{residual} and MC_{Sim}','FontSize',12)