% HRIT_I_Sim_v3.m
% I_Sim: Hierarchical Perceptual Inference — Physiological States
% HRIT v3 Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Simulation
%   Pre-computational: Hierarchical Perceptual Inference
%   Computational    : Two-level POMDP (spm_MDP_VB_X, MDP.link = 1)
%
% What changed from v2 (clinical baseline):
%   v2 modelled three conditions: Normal Waking, Integration Collapse,
%   Learning Flow — a verification that the architecture works.
%   v3 models six PHYSIOLOGICAL states in the healthy and sub-optimal brain,
%   matched to the I_Intero v4 physiological framework.
%
% Six conditions and their I_Sim physiological interpretation:
%   1. Waking Rest          — baseline hierarchical inference (reference)
%   2. Physical Exertion    — enhanced L1 precision (focused bodily attention)
%   3. Psychological Stress — reduced stability (hyperscanning, attentional jumps)
%   4. Sleep N3             — dramatically suppressed (dreamless sleep)
%   5. Flow / Optimal       — expert prior on sequence + peak precision
%   6. Allostatic Overload  — reduced precision from cumulative exhaustion
%
% Connection to Study 1c (Sleep EEG):
%   Condition 4 (Sleep N3): A1_prec and A2_prec dramatically reduced
%   → I_Sim ≈ 0.10-0.15, consistent with near-zero perceptual inference
%   in dreamless N3 sleep. Complements I_Intero v4 Sleep Onset condition.
%
% Architecture:
%   Level 1: tone identity per timestep (4 steps, 2 states, 2 obs)
%   Level 2: sequence pattern across trials (6 trials, 4 states, 2 obs)
%   MDP.link = 1: L1 posteriors → L2 observations
%
% Author: Hamza S. Almahi
% Version: v3 — June 2026

clear; close all; clc;

%% ── PARAMETERS ───────────────────────────────────────────────────────────

n_trials = 6;
T1       = 4;
n_seq    = 4;

obs_HHHL = [1 1 1 2];

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% Parameters:
%   A1_prec : Level-1 perceptual likelihood precision
%             (how accurately brain maps states to observations)
%   B1_stab : Level-1 state transition stability
%             (how sustained moment-to-moment attention is)
%   A2_prec : Level-2 sequence likelihood precision
%             (how accurately higher-level patterns are recognised)
%   B2_stab : Level-2 sequence state stability
%             (how sustained higher-level pattern expectations are)
%
% Physiological rationale:
%   Exertion    : A1_prec elevated — exercise focuses interoceptive attention
%   Stress      : B1_stab reduced — hyperscanning, attention rapidly redirected
%   Sleep N3    : all parameters dramatically reduced — near-absent inference
%   Flow        : concentrated D2 prior, full precision — expert prediction
%   Overload    : all parameters moderately reduced — exhaustion-driven impairment

colors = {[0.08 0.50 0.25], [0.88 0.50 0.05], [0.20 0.35 0.78], ...
          [0.40 0.18 0.70], [0.05 0.65 0.70], [0.70 0.12 0.15]};

conditions = {
%   label                          A1_prec B1_stab A2_prec B2_stab L1_obs
    'Waking Rest',                  0.95,   0.90,   0.90,   0.90,   obs_HHHL;
    'Physical Exertion',            0.97,   0.85,   0.90,   0.85,   obs_HHHL;
    'Psychological Stress',         0.95,   0.70,   0.85,   0.70,   obs_HHHL;
    'Sleep Onset (N3)',             0.20,   0.30,   0.30,   0.30,   obs_HHHL;
    'Flow / Optimal',               0.95,   0.70,   0.90,   0.70,   obs_HHHL;
    'Allostatic Overload',          0.70,   0.70,   0.70,   0.70,   obs_HHHL;
};

D2_priors = {
    ones(n_seq,1)/n_seq;               % Waking Rest: flat
    ones(n_seq,1)/n_seq;               % Exertion: flat
    ones(n_seq,1)/n_seq;               % Stress: flat
    ones(n_seq,1)/n_seq;               % Sleep N3: flat (no expectation)
    [0.90; 0.04; 0.03; 0.03];         % Flow: concentrated (expert prior)
    ones(n_seq,1)/n_seq;               % Overload: flat
};

L2_obs_ref = {1, 1, 1, 1, 1, 1};

n_cond = size(conditions, 1);

%% ── STORAGE ──────────────────────────────────────────────────────────────

acc_L2_all = zeros(1, n_cond);
acc_L1_all = zeros(1, n_cond);
PE_L2_all  = zeros(1, n_cond);
PE_L1_all  = zeros(1, n_cond);
F_L2_all   = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Sim v3: Physiological Hierarchical Perceptual Inference\n')
fprintf('═══════════════════════════════════════════════════════════════\n\n')

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
    A1 = A1 ./ sum(A1, 1);

    B1 = [B1_stab,   1-B1_stab;
          1-B1_stab, B1_stab  ];
    B1 = B1 ./ sum(B1, 1);

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
    A2 = [A2_prec,    1-A2_prec,  0.98,  0.02;
          1-A2_prec,  A2_prec,    0.02,  0.98];
    A2 = A2 ./ sum(A2, 1);

    B2_eye  = eye(n_seq);
    B2_flat = ones(n_seq) / n_seq;
    B2 = B2_stab * B2_eye + (1-B2_stab) * B2_flat;
    B2 = B2 ./ sum(B2, 1);

    C2 = zeros(2, n_trials);
    V2 = ones(n_trials-1, 1, 1);

    mdp2.A    = {A2};
    mdp2.B    = {B2};
    mdp2.C    = {C2};
    mdp2.D    = {D2};
    mdp2.V    = V2;
    mdp2.T    = n_trials;
    mdp2.MDP  = mdp1;
    mdp2.link = 1;

    %% ── RUN ──────────────────────────────────────────────────────────
    MDP2 = spm_MDP_VB_X(mdp2);

    %% ── EXTRACT LEVEL-2 QUANTITIES ───────────────────────────────────
    xn2    = MDP2.xn{1};
    acc_L2 = 0;

    for tau = 1:n_trials
        t_idx = min(tau, size(xn2, 4));
        q_s2  = squeeze(xn2(end, :, tau, t_idx));
        q_s2  = q_s2(:) / sum(q_s2(:));
        llik2 = log(A2(o2_ref,:)' + 1e-16);
        acc_L2 = acc_L2 + q_s2' * llik2;
    end
    acc_L2_all(c) = acc_L2 / n_trials;

    F_L2_all(c)  = mean(-MDP2.F);
    vn2          = MDP2.vn{1};
    PE_L2_all(c) = norm(vn2(end,:));

    %% ── EXTRACT LEVEL-1 QUANTITIES ───────────────────────────────────
    acc_L1 = 0;
    PE_L1  = 0;

    if isfield(MDP2, 'mdp') && ~isempty(MDP2.mdp)
        n_emb = numel(MDP2.mdp);
        for tr = 1:n_emb
            mdp1_tr = MDP2.mdp(tr);
            if isfield(mdp1_tr, 'xn') && ~isempty(mdp1_tr.xn)
                xn1    = mdp1_tr.xn{1};
                acc_tr = 0;
                for tau = 1:T1
                    t_idx = min(tau, size(xn1, 4));
                    q_s1  = squeeze(xn1(end,:,tau,t_idx));
                    q_s1  = q_s1(:) / sum(q_s1(:));
                    o1_t  = L1_obs(min(tau, length(L1_obs)));
                    llik1 = log(A1(o1_t,:)' + 1e-16);
                    acc_tr = acc_tr + q_s1' * llik1;
                end
                acc_L1 = acc_L1 + acc_tr / T1;
            end
            if isfield(mdp1_tr, 'vn') && ~isempty(mdp1_tr.vn)
                vn1   = mdp1_tr.vn{1};
                PE_L1 = PE_L1 + norm(vn1(end,:));
            end
        end
        acc_L1_all(c) = acc_L1 / n_emb;
        PE_L1_all(c)  = PE_L1 / n_emb;
    else
        acc_L1_all(c) = acc_L2_all(c);
        PE_L1_all(c)  = PE_L2_all(c);
    end

end

%% ── NORMALISE I_Sim ──────────────────────────────────────────────────────

acc_combined = 0.5 * acc_L1_all + 0.5 * acc_L2_all;
acc_baseline = acc_combined(1);   % Waking Rest = reference

I_Sim  = abs(acc_baseline) ./ abs(acc_combined);
PE_Sim = 0.5 * PE_L1_all + 0.5 * PE_L2_all;
MC_Sim = max(0, F_L2_all - abs(acc_L2_all));
MMN    = PE_L1_all;
P3b    = PE_L2_all;

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Sim v3: Results — Physiological States\n')
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('%-28s %7s %7s %7s\n', 'Condition', 'I_Sim', 'PE_Sim', 'MC_Sim')
fprintf('%-28s %7s %7s %7s\n', '---------', '-----', '------', '------')
for c = 1:n_cond
    fprintf('%-28s %7.3f %7.5f %7.5f\n', ...
            conditions{c,1}, I_Sim(c), PE_Sim(c), MC_Sim(c))
end
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('\nPhysiological targets:\n')
fprintf('  Waking Rest      : I_Sim = 1.000 (reference)\n')
fprintf('  Physical Exertion: I_Sim > 1.000 (elevated bodily precision)\n')
fprintf('  Psych Stress     : I_Sim ≈ 0.95-1.00 (intact but unstable)\n')
fprintf('  Sleep N3         : I_Sim ≈ 0.10-0.15 (near-absent inference)\n')
fprintf('  Flow / Optimal   : I_Sim > 1.10 (expert prior matches input)\n')
fprintf('  Allostatic Ovld  : I_Sim ≈ 0.60-0.70 (exhaustion impairment)\n\n')

%% ── FIGURE 1: I_Sim comparison ────────────────────────────────────────────

figure('Name', 'HRIT I_Sim v3: Physiological States', ...
       'Position', [50 50 1100 450]);

subplot(1,3,1)
b1 = bar(I_Sim, 'FaceColor', 'flat');
for c = 1:n_cond; b1.CData(c,:) = colors{c}; end
hold on; yline(1.0, 'k--', 'LineWidth', 1.5)
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('I\_Sim'); title('I\_Sim (normalised accuracy)'); grid on

subplot(1,3,2)
b2 = bar(MMN, 'FaceColor', 'flat');
for c = 1:n_cond; b2.CData(c,:) = colors{c}; end
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('MMN (PE_{L1})'); title('MMN: Level-1 PE'); grid on

subplot(1,3,3)
b3 = bar(P3b, 'FaceColor', 'flat');
for c = 1:n_cond; b3.CData(c,:) = colors{c}; end
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('P3b (PE_{L2})'); title('P3b: Level-2 PE'); grid on

sgtitle('HRIT I\_Sim v3: Physiological Hierarchical Perceptual Inference', ...
        'FontSize', 13)
