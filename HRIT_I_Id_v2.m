% HRIT_I_Id_v2.m
% I_Id: Autobiographical Self-Modelling — Physiological States
% HRIT v3 Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Identity
%   Pre-computational: Autobiographical Self-Modelling
%   Computational    : Deep Temporal Model (slow B, concentrated D)
%
% What changed from v1:
%   v1 verified three conditions (Normal, Degraded, Enhanced).
%   v2 models six PHYSIOLOGICAL states matched to I_Intero v4.
%
% Six conditions and their I_Id physiological interpretation:
%   1. Waking Rest      — stable baseline self-model (reference)
%   2. Physical Exertion — intact self-model + embodied presence
%   3. Psych Stress     — self-model under mild threat (B_stab ↓)
%   4. Sleep N3         — autobiographical self suspended (dreamless)
%   5. Flow / Optimal   — peak self-expression and continuity (B very slow)
%   6. Allostatic Ovld  — eroding self-model from chronic load (B_stab ↓↓)
%
% Author: Hamza S. Almahi
% Version: v2 — June 2026

clear; close all; clc;

T_steps  = 8;    % longer for temporal depth
n_trials = 10;

colors = {[0.08 0.50 0.25], [0.88 0.50 0.05], [0.20 0.35 0.78], ...
          [0.40 0.18 0.70], [0.05 0.65 0.70], [0.70 0.12 0.15]};

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% B_stability: diagonal of transition matrix
%   0.99 = very slow (stable identity — waking rest, exertion)
%   0.95 = slow but mildly threatened (psychological stress)
%   0.60 = unstable (sleep N3 — autobiographical processing suspended)
%   0.999= near-frozen (flow — deep self-expression)
%   0.80 = gradually degrading (allostatic overload)
%
% D_prior: initial belief about self-state
%   [0.90;0.10] = strong prior on coherent self (normal)
%   [0.80;0.20] = weakened prior (stress, mild threat to self)
%   [0.50;0.50] = flat (sleep N3 — no stable self-model active)
%   [0.99;0.01] = near-certain (flow — optimal self-model)
%   [0.70;0.30] = partially degraded (overload)
%
% A_precision: likelihood of observation given state (fixed across conditions)

conditions = {
%   label                  B_stab  D_prior
    'Waking Rest',          0.990,  [0.90; 0.10];
    'Physical Exertion',    0.990,  [0.90; 0.10];
    'Psychological Stress', 0.950,  [0.80; 0.20];
    'Sleep Onset (N3)',     0.600,  [0.50; 0.50];
    'Flow / Optimal',       0.999,  [0.99; 0.01];
    'Allostatic Overload',  0.800,  [0.70; 0.30];
};

A_precision = 0.90;   % fixed — interoception intact in all physiological states
n_cond = size(conditions, 1);

%% ── STORAGE ──────────────────────────────────────────────────────────────

I_Id_all  = zeros(1, n_cond);
PE_Id_all = zeros(1, n_cond);
MC_Id_all = zeros(1, n_cond);
acc_all   = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Id v2: Physiological Autobiographical Self-Modelling\n')
fprintf('════════════════════════════════════════════════════════\n\n')

for c = 1:n_cond

    label   = conditions{c,1};
    B_stab  = conditions{c,2};
    D_prior = conditions{c,3};

    fprintf('Running: %s\n', label)

    %% ── BUILD MDP ────────────────────────────────────────────────────
    A = [A_precision,   1-A_precision;
         1-A_precision, A_precision  ];
    A = A ./ sum(A, 1);

    B_eye  = eye(2);
    B_flat = ones(2) / 2;
    B = B_stab * B_eye + (1-B_stab) * B_flat;
    B = B ./ sum(B, 1);

    D = D_prior / sum(D_prior);

    C = zeros(2, T_steps);   % no outcome preference
    V = ones(T_steps-1, 1, 1);

    % Identity-specific observations: self-referential signal
    % Coherent self → observes coherent signal (obs=1)
    obs = ones(1, T_steps);

    mdp.A = {A};
    mdp.B = {B};
    mdp.C = {C};
    mdp.D = {D};
    mdp.V = V;
    mdp.T = T_steps;
    mdp.o = obs;

    %% ── AVERAGE OVER TRIALS ──────────────────────────────────────────
    acc_total = 0;
    PE_total  = 0;
    F_total   = 0;

    for tr = 1:n_trials
        MDP = spm_MDP_VB_X(mdp);

        % Accuracy: normalised log-likelihood of correct state inference
        xn     = MDP.xn{1};
        acc_tr = 0;
        for tau = 1:T_steps
            t_idx = min(tau, size(xn, 4));
            q_s   = squeeze(xn(end, :, tau, t_idx));
            q_s   = q_s(:) / sum(q_s(:));
            llik  = log(A(1,:)' + 1e-16);   % obs=1 → coherent self
            acc_tr = acc_tr + q_s' * llik;
        end
        acc_total = acc_total + acc_tr / T_steps;

        % PE_residual and MC
        vn       = MDP.vn{1};
        PE_total = PE_total + norm(vn(end,:));
        F_total  = F_total + mean(-MDP.F);
    end

    acc_all(c)   = acc_total / n_trials;
    PE_Id_all(c) = PE_total  / n_trials;
    MC_Id_all(c) = max(0, F_total/n_trials - abs(acc_all(c)));

end

%% ── NORMALISE I_Id ───────────────────────────────────────────────────────

acc_baseline = acc_all(1);   % Waking Rest = reference
I_Id_all     = abs(acc_baseline) ./ abs(acc_all);

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Id v2: Results — Physiological States\n')
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('%-28s %7s %7s %7s\n', 'Condition', 'I_Id', 'PE_Id', 'MC_Id')
fprintf('%-28s %7s %7s %7s\n', '---------', '----', '-----', '-----')
for c = 1:n_cond
    fprintf('%-28s %7.3f %7.5f %7.5f\n', ...
            conditions{c,1}, I_Id_all(c), PE_Id_all(c), MC_Id_all(c))
end
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('\nPhysiological targets:\n')
fprintf('  Waking Rest      : I_Id = 1.000 (reference)\n')
fprintf('  Physical Exertion: I_Id ≈ 1.000 (self-model intact)\n')
fprintf('  Psych Stress     : I_Id ≈ 0.85-0.95 (mild threat)\n')
fprintf('  Sleep N3         : I_Id ≈ 0.35-0.45 (self suspended)\n')
fprintf('  Flow / Optimal   : I_Id > 1.000 (peak self-expression)\n')
fprintf('  Allostatic Ovld  : I_Id ≈ 0.65-0.75 (eroding)\n\n')

%% ── FIGURE ───────────────────────────────────────────────────────────────

figure('Name', 'HRIT I_Id v2: Physiological Self-Modelling', ...
       'Position', [50 50 900 400]);

subplot(1,2,1)
b1 = bar(I_Id_all, 'FaceColor', 'flat');
for c = 1:n_cond; b1.CData(c,:) = colors{c}; end
hold on; yline(1.0, 'k--', 'LineWidth', 1.5)
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('I\_Id'); title('I\_Id: Autobiographical Self-Modelling')
grid on

subplot(1,2,2)
b2 = bar(PE_Id_all, 'FaceColor', 'flat');
for c = 1:n_cond; b2.CData(c,:) = colors{c}; end
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('PE_{Id}'); title('PE_{Id}: Self-referential PE residual')
grid on

sgtitle('HRIT I\_Id v2: Physiological Autobiographical Self-Modelling', 'FontSize', 13)
