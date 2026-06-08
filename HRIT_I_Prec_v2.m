% HRIT_I_Prec_v2.m
% I_Prec: Metacognitive Monitoring — Physiological States
% HRIT v3 Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Awareness
%   Pre-computational: Metacognitive Monitoring
%   Computational    : Second-order Precision Control POMDP
%
% What changed from v1:
%   v1 verified three conditions (Normal, Degraded, Enhanced).
%   v2 models six PHYSIOLOGICAL states matched to I_Intero v4.
%
% Six conditions and their I_Prec physiological interpretation:
%   1. Waking Rest      — baseline metacognitive monitoring (reference)
%   2. Physical Exertion — intact metacognition during exercise
%   3. Psych Stress     — slightly impaired (Arnsten 2015: stress reduces PFC)
%   4. Sleep N3         — metacognition suspended (dreamless; cf. Study 1c)
%   5. Flow / Optimal   — peak metacognitive efficiency (near meta-d'/d'=1)
%   6. Allostatic Ovld  — impaired from exhaustion (reduced second-order precision)
%
% Behavioural correlate: meta-d'/d' (Fleming & Dolan 2012)
%   High I_Prec: meta-d' ≈ d' (efficient)
%   Low I_Prec:  meta-d' < d' (impaired, as in anaesthesia/RSI)
%
% Author: Hamza S. Almahi
% Version: v2 — June 2026

clear; close all; clc;

T_steps  = 6;
n_trials = 10;

colors = {[0.08 0.50 0.25], [0.88 0.50 0.05], [0.20 0.35 0.78], ...
          [0.40 0.18 0.70], [0.05 0.65 0.70], [0.70 0.12 0.15]};

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% A2_precision: second-order likelihood precision
%   How accurately the metacognitive system observes its own confidence
%   0.90 = good metacognitive accuracy (waking rest, exertion)
%   0.80 = mildly impaired (psychological stress — PFC load)
%   0.55 = severely impaired (sleep N3 — metacognition suspended)
%   0.98 = near-perfect (flow — optimal metacognitive monitoring)
%   0.65 = moderately impaired (allostatic overload — exhaustion)
%
% B_stability: confidence state stability
%   Higher = confidence state more persistent; lower = rapidly switching
%
% D_prior: prior over confidence state
%   [0.90;0.10] = high confidence typical (waking, exertion, flow)
%   [0.70;0.30] = moderately degraded prior (stress)
%   [0.50;0.50] = flat (sleep N3 — no metacognitive baseline)
%   [0.65;0.35] = partially depleted (overload)

conditions = {
%   label                  A2_prec B_stab D_prior
    'Waking Rest',          0.90,   0.85,  [0.90; 0.10];
    'Physical Exertion',    0.90,   0.85,  [0.90; 0.10];
    'Psychological Stress', 0.80,   0.75,  [0.70; 0.30];
    'Sleep Onset (N3)',     0.55,   0.60,  [0.50; 0.50];
    'Flow / Optimal',       0.98,   0.85,  [0.90; 0.10];
    'Allostatic Overload',  0.65,   0.65,  [0.65; 0.35];
};

n_cond = size(conditions, 1);

%% ── STORAGE ──────────────────────────────────────────────────────────────

I_Prec_all  = zeros(1, n_cond);
PE_Prec_all = zeros(1, n_cond);
MC_Prec_all = zeros(1, n_cond);
acc_all     = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Prec v2: Physiological Metacognitive Monitoring\n')
fprintf('════════════════════════════════════════════════════════\n\n')

for c = 1:n_cond

    label      = conditions{c,1};
    A2_prec    = conditions{c,2};
    B_stab     = conditions{c,3};
    D_prior    = conditions{c,4};

    fprintf('Running: %s\n', label)

    %% ── BUILD MDP ────────────────────────────────────────────────────
    A = [A2_prec,   1-A2_prec;
         1-A2_prec, A2_prec  ];
    A = A ./ sum(A, 1);

    B_eye  = eye(2);
    B_flat = ones(2) / 2;
    B = B_stab * B_eye + (1-B_stab) * B_flat;
    B = B ./ sum(B, 1);

    D = D_prior / sum(D_prior);

    C = zeros(2, T_steps);
    V = ones(T_steps-1, 1, 1);

    obs = ones(1, T_steps);   % accurate metacognitive signal expected

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

        xn     = MDP.xn{1};
        acc_tr = 0;
        for tau = 1:T_steps
            t_idx = min(tau, size(xn, 4));
            q_s   = squeeze(xn(end, :, tau, t_idx));
            q_s   = q_s(:) / sum(q_s(:));
            llik  = log(A(1,:)' + 1e-16);
            acc_tr = acc_tr + q_s' * llik;
        end
        acc_total = acc_total + acc_tr / T_steps;

        vn       = MDP.vn{1};
        PE_total = PE_total + norm(vn(end,:));
        F_total  = F_total + mean(-MDP.F);
    end

    acc_all(c)     = acc_total / n_trials;
    PE_Prec_all(c) = PE_total  / n_trials;
    MC_Prec_all(c) = max(0, F_total/n_trials - abs(acc_all(c)));

end

%% ── NORMALISE I_Prec ─────────────────────────────────────────────────────

acc_baseline = acc_all(1);   % Waking Rest = reference
I_Prec_all   = abs(acc_baseline) ./ abs(acc_all);

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n════════════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Prec v2: Results — Physiological States\n')
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('%-28s %8s %8s %8s\n', 'Condition', 'I_Prec', 'PE_Prec', 'MC_Prec')
fprintf('%-28s %8s %8s %8s\n', '---------', '------', '-------', '-------')
for c = 1:n_cond
    fprintf('%-28s %8.3f %8.5f %8.5f\n', ...
            conditions{c,1}, I_Prec_all(c), PE_Prec_all(c), MC_Prec_all(c))
end
fprintf('════════════════════════════════════════════════════════════════════\n')
fprintf('\nPhysiological targets:\n')
fprintf('  Waking Rest      : I_Prec = 1.000 (reference)\n')
fprintf('  Physical Exertion: I_Prec ≈ 1.000 (metacognition intact)\n')
fprintf('  Psych Stress     : I_Prec ≈ 0.80-0.90 (mild PFC load)\n')
fprintf('  Sleep N3         : I_Prec ≈ 0.25 (suspended, as in anaesthesia)\n')
fprintf('  Flow / Optimal   : I_Prec >> 1.000 (peak metacognitive efficiency)\n')
fprintf('  Allostatic Ovld  : I_Prec ≈ 0.50-0.65 (exhaustion impairment)\n\n')

%% ── FIGURE ───────────────────────────────────────────────────────────────

figure('Name', 'HRIT I_Prec v2: Physiological Metacognitive Monitoring', ...
       'Position', [50 50 900 400]);

subplot(1,2,1)
b1 = bar(I_Prec_all, 'FaceColor', 'flat');
for c = 1:n_cond; b1.CData(c,:) = colors{c}; end
hold on; yline(1.0, 'k--', 'LineWidth', 1.5)
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('I\_Prec'); title('I\_Prec: Metacognitive Monitoring'); grid on

subplot(1,2,2)
b2 = bar(PE_Prec_all, 'FaceColor', 'flat');
for c = 1:n_cond; b2.CData(c,:) = colors{c}; end
set(gca, 'XTickLabel', conditions(:,1), 'XTickLabelRotation', 20)
ylabel('PE_{Prec}'); title('PE_{Prec}: Metacognitive PE residual'); grid on

sgtitle('HRIT I\_Prec v2: Physiological Metacognitive Monitoring', 'FontSize', 13)
