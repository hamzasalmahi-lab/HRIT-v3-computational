% HRIT_I_Prec_v1.m
% I_Prec: Second-order Precision Control (Metacognitive Monitoring)
% HRIT Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Awareness
%   Pre-computational: Metacognitive Monitoring
%   Computational    : Second-order Precision Control —
%                      A matrix precision at the meta-awareness level
%
% I_Prec captures how accurately the agent models its own inference
% process. Distinct from I_Sim (first-order perceptual accuracy) and
% I_Id (slow self-referential stability).
%
% Behavioural correlate: meta-d' (Fleming & Dolan 2012)
%   High I_Prec: meta-d' = d' (perfect metacognitive efficiency)
%   Low I_Prec : meta-d' < d' (metacognitive impairment — ketamine/RSI)
%
% Key parameter:
%   A2_precision: how accurately agent observes its own belief states
%                 (second-order likelihood mapping)
%   High = agent knows when it is certain vs uncertain
%   Low  = agent cannot distinguish high from low confidence states
%
% Hidden states: confidence state (1=high confidence, 2=low confidence)
% Observations : metacognitive signal (1=accurate, 2=inaccurate)
%
% Extractor (confirmed from I_Sim and I_Id):
%   I_Prec      = abs(acc_baseline) / abs(acc_current)
%   PE_residual = norm(MDP.vn{1}(end,:))
%   MC_Prec     = mean(-MDP.F) - abs(acc)
%
% Three conditions (baseline verification only):
%   Normal  : A2=0.90, B=0.85, concentrated D -> I_Prec = 1.0
%   Degraded: A2=0.55, B=0.60, flat D         -> I_Prec < 1.0
%   Enhanced: A2=0.98, B=0.85, concentrated D -> I_Prec > 1.0
%
% Author: Hamza S. Almahi

clear; close all; clc;

%% ── PARAMETERS ───────────────────────────────────────────────────────────

T_steps  = 6;     % intermediate length — metacognition operates between
                  % fast perception (I_Sim=4) and slow identity (I_Id=8)
n_trials = 10;

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% A2_precision: second-order likelihood precision
%   How accurately the metacognitive system observes its own confidence
%   Normal  : 0.90 (good metacognitive accuracy)
%   Degraded: 0.55 (near-flat — ketamine/RSI impairs metacognition)
%   Enhanced: 0.98 (near-perfect metacognitive monitoring)
%
% B_stability: how stable the confidence state is
%   0.85 = moderately stable (confidence fluctuates but persists)
%   0.60 = unstable (confidence state rapidly switching — degraded)
%
% D_prior: prior over confidence state
%   Normal/Enhanced: [0.85; 0.15] — agent expects to be in high-conf state
%   Degraded: [0.50; 0.50] — no prior (cannot predict own confidence)
%
% Observations: metacognitive accuracy signal
%   Normal/Enhanced: mostly 1s (agent accurately monitors own states)
%   Degraded: alternating (metacognitive signal unreliable)

obs_accurate   = ones(1, T_steps);
obs_unreliable = mod((1:T_steps), 2) + 1;

conditions = {
%   label       A2_prec  B_stab  D_prior          observations
    'Normal',   0.90,    0.85,   [0.85; 0.15],    obs_accurate;
    'Degraded', 0.55,    0.60,   [0.50; 0.50],    obs_unreliable;
    'Enhanced', 0.98,    0.85,   [0.95; 0.05],    obs_accurate;
};

n_cond = size(conditions, 1);
colors = {[0 0.65 0], [0.85 0 0], [0.2 0.4 0.9]};

%% ── STORAGE ──────────────────────────────────────────────────────────────

acc_raw = zeros(1, n_cond);
PE_out  = zeros(1, n_cond);
MC_out  = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════\n')
fprintf('HRIT I_Prec v1: Running conditions\n')
fprintf('═══════════════════════════════════════════════════\n\n')

for c = 1:n_cond

    label   = conditions{c,1};
    A2_prec = conditions{c,2};
    B_stab  = conditions{c,3};
    D_prior = conditions{c,4};
    obs     = conditions{c,5};

    fprintf('Running: %s\n', label)

    acc_trials = zeros(1, n_trials);
    PE_trials  = zeros(1, n_trials);
    F_trials   = zeros(1, n_trials);

    for tr = 1:n_trials

        %% ── A matrix: second-order precision ─────────────────────────
        % Rows = metacognitive observations (1=accurate, 2=inaccurate)
        % Columns = confidence states (1=high-conf, 2=low-conf)
        % A2_prec controls quality of metacognitive self-observation
        A_mat = [A2_prec,   1-A2_prec;
                 1-A2_prec, A2_prec  ];
        A_mat = A_mat ./ sum(A_mat, 1);

        %% ── B matrix: confidence state stability ─────────────────────
        B_mat = [B_stab,   1-B_stab;
                 1-B_stab, B_stab  ];
        B_mat = B_mat ./ sum(B_mat, 1);

        %% ── MDP structure ────────────────────────────────────────────
        mdp.A = {A_mat};
        mdp.B = {B_mat};
        mdp.C = {zeros(2, T_steps)};
        mdp.D = {D_prior};
        mdp.T = T_steps;
        mdp.V = ones(T_steps-1, 1, 1);
        mdp.o = obs;

        %% ── Run ──────────────────────────────────────────────────────
        MDP = spm_MDP_VB_X(mdp);

        %% ── Extract accuracy ─────────────────────────────────────────
        xn  = MDP.xn{1};
        acc = 0;
        for tau = 1:T_steps
            t_idx = min(tau, size(xn, 4));
            q_s   = squeeze(xn(end, :, tau, t_idx));
            q_s   = q_s(:) / sum(q_s(:));
            o_t   = obs(tau);
            llik  = log(A_mat(o_t, :)' + 1e-16);
            acc   = acc + q_s' * llik;
        end
        acc_trials(tr) = acc / T_steps;

        %% ── Extract F ────────────────────────────────────────────────
        F_trials(tr) = mean(-MDP.F);

        %% ── Extract PE_residual ──────────────────────────────────────
        vn = MDP.vn{1};
        PE_trials(tr) = norm(vn(end, :));

    end

    acc_raw(c) = mean(acc_trials);
    PE_out(c)  = mean(PE_trials);
    F_mean     = mean(F_trials);
    MC_out(c)  = max(0, F_mean - abs(acc_raw(c)));

end

%% ── NORMALISE I_Prec ─────────────────────────────────────────────────────

acc_baseline = acc_raw(1);
I_Prec = abs(acc_baseline) ./ abs(acc_raw);

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Prec v1: Results\n')
fprintf('═══════════════════════════════════════════════════════════════\n')
fprintf('%-15s %8s %10s %10s %10s\n', ...
        'Condition','I_Prec','PE_resid','MC_Prec','acc_raw')
fprintf('%-15s %8s %10s %10s %10s\n', ...
        '---------','------','--------','-------','-------')
for c = 1:n_cond
    fprintf('%-15s %8.3f %10.5f %10.5f %10.5f\n', ...
            conditions{c,1}, I_Prec(c), ...
            PE_out(c), MC_out(c), acc_raw(c))
end
fprintf('═══════════════════════════════════════════════════════════════\n')
fprintf('\nTargets:\n')
fprintf('  Normal  : I_Prec = 1.000  (reference)\n')
fprintf('  Degraded: I_Prec < 1.000  (metacognitive impairment)\n')
fprintf('  Enhanced: I_Prec > 1.000  (enhanced metacognitive monitoring)\n\n')

%% ── FIGURE 1: I_Prec bar chart ───────────────────────────────────────────

figure('Name','HRIT I_Prec: Normalised accuracy',...
       'Position',[50 50 700 500]);

b = bar(I_Prec, 'FaceColor', 'flat');
for c = 1:n_cond; b.CData(c,:) = colors{c}; end
hold on
yline(1.0, 'k--', 'LineWidth', 1.5)
set(gca,'XTickLabel', conditions(:,1), 'XTickLabelRotation', 12)
ylabel('I\_Prec (normalised accuracy)')
title('HRIT I\_Prec: Second-order Precision Control', 'FontSize', 12)
ylim([0 max(I_Prec)*1.3 + 0.1])
grid on

%% ── FIGURE 2: PE_residual and MC_Prec ───────────────────────────────────

figure('Name','HRIT I_Prec: PE_residual and MC_Prec',...
       'Position',[50 50 900 400]);

subplot(1,2,1)
b2 = bar(PE_out, 'FaceColor', 'flat');
for c = 1:n_cond; b2.CData(c,:) = colors{c}; end
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',12)
ylabel('PE_{residual}')
title('PE_{residual} (input to p(\Psi))')
grid on

subplot(1,2,2)
b3 = bar(MC_out, 'FaceColor', 'flat');
for c = 1:n_cond; b3.CData(c,:) = colors{c}; end
set(gca,'XTickLabel',conditions(:,1),'XTickLabelRotation',12)
ylabel('MC_{Prec}')
title('MC_{Prec} (Metabolic Integration Cost)')
grid on

sgtitle('HRIT I\_Prec: PE_{residual} and MC_{Prec}', 'FontSize', 12)