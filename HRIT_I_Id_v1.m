% HRIT_I_Id_v1.m
% I_Id: Autobiographical Self-Modelling (Deep Temporal Model)
% HRIT Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Identity
%   Pre-computational: Autobiographical Self-Modelling
%   Computational    : Deep Temporal Model — slow B matrix transitions
%                      with high-concentration D vector prior
%
% I_Id captures the stability of the agent's belief about its own
% persistent identity state across time. Distinct from I_Sim which
% tracks moment-to-moment perceptual inference.
%
% Key parameters:
%   B_stability: how slowly identity state changes (close to 1 = stable)
%   D_prior    : strength of prior over self-state
%   T_steps    : longer trial = more temporal depth
%
% Extractor (confirmed from I_Sim):
%   I_Id        = abs(acc_baseline) / abs(acc_current)
%   PE_residual = norm(MDP.vn{1}(end,:))
%   MC_Id       = mean(-MDP.F) - abs(acc)
%
% Three conditions (baseline verification only):
%   Normal    : B=0.99, concentrated D -> I_Id = 1.0 (reference)
%   Degraded  : B=0.50, flat D         -> I_Id < 1.0 (fragmented self-model)
%   Enhanced  : B=0.999, very concentrated D -> I_Id > 1.0
%
% Clinical state simulation deferred until all components integrated.
%
% Author: Hamza S. Almahi

clear; close all; clc;

%% ── PARAMETERS ───────────────────────────────────────────────────────────

% Longer trial for I_Id: temporal depth requires more timesteps
% Identity inference operates on a slower timescale than Simulation
T_steps  = 8;     % longer sequence captures slow temporal dynamics
n_trials = 10;    % trials to average

% Hidden states: self-state (1=coherent self, 2=fragmented self)
% Observations: self-referential signal (1=coherent, 2=fragmented)

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% B_stability: diagonal of transition matrix
%   I_Sim used 0.90 (moderate)
%   I_Id uses 0.99 (very slow — identity state barely changes)
%   Degraded: 0.50 (random — identity state unstable, dissociation/dementia)
%   Enhanced: 0.999 (extremely slow — rock-solid autobiographical continuity)
%
% D_prior: initial belief about self-state
%   Normal: [0.90; 0.10] — strong prior on coherent self
%   Degraded: [0.50; 0.50] — no prior (no stable self-model)
%   Enhanced: [0.99; 0.01] — near-certain prior on coherent self
%
% A_precision: likelihood of observation given state
%   Fixed at 0.90 across all conditions (interoception intact)
%   Only B and D vary — these are the identity-specific parameters
%
% Observations: coherent self-referential signal (1 repeated)
%   Normal/Enhanced: mostly 1s (coherent self-signal)
%   Degraded: alternating (no stable self-signal)

obs_coherent  = ones(1, T_steps);           % stable coherent self-signal
obs_fragmented = mod((1:T_steps), 2) + 1;  % alternating 1,2,1,2...

conditions = {
%   label          B_stab   D_prior          A_prec  observations
    'Normal',      0.990,   [0.90; 0.10],    0.90,   obs_coherent;
    'Degraded',    0.500,   [0.50; 0.50],    0.90,   obs_fragmented;
    'Enhanced',    0.999,   [0.99; 0.01],    0.90,   obs_coherent;
};

n_cond = size(conditions, 1);
colors = {[0 0.65 0], [0.85 0 0], [0.2 0.4 0.9]};

%% ── STORAGE ──────────────────────────────────────────────────────────────

acc_raw = zeros(1, n_cond);
PE_out  = zeros(1, n_cond);
MC_out  = zeros(1, n_cond);

%% ── RUN CONDITIONS ───────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════\n')
fprintf('HRIT I_Id v1: Running conditions\n')
fprintf('═══════════════════════════════════════════════════\n\n')

for c = 1:n_cond

    label   = conditions{c,1};
    B_stab  = conditions{c,2};
    D_prior = conditions{c,3};
    A_prec  = conditions{c,4};
    obs     = conditions{c,5};

    fprintf('Running: %s\n', label)

    acc_trials = zeros(1, n_trials);
    PE_trials  = zeros(1, n_trials);
    F_trials   = zeros(1, n_trials);

    for tr = 1:n_trials

        %% ── A matrix ─────────────────────────────────────────────────
        % Fixed precision across conditions
        % Only B and D distinguish I_Id conditions
        A_mat = [A_prec,   1-A_prec;
                 1-A_prec, A_prec  ];
        A_mat = A_mat ./ sum(A_mat, 1);

        %% ── B matrix: slow transitions (identity-specific) ───────────
        % Very high stability = identity state persists across time
        % This is the key distinction from I_Sim
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
            % Clamp tau index to available dimensions
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

%% ── NORMALISE I_Id ───────────────────────────────────────────────────────

acc_baseline = acc_raw(1);   % Normal = reference
I_Id = abs(acc_baseline) ./ abs(acc_raw);

%% ── PRINT RESULTS ────────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════════════════\n')
fprintf('HRIT I_Id v1: Results\n')
fprintf('═══════════════════════════════════════════════════════════════\n')
fprintf('%-15s %8s %10s %10s %10s\n', ...
        'Condition','I_Id','PE_resid','MC_Id','acc_raw')
fprintf('%-15s %8s %10s %10s %10s\n', ...
        '---------','----','--------','-----','-------')
for c = 1:n_cond
    fprintf('%-15s %8.3f %10.5f %10.5f %10.5f\n', ...
            conditions{c,1}, I_Id(c), ...
            PE_out(c), MC_out(c), acc_raw(c))
end
fprintf('═══════════════════════════════════════════════════════════════\n')
fprintf('\nTargets:\n')
fprintf('  Normal  : I_Id = 1.000  (reference)\n')
fprintf('  Degraded: I_Id < 1.000  (fragmented self-model)\n')
fprintf('  Enhanced: I_Id > 1.000  (strong autobiographical continuity)\n\n')

%% ── FIGURE 1: I_Id bar chart ─────────────────────────────────────────────

figure('Name','HRIT I_Id: Normalised accuracy',...
       'Position',[50 50 700 500]);

b = bar(I_Id, 'FaceColor', 'flat');
for c = 1:n_cond; b.CData(c,:) = colors{c}; end
hold on
yline(1.0, 'k--', 'LineWidth', 1.5)
set(gca,'XTickLabel', conditions(:,1), 'XTickLabelRotation', 12)
ylabel('I\_Id (normalised accuracy)')
title('HRIT I\_Id: Autobiographical Self-Modelling', 'FontSize', 12)
ylim([0 max(I_Id)*1.3 + 0.1])
grid on

%% ── FIGURE 2: PE_residual and MC_Id ──────────────────────────────────────

figure('Name','HRIT I_Id: PE_residual and MC_Id',...
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
ylabel('MC_{Id}')
title('MC_{Id} (Metabolic Integration Cost)')
grid on

sgtitle('HRIT I\_Id: PE_{residual} and MC_{Id}', 'FontSize', 12)