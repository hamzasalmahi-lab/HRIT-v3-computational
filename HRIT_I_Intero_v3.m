% HRIT_I_Intero_v3.m
% I_Intero: Allostatic Interoceptive Inference with Signed PDS
% HRIT Computational Framework
%
% Three-layer terminology:
%   Phenomenological : Vitality
%   Pre-computational: Allostatic Interoceptive Regulation
%   Computational    : Signed Predictive Coding ODE
%
% Clinical states:
%   Normal  : sigma_o small, transient perturbation, full recovery
%   DPDR    : sigma_o large (precision collapse), ICP near zero, PDS ~ 0
%   Cotard  : sigma_o moderate + sustained negative push,
%             ICP chronically below IBT, CAL accumulates, PDS -> -1
%
% Based on Tschantz et al. (2022) extended with signed threshold crossing
% PE_residual = |dF/dt| feeds into p(Psi) formula
%
% Author: Hamza S. Almahi

clear; close all; clc;

%% ── PARAMETERS ───────────────────────────────────────────────────────────

dt = 0.01;
T  = 12000;
t  = (1:T) * dt;

% Shared across all conditions
sigma_p   = 1.0;    % prior variance (fixed)
kappa     = 0.5;    % autonomic action learning rate
IBT       = 0.0;    % Interoceptive Baseline Threshold (setpoint)

% PDS equation parameters
w         = 2.0;    % ICP coupling weight
PSC       = 0.5;    % Presence Sensitivity Constant
beta_CAL  = 0.15;   % allostatic load sensitivity
ICP_ref   = 0.25;   % ICP reference: resting ICP >> this -> PDS positive
                    % Cotard ICP < this -> PDS starts negative immediately
CAL_decay = 0.003;  % rate of CAL recovery per timestep (Normal only)

% Perturbation: transient homeostatic challenge (all conditions)
t_perturb = 1000;   % timestep of transient perturbation

%% ── CONDITION DEFINITIONS ────────────────────────────────────────────────
%
% Normal:
%   sigma_o = 1.0  -> high precision -> autonomic action strong
%   push = 0       -> transient perturbation only
%   CAL recovers   -> full homeostatic recovery
%   Expected: PDS > 0.8 at steady state
%
% DPDR (Depth Collapse):
%   sigma_o = 80   -> low precision -> autonomic action weak
%   push = 0       -> same transient perturbation
%   CAL recovers   -> but ICP stays low because autonomic is weak
%   Expected: PDS near 0, does not recover to positive
%
% Cotard (Depth Inversion):
%   sigma_o = 5.0  -> moderately reduced precision
%   push = -0.5    -> sustained negative drive (chronic below-IBT state)
%   Combined: autonomic cannot fully compensate -> eps_o stays large
%             ICP chronically below ICP_ref -> PDS starts negative
%             CAL does not recover -> accumulates -> PDS -> -1 (attractor)
%   Expected: PDS < -0.85 at steady state

conditions = {
%   label                    sigma_o  push   CAL_recovers  F_CAL_thresh
    'Normal (Homeostasis)',   1.0,     0.0,   true,         2.0;
    'DPDR (Depth Collapse)',  20.0,  -0.075,   true,         0.5;
    'Cotard (Depth Inversion)',5.0,   -0.5,   false,        0.0;
};

n_cond  = size(conditions, 1);
colors  = {[0 0.65 0], [0 0.4 0.8], [0.85 0 0]};

% Pre-allocate outputs
PDS_all = zeros(n_cond, T);
ICP_all = zeros(n_cond, T);
F_all   = zeros(n_cond, T);
PE_all  = zeros(n_cond, T);  % PE_residual = |dF/dt|

%% ── SIMULATION LOOP ──────────────────────────────────────────────────────

for c = 1:n_cond

    label        = conditions{c,1};
    sigma_o      = conditions{c,2};
    push         = conditions{c,3};
    CAL_recovers = conditions{c,4};
    F_thresh     = conditions{c,5};   % CAL accumulates when F > this

    fprintf('Running: %s\n', label)

    % State variables
    mu_i  = 0.0;   % posterior interoceptive belief
    y_i   = 0.0;   % interoceptive sensory data
    a_i   = 0.0;   % autonomic action (previous step)
    CAL   = 0.0;   % cumulative allostatic load
    F_prev = 0.0;  % previous F for dF/dt

    for i = 2:T

        % ── Sensory data update ──────────────────────────────────────────

        % Sustained drive (Cotard: chronically push ICP below IBT)
        y_i = y_i + push * dt;

        % Transient homeostatic challenge (all conditions)
        if i == t_perturb
            y_i = y_i + 4.0;
        end

        % Autonomic action from previous timestep
        y_i = y_i + a_i * dt;

        % ── Prediction errors ────────────────────────────────────────────

        eps_o = y_i - mu_i;       % observation PE: data vs posterior
        eps_p = IBT - mu_i;       % prior PE: setpoint vs posterior

        % ── Posterior update (predictive coding gradient descent on F) ───

        mu_i = mu_i + dt * (eps_o / sigma_o + eps_p / sigma_p);

        % ── Autonomic action ─────────────────────────────────────────────
        % Minimises observation PE
        % Critical: divided by sigma_o
        %   Normal (sigma_o=1):  action strong  -> restores homeostasis
        %   DPDR   (sigma_o=80): action weak    -> homeostasis fails
        %   Cotard (sigma_o=5):  action moderate-> cannot compensate push

        a_i = -kappa * eps_o / sigma_o;

        % ── Free energy ──────────────────────────────────────────────────

        F_i = 0.5 * (eps_o^2 / sigma_o + eps_p^2 / sigma_p);
        F_all(c,i) = F_i;

        % ── PE_residual = |dF/dt| ────────────────────────────────────────
        % Feeds into p(Psi) = PE_residual_dominant / (MC_dominant + eps)
        % At homeostasis: |dF/dt| -> 0 (errors resolved)
        % During episode onset: |dF/dt| large (errors unresolved)
        % This is the DEWS early warning signal

        PE_all(c,i) = abs(F_i - F_prev) / dt;
        F_prev = F_i;

        % ── ICP proxy ────────────────────────────────────────────────────
        % ICP: quality of interoceptive coupling
        % ICP -> 1 when eps_o -> 0 (perfect coupling, high precision)
        % ICP -> 0 when eps_o large (coupling broken)

        ICP_i = 1.0 / (1.0 + abs(eps_o));
        ICP_all(c,i) = ICP_i;

        % ── Cumulative Allostatic Load ────────────────────────────────────
        % Accumulates when F exceeds condition-specific threshold
        % Decays only in Normal condition when near homeostasis

        F_excess = max(0, F_i - F_thresh);
        CAL = CAL + dt * F_excess;

        if CAL_recovers && F_i < 0.3
            CAL = max(0, CAL - CAL_decay * dt);
        end

        % ── PDS: Phenomenal Depth Signal ─────────────────────────────────
        % Signed allostatic depth
        % PDS > 0 : ICP above IBT-equivalent -> depth established
        % PDS ~ 0 : ICP at threshold -> depth collapsed (DPDR)
        % PDS < 0 : ICP below threshold + CAL elevated -> depth inverted
        % PDS ~ -1: stable Cotard attractor (tanh saturation)

        PDS_all(c,i) = tanh((w * (ICP_i - ICP_ref) - beta_CAL * CAL) / PSC);

    end

end

%% ── FIGURE 1: PDS comparison all three conditions ────────────────────────

figure('Name','HRIT I_Intero: PDS across Clinical States',...
       'Position',[50 50 1000 500]);

hold on
for c = 1:n_cond
    plot(t, PDS_all(c,:), 'Color', colors{c}, 'LineWidth', 2.5, ...
         'DisplayName', conditions{c,1})
end
yline( 0,  'k--', 'LineWidth', 1.5, 'DisplayName', 'Depth boundary (0)')
yline(-1,  'r:',  'LineWidth', 1.0, 'DisplayName', 'Cotard floor (-1)')
yline( 1,  'g:',  'LineWidth', 1.0, 'DisplayName', 'Max depth (+1)')
xline(t(t_perturb), 'k:', 'LineWidth', 1.5, 'DisplayName', 'Perturbation')
ylim([-1.15 1.15])
legend('Location', 'southwest', 'FontSize', 10)
title('HRIT I\_Intero: PDS (Phenomenal Depth Signal)', 'FontSize', 13)
xlabel('Time (a.u.)'); ylabel('PDS')
grid on

%% ── FIGURE 2: ICP, F, PDS, PE_residual per condition ─────────────────────

figure('Name','HRIT I_Intero: Detail per condition',...
       'Position',[50 50 1400 900]);

row_labels = {'ICP', 'Free Energy F', 'PDS', 'PE_{residual} = |dF/dt|'};
data_sets  = {ICP_all, F_all, PDS_all, PE_all};

for c = 1:n_cond
    for row = 1:4
        subplot(4, n_cond, (row-1)*n_cond + c)
        dat = data_sets{row};
        plot(t, dat(c,:), 'Color', colors{c}, 'LineWidth', 1.8)
        hold on
        xline(t(t_perturb), 'k:', 'LineWidth', 1)
        if row == 1
            yline(ICP_ref, 'r--', 'LineWidth', 1)
            ylim([0 1.05])
            title(conditions{c,1}, 'FontSize', 10)
        end
        if row == 3
            yline(0,  'k--', 'LineWidth', 1)
            yline(-1, 'r:',  'LineWidth', 1)
            ylim([-1.15 1.15])
        end
        if c == 1
            ylabel(row_labels{row}, 'FontSize', 9)
        end
        xlabel('Time')
        grid on
    end
end

sgtitle('HRIT I\_Intero: Detailed output per clinical state', 'FontSize', 13)

%% ── FIGURE 3: PE_residual detail (for p(Psi) formula) ───────────────────

figure('Name','HRIT I_Intero: PE_residual',...
       'Position',[50 50 1000 400]);

hold on
for c = 1:n_cond
    % Smooth PE_residual for visualisation (raw is very spiky)
    PE_smooth = movmean(PE_all(c,:), 50);
    plot(t, PE_smooth, 'Color', colors{c}, 'LineWidth', 2, ...
         'DisplayName', conditions{c,1})
end
xline(t(t_perturb), 'k:', 'LineWidth', 1.5)
legend('Location', 'northeast', 'FontSize', 10)
title('PE_{residual} = |dF/dt| (smoothed) — input to p(\Psi)', 'FontSize', 12)
xlabel('Time (a.u.)'); ylabel('PE_{residual}')
grid on

%% ── SUMMARY ──────────────────────────────────────────────────────────────

fprintf('\n═══════════════════════════════════════════════════\n')
fprintf('HRIT I_Intero v3: Steady-state results (last 2000 steps)\n')
fprintf('═══════════════════════════════════════════════════\n')
fprintf('%-30s %8s %8s %12s\n', 'Condition', 'PDS', 'ICP', 'PE_residual')
fprintf('%-30s %8s %8s %12s\n', '---------', '---', '---', '-----------')
for c = 1:n_cond
    pds_ss = mean(PDS_all(c, end-2000:end));
    icp_ss = mean(ICP_all(c, end-2000:end));
    pe_ss  = mean(PE_all(c,  end-2000:end));
    fprintf('%-30s %8.3f %8.3f %12.5f\n', ...
            conditions{c,1}, pds_ss, icp_ss, pe_ss)
end
fprintf('═══════════════════════════════════════════════════\n')
fprintf('\nTargets:\n')
fprintf('  Normal  : PDS > 0.80\n')
fprintf('  DPDR    : -0.10 < PDS < 0.25  (near zero)\n')
fprintf('  Cotard  : PDS < -0.85          (negative attractor)\n\n')