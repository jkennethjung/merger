clc;
clear;
close all;
echo off;

diary ../output/gen_data.log
rng(1);

% 0. Globals
global n_draw J T JT beta1_mean beta_mean beta_var alpha gamma0 gamma1 ...
  unobs_mean unobs_var;

n_draw = 1e2;
J = 4;
T = 600;
JT = J*T;

beta1_mean  = 1;
beta_mean = 4;
beta_var = 1;
unobs_mean = [0; 0];
unobs_var = [1, 0.25; 0.25, 1];
alpha = -2;

% 1. Start generating data
[j, t, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, unobs_mean, ...
  unobs_var);
p = ones(JT, 1); % arbitrary prices for now--fix later
data_mat = [j, t, x, sat, wire, p, w, xi, omega];
beta1 = repmat(beta1_mean, 1, n_draw);
beta2 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
beta3 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
alpha_vec = repmat(alpha, 1, n_draw);
const = ones(1, n_draw);
theta = [beta1; beta2; beta3; alpha_vec; const];
[s, ds_dp] = gen_shares(data_mat, theta, T, JT, n_draw);
full_data_mat = [data_mat, s];
disp(ds_dp(1:8,1:8))
writematrix(full_data_mat, "../output/data.csv");

diary off;

function [j, t, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, ...
  unobs_mean, unobs_var)
    j = repmat((1:J)', T, 1);
    t = repelem((1:T)', J);
    x = abs(normrnd(0, 1, JT, 1));
    sat = repmat([1; 1; 0; 0], T, 1);
    wire = ones(JT, 1) - sat;
    w = abs(normrnd(0, 1, JT, 1));
    unobs = mvnrnd(unobs_mean, unobs_var, JT);
    xi = unobs(:,1);
    omega = unobs(:,2);
end

function [s, ds_dp] = gen_shares(data_mat, theta, T, JT, n_draw)
    s = zeros(JT, 1);
    ds_dp = [0];
    for t = 1:T
        t_rows = (data_mat(:, 2) == t);
        t_mat = data_mat(t_rows, [3, 4, 5, 6, 8]);
        s_t = mkt_shares(t_mat, theta, n_draw);
        s(t_rows, 1) = s_t;
        dst_dpt = share_deriv_market(t_mat, theta, n_draw);
        ds_dp = blkdiag(ds_dp, dst_dpt);
    end
    ds_dp = ds_dp(2:(JT + 1), 2:(JT + 1));
end

function dst_dpt = share_deriv_market(t_mat, theta, n_draw)
    [s_t, sigma_t] = mkt_shares(t_mat, theta, n_draw);
    t_size = size(t_mat);
    J_t = t_size(1);
    dst_dpt = zeros(J_t, J_t);
    alpha = theta(4,1);
    for i = 1:n_draw
        sigma_it = sigma_t(:,i);
        dsigmait_dpt = -alpha*sigma_it*sigma_it';
        dsigma_own = alpha*sigma_it .* (1 - sigma_it);
        dsigma_own = diag(dsigma_own);
        dsigmait_dpt = dsigmait_dpt .* (ones(J_t) - eye(J_t)) + dsigma_own;
        dst_dpt = dst_dpt + dsigmait_dpt;
    end
    dst_dpt = dst_dpt / n_draw;
end

function [s_t, sigma_t] = mkt_shares(t_mat, theta, n_draw)
    size_t = size(t_mat);
    J_t = size_t(1);
    exp_delta = exp(t_mat * theta);
    sum_exp_delta = sum(exp_delta, 1) + ones(1, n_draw);
    sigma_t = zeros(J_t, n_draw);
    for j = 1:J_t
        for i = 1:n_draw
            sigma_t(j, i) = exp_delta(j,i)/sum_exp_delta(i);
        end
    end
    s_t = mean(sigma_t, 2);
end

