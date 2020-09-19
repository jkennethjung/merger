clc;
clear;
close all;
echo off;

diary ./gen_data.log

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

[j, t, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, unobs_mean, ...
  unobs_var);
p = ones(JT, 1); % arbitrary prices for now--fix later
data_mat = [j, t, x, sat, wire, p, w, xi, omega];
beta1 = repmat(beta1_mean, 1, n_draw);
beta2 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
beta3 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
alpha = repmat(-2, 1, n_draw);
const = ones(1, n_draw);
theta = [beta1; beta2; beta3; alpha; const];
sh = gen_shares(data_mat, theta, T, JT, n_draw);
full_data_mat = [data_mat, sh];
writematrix(full_data_mat, "../output/data.csv");

dsigmat_dpt = prob_deriv_matrix(-1, [0.2;0.2;0.2;0.2]);
disp(dsigmat_dpt);

diary off;

function [j, t, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, ...
  unobs_mean, unobs_var)
    j = repmat((1:J)', T, 1);
    t = repelem((1:T)', J);
    x = abs(normrnd(0, 1, JT, 1));
    sat = repmat([1; 1; 0; 0], T, 1);
    wire = ones(JT, 1) - sat;
    w = abs(normrnd(0, 1, JT, 1));
    unobs = mvnrnd(unobs_mean, sqrt(unobs_var), JT);
    xi = unobs(:,1);
    omega = unobs(:,2);
end

function sh = gen_shares(data_mat, theta, T, JT, n_draw)
    sh = zeros(JT, 1);
    for t = 1:T
        t_rows = (data_mat(:, 2) == t);
        t_mat = data_mat(t_rows, [3, 4, 5, 6, 8]);
        sh_t = mkt_shares(t_mat, theta, n_draw);
        sh(t_rows, 1) = sh_t;
    end
end

function sh_t = mkt_shares(t_mat, theta, n_draw)
    size_t = size(t_mat);
    J_t = size_t(1);
    exp_delta = exp(t_mat * theta);
    sum_exp_delta = sum(exp_delta, 1) + ones(1, n_draw);
    prob = zeros(J_t, n_draw);
    for j = 1:J_t
        for i = 1:n_draw
            prob(j, i) = exp_delta(j,i)/sum_exp_delta(i);
        end
    end
    sh_t = mean(prob, 2);
end

function dsigmat_dpt = prob_deriv_matrix(alpha, s_t)
    J_t_size= size(s_t);
    J_t = J_t_size(1);
    dsigmat_dpt = zeros(J_t, J_t);
    for j = 1:J_t
        for k = 1:J_t
            dsigmajt_dpkt = prob_deriv_scalar(alpha, s_t, j, k);
            dsigmat_dpt(j, k) = dsigmajt_dpkt;
        end
    end
end

function dsigmajt_dpkt = prob_deriv_scalar(alpha, s_t, j, k)
    if j == k
        dsigmajt_dpkt = alpha*(s_t(j) - s_t(j)^2); 
    else
        dsigmajt_dpkt = -alpha*(s_t(j)*s_t(k));
    end
end

