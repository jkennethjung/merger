clc;
clear;
close all;
echo off;

diary ../output/gen_data.log
diary on;
rng(1);

% 0. Globals
global n_draw ZETA_TOL J T JT H_t beta1_mean beta_mean beta_var alpha ...
  gamma0 gamma1 unobs_mean unobs_var theta data_mat mc mkt_rows opts ...
  j_vec t_vec x sat wire w xi omega;

J = 4;
T = 600;
JT = J*T;
H_t = eye(J); % ownership matrix 
% fsolve algorithm: choose from 'levenberg-marquardt' or 'trust-region-dogleg'
opts = optimoptions('fsolve', 'Algorithm', 'trust-region-dogleg');

beta1_mean  = 1;
beta_mean = 4;
beta_var = 1;
unobs_mean = [0; 0];
unobs_var = [1, 0.25; 0.25, 1];
alpha = -2;
gamma0 = 0.5;
gamma1 = 0.25;

% 1. Generate exogenous data
[j_vec, t_vec, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, ...
  unobs_mean, unobs_var);

% 2. Generate endogenous data
ZETA_TOL = 1e-10;

n_draw = 1e2;
disp("Beginning dataset creation, N =");
disp(n_draw);
df = simulate('fsolve', '../output/fsolve_100.csv');
n_draw = 2e2;
disp("Beginning dataset creation, N =");
disp(n_draw);
df = simulate('fsolve', '../output/fsolve_200.csv');
n_draw = 5e2;
disp("Beginning dataset creation, N =");
disp(n_draw);
df = simulate('fsolve', '../output/fsolve_500.csv');
n_draw = 1e3;
disp("Beginning dataset creation, N =");
disp(n_draw);
df = simulate('fsolve', '../output/fsolve_1000.csv');
disp("Beginning dataset creation, N =");
disp(n_draw);
df = simulate('zeta', '../output/zeta_1000.csv');
diary off;

function full_data_mat = simulate(PRICING, save_as)
    global n_draw J T JT H_t beta1_mean beta_mean beta_var alpha ...
      gamma0 gamma1 unobs_mean unobs_var theta data_mat mc mkt_rows opts ...
      j_vec t_vec x sat wire w xi omega;
    % A. Draw parameters
    beta1 = repmat(beta1_mean, 1, n_draw);
    beta2 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
    beta3 = normrnd(beta_mean, sqrt(beta_var), 1, n_draw);
    alpha_vec = repmat(alpha, 1, n_draw);
    const = ones(1, n_draw);
    theta = [beta1; beta2; beta3; alpha_vec; const];

    % B. Generate prices
    mc = exp(gamma0*ones(JT,1) + gamma1*w + omega/8);
    p0 = 1.2*mc; % initial guess (no markup); 
    data_mat = [j_vec, t_vec, x, sat, wire, p0, w, xi, omega];
    [s, ds_dp] = gen_shares(data_mat, theta, T, JT, n_draw); % initial guess
    p = zeros(JT, 1);
    for t = 1:T
        mkt_rows = (data_mat(:, 2) == t);
        p0_t = p0(mkt_rows, 1);
        if strcmp(PRICING, 'fsolve')
            [p_t, dPI, flag, output] = fsolve(@foc, p0_t, opts);
            if flag ~= 1
                disp("WARNING! Price could not be solved for the following market:")
                disp(t)
            end
        elseif strcmp(PRICING, 'zeta')
            p_t = iterate_zeta(p0_t);
        else
            disp('PRICING must take a valid value');
            assert(strcmp(PRICING, 'zeta') | strcmp(PRICING, 'fsolve'));
        end 
        p(mkt_rows, 1) = p_t;
    end
    data_mat(:, 6) = p;
    [s, ds_dp] = gen_shares(data_mat, theta, T, JT, n_draw); 
    ds_dp_own = diag(ds_dp);
    own_price_e = ds_dp_own ./s .* p;
    div_ratio = repmat([0], 1, J);
    ds0_dj = - sum(ds_dp, 2); 
    Dj0 = - ds0_dj ./ diag(ds_dp);
    for t = 1:T
        mkt_rows = (data_mat(:, 2) == t);
        dst_dpt = ds_dp(mkt_rows, mkt_rows);
        dst_dpt_own = ds_dp_own(mkt_rows, 1); 
        div_ratio_t = diag(Dj0(mkt_rows, :)); 
        for j = 1:J
            for k = 1:J
                if k~= j
                    div_ratio_t(j,k) = - dst_dpt(j,k) / dst_dpt(j,j);
                end
            end
        end
        div_ratio = [div_ratio; div_ratio_t];
    end
    div_ratio = div_ratio(2:(JT + 1), :);
    full_data_mat = [data_mat, s, mc, own_price_e, div_ratio];
    writematrix(full_data_mat, save_as);
    disp("Variance-covariance matrix: ");
    cov(beta2, beta3)
end

function [j_vec, t_vec, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, ...
  unobs_mean, unobs_var)
    j_vec = repmat((1:J)', T, 1);
    t_vec = repelem((1:T)', J);
    x = abs(normrnd(0, 1, JT, 1));
    sat = repmat([1; 1; 0; 0], T, 1);
    wire = ones(JT, 1) - sat;
    w = abs(normrnd(0, 1, JT, 1));
    unobs = mvnrnd(unobs_mean, unobs_var, JT);
    xi = unobs(:,1);
    omega = unobs(:,2);
end

function dPI = foc(p_t)
    global data_mat mkt_rows theta n_draw mc;
    t_mat = data_mat(mkt_rows, [3, 4, 5, 6, 8]);
    t_mat(:, 4) = p_t;
    s_t = mkt_shares(t_mat, theta, n_draw);
    [dst_dpt, ~, ~] = share_deriv_market(t_mat, theta, n_draw);
    dPI = (p_t - mc(mkt_rows, 1)).*diag(dst_dpt) + s_t;
end

function p = iterate_zeta(p0_t)
    global ZETA_TOL;
    tol = ZETA_TOL;
    max_iter = 1e3;
    diff = 1;
    i = 0;
    while max(abs(diff)) > tol & i < max_iter
        p = p_zeta(p0_t);
        diff = p - p0_t; 
        p0_t = p;
        i = i + 1;
    end
    if i >= max_iter
        disp('WARNING: Zeta fixed point iteration exceeds max iterations.');
    end
end 

function p = p_zeta(p_t)
    global data_mat mkt_rows theta n_draw mc H_t;
    t_mat = data_mat(mkt_rows, [3, 4, 5, 6, 8]);
    t_mat(:, 4) = p_t;
    c_t = mc(mkt_rows, 1);
    s_t = mkt_shares(t_mat, theta, n_draw);
    [dst_dpt, Lambda_t, Gamma_t] = share_deriv_market(t_mat, theta, n_draw);
    invLambda_t = Lambda_t^-1;
    zeta_t = invLambda_t*(H_t .* Gamma_t)*(p_t - c_t) - invLambda_t*s_t;
    p = c_t + zeta_t; 
end

function [s, ds_dp] = gen_shares(data_mat, theta, T, JT, n_draw)
    s = zeros(JT, 1);
    ds_dp = [0];
    for t = 1:T
        t_rows = (data_mat(:, 2) == t);
        t_mat = data_mat(t_rows, [3, 4, 5, 6, 8]);
        s_t = mkt_shares(t_mat, theta, n_draw);
        s(t_rows, 1) = s_t;
        [dst_dpt, ~, ~] = share_deriv_market(t_mat, theta, n_draw);
        ds_dp = blkdiag(ds_dp, dst_dpt);
    end
    ds_dp = ds_dp(2:(JT + 1), 2:(JT + 1));
end

function [dst_dpt, Lambda_t, Gamma_t] = share_deriv_market(t_mat, theta, n_draw)
    [s_t, sigma_t] = mkt_shares(t_mat, theta, n_draw);
    alpha = theta(4,1);
    t_size = size(t_mat);
    J_t = t_size(1);
    dst_dpt = zeros(J_t, J_t);
    Lambda_t = zeros(J_t, J_t);
    Gamma_t = zeros(J_t, J_t);
    for i = 1:n_draw
        sigma_it = sigma_t(:,i);
        Gamma_it = alpha*sigma_it*sigma_it';
        Lambda_it = alpha*sigma_it;
        dsigma_own = Lambda_it .* (1 - sigma_it);
        dsigma_own = diag(dsigma_own);
        dsigmait_dpt = -Gamma_it .* (ones(J_t) - eye(J_t)) + dsigma_own;
        Gamma_t = Gamma_t + Gamma_it; 
        Lambda_t = Lambda_t + diag(Lambda_it); 
        dst_dpt = dst_dpt + dsigmait_dpt;
    end
    Gamma_t = Gamma_t / n_draw;
    Lambda_t = Lambda_t / n_draw;
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
