clc;
clear;
close all;
echo off;

diary ./gen_data.log

% 0. Globals
global n_draw J T JT beta1_mean beta2_mean beta3_mean alpha gamma0 gamma1;
global unobs_mean unobs_var;

n_draw = 1e5;
J = 4;
T = 600;
JT = J*T;
unobs_mean = [0; 0];
unobs_var = [1, 0.25; 0.25, 1];

[x, w, xi, omega] = draw_chars(JT, unobs_mean, unobs_var);
dsigmat_dpt = prob_deriv_matrix(-1, [0.2;0.2;0.2;0.2]);
disp(dsigmat_dpt);

diary off;

function [x, w, xi, omega] = draw_chars(JT, unobs_mean, unobs_var)
    x = abs(normrnd(0, 1, JT, 1));
    w = abs(normrnd(0, 1, JT, 1));
    unobs = mvnrnd(unobs_mean, unobs_var, JT);
    xi = unobs(:,1);
    omega = unobs(:,2);
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

