clc;
clear;
close all;
echo off;

diary ./gen_data.log

% 0. Globals
global J T JT beta1_mean beta2_mean beta3_mean alpha gamma0 gamma1;
global theta_mean theta_var;

J = 4;
T = 600;
JT = J*T;
theta_mean = [0; 0];
theta_var = [1, 0.25; 0.25, 1];

[x, w, xi, omega] = draw_chars(JT, theta_mean, theta_var);

diary off;

function [x, w, xi, omega] = draw_chars(JT, theta_mean, theta_var)
    x = abs(normrnd(0,1,JT,1));
    w = abs(normrnd(0,1,JT,1));
    theta = mvnrnd(theta_mean, theta_var, JT);
    xi = theta(:,1);
    omega = theta(:,2);
end

