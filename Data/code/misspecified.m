clc;
clear;
close all;
echo off;

J = 4;
T = 600;
JT = J*T;


full_data_mat = readmatrix("../output/data.csv");
%full_data_mat = cell2mat(struct2cell(full_data_mat));


full_data_mat = [full_data_mat, delta(full_data_mat, JT, T,J)];
full_data_mat = [full_data_mat, within_group_share(full_data_mat, JT, T)];
full_data_mat = [full_data_mat, opponent(full_data_mat, JT)];




%[j, t, x, sat, wire, p, w, xi, omega, s, mc, delta, wgs, lnwgs, lnwgs*sat, lnwgs*wire]

mat = full_data_mat(:,[12, 3, 4, 5, 6, 7, 14, 15, 16, 17, 18]);
tbl = array2table(mat,'VariableNames', {'delta','x','sat','wire','price','w','lnwgs', 'lnwgs_sat','lnwgs_wire', 'x_opp','w_opp'});

writetable(tbl, "../output/data_misspecified.csv");

%OLS
ols = fitlm(tbl, 'delta~x+sat+wire+price', 'Intercept',false);

ols

%IV (Conduct 2SLS manually)
IV1 = fitlm(tbl, 'price~x+w+sat+wire', 'Intercept',false);
phat = predict(IV1,tbl);
phat = array2table(phat, 'VariableNames',{'phat'});
tbl2 = [tbl, phat];
IV2 = fitlm(tbl2, 'delta~x+sat+wire+phat', 'Intercept',false);

IV2

%Nested Logit1 (correlation is identical across groups)
NL1 = fitlm(tbl, 'price~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
phat_NL = predict(NL1,tbl);
phat_NL = array2table(phat_NL, 'VariableNames',{'phat_NL'});
tbl3 = [tbl, phat_NL];

NL2 = fitlm(tbl, 'lnwgs~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
lnwgshat = predict(NL2,tbl3);
lnwgshat = array2table([lnwgshat,lnwgshat .* tbl.sat, lnwgshat .* tbl.wire], 'VariableNames',{'lnwgshat','lnwgshat_s','lnwgshat_w'});
tbl4 = [tbl3, lnwgshat];

NL3 = fitlm(tbl4, 'delta~x+sat+wire+phat_NL+lnwgshat', 'Intercept',false);


NL3


%Nested Logit 2 (allow correlation to be different across groups)
NL4 = fitlm(tbl, 'lnwgs_sat~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
ls_new = predict(NL4,tbl);
ls_new = array2table(ls_new, 'VariableNames',{'ls_hat'});


NL5 = fitlm(tbl, 'lnwgs_wire~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
lw_new = predict(NL5,tbl);
lw_new = array2table(lw_new, 'VariableNames',{'lw_hat'});
tbl5 = [tbl3, ls_new, lw_new];

NL6 = fitlm(tbl5, 'delta~x+sat+wire+phat_NL+ls_hat + lw_hat', 'Intercept',false);


NL6


% calculate delta = ln(s_j) - ln(s_0)
function output = delta(full_data_mat, JT, T, J)
    dummy = zeros(JT,1);
    %full_data_mat = [full_data_mat, zero];
    for t = 1:T
        t_rows = (full_data_mat(:, 2) == t);
        t_mat = full_data_mat(t_rows, :);
        s_0 = 1 - sum(t_mat(:,10));
        delta_t = log(t_mat(:,10))-log(s_0)*ones(J, 1);
        dummy(t_rows,1) = delta_t;
    end
    output = dummy;
end

% opponent's characteristics

function output = opponent(full_data_mat, JT)
    dummy = zeros(JT,2);
    for i =1:(JT/2)
        dummy((2*i-1),1) = full_data_mat((2*i),3);
        dummy((2*i-1),2) = full_data_mat((2*i),7);
        dummy((2*i),1) = full_data_mat((2*i-1),3);
        dummy((2*i),2) = full_data_mat((2*i-1),7);
    end
    output = dummy;
end

% calculate within group share

function output = within_group_share(full_data_mat, JT, T)
    dummy = zeros(JT,4);
    for t =1:T
        for n = 0:1
        tn_rows = (full_data_mat(:, 2) == t) .* (full_data_mat(:, 4+n) == 1);
        tn_mat = full_data_mat(logical(tn_rows), :);
        gs = sum(tn_mat(:,10)); %group share
        wgs = tn_mat(:,10)./gs; %within group share
        dummy(logical(tn_rows),1) = wgs;
        end
    end
    dummy(:,2) = log(dummy(:,1));
    dummy(:,3) = full_data_mat(:,4).*dummy(:,2);
    dummy(:,4) = full_data_mat(:,5).*dummy(:,2);
    output = dummy;
end