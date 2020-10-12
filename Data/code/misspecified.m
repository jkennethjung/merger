clc;
clear;
close all;
echo off;

J = 4;
T = 600;
JT = J*T;


%%% Import Data

full_data_mat = readmatrix("../output/fsolve_1000.csv");
%full_data_mat = cell2mat(struct2cell(full_data_mat));


%%% Generate Variables Needed for Estimation

full_data_mat = [full_data_mat, delta(full_data_mat, JT, T,J)]; % delta
full_data_mat = [full_data_mat, within_group_share(full_data_mat, JT, T)]; %wgs, ln_wgs, lnwgs*sat, lnwgs*wire
full_data_mat = [full_data_mat, opponent(full_data_mat, JT)]; % x_opp, w_opp

%[j, t, x, sat, wire, p, w, xi, omega, s, mc, own_price_e, div_ratio, delta, wgs, lnwgs, lnwgs*sat, lnwgs*wire]
mat = full_data_mat(:,[17, 3, 4, 5, 6, 7, 19, 20, 21, 22, 23]);
%RowsToDelete = full_data_mat(:,17);
%mat(RowsToDelete == 1, :) = [];
tbl = array2table(mat,'VariableNames', {'delta','x','sat','wire','price','w','lnwgs', 'lnwgs_sat','lnwgs_wire', 'x_opp','w_opp'});

writetable(tbl, "../output/data_misspecified.csv");


%%% Estimation of Misspecified Models (I DID THIS PART WITH STATA TOO. JUST IGNORE THIS PART.)

%OLS
ols = fitlm(tbl, 'delta~x+sat+wire+price', 'Intercept',false);

%IV (Conduct 2SLS manually)
IV1 = fitlm(tbl, 'price~x+w+sat+wire', 'Intercept',false);
phat = predict(IV1,tbl);
phat = array2table(phat, 'VariableNames',{'phat'});
tbl2 = [tbl, phat];
IV2 = fitlm(tbl2, 'delta~x+sat+wire+phat', 'Intercept',false);

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


%Nested Logit 2 (allow correlation to be different across groups)
NL4 = fitlm(tbl, 'lnwgs_sat~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
ls_new = predict(NL4,tbl);
ls_new = array2table(ls_new, 'VariableNames',{'ls_hat'});


NL5 = fitlm(tbl, 'lnwgs_wire~x+w+sat+wire+x_opp+w_opp', 'Intercept',false);
lw_new = predict(NL5,tbl);
lw_new = array2table(lw_new, 'VariableNames',{'lw_hat'});
tbl5 = [tbl3, ls_new, lw_new];

NL6 = fitlm(tbl5, 'delta~x+sat+wire+phat_NL+ls_hat + lw_hat', 'Intercept',false);

beta = NL6.Coefficients.Estimate;

%%% Price Elasticities Based on Nested Logit Model
full_data_mat = [full_data_mat, add_elas(full_data_mat, JT, T, beta)]; %
full_data_mat(:,17) = [];
data = full_data_mat(:,[1,12,13,14,15,16,27,32,33,34,35]);

% Comparison btw true and estimated (based on NLM) own price elasticities
% and diversion ratio
A = zeros(4,11);% mean
B = zeros(4,5);% correlation coefficients 
C = zeros(4,5);% max deviation
for j = 1:4
   j_rows = (data(:,1) == j); 
   j_mat = data(j_rows, :);
   A(j,:) = mean(j_mat);
   for k = 1:5
       B(j,k) = corr(j_mat(:,1+k), j_mat(:,6+k));
       C(j,k) = max(abs(j_mat(:,1+k)-j_mat(:,6+k)));
   end
end

D = [B,C];

table2latex(array2table(A), '../output/elas_div1.tex');
table2latex(array2table(D), '../output/elas_div2.tex');


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

function output = price_elasticities(tmat, beta)
    t_size = size(tmat);
    J_t = t_size(1);
    dst_dpt_own = zeros(J_t, 1);
    dst_dpt = zeros(J_t, J_t);
    els = zeros(J_t, J_t);
    div_ratio_t = -ones(J_t);
    alpha = beta(4);
    sigma_1 = beta(5);
    sigma_2 = beta(6);
    for j = 1:J_t
       for k = 1:J_t
           sig = tmat(j,4)*sigma_1 + tmat(j,5)*sigma_2;
           dst_dpt(j,k) =  (j == k)*alpha/(1-sig)*tmat(j,10)...
           - (tmat(j,4)*tmat(k,4)+tmat(j,5)*tmat(k,5))*alpha*sig/(1-sig)*tmat(j,10)*tmat(k,13)...
           - alpha*tmat(j,10)*tmat(k,10);
           els(j,k) = dst_dpt(j,k)/(tmat(j,10)/tmat(k,6));
       end
       els_own(j,1)=els(j,j);
    end
    for j = 1:J_t
        for k = 1:J_t
            div_ratio_t(j,k) = - dst_dpt(j,k) / dst_dpt(j,j);
        end
    end
    output = [dst_dpt, els_own, els, div_ratio_t];
end

function dummy = add_elas(full_data_mat, JT, T, beta)
    dummy = zeros(JT,13);
    for t =1:T
        tn_rows = (full_data_mat(:, 2) == t);
        tn_mat = full_data_mat(logical(tn_rows), :);
        dummy(logical(tn_rows),:) = price_elasticities(tn_mat, beta);
    end
end

function table2latex(T, filename)
    
    % Error detection and default parameters
    if nargin < 2
        filename = 'table.tex';
        fprintf('Output path is not defined. The table will be written in %s.\n', filename); 
    elseif ~ischar(filename)
        error('The output file name must be a string.');
    else
        if ~strcmp(filename(end-3:end), '.tex')
            filename = [filename '.tex'];
        end
    end
    if nargin < 1, error('Not enough parameters.'); end
    if ~istable(T), error('Input must be a table.'); end
    
    % Parameters
    n_col = size(T,2);
    col_spec = [];
    for c = 1:n_col, col_spec = [col_spec 'l']; end
    col_names = strjoin(T.Properties.VariableNames, ' & ');
    row_names = T.Properties.RowNames;
    if ~isempty(row_names)
        col_spec = ['l' col_spec]; 
        col_names = ['& ' col_names];
    end
    
    % Writing header
    fileID = fopen(filename, 'w');
    fprintf(fileID, '\\begin{tabular}{%s}\n', col_spec);
    fprintf(fileID, '%s \\\\ \n', col_names);
    fprintf(fileID, '\\hline \n');
    
    % Writing the data
    try
        for row = 1:size(T,1)
            temp{1,n_col} = [];
            for col = 1:n_col
                value = T{row,col};
                if isstruct(value), error('Table must not contain structs.'); end
                while iscell(value), value = value{1,1}; end
                if isinf(value), value = '$\infty$'; end
                temp{1,col} = num2str(value);
            end
            if ~isempty(row_names)
                temp = [row_names{row}, temp];
            end
            fprintf(fileID, '%s \\\\ \n', strjoin(temp, ' & '));
            clear temp;
        end
    catch
        error('Unknown error. Make sure that table only contains chars, strings or numeric values.');
    end
    
    % Closing the file
    fprintf(fileID, '\\hline \n');
    fprintf(fileID, '\\end{tabular}');
    fclose(fileID);
end