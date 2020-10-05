diff --git a/Data/code/gen_data.m b/Data/code/gen_data.m
index fdd5bc3..a538ab9 100644
--- a/Data/code/gen_data.m
+++ b/Data/code/gen_data.m
@@ -1,4 +1,4 @@

+% 4. Generate own-price elasticities and matrix of diversion ratios
+
+% add a column of own price elasticity
+own_price_e = diag(ds_dp) ./s .* p;
+full_data_mat = [full_data_mat, own_price_e];
+
+% diversion ratio matrix
+D = zeros(2400,4);
+ds0_dj = sum(ds_dp, 2); %derivative of shares of outside good wrt price
+% of product j
+Dj0 = - ds0_dj ./ diag(ds_dp);
+
+for t=1:600
+    d = zeros(4,4);
+    d = d + diag( Dj0(1+4*(t-1):4*t,:) );
+    ds_dp_matrix = ds_dp(1+4*(t-1):4*t,1+4*(t-1):4*t);
+    for j = 1:4
+        for k = 1:4
+            if k ~= j
+                d(j,k) = - ds_dp_matrix(j,k) / ds_dp_matrix(j,j);
+            end
+        end
+    end                
+    D(1+4*(t-1):4*t,1:4) = D(1+4*(t-1):4*t, 1:4) + d;
+end
+
+full_data_mat = [full_data_mat, D];
 writematrix(full_data_mat, "../output/data.csv");
 
 diary off;
 
+%%
+
 function [j, t, x, sat, wire, w, xi, omega] = draw_chars(J, T, JT, ...
   unobs_mean, unobs_var)
     j = repmat((1:J)', T, 1);
@@ -136,5 +170,4 @@
         end
     end
     s_t = mean(sigma_t, 2);
-end
-
+end
\ No newline at end of file
