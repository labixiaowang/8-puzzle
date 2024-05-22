```Matlab
% 定义系统矩阵 A
A = [1 2; 5 -4]; % [-1 2;-3 -4]是稳定的
% 选择一个对称正定矩阵 Q
Q = eye(2);  % 可以选择单位矩阵
% 求解Lyapunov方程 A'P + PA = -Q
P = lyap(A', Q);
% 检查 P 是否对称正定
isSymmetric = isequal(P, P');  % 检验是否对称
isPositiveDefinite = all(eig(P) > 0);  % 检验是否正定
if isSymmetric && isPositiveDefinite
    disp('系统是Lyapunov稳定的');
else
    disp('系统不是Lyapunov稳定的');
end
% 显示 P 矩阵
disp('P 矩阵:');
disp(P);
```