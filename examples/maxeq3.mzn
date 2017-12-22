% Equation value maximization (size 3)
% Elia Calligaris

int: n = 3;
int: m = 10;

var int: x;
array [1..n] of var 0..m: ys;
array [1..n] of int: coeff;
coeff = [3,2,-1];

constraint
    x = sum(i in 1..n)(ys[i]*coeff[i]);
    
solve maximize x;

