% Equation value maximization
% Elia Calligaris

int: n = 100;

var int: x;
array [1..10] of var 0..100: ys;
array [1..10] of int: coeff;
coeff = [8,2,-3,6,5,-1,-7,3,9,-3];

constraint
    x = sum(i in 1..10)(ys[i]*coeff[i]);
    
solve maximize x;

