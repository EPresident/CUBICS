% Simple maximization test for the IntAbs constraint
% Elia Calligaris

int: n = 100;

var int: x;

constraint
    abs(x) = n;
    
solve maximize x;

