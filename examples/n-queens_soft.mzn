% n-queens example in MiniZinc using CP techniques
% vim: ft=zinc ts=4 sw=4 et tw=0
% By Reza Rafeh July 2005
% MiniZinc version
% Peter Stuckey September 30 2006
% Soft-constrained/optimization version
% Elia Calligaris

int: n;                                 % The number of queens.

array [1..n] of var 1..n: q;
var int: attacks;

function var int: 
    countattacks(int: i, int: j, var int: qi, var int: qj) =
    (
     bool2int(qi == qj) +
     bool2int(abs(qi - qj) == abs(i - j))
    ) * bool2int(i!=j);

constraint
    attacks = sum(i in 1..n, j in 1..n)(countattacks(i, j, q[i], q[j]));
    
    constraint attacks = 0;
    
 
%constraint attacks = 0;
%solve satisfy;
solve minimize attacks;

% output	["n queens, soft version:\n"] ++
% 	[	if fix(q[i]) = j then "Q " else ". " endif ++
% 	 	if j = n then "\n" else "" endif
% 	|	i, j in 1..n
% 	]++
% 	["Attacks: ", show(attacks)]; 
