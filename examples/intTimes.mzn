int: n;

var 0..n: x;
var 0..n: y;
var 0..n*n: z;

constraint z = x * y;

solve maximize z;

output [show(z)];
