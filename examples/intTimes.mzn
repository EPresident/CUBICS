int: n;

var-n..n: x;
var -n..n: y;
var -n*n..n*n: z;

constraint z = x * y;

solve maximize z;

output [show(z)];
