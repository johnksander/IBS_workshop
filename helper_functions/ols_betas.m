function B = ols_betas(x,y)

int = ones(size(y)); %create a vector of ones the same size as y
x = [int x]; %concatenate it with x for an intercept  

%calculate the slope coefficients 
B = ((x' * x)^-1) * (x' * y);

%matrix vs. element-wise operations:
%   using "*" applies matrix multiplication (so we can get sum of squares)
%               x' * x  
%   using ".*" applies element-wise multiplication (entries multiplied individually)
%               x .* x  
%   the same notation works for division "/" and "./"

%other special operators here:
%   the apostrophe transposes an array
%   the caret gives exponentiation 


