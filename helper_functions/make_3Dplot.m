function make_3Dplot(x,y)
%takes a Nx2 predictor matrix x, and Nx1 outcome vector y
%makes a 3D mesh plot with regression plane 
%the axes are:
%X: x1
%Y: x2
%z: x3

%There's a walkthrough on making these 3D regression plots here:
%https://www.mathworks.com/help/stats/regress.html
%you can make cool non-linear planes etc. 

x1 = x(:,1); %seperate out predictors, so there's less typing below 
x2 = x(:,2);
scatter3(x1,x2,y,'filled') %make a 3D scatter plot
hold on 
%make vectors of 100 equally spaced values within the range of Xi
x1fit = linspace(min(x1),max(x2));
x2fit = linspace(min(x2),max(x2));
%get the x, y coordinates for the X plane 
[X1FIT,X2FIT] = meshgrid(x1fit,x2fit);
%get OLS betas for this model (with an intercept)
betas = ols_betas(x,y);
%get the Z coordinates for regression plane 
YFIT = betas(1) + betas(2)*X1FIT + betas(3)*X2FIT;
%plot the regression plane  
mesh(X1FIT,X2FIT,YFIT,'FaceAlpha',.5)
hold off
