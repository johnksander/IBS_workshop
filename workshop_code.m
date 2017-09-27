clear          %remove all variables from the workspace
clc            %clears the command window
format compact %removes extra spaces from the command window's output 
%always start scripts with at least a clear!


%----specify the directories needed in your analysis----
%I always use absolute directories, this avoids unintended consequences with
%duplicate files, mixing up working directories, etc. 
%I often start by hardcoding one "base" directory, then building my
%directory tree out from there. 

base_dir = '/Users/ksander/Desktop/IBS_workshop';
data_dir = fullfile(base_dir,'data'); %fullfile() is an easy way to create directories 
func_dir = fullfile(base_dir,'helper_functions');
%If you want to use any extra functions or libraries, they must be in
%matlab's path. Matlab won't know they exist otherwise 
addpath(func_dir) %add our helper_functions directory to the path
data_fn = fullfile(data_dir,'boston_housing.xlsx');


%----import the data----
%There's a few different ways to import data, this is a good reference:
%https://www.mathworks.com/help/matlab/import_export/supported-file-formats.html
%import functions differ in how they import data, and supported formats

help xlsread 

%multiple function outputs are called using brackets. We only want the third 
%xlsread() output here, so we'll put the "is not" operator (~) in the first two spots. 
[~,~,data] = xlsread(data_fn); 

%data is a cell array. Cell arrays may be indexed like:
data(1,:) %show all cells in the first row 
data{1,2} %show the contents of the cell in row 1 & column 2 (note curly brackets) 

var_info = data(1,:); %assign the column headers to a new variable
data = data(2:end,:); %reassign data without the 1st row (rows 2 through end)

%now that data is entirely numeric, convert the variable to a matrix 
data = cell2mat(data); 
data(1,:) %matricies are only index with parentheses
data(1,2)


%----select the data for analysis----
%let's model housing value as a function of crime rate, distance to employment
%centers, and pupil-teacher ratio 

price_var = strcmp('MEDV',var_info); 
%these are returned as a logical vectors (booleans)
price_var
%you can index with logicals as well 
data(:,price_var) %returns only the column in data where price_var is true 

%get logicals for the predictor variables as well 
crime_var = strcmp('CRIM',var_info); 
dist_var = strcmp('DIS',var_info); 
PTR_var = strcmp('PTRATIO',var_info); 
%get a logical for all the predictors using the "or" logical operator 
pred_vars = crime_var | dist_var | PTR_var;

%another way we could've done this is with ismember()
vars2select = {'CRIM','DIS','PTRATIO'}; %make a cell array with variable names {}
pred_vars = ismember(var_info,vars2select); %returns the same logical 


%----inspecting for outliers/missing values----
%plot the housing value distribution with histogram()
histogram(data(:,price_var)) %input is data indexed with the price_var logical 

%identify the outlier price values 
outliers = data(:,price_var) > 100; %relational operators always give logicals 

sum(outliers) %check how many outliers there are by summing the logical vector 

%find missing values in the predictor variables 
%missing entries give: NaN "Not a Number" values 
missing_vals = isnan(data(:,pred_vars)); %we get a logical matrix since there's three columns 

sum(missing_vals) %you can see sum() operates column-wise by default  


%----cleaning the data----
%we want to remove rows from the dataset containing outliers or missing values 

%turn the missing_vals logical matrix into a logical vector with T/F for each row
missing_vals = sum(missing_vals,2) > 0; %sum(x,2) operates row-wise 

bad_obs = outliers | missing_vals; %now we have one logical for all rows with bad values

data = data(~bad_obs,:); %remove these rows using the "is not" operator 

%----prepare the remaining data for regression----
y = data(:,price_var); 
x = data(:,pred_vars);
%if you want to double-check the column ordering of x
pred_names = var_info(pred_vars);
pred_names

%let's see if the predictor variables are normally distributed 
histogram(x(:,1))
histogram(x(:,2)) 

x = log2(x); %log transform the predictors 

histogram(x(:,2)) 


%----fit a linear model----
%great documentation
%https://www.mathworks.com/help/stats/linear-regression-model-workflow.html

mdl = fitlm(x,y);
mdl  %this is a structure datatype (that gives special command prompt output)
mdl.Coefficients.Estimate %fields within the structure can be accessed with a "."

plot(mdl) %matlab has a lot of built in ways to visualize linear models 
plotResiduals(mdl,'probability') %plotInteraction(), plotadded() are some others

%try making ols_slopes() function that returns beta coefficients
%betas = ols_slopes(x,y);

betas = ols_betas(x,y);


%----model selection and plotting----
%distance to employment centers is a poor predictor, so let's refit the
%model on just x1 and x3. 

%assigning matrix entries to empty brackets is another way to remove entries
x(:,2) = []; 

mdl = fitlm(x,y); %refit the model 
mdl

betas = ols_betas(x,y); %new slopes 

%let's plot how the model fits crime rate vs. housing price 
scatter(x(:,1),y,'filled') %scatter plot for the datapoints 
hold on %for adding things on top of existing plots ("hold on to the figure")

%the plot() function plots lines using x and y coordinate inputs
%we need a vector of x coordinates for crime rate data's range 
%linspace(a,b) gives a linearly spaced vector from a -> b (effectively a
%line with a slope of 1)
Xline = linspace(min(x(:,1)), max(x(:,1))); %line from min(x1) to max(x1)
%get the predicted housing price values for that line (y coordinates)
yhat = betas(1) + betas(2)*Xline + betas(3)*mean(x(:,2)); %y = 1 + x2 + x2

plot(Xline,yhat,'linewidth',4) 
ylim([0,55]) %make the y axis limits a bit bigger
%refline() gives similar functionality to what we just did here 

%add a legend
leg_labels = {'town data','model fit at mean PT ratio'};
legend(leg_labels,'FontSize',14,'location','southwest')

%add some axes labels 
ylabel('median house price ($k)','FontSize',14)
xlabel('crime rate (log2)','FontSize',14)

%and a title 
title('Boston area real estate','FontSize',16,'FontWeight','b')

%remove the little ticks if you don't like them 
set(gca,'TickLength',[0 0])

hold off %remember to turn the hold off, or you'll keep adding to the figure!

fig_fn = fullfile(base_dir,'model_fit'); %file name for the figure
print(fig_fn,'-djpeg') 
%the second argument specifies the output filetype. The format is a little 
%funny, but it's specifying the file driver so you can remember it
%like "-driver jpeg". Likewise, for pdfs it's '-dpdf' like "-driver pdf".


%let's plot how the model fits housing price vs. crime rate & PT ratio
make_3Dplot(x,y)
%makes a 3D mesh plot with regression plane, the axes are:
%X: x1
%Y: x2
%Z: y

rotate3d on %this will let you freely rotate the 3d figure using the mouse    

%add some axes labels 
xlabel('crime rate (log2)','FontSize',14)
ylabel('pupil-teacher ratio (log2)','FontSize',14)
zlabel('median house price ($k)','FontSize',14)

%once you've found a nice angle, turn the rotation off 
rotate3d off

fig_fn = fullfile(base_dir,'model_fit3d'); %make a new filename for this figure
print(fig_fn,'-djpeg') %print it 
%----check out----
%https://www.mathworks.com/help/stats/regress.html
%for a walkthrough on making these 3d plots 


%:::::::BONUS if there's time left:::::::
%let's see if we can predict whether a town's on the charles river
%using cross-validated LASSO regression

%----prepare the data----
charles_var = strcmp('CHAS',var_info); 
y = data(:,charles_var);
x = data(:,~charles_var);

sum(y) 
%there's about 35 towns on the charles, so for this quick & dirty example
%let's dump some of the non-charles observations just so the class sizes
%aren't as crazy imbalanced

[y,sorting_order] = sort(y); %sort y and return the sorting order 
x = x(sorting_order,:); %sort the rows of x according to that sorting order  

%now all the charles-river towns are at the end of these vectors because
%the sort() default order is ascending. Let's trim like 300 observations
%off the top. Naturally, this is NOT how you'd solve this problem in real life!!!  
y = y(300:end,:);
x = x(300:end,:); 

x = zscore(x); %standardize the predictor variables


%----set up analysis parameters----
k_folds = 10; %10 cross-validation folds (also for lambda tuning)
num_lambda = 25; %number of lambda values to evaluate (this is low & may throw warnings)
%matlab's cvpartition() will automatically try and balance how these 
%observations divvied up across all 10 CV folds. That's important here
%since there's so few charles observations. 
CV = cvpartition(y,'KFold',k_folds);
guesses = cell(k_folds,1); %preallocate a cell array for CV fold results 

%----do cross-validation----
%iterate over each fold with a loop
for idx = 1:k_folds
    %our looping/iterating variable is idx
    
    %----split up the data----
    %we want to get logicals for this fold's training & testing data 
    train_inds = training(CV,idx); %pass CV plus the fold index (idx) 
    test_inds = test(CV,idx);
    %seperate the data out into testing & training sets 
    Ytrain = y(train_inds); 
    Xtrain = x(train_inds,:);
    Ytest = y(test_inds);
    Xtest = x(test_inds,:);
    
    %lambda is optimized on the training data with cross-validation as well
    %make another cvpartition() object with the training labels 
    lambda_CV = cvpartition(Ytrain,'KFold',k_folds);
    
    %----train the model----
    %pass the training data, lambda CV info, and num lambda parameters  
    %specifying a binomial outcome distribution defaults to a logit link function
    [Breg,fit] = lassoglm(Xtrain,Ytrain,'binomial','CV',lambda_CV,'NumLambda',num_lambda);
    %output is a matrix of regularized betas (predictors x lambda values)
    %and a structure with additional information about the model's fit
    
    %----test the model----
    %find the optimal regularized betas 
    Lreg = fit.Index1SE; %best lambda value index
    Breg = Breg(:,Lreg); %optimal regularized betas
    Ireg = fit.Intercept(Lreg); %optimal regularized intercept 
    Breg = [Ireg;Breg]; %concatenate Ireg ontop of Breg with ";" 
    %estimate the Y labels for Xtest using these regularized betas
    %specify estimation with a logit link function 
    label_predictions = glmval(Breg,Xtest,'logit');
    %round the label probabilties for descrete class predictions 
    label_predictions = round(label_predictions); 
    %calculate and display model accuracy for this fold 
    CVacc = sum(label_predictions == Ytest) / numel(Ytest);
    disp(['CV fold accuracy = ' num2str(CVacc*100) '%'])
    %store the label predictions and true Y labels 
    guesses{idx} = [label_predictions,Ytest];
end

guesses = cell2mat(guesses); %convert to a matrix
%quick & dirty way to visualize the model's prediction behavior 
imagesc(guesses) 
%calculate and display overall cross-validation accuracy 
accuracy = sum(guesses(:,1) == guesses(:,2)) / numel(guesses(:,1));
disp(['overall accuracy = ' num2str(accuracy*100) '%'])

%next we'd want to do permutation testing in order to evaluate whether our
%model truly works well. 





