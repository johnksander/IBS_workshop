clear          %remove all variables from the workspace
clc            %clears the command window
format compact %removes extra spaces from the command window's output 
%always start scripts with at least a clear!


%----specify the directories needed in your analysis----
%I always use absolute directories, this avoids unintended consequences with
%duplicate files, mixing up working directories, etc. 
%I often start by hardcoding one "base" directory, then building my
%directory tree out from there. 

base_dir = 'C:\Users\jksander.000\Desktop\IBS_workshop\admin\IBS_workshop';
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


%----importing and handling multiple data files----
%set the data directory to IBS_workshop/data/file_pieces
files_dir = fullfile(data_dir,'file_pieces');  
%find all the .xlsx files in file_pieces/
FNs = dir(fullfile(files_dir,'*.xlsx'));
FNs.name %this is the field we want
%take the file names out and put them in a cell array with { }
FNs = {FNs.name};
%now we know how many data files we have 
num_files = numel(FNs); %numel() gives the number of elements, 
%we could get the same answer with num_files = size(FNs,1) as well 

%set up a cell array that we can load each datafile into 
all_data = cell(num_files,1);

%now we want to iteratively load each file, but only load the files with
%the correct number of variables 
num_vars = numel(var_info); %could also use size() instead of numel() here 

for idx = 1:num_files
   
    %specify the current datafile to load 
    current_file = FNs{idx};
    %load the data from that file 
    [~,~,curr_data] = xlsread(fullfile(files_dir,current_file)); 
    
    %Here current_file and curr_data are "temporary variables". 
    %They're declared within the loop, and will change with every iteration
    
    %remove the header info, we don't need the header for each file 
    curr_data = curr_data(2:end,:);
    %convert from cell array to matrix 
    curr_data = cell2mat(curr_data);  
    
    %Determine if the file's data is correct. Store correct data, or display
    %a message indicating the file has bad data so we know what happened.

    if size(curr_data,2) == num_vars
        %the number of columns matches our expected number of varaibles 
        all_data{idx} = curr_data; %store the imported data 
    
    else
        %"else" captures any condition not previously met in this if-statement
        %here, that must mean an incorrect number of variables
        
        %square brackets [] concatenate any datatype 
        %construct a message by concatenating strings 
        message = [current_file, ' only has ', num2str(size(curr_data,2)), ' variables'];
        disp(message)
    end
    
end


%each cell has numeric data with the same number of columns
data = cell2mat(all_data); %so convert to a matrix 


%----select the data for analysis----
%Let's look at housing value as a function of distance to employment

price_var = strcmp('MEDV',var_info); 
%these are returned as a logical vectors (booleans)
price_var
%you can index with logicals as well 
data(:,price_var) %returns only the column in data where price_var is true 

dist_var = strcmp('DIS',var_info); 
price_or_dist = price_var | dist_var;


%----inspecting for outliers/missing values----
%identify all the missing values in the dataset 
%missing entries give: NaN "Not a Number" values 
missing_vals = isnan(data); %we get a logical matrix back, same size as data 

%check how many outliers there are by summing the logical matrix 
sum(missing_vals) %you can see sum() operates column-wise by default  

%we want to remove rows from the dataset containing  missing values 
%turn the missing_vals logical matrix into a logical vector with T/F for each row
missing_vals = sum(missing_vals,2) > 0; %sum(x,2) operates row-wise 

data = data(~missing_vals,:); %remove these rows using the "is not" operator 

%identify all outlier values in the dataset 
outlier_thresh = mean(data) + (4 * std(data)); %note: high threshold just for the purposes of this excersize 

%the result is a threshold value for each variable since mean() and std() 
%operate column-wise. We want to do the operation "data > outliers", and matlab will 
%infer that we're doing a column-wise "greater than" operation based on the
%matrix dimensions 

outliers = data > outlier_thresh; %now we have a full logical matrix 
outliers = sum(outliers,2) > 0; %find the bad observations (mark the rows)

data = data(~outliers,:); %remove these rows using the "is not" operator 

%Note: I only put high outliers in this data, but you'd want to do this 
%for outliers on the low-end as well. You could also use absolute
%difference to do both hi & lo in one pass. Something like this:
%abs(data - mean(data)) > (4 * std(data))

%----correlation & permutation testing with price & distance----

price_data = data(:,price_var);
dist_data = data(:,dist_var);
scatter(price_data,dist_data,'filled')
xlabel('price')
ylabel('distance to employment')

%find the spearman correlation
r = corr(price_data,dist_data);

%test whether this correlation is meaningful with a permutation test
num_perms = 15000; %15k permutations
null_dist = NaN(num_perms,1); %preallocate a vector for our null distriubion
num_obs = numel(price_data); %we'll need this for randperm()

%for 15k permuted orderings, take our data & randomly shuffle the
%observations. Then, find the correlation of our permuted data 
for idx = 1:num_perms
    shuffled_order = randperm(num_obs);
    price_shuffled = price_data(shuffled_order); %the indexing reorders price_data
    %find the permuted r statistic, and store in null_dist
    null_dist(idx) = corr(price_shuffled,dist_data);    
end

%plot the null distribution 
histogram(null_dist)
%plot the real value on the nulll
hold on 
scatter(r,10,50,'filled')
hold off
%find the p-value for our r statistic 
p = (sum(abs(null_dist) > r) + 1) / (numel(null_dist) + 1); %two tailed, make sure no zero p values



%:::::::BONUS if there's time left:::::::
%let's model housing price with all our other predictiors, and see if lasso
%regularization can tell us what's really driving housing prices in boston.


%----prepare the data----
charles_var = strcmp('CHAS',var_info); %for removing the categorical predictor  
Y = data(:,price_var);
X = data(:,~charles_var & ~price_var); %use everything else
pred_names = var_info(~charles_var & ~price_var); %keep the names for later 

Y = zscore(Y); %standardize the predictor & outcome variables
X = zscore(X); 


%----set up analysis parameters----
num_obs = numel(Y);
k_folds = 10; %10 cross-validation folds for lambda tuning
num_lambda = 250; %number of lambda values to evaluate 
target_feats = 4; %identify 4 most important predictors 
%matlab's cvpartition() will automatically try and balance how these 
%observations divvied up across all 10 CV folds. 
lambda_CV = cvpartition(num_obs,'KFold',k_folds);

%----train the model----
%pass the X and Y data, lambda CV info, and num lambda parameters
%specifying a normal outcome distribution defaults to an identity link function
[B,fit] = lassoglm(X,Y,'normal','CV',lambda_CV,'NumLambda',num_lambda,'DFmax',target_feats);
%output is a matrix of regularized betas (predictors x lambda values)
%and a structure with additional information about the model's fit
lassoPlot(B,fit,'plottype','CV');
%----test the model----
%find the optimal regularized betas
Lreg = fit.Index1SE; %best lambda value index
Breg = B(:,Lreg); %optimal regularized betas
Ireg = fit.Intercept(Lreg); %optimal regularized intercept
Breg = [Ireg;Breg]; %concatenate Ireg ontop of Breg with ";"
%estimate the Y labels for Xtest using these regularized betas
%specify estimation with a logit link function

y_hat = glmval(Breg,X,'identity'); %get predictions
R2 = 1 - (sumsqr(Y-y_hat) / sumsqr(Y-mean(Y))); %check out the fit 

Breg = Breg(2:end); %remove the intercept from regularized betas 
most_important = pred_names(Breg ~= 0); %it even beat our target number! 

%We'd proabably also want to do some cross-validation with the
%full model to avoid overfitting... 




