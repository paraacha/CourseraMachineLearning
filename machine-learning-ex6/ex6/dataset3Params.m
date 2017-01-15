function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%


% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

choiceC = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
choiceSig = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

bestC = 0;
bestSig = 0;

currentError = 10000000000;

for i = 1:8
    
    for j = 1:8
        
        model= svmTrain(X, y, choiceC(i), @(x1, x2) gaussianKernel(x1, x2, choiceSig(j)));
        
        predictions = svmPredict(model, Xval);
        
        newError = mean(double(predictions ~= yval));
        
        if newError < currentError
        
            currentError = newError;
            
            bestC = choiceC(i);
            bestSig = choiceSig(j);
            
        end        
        
        
    end


end

C = bestC; sigma = bestSig;
display(bestC); display(bestSig);

% =========================================================================

end
