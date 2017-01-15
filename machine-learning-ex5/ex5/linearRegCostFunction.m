function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

for i = 1:m
   
    hypothesis = X(i,:) * theta;
   
    J = J + (hypothesis - y(i,:))^2;
    
end

J = J / (2*m);

regTerm = (theta .* theta);
regTerm = regTerm(2:end,:);
regTerm = (sum(regTerm)) / (2*m);
regTerm = regTerm * lambda;

J = J + regTerm;



for l=1:length(theta)

    for t = 1:m
    
    hypothesis = X(t,:) * theta;
    grad(l) = grad(l) + (hypothesis - y(t,:))*X(t,l);
    
    
    end
    
    grad(l) = grad(l) / m;
    
end

for s=2:length(theta)
grad(s) = grad(s) + ( theta(s,:) * lambda / m );
end

% =========================================================================

grad = grad(:);

end
