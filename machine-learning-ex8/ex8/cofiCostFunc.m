function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%


for i = 1:num_movies
   
    for j = 1:num_users
        
       if R(i,j) == 1
          
           J = J + (Theta(j,:)*X(i,:)' - Y(i,j))^2;
           
       end
        
        
    end
    
    
end

J = J / 2;



for l = 1:num_movies
   
    for b = 1:num_features
       
       
       X_grad(l,b) = 0;
        
        
        for allJ = 1:num_users
           
            if R(l,allJ) == 1
                
              X_grad(l,b) = X_grad(l,b) + (Theta(allJ,:)*X(l,:)' - Y(l,allJ))*Theta(allJ,b);             
                      
            end
            
        end
        
        X_grad(l,b) = X_grad(l,b) + (lambda*X(l,b));
        
        
    end
   
    
end




for nU = 1:num_users
   
    for c = 1:num_features
       
       
       Theta_grad(nU,c) = 0;
        
        
        for allI = 1:num_movies
           
            if R(allI,nU) == 1
                
              Theta_grad(nU,c) = Theta_grad(nU,c) + (Theta(nU,:)*X(allI,:)' - Y(allI,nU))*X(allI,c);             
                      
            end
            
        end
        
        Theta_grad(nU,c) = Theta_grad(nU,c) + (lambda*Theta(nU,c));
        
        
    end
   
    
end



Jreg = 0;

for dd=1:num_users
   
    for nF = 1:num_features
    
        Jreg = Jreg + Theta(dd,nF)^2;
    
    end
    
end

Jreg = Jreg * lambda/ 2;

J = J + Jreg;

Jreg = 0;

for jj=1:num_movies
    
   for ii=1:num_features
      
       Jreg = Jreg + X(jj,ii)^2;
       
   end
    
end

Jreg = Jreg * lambda / 2;

J = J + Jreg;



% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
