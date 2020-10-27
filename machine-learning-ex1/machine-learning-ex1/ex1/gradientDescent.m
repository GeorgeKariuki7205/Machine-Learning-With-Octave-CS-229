function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %


    calculatedCosts = X*theta;

    errorRate = (calculatedCosts.-y);
    errorRateSumThetaZero = sum(errorRate);
    errorRateSumThetaOne = sum(errorRate.*X(:, 2));    
    multiplyByAlphaThetaZero = (alpha*errorRateSumThetaZero)/m;
    multiplyByAlphaThetaOne = (alpha*errorRateSumThetaOne)/m;
    
    theta0 = theta(1)- multiplyByAlphaThetaZero;
    theta1 = theta(2) - multiplyByAlphaThetaOne;
    
    theta(1) = theta0;
    theta(2) = theta1;




    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
