function HK = hK(X, exampleNum, theta1, theta2, k)

X = X(exampleNum,:)';

a1 = [1; X];

a2 = sigmoid(theta1 * a1);

a2 = [1; a2];

a3 = sigmoid(theta2 * a2);

HK = a3(k,:);


end