function YK = yK(y, exampleNum, k)


y = y(exampleNum,:);

output = [1;2;3;4;5;6;7;8;9;10];

YK = (output == y);

YK = YK(k,:);

end
