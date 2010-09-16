
net = NeuralNet([2 2 1]);

xin = [0 0;
       0 1;
       1 0;
       1 1];
tout = [0; 0; 1; 1];

for i=1:2000,
    net.learnAll(xin, tout);
end

net.insert([0; 0])
net.insert([0; 1])
net.insert([1; 0])
net.insert([1; 1])