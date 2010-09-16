clear;
%load_data;
%digit1ls(1,:)

xin  = [1 1; 0 1; 1 0; 0 0];
tout = [0 1; 1 0; 1 0; 0 1];

net = NeuralNet([2 2 2]);

[dEdw,E] = net.evaluateError(xin, tout);

for i=1:length(dEdw),
    dEdw{i}
    E{i}
end

n = 5000;
allE = zeros(n,1);
for i=1:n,
    allE(i) = net.learn(xin, tout);
end

figure(1);
plot(allE, '*');

net.test([0; 1])
net.test([0; 0])