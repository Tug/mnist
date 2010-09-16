clear;
%load_data;
%digit1ls(1,:)

xin = [1 1; 0 1; 1 0; 0 0];
tout = [0; 1; 1; 0];

net = NeuralNet([2 2 1]);

[dw, dEdw] = net.evaluateError(xin, tout);

for i=1:length(dEdw),
    dw{i} - dEdw{i}
end

net.w{1} = [0.13776874061  -0.0317713676677 0.00450988854744;
            0.103181761176 -0.0964332998828 -0.0380263450198];
net.w{2} = [1.13519435614 -0.786749095684 1.48263390523];
net.eta = 0.1;
net.m = 0.9;

n = 10000;
allE = zeros(n,1);
for i=1:n,
    allE(i) = net.learn(xin, tout);
end

figure(1);
plot(allE, '*');

net.test([0; 1])
net.test([0; 0])