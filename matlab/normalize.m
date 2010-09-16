
for d=0:9,
    load(['test' num2str(d) '.mat'],'D','-mat');
    sizeD = size(D)
    D2 = zeros(sizeD);
    u = ones(sizeD(1),1)*mean(D);
    sd = ones(sizeD(1),1)*std(D);
    D = (D - u)/(sd.^2);
    save(['testb' num2str(d) '.mat'],'D','-mat');
end

for d=0:9,
    load(['digit' num2str(d) '.mat'],'D','-mat');
    sizeD = size(D)
    D2 = zeros(sizeD);
    u = ones(sizeD(1),1)*mean(D);
    var = ones(sizeD(1),1)*var(D);
    D = (D - u)/var;
    save(['digitb' num2str(d) '.mat'],'D','-mat');
end
