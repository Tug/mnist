S = load('digitb1.mat');
digit1 = S.D;
S = load('digitb3.mat');
digit3 = S.D;
S = load('digitb7.mat');
digit7 = S.D;

S = load('testb1.mat');
test1 = S.D;
S = load('testb3.mat');
test3 = S.D;
S = load('testb7.mat');
test7 = S.D;

%imshow(255*getImage( digit1, 533 ));

limit = 1/2;
sized1 = size(digit1);
limitLine = uint8(sized1(1) * limit);

digit1ls = digit1(1:limitLine,:);
digit1val = digit1(limitLine+1:end,:);

digit3ls = digit3(1:limitLine,:);
digit3val = digit3(limitLine+1:end,:);

digit7ls = digit7(1:limitLine,:);
digit7val = digit7(limitLine+1:end,:);

%imshow(uint8(getImage( digit1, 533 )));
