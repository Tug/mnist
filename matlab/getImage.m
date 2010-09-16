function [ image ] = getImage( raw_data, nb )
%GETIIMAGE Summary of this function goes here
%   Detailed explanation goes here

image = uint8(reshape(raw_data(nb,1:784),28,28))';

end

