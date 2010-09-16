function image = binarize(im)
    minv = min(min(im));
    maxv = max(max(im));
    sizeInterval = maxv-minv+2;
    image = (im - minv + 1 )*255/sizeInterval;
    image = image > 128;
