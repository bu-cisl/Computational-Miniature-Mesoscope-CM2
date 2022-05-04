function [img,bg] = bg_removal(image, kernel_size)
% bg = imopen(image, strel('disk', kernel_size));
bg = imopen(image, strel('disk', kernel_size,8));
img = imsubtract(image, bg);
end

