function [output, val_max] = enhance_contrast_high_mat(data, high_threshold)
data = double(data);
if nargin == 1
    high_threshold = 0.9999;
end
hist_obj = histogram(data(:),2^14);
bin_edges = hist_obj.BinEdges;
bin_counts = hist_obj.BinCounts;
% close all;
pdf = bin_counts ./ sum(bin_counts(:));
cdf = cumsum(pdf);
[~,idx] = min(abs(cdf - high_threshold));
high_clip_val = 0.5 * (bin_edges(idx) + bin_edges(idx + 1));
data(data >= high_clip_val) = high_clip_val;
val_max = high_clip_val;
output = data;
close all;
end