function output = average_shrink(input, shrink_size)

[rows,cols] = size(input);
if mod(rows,shrink_size) || mod(cols,shrink_size)
    input = padarray(input,...
        [shrink_size-mod(rows,shrink_size),...
        shrink_size-mod(cols,shrink_size)],'post');
end

[rows,cols] = size(input);


kernel = ones(shrink_size);
kernel = kernel./sum(kernel(:));
output = conv2(input,kernel);
output = output(shrink_size:shrink_size:rows,shrink_size:shrink_size:cols);

% output = zeros(rows/shrink_size, cols/shrink_size);
% for i = 1:size(output,1)
%     for j = 1:size(output,2)
%         block = input((i-1)*shrink_size+1:...
%             (i-1)*shrink_size+shrink_size,...
%             (j-1)*shrink_size+1:...
%             (j-1)*shrink_size+shrink_size);
%         output(i,j) = mean(block(:));
%     end
% end


end

