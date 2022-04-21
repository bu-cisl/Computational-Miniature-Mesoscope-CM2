function y = cm2_forward_gpu(x,psfs,verbose)
F2D = @(x) fftshift(fft2(ifftshift(x)));
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));
pad2d = @(x) padarray(x,0.5*size(x));
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);
conv2d = @(obj,psf) crop2d(real(Ft2D(F2D(pad2d(obj)).*F2D(pad2d(psf)))));
% 
% t0 = tic;
% x_gpu = gpuArray(x);
% psfs_gpu = gpuArray(psfs);
% tn = toc;
% disp(['data transfer takes: ',num2str(tn-t0), ' sec']);

y = zeros(size(x,1),size(x,2));
for i = 1:size(x,3)
    y = y + conv2d(gpuArray(x(:,:,i)), gpuArray(psfs(:,:,i)));
    if verbose
        disp(i);
    end
end

end

