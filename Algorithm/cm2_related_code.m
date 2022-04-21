%% CM2 related code on forward simulation and 3d reconstruction

%% useful functions
VEC = @(x) x(:);
clip = @(x, vmin, vmax) max(min(x, vmax), vmin);
F2D = @(x) fftshift(fft2(ifftshift(x)));
Ft2D = @(x) fftshift(ifft2(ifftshift(x)));
pad2d = @(x) padarray(x,0.5*size(x));
crop2d = @(x) x(1+size(x,1)/4:size(x,1)/4*3,1+size(x,2)/4:size(x,2)/4*3);
conv2d = @(obj,psf) crop2d(real(Ft2D(F2D(pad2d(obj)).*F2D(pad2d(psf)))));
unit_norm = @(x) x./norm(x(:));
auto_corr = @(x) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(x))))));
x_corr = @(x,y) crop2d(Ft2D(F2D(pad2d(unit_norm(x))).*conj(F2D(pad2d(unit_norm(y))))));
linear_normalize = @(x) (x - min(x(:)))./(max(x(:))-min(x(:)));

%% load 3d psfs data
load('data/downsampled_3d_psfs.mat');

%% down sample to further reduce the scale
psfs = psfs_ds(:,:,81:320); 
clear psfs_ds
psfs_tmp = zeros(324,432,16);
for i = 0:15
    tmp = psfs(:,:,i*15+1:i*15+15);
    tmp = sum(tmp,3);
    tmp = average_shrink(tmp,3);
    psfs_tmp(:,:,i+1) = tmp;
end
psfs = psfs_tmp;
clear psfs_tmp

% psfs is used in forward model. it has very small background values so
% that can be used to mimic the background; psfs_recon has all zeros on the
% background, which is necessary for reconstruction
[rows,cols,depth] = size(psfs);
psfs_recon = psfs;
psfs_recon(psfs_recon<=1.5e-4) = 0;
for i = 1:depth
    tmp = psfs(:,:,i);
    tmp = tmp ./ sum(tmp(:));
    psfs(:,:,i) = tmp;
    tmp = psfs_recon(:,:,i);
    tmp = tmp ./ sum(tmp(:));
    psfs_recon(:,:,i) = tmp;
end

%% synthesize a single 2D measurement by depth-wise convolution between
% PSFs and objects (here it is a volume of 1.5pixel radius spheres)
% sbr = 1.2; % signal to background ratio
[xx,yy] = meshgrid([-cols/2:1:cols/2-1], [-rows/2:1:rows/2-1]);%% coordinates of full fov in pixel
[sx,sy,sz] = meshgrid([-17:17].*(1/5), [-17:17].*(1/5), [-1:1]);%% coordinates of a small sphere volume
num_particles = 200; 
current_radius = 1.5; % radius of sphere: 1.5 pixels
% to fine sampling the sphere
sphere = zeros(7*5,7*5,3); % generate a dense sphere volume
sphere(sqrt(sx.^2+sy.^2+sz.^2)<=current_radius) = 1;
sphere2 = zeros(7,7,3);
for i = 1:3
    sphere2(:,:,i) = average_shrink(sphere(:,:,i),5);
end
sphere = sphere2;
sphere = sum(sphere,3);
clear sphere2

gt_volume = zeros(rows,cols,depth);
gt_locations = zeros(num_particles,4);
std_lum = 0.3;
for i = 1:num_particles

    rr = randi([7,rows-7]);
    cc = randi([7,cols-7]);
    
    while true 
        rr = randi([7,rows-7]);
        cc = randi([7,cols-7]);
        if sqrt(xx(rr,cc).^2 + yy(rr,cc).^2) <= 100
            break
        end
    end

    zz = randi([2, depth-2]);
    gt_locations(i,1) = rr;
    gt_locations(i,2) = cc;
    gt_locations(i,3) = zz;

    tmp_lum = 1 + std_lum*randn(1);
    tmp_lum = clip(tmp_lum, 0,1.5);
    gt_locations(i,4) = tmp_lum;
    gt_volume(rr-3:rr+3,cc-3:cc+3,zz)=gt_volume(rr-3:rr+3,cc-3:cc+3,zz)+tmp_lum.*sphere;
end
% 
% gt_volume_bg = zeros(rows*2,cols*2,depth);
% num_bg = 50000000;
% rr = randi([1,rows*2],[num_bg,1]);
% cc = randi([1, cols*2], [num_bg,1]);
% zz = randi([1,depth], [num_bg,1]);
% tmp_lum = clip(1 + 0.1*randn([num_bg,1]), 0,1.5);
% 
% for i = 1:num_bg
%     gt_volume_bg(rr(i), cc(i), zz(i)) = gt_volume_bg(rr(i), cc(i), zz(i)) + tmp_lum(i);
% end

y_part1 = gather(cm2_forward_gpu(gt_volume,psfs,false)); % can change to CPU version
% y_part2 = crop2d(gather(cm2_forward_bg_gpu(gt_volume_bg,psfs,false)));
% 
% mean_signal = mean(mean(y_part1(y_part1>=0.1)));
% mean_bg = mean(y_part2(:));
% factor = mean_signal/(sbr-1)/mean_bg;
% 
% y = y_part1 + factor*y_part2;

y = y_part1;

y = y./max(y(:));
y = poissrnd(y.*1000)/1000;
noise_std = 0.02;
y = y + noise_std*randn(size(y));
y = clip(y,0,1);

%% reconstruction
psfs_recon = single(psfs_recon);
y = single(y);
y_bgsub = single(bg_removal(y,4));

para = [];
para.mu1 = 1;
para.mu2 = 1;
para.mu3 = 1;
para.mu4 = 1;
para.clip_min = 0;
para.clip_max = 100;
para.color = jet(256);
para.tau_l1 = 0.03;
para.tau_tv = 0.01;
para.rtol = 1.5;
para.mu_ratio = 1.1;
para.display_flag = 1;
para.termination_ratio = 0.01;
para.plateau_tolerence = 4;
para.maxiter = 128;
para.img_save_period = 80;
para.img_save_path = '';
xhat = ADMM_LSI_deconv_3D(y_bgsub,psfs_recon,para); % it save result as tif

%% save gt vol as tif
xhat(xhat<=0)=0;
xhat = xhat(rows/2+1:rows/2+rows,cols/2+1:cols/2+cols,end:-1:1); % need to reverse z axis to match to gt
xhat = uint8(255*linear_normalize(xhat));
write_mat_to_tif(xhat, 'xhat_cropped.tif');
write_mat_to_tif(uint8(255*linear_normalize(gt_volume)), 'gt_vol.tif');














