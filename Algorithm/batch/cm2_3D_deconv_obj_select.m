function cm2_3D_deconv_obj_select(tau_tv,tau_l1, obj_idx)
linear_normalize = @(x) (x - min(x(:)))./(max(x(:))-min(x(:)));

save_folder = ['save/'];
load(['expt_1013.mat']);

y = single(ys(:,:,obj_idx));
[rows,cols] = size(y);
para.tau_tv = 10^(-1*tau_tv);
para.tau_l1 = 10^(-1*tau_l1);

para.maxiter = 128;
para.display_flag = 0;
para.img_save_period = 120;
para.img_save_path = ['save/temp/tv_',num2str(tau_tv),'_l1_',num2str(tau_l1),'_obj_',num2str(obj_idx)];
disp(['tau tv: ',num2str(tau_tv),', tau l1:',num2str(tau_l1),', obj:',num2str(obj_idx)]);
xhat = ADMM_LSI_deconv_3D(y,psfs,para);
xhat = uint8(255*linear_normalize(xhat));
xhat = xhat(rows/2+1:rows/2+rows, cols/2+1:cols/2+cols, :);
write_mat_to_tif(xhat,[save_folder,'tv_',num2str(tau_tv),'_l1_',num2str(tau_l1),'_obj_',num2str(obj_idx),'.tif']);

end