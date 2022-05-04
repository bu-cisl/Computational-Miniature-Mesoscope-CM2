% read tiff file to mat
function y = read_tif_to_mat(filename, verbose)

nz = size(imfinfo(filename),1);

for i = 1:nz
   y(:,:,i) = imread(filename,i); 
   if verbose
       disp([num2str(i),'/',num2str(nz)]);
   end
end

end
