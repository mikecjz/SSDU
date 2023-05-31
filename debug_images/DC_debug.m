load('/home/jc_350/zfan0804_712/Zhehao/Accelerated-VWI/Patient-017-CE/DL_data/Slice_138.mat')
% load('/home/jc_350/zfan0804_712/Zhehao/Accelerated-VWI/Patient-017-CE/DL_data/Slice_52.mat')
% load('/home/jc_350/zfan0804_712/Zhehao/Accelerated-VWI/Patient-028/DL_data/Slice_138.mat')
load('trn_loss_masks.mat')

HOMEDIR = getenv('HOME');
BART_dir = fullfile(HOMEDIR,'BART/bart-0.7.00');
working_dir = pwd;
cd(BART_dir);
startup;%Bart startup
cd(working_dir);
sizes = size(MC_kspace_slice);

% kspace_slice = reshape(MC_kspace_slice,1,sizes(1), sizes(2), sizes(3));
% calibs  = bart('ecalib','-r 24 -m1',kspace_slice);

MC_kspace_slice = MC_kspace_slice.* abs(trn_mask);

SC_image = sense1(MC_kspace_slice, SE);

SC_image_EhE = EhE_Op(SC_image, SE, trn_mask);
SC_image_EhE2 = EhE_Op(SC_image_EhE, SE, trn_mask);

calibs = reshape(SE, 1,sizes(1), sizes(2), sizes(3),[]);
kspace_slice = reshape(MC_kspace_slice,1,sizes(1), sizes(2), sizes(3));
SC_BART_SENSE =  bart('pics', kspace_slice, calibs);
SC_BART_SENSE = squeeze(SC_BART_SENSE);
calibs = squeeze(calibs);

%%
SC_CG_SENSE = CG_SENSE(MC_kspace_slice, calibs, trn_mask, 15);

%%
SC_CG_SENSE_iteration = zeros([size(SC_CG_SENSE),30], 'like', 1i);
for i = 1:30
    SC_CG_SENSE_iteration(:,:,:,i) = CG_SENSE(MC_kspace_slice, calibs, trn_mask, i);
end
%%
function SC_image = sense1(MC_kspace_slice, SE)
MC_image = fftshift(fftshift(ifft(ifft(ifftshift(ifftshift(MC_kspace_slice,1),2),[],1),[],2),1),2);

SC_image = squeeze(sum(MC_image.* conj(SE),3));
end

function SC_image = EhE_Op(SC_image, SE, mask)
sizes = size(SC_image);
SC_image = reshape(SC_image, sizes(1), sizes(2), 1, []);

MC_image = SC_image .* SE;
MC_kspace_slice = ifftshift(ifftshift(fft(fft(fftshift(fftshift(MC_image,1),2),[],1),[],2),1),2);

MC_kspace_slice = sum(MC_kspace_slice, 4);

MC_kspace_slice = MC_kspace_slice .* abs(mask);

SC_image = sense1(MC_kspace_slice, SE);

end

function SC_image = CG_SENSE(MC_kspace_slice, SE, mask, max_itr)
%Initialization
sizes = size(SE);
nCoil = sizes(3);
image_sizes = sizes;
image_sizes(3) = [];

v0 = zeros(image_sizes, 'like', 1i);
b = sense1(MC_kspace_slice, SE);
r0 = b-EhE_Op(v0, SE, mask);
p0 = r0;
r = r0;
v = v0;
alpha = [];
beta = [];
p = p0;

%Conjugate gradient descent loop
for i = 1:max_itr
    disp(['Loop iteration: ', num2str(i)])
    
    %
    sigma = sum(r.*conj(r),'all') ./ sum(r0.*conj(r0),'all');
    
    %
    Ap = EhE_Op(p, SE, mask);
    alpha = sum(r.*conj(r),'all') ./ sum(Ap.*conj(p),'all');
    
    %
    v = v + alpha * p;
    
    %
    r_old = r;
    r = r - alpha * Ap;
    
    %
    beta = sum(r.*conj(r),'all') ./ sum(r_old.*conj(r_old),'all');
    
    %
    p = r + beta * p;
    
end

SC_image = v;

end


