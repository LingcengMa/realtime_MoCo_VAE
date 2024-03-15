clc;
load binning_GT.mat
load realtime_GT_imgs.mat
U_init_matrix = reshape(U_init,[],48);
Img = U_init_matrix * Phi_rt_init(:,1:1000);
Img = reshape(Img,320,320,1000);
Max = max(abs(Img(:)));
U = U_init_matrix' * U_init_matrix;
[~,S,V] = svd(U_init_matrix,"econ");
M = S*V';
M = pinv(M);
Phi_rt_feature = S*V'*Phi_rt_init;
save("M.mat", "M")
save("feature.mat", "Phi_rt_feature")
% plot(real(Phi_rt_feature(1,:)));
% plot(real(Phi_rt_init(1,:)));
Segidx = mod((1:12600)-1,360)+1; %inversion time input
Segidx = Segidx';


% imshow(U);
% implay(abs(Img)/Max)

fft_result = fft(Segidx);
n = length(Segidx);  % Length of the signal
sampling_rate = 2.5; % Replace with your actual sampling rate
freq = fftshift(fftfreq(n, 1/sampling_rate));

figure;
plot(freq, abs(fft_result));
xlabel('Frequency (Hz)');
ylabel('Magnitude');
title('FFT of Signal');
grid on;
