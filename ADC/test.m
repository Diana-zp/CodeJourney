
% load('processed_data.mat');
% secs = seconds(t);
% simin_Ps = timetable(secs, Ps_noise);
% simin_Pt = timetable(secs, Pt_noise);
% simin_Tt = timetable(secs, Tt_noise);
% 
% load_system('testmodel.slx'); 
% out = sim('testmodel');
% Ps_sim = out.simout.Data;
% Pt_sim = out.simout1.Data;
% Tt_sim = out.simout2.Data;

% %testmodel只是产生3个窄带高斯白噪声，功率分别为0.3,0.2,0.4，与下面噪声相对应
% load_system('testmodel.slx'); 
% out = sim('testmodel');
% noise1 = out.simout3.Data;
% noise2 = out.simout4.Data;
% noise3 = out.simout5.Data;
% noise_band=[noise1(:)';noise2(:)';noise3(:)'];
% noise_randn=[sqrt(0.3),0,0;0,sqrt(0.4),0;0,0,sqrt(0.2)]*randn(size(noise_band));








% % 假设你有一个离散的信号向量 signal
% % 例如，这里我们创建一个简单的离散时间信号
% 
% 
% signal=Hp_sim(1:end-1);
% L=size(signal,1);
% Fs=10;
% T=1/Fs;
% t=T*(0:L-1);
% %signal=filter(LP,signal);


% 参数设置
num_points = 20001;  % 信号总时间点
sampling_time = 0.1; % 采样时间 (s)
total_time = num_points * sampling_time; % 总时间 (s)

% 生成时间轴
t = linspace(0, total_time, num_points);
% 生成信号 (例如，一个简单的正弦波)
signal_freq = 3; % 信号频率 (Hz)
signal = Hp_sim(1:end-1);

% 生成噪声 (高斯噪声)
noise = 0.5 * randn(1, num_points); % 均值为0，标准差为0.5

% 带噪声的信号
noisy_signal = signal ;
% 计算FFT
signal_fft = fft(noisy_signal);

% 计算频率轴
frequencies = (0:num_points-1) * (1 / sampling_time) / num_points;

% 只取正频率部分
positive_freq = frequencies(1:floor(num_points/2));
positive_fft = 2.0 / num_points * abs(signal_fft(1:floor(num_points/2)));

% 绘制频谱图
figure;
plot(positive_freq, positive_fft);
title('Frequency Spectrum of Noisy Signal');
xlabel('Frequency (Hz)');
ylabel('Amplitude');
grid on;







