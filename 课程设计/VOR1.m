clear; clc;
%% VOR等效基带仿真参数
f_subcarrier = 9960;     % 子载波频率 (9960 Hz)
f_ref = 30;              % 参考信号频率 (30 Hz)
deviation = 480;         % 频偏 (480 Hz)
% 基带仿真参数
fs = 50e3;               % 采样频率 (50 kHz)
t_total = 10;            % 总模拟时间延长到10秒
t = 0:1/fs:t_total-1/fs; % 时间序列(避免长度问题)
true_bearing = 50;       % 固定真实方位角 (度)

%% 等效基带信号生成
ref_30Hz = sin(2*pi*f_ref*t); 
ref_fm_sub = fmmod(ref_30Hz, f_subcarrier, fs, deviation);

% 可变信号: 固定方位角
var_am = 0.3 * sin(2*pi*f_ref*t + deg2rad(true_bearing));

% 合成基带VOR信号
tx_baseband = ref_fm_sub + var_am;
figure
plot(t,tx_baseband);
xlabel('时间(s)');
ylabel('幅度');
title('基带VOR信号')
%% 信道传输模拟
snr = 10;                % 信噪比(dB)
rx_baseband = awgn(tx_baseband, snr, 'measured');

%% 接收机处理
% 分离参考信号
[b_bpf9960, a_bpf9960] = butter(4, [9800 10100]/(fs/2), 'bandpass');
ref_subcarrier = filtfilt(b_bpf9960, a_bpf9960, rx_baseband);

% FM解调
ref_30Hz_demod = fmdemod(ref_subcarrier, f_subcarrier, fs, deviation);

% 提取可变信号
[b_lpf, a_lpf] = butter(4, 40/(fs/2));
var_30Hz_demod = filtfilt(b_lpf, a_lpf, rx_baseband);

%% 30Hz信号提纯
n = 100;                 % 滤波器阶数
f_pass = [28, 32];       % 通带频率
b30 = fir1(n, 2*f_pass/fs, 'bandpass');

ref_30Hz_filt = filtfilt(b30, 1, ref_30Hz_demod);
var_30Hz_filt = filtfilt(b30, 1, var_30Hz_demod);

% 归一化信号
ref_30Hz_filt = ref_30Hz_filt / max(abs(ref_30Hz_filt));
var_30Hz_filt = var_30Hz_filt / max(abs(var_30Hz_filt));

figure
L=round(length(t)/7);
plot(t(1:L),ref_30Hz_filt(1:L),'LineWidth', 2);
hold on;
plot(t(1:L),var_30Hz_filt(1:L),'r--', 'LineWidth', 1.5);
xlabel('时间(s)');
ylabel('幅度');
title('接收到的可变相位信号和基准相位信号');
legend('基准相位信号','可变相位信号');


period_samples = round(fs / f_ref);  
num_periods = floor(length(t) / period_samples);  


period_bearings = zeros(1, num_periods);
period_times = zeros(1, num_periods);

for i = 1:num_periods
    % 提取当前周期数据
    start_idx = (i-1)*period_samples + 1;
    end_idx = i*period_samples;
    idx_range = start_idx:end_idx;
    
    ref_segment = ref_30Hz_filt(idx_range);
    var_segment = var_30Hz_filt(idx_range);
    
    % 获取解析信号
    ref_analytic = hilbert(ref_segment);
    var_analytic = hilbert(var_segment);
    
    phase_diff = angle(mean(var_analytic .* conj(ref_analytic)));
        
    % 计算当前周期方位角
    bearing = mod(rad2deg(phase_diff), 360);
    period_bearings(i) = bearing;
    period_times(i) = mean(t(idx_range));
end

mean_vector = mean(exp(1i*deg2rad(period_bearings)));
final_bearing = mod(rad2deg(angle(mean_vector)), 360);


%% 结果可视化

% 方位角跟踪结果
figure
plot(period_times, repmat(true_bearing, size(period_times)), 'LineWidth', 2);
hold on;
plot(period_times, period_bearings, 'o', 'MarkerSize', 4);
plot([0 t_total], [final_bearing final_bearing], 'r--', 'LineWidth', 1.5);
title(['VOR方位角估计 (真实: ' num2str(true_bearing) '°, 估计: ' num2str(round(final_bearing,1)) '°)']);
xlabel('时间 (s)');
ylabel('方位角 (度)');
legend('真实方位角', '周期测量值', '平均结果', 'Location', 'best');
grid on;
ylim([0, 360]);





