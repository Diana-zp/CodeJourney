data = table2array(readtable('测试数据.csv'));
t = data(:,1);
Ps = data(:,2)./1000;
Pt = data(:,3)./1000;
Tt = data(:,4);
Hp = data(:,5);
Vi = data(:,6);
Vt = data(:,7);
M = data(:,8);
Vh = data(:,9);
Ts = data(:,10);



secs = seconds(t);
simin_Ps = timetable(secs, Ps+sqrt(0)*randn(size(Ps)));
simin_Pt = timetable(secs, Pt+sqrt(0)*randn(size(Pt)));
simin_Tt = timetable(secs, Tt+sqrt(0)*randn(size(Tt)));

% 运行模型main.slx, 输出结果到工作区位out
load_system('main.slx'); 
out = sim('main');
% 仿真结果
Hp_sim = out.simout_Hp.Data;
Vi_sim = out.simout1_Vi.Data;
Vt_sim = out.simout_Vt.Data;
Ma_sim = out.simout_Ma.Data;
Vh_sim = out.simout_Vh.Data;
Ts_sim = out.simout_Ts.Data;
% 获取时间向量
t_sim = out.tout;

% 气压高度比较
figure;
plot(t, Hp, 'b', t_sim, Hp_sim, 'r');
legend('Real Hp', 'Simulated Hp');
title("气压高度比较");
xlabel('Time (s)');
ylabel('气压高度 (m)');
saveas(gcf, 'result/plot/with_kalman_Hp.png');

% 指示空速比较
figure;
plot(t, Vi, 'b', t_sim, Vi_sim, 'r');
legend('Real Vi', 'Simulated Vi');
title("指示空速比较");
xlabel('Time (s)');
ylabel('指示空速 (m/s)');
saveas(gcf, 'result/plot/with_kalman_Vi.png');

% 真空速比较
figure;
plot(t, Vt, 'b', t_sim, Vt_sim, 'r');
legend('Real Vt', 'Simulated Vt');
title("真空速比较");
xlabel('Time (s)');
ylabel('真空速 (m/s)');
saveas(gcf, 'result/plot/with_kalman_Vt.png');

% 马赫数比较
figure;
plot(t, M, 'b', t_sim, Ma_sim, 'r');
legend('Real M', 'Simulated M');
title("马赫数比较");
xlabel('Time (s)');
ylabel('Mach Number');
saveas(gcf, 'result/plot/with_kalman_M.png');

% 垂直速度比较
figure;
plot(t, Vh, 'b', t_sim, Vh_sim, 'r');
legend('Real Vh', 'Simulated Vh');
title("垂直速度比较");
xlabel('Time (s)');
ylabel('Vertical Speed (m/s)');
saveas(gcf, 'result/plot/with_kalman_Vh.png');

% 大气静温比较
figure;
plot(t, Ts, 'b', t_sim, Ts_sim, 'r');
legend('Real Ts', 'Simulated Ts');
title("大气静温比较");
xlabel('Time (s)');
ylabel('Static Air Temperature (K)');
saveas(gcf, 'result/plot/with_kalman_Ts.png');

% 创建表格来存储所有的统计信息
stats_table = table('Size',[5 5], 'VariableTypes',{'string','double','double','double','double'}, ...
    'VariableNames',{'参数', 'RMSE', 'MAE', 'MAPE','R2'});

parameters = {'气压高度', '指示空速', '真空速', '马赫数', '垂直速度', '大气静温'};
data_pairs = {[Hp, Hp_sim]; [Vi, Vi_sim]; [Vt, Vt_sim]; [M, Ma_sim]; [Vh, Vh_sim]; [Ts, Ts_sim]};
for i = 1:length(parameters)
    [rmse, mae, mape, r2] = calculate_stats(data_pairs{i}(:,1), data_pairs{i}(:,2));
    stats_table(i,:) = {parameters{i}, rmse, mae,mape, r2};
end

writetable(stats_table, 'Comparison_Statistics_with_kalman.csv');

time_series_table = table(t, t_sim, ...
    Hp, Hp_sim, ...
    Vi, Vi_sim, ...
    Vt, Vt_sim, ...
    M, Ma_sim, ...
    Vh, Vh_sim, ...
    Ts, Ts_sim, ...
     'VariableNames', {'时间', '仿真时间', ...
                      '真实压高度', '仿真气压高度', ...
                      '真实指示空速', '仿真指示空速', ...
                      '真实真空速', '仿真真空速', ...
                      '真实马赫数', '仿真马赫数', ...
                      '真实垂直速度', '仿真垂直速度', ...
                      '真实大气静温', '仿真大气静温'});

writetable(time_series_table, 'Time_Series_Comparison_with_kalman.csv');

function [rmse, mae, mape, r2] = calculate_stats(real_data, sim_data)
    % 均方根误差 (RMSE)
    rmse = sqrt(mean((real_data - sim_data).^2));
    
    % 平均绝对误差 (MAE)
    mae = mean(abs(real_data - sim_data));
    % 决定系数 (R2)
    mdl = fitlm(real_data, sim_data);
    r2 = mdl.Rsquared.Ordinary;

    valid_indices = real_data ~= 0;
  
    if ~any(valid_indices)
        warning('All real data values are zero, MAPE cannot be computed.');
        mape = NaN;
        return;
    end
    % 平均绝对百分比误差 (MAPE)，仅在有效索引上操作
    mape = mean(abs((real_data(valid_indices) - sim_data(valid_indices)) ./ real_data(valid_indices))) * 100; % 转换为百分比    
end