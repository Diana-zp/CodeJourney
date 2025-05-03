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

f1=0.009;
f2=0.007;
f3=0.008;
size1=size(Ps);
var1=sqrt(0.3.*(1+1/2.*sin(2*pi*f1*t)));
var2=sqrt(0.2.*(1+1/2.*sin(2*pi*f3*t)));
var3=sqrt(0.4.*(1+1/2.*sin(2*pi*f3*t)));

noise1=var1.*randn(size1);
noise2=var2.*randn(size1);
noise3=var3.*randn(size1);

secs = seconds(t);
simin_Ps = timetable(secs, Ps+noise1);
simin_Pt = timetable(secs, Pt+noise2);
simin_Tt = timetable(secs, Tt+noise3);

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

cov1=var1.^2;
cov2=var2.^2;
cov3=var3.^2;
cov1_sim=zeros(size1);
cov2_sim=zeros(size1);
cov3_sim=zeros(size1);

R=out.simout1_R.Data;

for k=1:length(t)
    cov1_sim(k)=R(1,1,k);
    cov2_sim(k)=R(2,2,k);
    cov3_sim(k)=R(3,3,k);
end






% 绘制 cov1 和 cov1_sim 的比较图
figure;
plot(t, cov1, 'b', 'LineWidth', 1.5); % 真实值，蓝色
hold on;
plot(t, cov1_sim, 'r--', 'LineWidth', 1.5); % 仿真值，红色虚线
xlabel('时间 (s)');
ylabel('方差');
title('cov1 与 cov1\_sim 的比较');
legend('真实值', '仿真值');
grid on;
saveas(gcf, 'cov1_comparison.png'); % 保存图像

% 绘制 cov2 和 cov2_sim 的比较图
figure;
plot(t, cov2, 'b', 'LineWidth', 1.5); % 真实值，蓝色
hold on;
plot(t, cov2_sim, 'r--', 'LineWidth', 1.5); % 仿真值，红色虚线
xlabel('时间 (s)');
ylabel('方差');
title('cov2 与 cov2\_sim 的比较');
legend('真实值', '仿真值');
grid on;
saveas(gcf, 'cov2_comparison.png'); % 保存图像

% 绘制 cov3 和 cov3_sim 的比较图
figure;
plot(t, cov3, 'b', 'LineWidth', 1.5); % 真实值，蓝色
hold on;
plot(t, cov3_sim, 'r--', 'LineWidth', 1.5); % 仿真值，红色虚线
xlabel('时间 (s)');
ylabel('方差');
title('cov3 与 cov3\_sim 的比较');
legend('真实值', '仿真值');
grid on;
saveas(gcf, 'cov3_comparison.png'); % 保存图像




% 创建表格来存储所有的统计信息
stats_table = table('Size',[5 5], 'VariableTypes',{'string','double','double','double','double'}, ...
    'VariableNames',{'参数', 'RMSE', 'MAE', 'MAPE','R2'});

parameters = {'气压高度', '指示空速', '真空速', '马赫数', '垂直速度', '大气静温'};
data_pairs = {[Hp, Hp_sim]; [Vi, Vi_sim]; [Vt, Vt_sim]; [M, Ma_sim]; [Vh, Vh_sim]; [Ts, Ts_sim]};
for i = 1:length(parameters)
    [rmse, mae, mape, r2] = calculate_stats(data_pairs{i}(:,1), data_pairs{i}(:,2));
    stats_table(i,:) = {parameters{i}, rmse, mae,mape, r2};
end

writetable(stats_table, 'Comparison_Statistics_time_varying_noise.csv');

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

writetable(time_series_table, 'Time_Series_Comparisontime_varying_noise.csv');

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
