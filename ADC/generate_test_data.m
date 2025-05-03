L=20001;
Fs=0.1;
t=Fs.*(0:L-1)';
freq_Vh = 0.01; 
freq_V = 0.015; 
Vh=6.25+2.*cos(2*pi*freq_Vh.*t);
% 计算 Vh 的数值积分 H，初始值为 0
H = cumtrapz(t, Vh);
V=220+40.*cos(2*pi*freq_Vh.*t);
secs = seconds(t);
simin_H = timetable(secs, H);
simin_V = timetable(secs, V);
% 运行模型
load_system('generate_test_data_model.slx'); 
out_test_data = sim('generate_test_data_model.slx');
Ps = out_test_data.simout_Ps.Data;
Pt=out_test_data.simout_Pt.Data;
Tt=out_test_data.simout_Tt.Data;
Ma=out_test_data.simout_Ma.Data;
Ts=out_test_data.simout_Ts.Data;
[Hp, Vi, Vt, M, Vh2, Ts2] = ADC11(Ps./1000, Pt./1000, Tt, 0, t);

T=table(t,Ps,Pt,Tt,H,Vi,V,Ma,Vh,Ts,'Variablename',{'时间','静压','总压','总温','气压高度','指示空速','真空速','马赫数','垂直速度','大气静温'});
writetable(T,'测试数据.csv');

function [Hp, Vi, Vt, M, Vh, Ts] = ADC11(Ps, Pt, Tt, h0, t)%单位KPa,K,m
    P0 = 101.325;
    T0=288.15;
    % Hp计算
    Hp = zeros(length(t), 1);
    for i = 1:length(Hp)
        if i == 1
            flag = h0;
        else
            flag = Hp(i - 1);
        end
        if flag < 11000
            Hp(i) = 44330.7216 * (1 - (Ps(i) / P0)^(0.19026));
        elseif flag >= 11000 && flag < 20000
            Hp(i) = 11000 + 6337.22 * log(22.632 / Ps(i));
        elseif flag >= 20000 && flag < 32000
            Hp(i) = 22000 - 216650 * (1 - (Ps(i) / 5.5293)^(-0.029271));
        end
    end
    Pq = Pt - Ps;
    %注意单位，m/s
    Vi = 1225.08 .* sqrt(5.* ((1 + Pq ./ P0).^(2/7) - 1))./3.6;
    M = (5 * ((Pt ./ Ps).^(2/7) - 1)).^0.5;
    if length(t) > 1
        Vh = gradient(Hp, t);
    else
        Vh = 0; 
    end
    Ts = Tt ./ (1 + 0.2 * M.^2);
    Vt=Vi.*sqrt(P0./Ps.*Ts./T0);
end
