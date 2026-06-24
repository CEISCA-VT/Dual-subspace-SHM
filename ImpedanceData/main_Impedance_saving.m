clc;
clear;
close all;
tic

% coluna 1: frequencia;
% coluna 2: módulo;
% coluna 3: parte real;
% coluna 4: parte imaginária;

%% Temperature of 24 ºC  

disp('Temperature of 24 ºC - Healthy')
load 'Features_Data\t24_sf_1.lvm'
freq_24H = t24_sf_1(:,1);       
real_24H = t24_sf_1(:,3);       
imag_24H = t24_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI24H.mat freq_24H real_24H imag_24H
clear

disp('Temperature of 24 ºC - Damage 1')
load 'Features_Data\t24_cf1_1.lvm'

freq_24D1 = t24_cf1_1(:,1);       
real_24D1 = t24_cf1_1(:,3);       
imag_24D1 = t24_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T24degrees\EMI24D1.mat freq_24D1 real_24D1 imag_24D1
clear

disp('Temperature of 24 ºC - Damage 2')
load 'Features_Data\t24_cf2_5.lvm'

freq_24D2 = t24_cf2_5(:,1);       
real_24D2 = t24_cf2_5(:,3);       
imag_24D2 = t24_cf2_5(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T24degrees\EMI24D2.mat freq_24D2 real_24D2 imag_24D2
clear

disp('Temperature of 24 ºC - Damage 3')
load 'Features_Data\t24_cc1_1.lvm'

freq_24D3 = t24_cc1_1(:,1);       
real_24D3 = t24_cc1_1(:,3);       
imag_24D3 = t24_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T24degrees\EMI24D3.mat freq_24D3 real_24D3 imag_24D3
clear

disp('Temperature of 24 ºC - Damage 4')
load 'Features_Data\t24_cc2_1.lvm'

freq_24D4 = t24_cc2_1(:,1);       
real_24D4 = t24_cc2_1(:,3);       
imag_24D4 = t24_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T24degrees\EMI24D4.mat freq_24D4 real_24D4 imag_24D4
clear

%% Temperature of 40 ºC  

disp('Temperature of 40 ºC - Healthy')
load 'Features_Data\t40_sf_1.lvm'
freq_40H = t40_sf_1(:,1);       
real_40H = t40_sf_1(:,3);       
imag_40H = t40_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI40H.mat freq_40H real_40H imag_40H
clear

disp('Temperature of 40 ºC - Damage 1')
load 'Features_Data\t40_cf1_1.lvm'
freq_40D1 = t40_cf1_1(:,1);       
real_40D1 = t40_cf1_1(:,3);       
imag_40D1 = t40_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T40degrees\EMI40D1.mat freq_40D1 real_40D1 imag_40D1
clear

disp('Temperature of 40 ºC - Damage 2')
load 'Features_Data\t40_cf2_1.lvm'
freq_40D2 = t40_cf2_1(:,1);       
real_40D2 = t40_cf2_1(:,3);       
imag_40D2 = t40_cf2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T40degrees\EMI40D2.mat freq_40D2 real_40D2 imag_40D2
clear

disp('Temperature of 40 ºC - Damage 3')
load 'Features_Data\t40_cc1_1.lvm'
freq_40D3 = t40_cc1_1(:,1);       
real_40D3 = t40_cc1_1(:,3);       
imag_40D3 = t40_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T40degrees\EMI40D3.mat freq_40D3 real_40D3 imag_40D3
clear

disp('Temperature of 40 ºC - Damage 4')
load 'Features_Data\t40_cc2_1.lvm'
freq_40D4 = t40_cc2_1(:,1);       
real_40D4 = t40_cc2_1(:,3);       
imag_40D4 = t40_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T40degrees\EMI40D4.mat freq_40D4 real_40D4 imag_40D4
clear

%% Temperature of 55 ºC  
disp('Temperature of 55 ºC - Healthy')
load 'Features_Data\t55_sf_1.lvm';
freq_55H = t55_sf_1(:,1);       
real_55H = t55_sf_1(:,3);       
imag_55H = t55_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI55H.mat freq_55H real_55H imag_55H
clear

disp('Temperature of 55 ºC - Damage 1')
load 'Features_Data\t55_cf1_1.lvm';
freq_55D1 = t55_cf1_1(:,1);       
real_55D1 = t55_cf1_1(:,3);       
imag_55D1 = t55_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T55degrees\EMI55D1.mat freq_55D1 real_55D1 imag_55D1
clear

disp('Temperature of 55 ºC - Damage 2')
load 'Features_Data\t55_cf2_1.lvm';
freq_55D2 = t55_cf2_1(:,1);       
real_55D2 = t55_cf2_1(:,3);       
imag_55D2 = t55_cf2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T55degrees\EMI55D2.mat freq_55D2 real_55D2 imag_55D2
clear

disp('Temperature of 55 ºC - Damage 3')
load 'Features_Data\t55_cc1_1.lvm'
freq_55D3 = t55_cc1_1(:,1);       
real_55D3 = t55_cc1_1(:,3);       
imag_55D3 = t55_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T55degrees\EMI55D3.mat freq_55D3 real_55D3 imag_55D3
clear

disp('Temperature of 55 ºC - Damage 3')
load 'Features_Data\t55_cc2_1.lvm'
freq_55D4 = t55_cc2_1(:,1);       
real_55D4 = t55_cc2_1(:,3);       
imag_55D4 = t55_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T55degrees\EMI55D4.mat freq_55D4 real_55D4 imag_55D4
clear

%% Temperature of 70 ºC  
disp('Temperature of 70 ºC - Healthy')
load 'Features_Data\t70_sf_1.lvm';
freq_70H = t70_sf_1(:,1);       
real_70H = t70_sf_1(:,3);       
imag_70H = t70_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI70H.mat freq_70H real_70H imag_70H
clear

disp('Temperature of 70 ºC - Damage 1')
load 'Features_Data\t70_cf1_1.lvm';
freq_70D1 = t70_cf1_1(:,1);       
real_70D1 = t70_cf1_1(:,3);       
imag_70D1 = t70_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T70degrees\EMI70D1.mat freq_70D1 real_70D1 imag_70D1
clear

disp('Temperature of 70 ºC - Damage 2')
load 'Features_Data\t70_cf2_1.lvm';
freq_70D2 = t70_cf2_1(:,1);       
real_70D2 = t70_cf2_1(:,3);       
imag_70D2 = t70_cf2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T70degrees\EMI70D2.mat freq_70D2 real_70D2 imag_70D2
clear

disp('Temperature of 70 ºC - Damage 3')
load 'Features_Data\t70_cc1_1.lvm'
freq_70D3 = t70_cc1_1(:,1);       
real_70D3 = t70_cc1_1(:,3);       
imag_70D3 = t70_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T70degrees\EMI70D3.mat freq_70D3 real_70D3 imag_70D3
clear

disp('Temperature of 70 ºC - Damage 4')
load 'Features_Data\t70_cc2_1.lvm'
freq_70D4 = t70_cc2_1(:,1);       
real_70D4 = t70_cc2_1(:,3);       
imag_70D4 = t70_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T70degrees\EMI70D4.mat freq_70D4 real_70D4 imag_70D4
clear

%% Temperature of 85 ºC  
disp('Temperature of 85 ºC - Healthy')
load 'Features_Data\t85_sf_1.lvm';
freq_85H = t85_sf_1(:,1);       
real_85H = t85_sf_1(:,3);       
imag_85H = t85_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI85H.mat freq_85H real_85H imag_85H
clear

disp('Temperature of 85 ºC - Damage 1')
load 'Features_Data\t85_cf1_1.lvm';
freq_85D1 = t85_cf1_1(:,1);       
real_85D1 = t85_cf1_1(:,3);       
imag_85D1 = t85_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T85degrees\EMI85D1.mat freq_85D1 real_85D1 imag_85D1
clear

disp('Temperature of 85 ºC - Damage 2')
load 'Features_Data\t85_cf2_1.lvm';
freq_85D2 = t85_cf2_1(:,1);       
real_85D2 = t85_cf2_1(:,3);       
imag_85D2 = t85_cf2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T85degrees\EMI85D2.mat freq_85D2 real_85D2 imag_85D2
clear

disp('Temperature of 85 ºC - Damage 3')
load 'Features_Data\t85_cc1_1.lvm'
freq_85D3 = t85_cc1_1(:,1);       
real_85D3 = t85_cc1_1(:,3);       
imag_85D3 = t85_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T85degrees\EMI85D3.mat freq_85D3 real_85D3 imag_85D3
clear

disp('Temperature of 85 ºC - Damage 4')
load 'Features_Data\t85_cc2_1.lvm'
freq_85D4 = t85_cc2_1(:,1);       
real_85D4 = t85_cc2_1(:,3);       
imag_85D4 = t85_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T85degrees\EMI85D4.mat freq_85D4 real_85D4 imag_85D4
clear

%% Temperature of 100 ºC  
disp('Temperature of 100 ºC - Healthy')
load 'Features_Data\t100_sf_1.lvm';
freq_100H = t100_sf_1(:,1);       
real_100H = t100_sf_1(:,3);       
imag_100H = t100_sf_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\HealthyCondition\EMI100H.mat freq_100H real_100H imag_100H
clear

disp('Temperature of 100 ºC - Damage 1')
load 'Features_Data\t100_cf1_1.lvm';
freq_100D1 = t100_cf1_1(:,1);       
real_100D1 = t100_cf1_1(:,3);       
imag_100D1 = t100_cf1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T100degrees\EMI100D1.mat freq_100D1 real_100D1 imag_100D1
clear

disp('Temperature of 100 ºC - Damage 2')
load 'Features_Data\t100_cf2_1.lvm';
freq_100D2 = t100_cf2_1(:,1);       
real_100D2 = t100_cf2_1(:,3);       
imag_100D2 = t100_cf2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T100degrees\EMI100D2.mat freq_100D2 real_100D2 imag_100D2
clear

disp('Temperature of 100 ºC - Damage 3')
load 'Features_Data\t100_cc1_1.lvm'
freq_100D3 = t100_cc1_1(:,1);       
real_100D3 = t100_cc1_1(:,3);       
imag_100D3 = t100_cc1_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T100degrees\EMI100D3.mat freq_100D3 real_100D3 imag_100D3
clear

disp('Temperature of 100 ºC - Damage 4')
load 'Features_Data\t100_cc2_1.lvm'
freq_100D4 = t100_cc2_1(:,1);       
real_100D4 = t100_cc2_1(:,3);       
imag_100D4 = t100_cc2_1(:,4);       

save C:\Users\Asus\Desktop\Projeto_Impedance\ImpedanceData\DamagedCondition\T100degrees\EMI100D4.mat freq_100D4 real_100D4 imag_100D4
clear

toc