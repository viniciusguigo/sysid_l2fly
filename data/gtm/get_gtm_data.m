% %
%-------------- Get data from NASA GTM model ------------
%
% States: p, q, r
% Controls: delta_a, delta_e, delta_r
%
% Run different doublets and save as a csv file.

% Vinicius Guimaraes Goecks
% April 18, 2017

% clean everything
clear all
close all
clc

% PARAMETERS
%
% TRAIN: 30 sec, doublets at [1 10 20]
% TEST: 10 sec, doublets at [1 4 7]
% TEST2: 5 sec, doublets at [1 1 1]
% TEST3: 15 sec, doublets at [3 1 6]

% add GTM path
addpath('/home/vinicius/Projects/GTM_DesignSim/gtm_design')

% load nominal starting point into simulation model workspace
loadmws(init_design('GTM_T2'));

% Trim to nominal condition: level flight, alpha=3
SimVars=trimgtm(struct('alpha',3, 'gamma',0));

% Load Simulation Variables (at trim condition) into Model Workspace
loadmws(SimVars);

% % Linearize model about this condition
% fprintf(1,'Linearizing...')
% [sys,londyn,latdyn]=linmodel();
% fprintf(1,' Done\n');

% Construct 1 degree double sequence.
% f=100;    % 100 Hz input sampling on sequence
% d=0.5;      % 2 sec pulse duration
a=[1 1 1];% pulse amplitude(deg), [ele,rud,ail]
% tf = 30; % seconds (TRAIN)
tf = 15; % seconds (TEST)

% Construct same doublet sequence via simulink
% set_param('gtm_design/Input Generator/Doublet Generator','timeon','[1 10 20]'); % seconds (TRAIN)
set_param('gtm_design/Input Generator/Doublet Generator','timeon','[3 1 6]'); % seconds (TEST)
set_param('gtm_design/Input Generator/Doublet Generator','pulsewidth','[0.5 0.5 0.5]'); % hertz
set_param('gtm_design/Input Generator/Doublet Generator','amplitude',sprintf('[%f %f %f]',a(1),a(3),a(2)));
[tsim,xsim,ysim]=sim('gtm_design',[0 tf]);

% Turn simulink doublet generation off
set_param('gtm_design/Input Generator/Doublet Generator','amplitude','[0 0 0]');

% Grab state in ysim First 6 are trim outputs, next 12 are state.
ysim_full = ysim;
ysim=ysim(:,7:18);


% Plot results
% FIGURE: LINEAR AND ANGULAR VELOCITIES
set(figure(1),'Position',[20 80 900 700]);

subplot(331),
plot(tsim,ysim(:,1)); grid on
title('Linear Velocity');
ylabel('u (ft/sec)')

subplot(334),
plot(tsim,ysim(:,2)); grid on
ylabel('v (ft/sec)')

subplot(337),
plot(tsim,ysim(:,3)); grid on
xlabel('Time (sec)');ylabel('w (ft/sec)')

subplot(332),
plot(tsim,180/pi*ysim(:,4)); grid on
title('Angular Velocity');
ylabel('p (deg/sec)')

subplot(335),
plot(tsim,180/pi*ysim(:,5)); grid on
ylabel('q (deg/sec)')

subplot(338),
plot(tsim,180/pi*ysim(:,6)); grid on
xlabel('Time (sec)');ylabel('r (deg/sec)')

subplot(333),
plot(tsim,doublet_sent(:,1)); grid on
title('Doublet Sequence');
ylabel('\delta_e (unit)')

subplot(336),
plot(tsim,doublet_sent(:,2)); grid on
ylabel('\delta_r (unit)')

subplot(339),
plot(tsim,doublet_sent(:,3)); grid on
xlabel('Time (sec)');ylabel('\delta_a (unit)')

if (exist('AutoRun','var'))
    pause(.2);
    orient portrait; print -dpng exampleplot01;
end

% Plot results
% FIGURE: LAT/LONG/ALT AND EULER ANGLES
set(figure(2),'Position',[20 80 900 700]);

subplot(321),
plot(tsim,180/pi*ysim(:,7)); grid on
title('Position');
ylabel('Latitude (deg)')

subplot(323),
plot(tsim,180/pi*ysim(:,8)); grid on
ylabel('Longitude (deg)')

subplot(325),
plot(tsim,ysim(:,9)); grid on
xlabel('Time (sec)');ylabel('Altitude (ft)')

subplot(322),
plot(tsim,180/pi*ysim(:,10)); grid on
title('Euler Angles');
ylabel('\phi (roll) (deg)')

subplot(324),
plot(tsim,180/pi*ysim(:,11)); grid on
ylabel('\theta (pitch) (deg)')

subplot(326),
plot(tsim,180/pi*ysim(:,12)); grid on
xlabel('Time (sec)');ylabel('\psi (yaw) (deg)')

if (exist('AutoRun','var'))
    pause(.2);
    orient portrait; print -dpng exampleplot01;
end

% save data to csv file
csvwrite('linear_velocity_test3.csv', ysim(:,1:3))
csvwrite('angular_velocity_test3.csv', 180/pi*ysim(:,4:6))
csvwrite('doublet_sent_test3.csv', doublet_sent)
csvwrite('states_test3.csv', ysim(:,1:12))

% csvwrite('linear_velocity_train.csv', ysim(:,1:3))
% csvwrite('angular_velocity_train.csv', 180/pi*ysim(:,4:6))
% csvwrite('doublet_sent_train.csv', doublet_sent)
% csvwrite('states_train.csv', ysim(:,1:12))
