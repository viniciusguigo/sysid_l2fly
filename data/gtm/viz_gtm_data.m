% Function to load and plot csv data.
% Labels/Titles designed to match states of GTM aircraft

clear all;
close all;
clc;

% load csv
ysim = csvread('states_train.csv');
doublet_sent = csvread('doublet_sent_train.csv');
tsim = 0:length(ysim)-1;

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
