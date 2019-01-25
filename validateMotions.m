% validateMotions.m     E.Anderlini@ucl.ac.uk     24/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used to validate the model for the cart-pole.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

%% Input data:
dt = 0.1;
N = 20;

%% Initialization:
x0 = [0,0];
x = zeros(N,2);
t = zeros(N,1);
p = InvertedPendulum();

%% Run the simulation:
for i=2:20
    t(i) = (i-1)*dt;
    u = 0;
    x0 = p.updateMotions(x0,u);
    x(i,:) = x0;
end

%% Plot the results:
figure;
subplot(2,1,1);
plot(t,x(:,1));
ylabel('$\theta$ [rad]','Interpreter','Latex');
grid on;
set(gca,'TickLabelInterpreter','Latex');
subplot(2,1,2);
plot(t,x(:,2),'Color',[0.8500,0.3250,0.0980]);
xlabel('$t$ (s)','Interpreter','Latex');
ylabel('$\dot{\theta}$ [rad/s]','Interpreter','Latex');
set(gca,'TickLabelInterpreter','Latex');
grid on;
set(gcf,'color','w');