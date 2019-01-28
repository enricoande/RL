% InvertedPendulum.m     E.Anderlini@ucl.ac.uk     24/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to model the motions of the classical inverted
% pendulum.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class definition:
classdef InvertedPendulum
    
    %% Protected properties:
    properties (Access = protected)
        % Constants:
        g         % gravitational accelaration [m/s2]
        m         % mass of the pendulum [kg]
        M         % mass of the cart [kg]
        alpha
        l         % pendulum length [m]
        dt        % time step [s]
        
        % Variables:
        x         % state vector
        u         % input vector
    end
    
    %% Public methods:
    methods
        %% Class constructor:
        function obj = InvertedPendulum(g,m,M,l,dt)
            if nargin < 1
                obj.g = 9.8;
                obj.m = 2;
                obj.M = 8;
                obj.l = 0.5;
                obj.dt = 0.1;
            else
                obj.g = g;
                obj.m = m;
                obj.M = M;
                obj.l = l;
                obj.dt = dt;
            end
            obj.alpha = 1/(obj.m+obj.M);
        end
        
        %% Update the body motions with a 4th order Runge-Kutta scheme:
        function xn = updateMotions(obj,x0,u)
            % Store the variables in the class:
            obj.u = u;
            obj.x = x0;
            
            % Estimates of the ODE at different time steps (t,t+dt/2,t+dt):
            dxdt1 = obj.f();
            x1 = x0+dxdt1*obj.dt/2;
            obj.x = x1;
            dxdt2 = obj.f();
            x2 = x0+dxdt2*obj.dt/2;
            obj.x = x2;
            dxdt3 = obj.f();
            x3 = x0+dxdt3*obj.dt;
            obj.x = x3;
            dxdt4 = obj.f();

            % Estimate of the system at the next time step:
            xn = x0 + (dxdt1+2*(dxdt2+dxdt3)+dxdt4)*obj.dt/6; 
        end
    end
    
    %% Protected Methods:
    methods (Access = protected)
        %% Return the derivative of the state:
        function dxdt = f(obj)
            dxdt = zeros(1,2);
            dxdt(1) = obj.x(2); 
            dxdt(2) = (obj.g*sin(obj.x(1))-obj.alpha*obj.m*obj.l*...
                obj.x(2)^2*sin(2*obj.x(1))/2-obj.alpha*cos(obj.x(1))*...
                obj.u)/(4*obj.l/3-obj.alpha*obj.m*obj.l*(cos(obj.x(1)))^2);
        end
    end
end