% InvertedPendulum.m     E.Anderlini@ucl.ac.uk     25/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to model the reinforcement learning states,
% actions and rewards of the classical cart-pole balancing problem as
% available on OpenAI Gym.
%
% Original code by OpenAI Gym available at:
% https://github.com/openai/gym/blob/master/gym/envs/classic_control/
% cartpole.py
%
% This script has been modified from Girish Joshi's original class file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class definition:
classdef CartPole < handle
    
    %% Protected properties:
    properties (Access = 'protected')
        state = [];
        timeStep = [];
        massCart = [];
        massPole = [];
        totalMass = [];
        poleLength = [];
        poleMassLength = [];
        gravity = [];
        substeps = [];
        actionCardinality = [];
        actions = [];
        pendLimitAngle = [];
        cartLimitRange = [];
        reward = [];
        bonus = [];
%         goal = [];
        resetCode = [];
        simOnOff;
        panel;
        cart;
        pole;
        dot;
        arrow;
        seed;
    end
    
    %% Protected methods:
    methods (Access = 'protected')
        %% Class constructor:
        function obj = CartPole(startPoint,simChoice,seed)
            if nargin == 0  
                % Default initialisation:
                obj.state = [0,0,0,0];
                obj.simOnOff = false;
                obj.seed = 0;
            else    
                % Otherwise specify initial state and animation boolean:
                obj.state = startPoint;
                obj.simOnOff = simChoice;   
                obj.seed = seed;
            end
            % Initialise the default properties as in OpenAI Gym:
            obj.timeStep = 0.02;
            obj.massCart = 1;
            obj.massPole = 0.1;
            obj.gravity = 9.8;
            obj.poleLength = 0.5;
            obj.substeps = 1;
            obj.reward = 0;
            obj.bonus = 0;
            obj.actions = [-10,10];
            obj.actionCardinality = length(obj.actions);
            obj.totalMass = obj.massCart+obj.massPole;
            obj.poleMassLength = obj.poleLength*obj.massPole;
            
            % Specify the thresholds at which the episode fails:
            obj.pendLimitAngle = deg2rad(12);
            obj.cartLimitRange = 2.4;
            
            if obj.simOnOff
                % Initialise the animation:
                obj.initSim
            end
            
            % Initialise the random generator with the given seed:
            rng(seed);
        end
        
        %% Perform an action:
        function [state,Action,reward,next_state,done]=doAction(obj,Action)
            % Return the state, new state, reward and episode flag:
            state = obj.state;            
            next_state = obj.RK4(Action);   
            obj.checkIfGoalReached();            
            reward = obj.reward;            
            done = obj.resetCode;            
            if obj.simOnOff
                obj.simCartpole(Action);
            end
        end
        
        %% Initialise the state randomly as in OpenAI Gym:
        function randomInitState(obj)
            % As in OpenAI Gym, all states are assigned a uniform random
            % value within +/-0.05:
            obj.state = 0.05*(2*rand(1,4)-1);
        end
        
        %% Check if the episode has been completed and get the reward:
        function checkIfGoalReached(obj)
            % N.B: The reward function and episode completion is kept
            % identical to that of OpenAI Gym. However, superior reward
            % functions may be developed.
            
            % Generate initial reward as in OpenAI Gym:
            obj.generateReward();
            
            % Determine whether the episode ends:
%             if norm([obj.state(3),obj.state(4)]) < 0.01                
%                 obj.bonus = 10;
%                 obj.goal = true;
%                 obj.resetCode = false;
        
            if abs(obj.state(3)) >  obj.pendLimitAngle     
                % Pole Angle is more than ±12°:
%                 obj.bonus = -10;     %punishement for falling down
                obj.resetCode = true;
                
            elseif abs(obj.state(1)) > obj.cartLimitRange   
                % Cart Position is more than ±2.4:
%                 obj.bonus = -10;     %punishement for moving too far
                obj.resetCode = true;
                
            else
                obj.bonus = 0;
                obj.resetCode = false;
            end
            
            % The bonus is now set to 0 to reflect OpenAI Gym's code:
%             obj.reward = obj.reward + obj.bonus;
%             obj.goal = false;
%             obj.bonus = 0;
        end
        
        %% Initialise the animation:
        function initSim(obj)
            obj.panel = figure;
            obj.panel.Position = [100,100,1200,600];
            obj.panel.Color = [1,1,1];
            hold on;
            obj.cart = plot(0,0,'m','Linewidth',50); % cart
            obj.pole = plot(0,0,'b','LineWidth',10); % pendulum stick
            axPend = obj.pole.Parent;
            axPend.XTick = [];
            axPend.YTick = [];
            axPend.Visible = 'off';
            axPend.Position = [0.35,0.4,0.3,0.3];
            axPend.Clipping = 'off';
            axis equal;
            axis([-10,10,-5,5]);
            obj.dot = plot(0,0,'.k','MarkerSize',50);
            obj.arrow = quiver(0,0,-3,0,'linewidth',7,'color','r',...
                'MaxHeadSize',15);
            hold off;
        end
    end
    
    %% Private methods:
    methods(Access = 'private')
        %% Return the state vector derivative:
        function Xdot = dynamicsCP(obj,state,Action)
            theta = state(3);            
            theta_dot = state(4);            
            A = [cos(theta),obj.poleLength;...
                obj.totalMass,obj.poleMassLength*cos(theta)];            
            B = [-obj.gravity*sin(theta);...
                Action+obj.poleMassLength*theta_dot^2*sin(theta)];
            dynamic = A\B;
            Xdot = [state(2),dynamic(1),state(4),dynamic(2)];
        end
        
        %% Integrate in time with a 4th order Runge-Kutta scheme:
        function Xstep = RK4(obj,Action)
            % N.B.: OpenAI Gym uses only a 1st order Euler scheme
            for i = 1:obj.substeps
                k1 = obj.dynamicsCP(obj.state,Action);
                k2 = obj.dynamicsCP(obj.state+obj.timeStep/2*k1,Action);
                k3 = obj.dynamicsCP(obj.state+obj.timeStep/2*k2,Action);
                k4 = obj.dynamicsCP(obj.state+obj.timeStep*k3,Action);                
                Xstep = obj.state + obj.timeStep/6*(k1 + 2*k2 + 2*k3 + k4);        
                % Map pendulum Angle from 0 to 360 to -180 to 180:
                Xstep(3) = wrapToPi(Xstep(3));          
            end            
        end
        
        %% Generate the initial reward as in OpenAI Gym:
        function generateReward(obj)   
            % The reward is 1 for all steps, including the termination step
            obj.reward = 1;
        end
        
        %% Update the animation plot:
        function simCartpole(obj,Action)
            set(obj.pole,'XData',[obj.state(1),...
                obj.state(1)-10*sin(obj.state(3))]);
            set(obj.pole,'YData',[0,10*cos(obj.state(3))]);
            set(obj.cart,'XData',[obj.state(1)-2.5,obj.state(1)+2.5]);
            set(obj.cart,'YData',[0,0]);
            set(obj.dot,'XData',obj.state(1));
            set(obj.dot,'YData',0);
            set(obj.arrow,'XData',obj.state(1)+sign(Action)*3);
            set(obj.arrow,'YData',0);
            set(obj.arrow,'UData',sign(Action)*3);
            drawnow;
        end        
    end    
end