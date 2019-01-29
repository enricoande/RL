% LunarLander.m     E.Anderlini@ucl.ac.uk     25/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to model the reinforcement learning states,
% actions and rewards of the classical lunar lander problem with 
% continuous actions. Although inspiration is taken from OpenAI Gym, for
% simplicity at the moment the script follows an assignment by the
% University of Sydney.
% However, note that noise is added to the action as in OpenAI Gym.
%
% Lunar lander simpliefied problem statement:
% http://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/
% tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf
%
% Original code by OpenAI Gym available at:
% https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class definition:
classdef LunarLander < handle
    
    %% Protected properties:
    properties (Access = 'protected')
        % Reinforcement learning properties:
        seed;
        state;
        actions;
        actionCardinality;
        reward;
        bonus;
        resetCode;          % boolean for end of episode
        simOnOff;           % boolean for simulation data storage
        % Dynamic model properties:
        timeStep;           % time step duration [s]
        substeps;           % no. substeps for the simulation
        max_lateral_force;  % maximum lateral force [kN]
        max_vertical_force; % maximum vertical force [kN]
        
        
        
        
        scale;             % affects how fast-paced the game is
        main_engine_power; % power of the main engine
        side_engine_power; % power of the side engine
        
        % End conditions for simulation:
        maxIter;           % max. no. of iterations
    end
    
    %% Protected methods:
    methods (Access = 'protected')
        %% Class constructor:
        function obj = LunarLander(startPoint,simChoice,seed)
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
            obj.substeps = 1;
            obj.reward = 0;
            obj.bonus = 0;
            obj.actions = [0,0];
            
            obj.scale = 30;
            obj.main_engine_power = 13;
            obj.side_engine_power = 0.6;
            
            obj.actionCardinality = length(obj.actions);
            obj.totalMass = obj.massCart+obj.massPole;
            obj.poleMassLength = obj.poleLength*obj.massPole;
            % Initialise the default end conditions:
            obj.maxIter = 200;
            
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
                obj.simLunarLander(Action);
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    
    %% Private methods:
    methods(Access = 'private')
        %% Return the state vector derivative:
        % The state vector is x = [x;y;theta;u;v;q;m].
        % The input vector is u = [Fl;Ft]. 
        % Random noise is added to the thrust values.
        function Xdot = dynamicsLL(obj,state,Action)
            % Map the actions to the thrust values:
            Fl = 0.5*(Action(1)+1)*(Action(1)>0)*obj.max_lateral_force;
            Fv = Action(2)*(abs(Action(2))>=0.5)*obj.max_vertical_force;
            
            
            
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
                k1 = obj.dynamicsLL(obj.state,Action);
                k2 = obj.dynamicsLL(obj.state+obj.timeStep/2*k1,Action);
                k3 = obj.dynamicsLL(obj.state+obj.timeStep/2*k2,Action);
                k4 = obj.dynamicsLL(obj.state+obj.timeStep*k3,Action);                
                Xstep = obj.state + obj.timeStep/6*(k1 + 2*k2 + 2*k3 + k4);
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % Map pendulum Angle from 0 to 360 to -180 to 180:
                Xstep(3) = wrapToPi(Xstep(3));    
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            end            
        end
        
        %% Generate the initial reward as in OpenAI Gym:
        function generateReward(obj)   
            % The reward is 1 for all steps, including the termination step
            obj.reward = 1;
        end
        
        %% Update the animation plot:
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        function simLunarLander(obj,Action)
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
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end    
end