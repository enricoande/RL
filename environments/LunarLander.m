% LunarLander.m     E.Anderlini@ucl.ac.uk     29/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to model the reinforcement learning states,
% actions and rewards of the classical lunar lander problem with 
% continuous actions. Although inspiration is taken from OpenAI Gym, for
% simplicity at the moment the script follows an assignment by the
% University of Sydney. The lander is expected to land at (0,0).
% Note that noise is added to the action as in OpenAI Gym.
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
        seed;               % seed to the random number generator
        state;              % current state
        action;             % current action
        reward;             % current reward
        resetCode;          % boolean for end of episode
        % Action and state spaces:
        stateCardinality;   % size of the state space
        actionCardinality;  % size of the action space
        minStates;          % lower boundary of all states
        maxStates;          % upper boundary of all states
        minActions;         % lower boundary of all actions
        maxActions;         % upper boundary of all actions
        stateHistory;       % states for all time steps
        actionHistory;      % actions for all time steps 
        % End conditions for simulation:
        maxIter;            % max. no. of iterations
        end_position;       % coordinates of the landing platform [m]
        minimum_mass;       % minimum allowable mass for departure [kg]
        maximum_horizontal_velocity; % max allowable hori velocity [m/s]
        maximum_vertical_velocity;   % max allowable vert velocity [m/s]
        maximum_rotational_velocity; % max pitch velocity [rad/s]
    end
    
    %% Private properties:
    properties (Access = 'private')
        % Dynamic model properties:
        timeStep;           % time step duration [s]
        substeps;           % no. substeps for the simulation
        max_lateral_force;  % maximum lateral force [N]
        max_vertical_force; % maximum vertical force [N]
        gravity;            % lunar gravitational acceleration [m/s^2]
        moment_inertia;     % moment of inertia [kg.m^2]
        thruster_impulse;   % rocket thruster specific impulse [N.s/kg]
    end
    
    %% Protected methods:
    methods (Access = 'protected')
        %% Class constructor:
        function obj = LunarLander(startPoint,seed)
            if nargin == 0  
                % Default initialisation:
                obj.state = [500,16e04,0,0,-700,0,15e03];
                obj.seed = 0;
            else    
                % Otherwise specify initial state and animation boolean:
                obj.state = startPoint;
                obj.seed = seed;
            end
            % Initialise the default properties for the dynamic model:
            obj.timeStep = 0.1;             % [s]
            obj.substeps = 1;
            obj.max_lateral_force = 500;           % [N]
            obj.max_vertical_force = 44e03;        % [N]
            obj.gravity = 1.6;                     % [m/s^2]
            obj.moment_inertia = 1e05;             % [kg m^2]
            obj.thruster_impulse = 3e03;           % [N.s/kg]      
            
            % Initialise the reinforcement learning properties:
            obj.reward = 0;
            obj.action = [0,0];
            obj.actionCardinality = 2;
            obj.stateCardinality = 7;
            if length(obj.action)~=obj.actionCardinality || ...
                    length(obj.state)~=obj.stateCardinality
                error('Inconsistent size of state & action spaces');
            end
            obj.minStates = [-1e03,0,-pi,-100,-700,-2,0];
            obj.maxStates = [1e03,16e04,pi,100,300,2,15e03];
            obj.minActions = [-1,-1];
            obj.maxActions = [1,1];
            
            % Initialise the default end conditions:
            obj.maxIter = 10000;
            obj.end_position = [0,0];              % [m]
            obj.minimum_mass = 11e03;              % [kg]
            obj.maximum_horizontal_velocity = 0.5; % [m/s]
            obj.maximum_vertical_velocity = 1;     % [m/s]
            obj.maximum_rotational_velocity = 0.1; % [rad/s]
            
            % Initialise the state and action history:
            obj.stateHistory = nan(obj.maxIter,obj.stateCardinality);
            obj.actionHistory = nan(obj.maxIter,obj.actionCardinality);
            
            % Initialise the random generator with the given seed:
            rng(obj.seed);
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
    end
    
    %% Private methods:
    methods(Access = 'private')
        %% Return the state vector derivative:
        % The state vector is x = [x;z;theta;u;w;q;m].
        % The input vector is u = [Fl;Ft]. 
        % Random noise is added to the thrust values.
        function Xdot = dynamicsLL(obj,state,Action)
            % Map the actions to the thrust values:
            Fl = 0.5*(Action(1)+1)*(Action(1)>0)*obj.max_lateral_force;
            Fv = Action(2)*(abs(Action(2))>=0.5)*obj.max_vertical_force;
            
            %%%%%%%%%%%%
            % Add noise
            %%%%%%%%%%%%
            
            % Express the equations of motion of the lunar lander:
            Xdot = [state(4),state(5),state(6);...
                (Fl*cos(state(3))-Fv*sin(state(3)))/state(7),...
                (Fl*sin(state(3))+Fv*cos(state(3)))/state(7)...
                -obj.gravity,...
                4*Fl/obj.moment_inertia,(Ft+Fl)/obj.thruster_impulse];
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
                % Map pitch angle from 0 to 360 to -180 to 180:
                Xstep(3) = wrapToPi(Xstep(3));    
            end            
        end
        
        %% Generate the initial reward as in OpenAI Gym:
        function generateReward(obj)   
            % The reward is 1 for all steps, including the termination step
            obj.reward = 1;
        end
    end    
end