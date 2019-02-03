% LunarLander.m     E.Anderlini@ucl.ac.uk     30/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to model the reinforcement learning states,
% actions and rewards of the classical lunar lander problem with 
% continuous actions. Although inspiration is taken from OpenAI Gym, for
% simplicity at the moment the script follows an assignment by the
% University of Sydney. The lander is expected to land at (0,0).
%
% Lunar lander simpliefied problem statement:
% http://web.aeromech.usyd.edu.au/AMME3500/Course_documents/material/
% tutorials/Assignment%204%20Lunar%20Lander%20Solution.pdf
%
% Original code by OpenAI Gym available at:
% https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
%
% To do:
% * add noise to the state - split reinforcement learning and dynamic
%   states.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Class definition:
classdef LunarLander < handle
    
    %% Public properties:
    properties (Access = 'public')
        stateHistory;       % states for all time steps
        actionHistory;      % actions for all time steps 
    end
    %% Protected properties:
    properties (Access = 'protected')
        % Reinforcement learning properties:
        seed;               % seed to the random number generator
        state;              % current state - observed (with noise)
        action;             % current action
        reward;             % current reward
        resetCode;          % boolean for end of episode
%         % Dynamic model property:
%         actual_state;       % actual state used in simulation
        % Action and state spaces:
        stateCardinality;   % size of the state space
        actionCardinality;  % size of the action space
        minStates;          % lower boundary of all states
        maxStates;          % upper boundary of all states
        minActions;         % lower boundary of all actions
        maxActions;         % upper boundary of all actions
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
%             % Set the actual state to be equal to the observed state:
%             obj.actual_state = obj.state;
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
            obj.minStates = [-750,0,-pi,-100,-1000,-2,0];
            obj.maxStates = [750,16e04,pi,100,500,2,15e03];
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
        function [state,action,reward,next_state,done]=doAction(obj,action)
            % Return the state, new state, reward and episode flag:
            state = obj.state;            
            next_state = obj.RK4(action);   
            obj.checkIfGoalReached();            
            reward = obj.reward;            
            done = obj.resetCode;
        end
        
        %% Initialise the state randomly as in OpenAI Gym:
        % N.B.: Whereas the initial vertical position and velocity and the
        % corresponding initial mass are not modified, the horizontal
        % position is changed randomly.
        function randomInitState(obj)
            % Randomly set the initial horizontal position of the lander:
            obj.state(1) = (rand-0.5)*1000; % range: +/-500 m
%             obj.actual_state(1) = obj.state(1);
        end
        
        %% Check if the episode has been completed and get the reward:
        % The reward function is expressed as follows:
        % * a living cost of -0.01 at every time step,
        % * a penalty of -|x_f - 0|,
        % * a penalty of -10*|u_f|,
        % * a penalty of -|w_f|,
        % * a penalty of -10*|q_f|,
        % * a reward of +m_f/100,
        % * a reward of +100 if |x_f| < 10,
        % * a reward of +50 if |u_f| < limit,
        % * a reward of +200 if |w_f| < limit,
        % * a reward of +50 if |q_f| < limit,
        % * a reward of +200 if m_f > limit.
        function checkIfGoalReached(obj)
            % Initialise the reward with the living cost:
            obj.reward = -0.001;
            
            % Check if the episode is completed, i.e. if the lander has hit
            % the ground:
            if obj.state(2) < obj.end_position(2)
                % Set the boolean to episode completion:
                obj.resetCode = true;
                % Update the reward based on the final state:
                obj.reward = obj.reward - 10*abs(obj.state(4));
                obj.reward = obj.reward - abs(obj.state(5));
                obj.reward = obj.reward - 10*abs(obj.state(6));
                obj.reward = obj.reward + obj.state(7)/100;
                % Add extra bonus for the final horizontal position:
                if abs(obj.state(1)) < obj.end_position(1)
                    obj.reward = obj.reward + 100;
                end
                % Add extra bonus for the final horizontal velocity:
                if abs(obj.state(4)) < obj.maximum_horizontal_velocity
                    obj.reward = obj.reward + 50;
                end
                % Add extra bonus for the final vertical velocity:
                if abs(obj.state(5)) < obj.maximum_vertical_velocity
                    obj.reward = obj.reward + 200;
                end
                % Add extra bonus for the final rotational velocity:
                if abs(obj.state(6)) < obj.maximum_rotational_velocity
                    obj.reward = obj.reward + 50;
                end
                % Add extra bonus for the final mass:
                if abs(obj.state(7)) > obj.minimum_mass
                    obj.reward = obj.reward + 200;
                end
            end
        end
    end
    
    %% Private methods:
    methods(Access = 'private')
        %% Return the state vector derivative:
        % The state vector is x = [x;z;theta;u;w;q;m].
        % The input vector is u = [Fl;Ft].
        function Xdot = dynamicsLL(obj,state,action)
            % Map the actions to the thrust values:
            Fl = 0.5*(action(1)+1)*(action(1)>0)*obj.max_lateral_force;
            Fv = action(2)*(abs(action(2))>=0.5)*obj.max_vertical_force;
            
            % Express the equations of motion of the lunar lander:
            Xdot = [state(4),state(5),state(6);...
                (Fl*cos(state(3))-Fv*sin(state(3)))/state(7),...
                (Fl*sin(state(3))+Fv*cos(state(3)))/state(7)...
                -obj.gravity,...
                4*Fl/obj.moment_inertia,(Ft+Fl)/obj.thruster_impulse];
        end
        
        %% Integrate in time with a 4th order Runge-Kutta scheme:
        function Xstep = RK4(obj,action)
            for i = 1:obj.substeps
                k1 = obj.dynamicsLL(obj.state,action);
                k2 = obj.dynamicsLL(obj.state+obj.timeStep/2*k1,action);
                k3 = obj.dynamicsLL(obj.state+obj.timeStep/2*k2,action);
                k4 = obj.dynamicsLL(obj.state+obj.timeStep*k3,action);                
                Xstep = obj.state + obj.timeStep/6*(k1 + 2*k2 + 2*k3 + k4);
                % Map pitch angle from 0 to 360 to -180 to 180:
                Xstep(3) = wrapToPi(Xstep(3));    
            end            
        end
        
        %% Add sensor noise to the actual state:
    end    
end