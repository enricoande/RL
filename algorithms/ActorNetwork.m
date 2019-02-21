% ActorNetwork.m     E.Anderlini@ucl.ac.uk     21/02/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class sets up the functions required for the training and use of the
% actor's deep neural network (DNN). This DNN can be considered to be an
% approximation for the agent's policy. The input to the network is the
% state, while the output is the action under a deterministic policy.
% The output layer activation is a hyperbolic tangent to keep the action
% between the prescribed bounds.
%
% References:
% https://spinningup.openai.com/
% https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef ActorNetwork
    %% Properties:
    properties (Access = private)
        % State-action space:
        action_bound;
        action_dim;
        state_dim;
        % Reinforcement learning parameters:
        batch_size;
        learning_rate;
        tau;
    end
    
    %% Methods:
    methods (Access = public)
        %% Class constructor:
        function obj = ActorNetwork(state_dim,action_dim,action_bound,...
                learning_rate,tau,batch_size)
            % Initialise the class properties:
            obj.action_bound = action_bound;
            obj.action_dim = action_dim;
            obj.state_dim = state_dim;
            obj.batch_size = batch_size;
            obj.learning_rate = learning_rate;
            obj.tau = tau;
            
            % Initialise the actor network:
            
            % Initialise the target network:
            
        end
    end
end