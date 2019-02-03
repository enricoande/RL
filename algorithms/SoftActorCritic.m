% SoftActorCritic.m     E.Anderlini@ucl.ac.uk     03/02/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class is designed to have an actor learn how to perform an episodic
% task using soft actor critic reinforcement learning. The implementation
% here is adapted from the code by OpenAI in their Spinning Up project:
%
% https://spinningup.openai.com/en/latest/algorithms/sac.html
%
% N.B.: At the moment, the 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef SoftActorCritic < LunarLander
    
    %% Private properties:
    properties (Access = 'private')
        % Reinforcement learning parameters:
        episodes;        % no. of episodes     
        gamma;           % discount factor
        polyak;          % interpolation factor
        lr;              % learning rate
        alpha;           % entropy regularization coefficient
        sampleSize;      % size of the sample
        batchSize;       % batch of data for learning
        replayBuffer;    % buffer for experience replay
        bufferSize;      % size of the buffer
        
        % Reinforcement learning variables:
        
        
        batch;
        Q;               % Q-value
        newQ;            % new Q-value
        targetQ;         % target Q-value (for training)
        maxQ;            % max(Q(s,:))
        epsilon;         % exploration rate
        epsilonDecay;    % exploration rate decay
        annealing;       % annealing rate
        totalReward;     % total reward
        testtotalReward; % total reward of the test run
        UpdateIndex;     % 
        TDerror;         % temporal differences error
        net;             % neural network object
        net_prev;        % previous neural network object
        
        
        
        
        netWeights;      % weight of the neural network
    end
    
    %% Public properties:
    properties (Access = 'public')
        episodeTotReward;  % total reward for the current episode
        episodeLength;     % episode length
    end
    
    %% Public methods:
    methods (Access = 'public')
        
        %% Constructor:
        function obj = SoftActorCritic(episodes,gamma,polyak,lr,alpha,...
                batchSize,bufferSize)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
            obj = obj@LunarLander();
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if nargin == 0    
                obj.episodes = 100;   
                obj.gamma = 0.99;
                obj.polyak = 0.995;
                obj.lr = 0.001;
                obj.alpha = 0.2;
                obj.batchSize = 100;
                obj.bufferSize = 1000;
            else                
                obj.episodes = episodes;   
                obj.gamma = gamma;
                obj.polyak = polyak;
                obj.lr = lr;
                obj.alpha = alpha;
                obj.batchSize = batchSize;
                obj.bufferSize = bufferSize;
            end
            
            obj.batch = 1;
            obj.epsilonDecay = 0.999;
            obj.totalReward = 0;
            obj.resetCode = false;
            
            % Initialize the Q-Network:
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % change to RELUs and Adam
            layer1Size = 10;
            layer2Size = 20;            
            obj.net = fitnet([layer1Size,layer2Size],'trainlm');
            obj.net_prev = obj.net;
            obj.net.trainParam.lr = 0.1;
            obj.net.trainParam.epochs = 10;
            obj.net.trainParam.showWindow = false;
            obj.net.trainParam.lr_dec = 0.8;  % ratio for LR decrease
            obj.net = train(obj.net,rand(length(obj.state),25),...
                rand(length(obj.actions),25));
            obj.netWeights = getwb(obj.net);
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
            % Initialize Data Recording
            obj.episodeLength = zeros(1,obj.maxEpisode);
            obj.episodeTotReward = zeros(1,obj.maxEpisode);
            
            % Initialize the replay Buffer
            [state,Action,reward,next_state,done] = obj.doAction(1);
            obj.bufferSize = 10000;
            obj.replayBuffer = [state,Action,reward,next_state,done];
            obj.sampleSize = 100;
        end
        
        %% Run Deep Q Neural learning:
        function QLearningTrain(obj)
            for episode = 1:obj.maxEpisode                
                % Reset the parameters:
                obj.totalReward = 0;
                obj.bonus = 0;
                % Exploration rate decay:              
                obj.epsilon = obj.epsilon*obj.epsilonDecay;                
                % Initialise the state randomly:
                obj.randomInitState();
                
                % Run DQN for all episodes:
                for itr_no = 1:obj.maxIter
                    % Select the action with an epsilon-greedy policy:
                    action = obj.selectAction();
                    % Perform the action, land in new state and get reward:
                    [state,Action,reward,next_state,done] = ...
                        obj.doAction(action);
                    % Add the data to the buffer memory:
                    obj.addtoReplaybuffer(state,Action,reward,...
                        next_state,done);        
                    % Update the state:
                    obj.state = next_state;
                    % Aggregate the total reward for every episode:
                    obj.totalReward = obj.totalReward + reward;
                    % Break the loop if the episode is completed:
                    if obj.resetCode
                        break;
                    end
                end
                
                % Store the data used to assess the method's performance:
                obj.episodeTotReward(episode) = obj.totalReward;
                obj.episodeLength(episode) = itr_no;
                
                % Display results summary for each episode:
                if itr_no == obj.maxIter
                    % If the maximum number of iterations is obtained:
                    disp(['Episode ',num2str(episode),...
                        ': Successful Balance Achieved!',...
                        ' - Total Reward:',...
                        num2str(obj.episodeTotReward(episode))]);                    
                elseif obj.resetCode == true     
                    % If the episide is completed due to failure:
                    disp(['Episode ',num2str(episode),...
                        ': Reset Condition reached!!!',...
                        ' - Total Reward:',...
                        num2str(obj.episodeTotReward(episode))]);
                    obj.resetCode = false;                    
                end
                
                % Check if the end conditions of the simulation are met:
                if episode>obj.desiredAvgWindow
                    if mean(obj.episodeTotReward(episode+1-...
                            obj.desiredAvgWindow:episode)) > ...
                            obj.desiredAvgReward
                        disp(['Solution criteria met at episode ',...
                            num2str(obj.episodeTotReward(episode))]);
                    end
                end
                
                % Train the neural network on the buffer data:
                obj.trainOnBuffer()               
            end
        end
        
        %% Add the current step to the buffer data for experience replay:
        % N.B.: This function will need to be modified or split into two
        % functions instead. The idea is to sample randomly from the
        % buffer, although it is important to keep end-of-episode data,
        % since that will present a very different reward.
        function addtoReplaybuffer(obj,state,Action,reward,next_state,done)            
            if length(obj.replayBuffer) < obj.bufferSize 
                % Add data without replacement:
                obj.replayBuffer = [obj.replayBuffer;...
                    [state,Action,reward,next_state,done]];
            else
                % Add data with replacement:
                obj.replayBuffer(1,:) = [];
                obj.replayBuffer = [obj.replayBuffer;...
                    [state,Action,reward,next_state,done]];
                % N.B.: A smarter way to store data may be sought.
            end
        end
        
        %% Run an episode with the trained DQN:
        function DQNTest(obj,simLength)
            obj.simOnOff = true;
            obj.testtotalReward = 0;
            obj.epsilon = -Inf;
            obj.initSim();
            obj.randomInitState();            
            
            for testIter  = 1:simLength                
                action = obj.selectAction();
                [~,~,reward,next_state,done] = obj.doAction(action);
                obj.state = next_state;
                obj.testtotalReward = obj.testtotalReward + reward;
                if done
                    break; % If Reset Condition is Reached; Break
                end
            end
        end
    end
    
    %% Private methods:
    methods (Access = 'private')
        
        %% Get the Q-value:
        function Qval = genQvalue(obj,state)            
            Qval = obj.net(state');            
        end
        
        %% Train the network from buffer memory:
        function trainOnBuffer(obj)
            % Get sample data from buffer memory:
            sampledrawfromBuffer = datasample(obj.replayBuffer,...
                min(obj.sampleSize,length(obj.replayBuffer)));
            stateBatch = sampledrawfromBuffer(:,(1:4));
            actionBatch = sampledrawfromBuffer(:,5);
            rewardBatch = sampledrawfromBuffer(:,6);
            nextstateBatch = sampledrawfromBuffer(:,(7:10));
            doneBatch = sampledrawfromBuffer(:,11);
            valueBatch = zeros(length(obj.actions),1);
            
            for count = 1:length(sampledrawfromBuffer)
                % Get the Q-value:
                value = obj.genQvalue(stateBatch(count,:));
                % Get the action index:
                aIdx = find(~(obj.actions-actionBatch(count)));
                if doneBatch(count)
                    % For last action, Q(s,a) = r
                    value(aIdx) = rewardBatch(count);
                else
                    % Otherwise, Q(s,a) = r + gamma * max(s',:)
                    value(aIdx) = rewardBatch(count) + obj.gamma...
                        *max(obj.genQvalue(nextstateBatch(count,:)));
                end
                valueBatch(:,count) = value;
            end
            % Retrain the neural network with the new batch:
            obj.net = setwb(obj.net,obj.netWeights);
            obj.net = train(obj.net, stateBatch',valueBatch);
            % Update the weights of the neural network in memory:
            obj.netWeights = getwb(obj.net);
        end
        
        %% Select an action with an epsilon-greedy policy:
        function selectedAction = selectAction(obj)
            if rand <= obj.epsilon  
                % Perform a random action:
                actionIndex = randi(obj.actionCardinality,1);                
            else
                % Select the greedy action:
                obj.Q = obj.genQvalue(obj.state);                
                [~,actionIndex] = max(obj.Q);                
            end
            % Apply the corresponding action:
            selectedAction = obj.actions(actionIndex);
        end
    end
end