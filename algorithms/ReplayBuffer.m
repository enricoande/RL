% ReplayBuffer.m     E.Anderlini@ucl.ac.uk     21/02/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This class prepares a data structure for implementing experience replay.
%
% References:
% https://spinningup.openai.com/
% https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef ReplayBuffer
    %% Properties:
    properties (Access = private)
        buffer_size;       % max. size of the buffer for experience replay
        count;             % no. of  items currently in the buffer
        count_1;           % no. of  items currently in the buffer - 1 
        buffer;            % buffer memory for double values
        buffer_terminal;   % buffer memory for boolean value
    end
    
    %% Methods:
    methods (Access = public)
        %% Class constructor:
        function obj = ReplayBuffer(buffer_size)
            % Set the default buffer size:
            if nargin<1
                buffer_size = 10000;
            end
            
            % Initialise the class properties:
            obj.buffer_size = buffer_size;
            obj.count = 0;
            obj.count_1 = 0;
            obj.buffer = nan(obj.buffer_size,4);
            obj.buffer = false(obj.buffer_size,1);
        end
        
        %% Function that adds experience to the buffer:
        function obj = add(obj,s,a,r,sp,d)
            if obj.count < obj.buffer_size
                % Add the current point to the buffer:
                obj.count_1 = obj.count;
                obj.count = obj.count + 1;
                obj.buffer(obj.count,:) = [s,a,r,sp];
                obj.buffer_terminal(obj.count) = d;
            else
                % Replace the first point with the current one:
                obj.buffer(1:obj.count_1,:) = obj.buffer(2:obj.count,:);
                obj.buffer_terminal(1:obj.count_1) = ...
                    obj.buffer_terminal(2:obj.count);
                obj.buffer(obj.count,:) = [s,a,r,sp];
                obj.buffer_terminal(obj.count) = d;
            end
        end
            
        %% Function that returns the number of points in the buffer:
        function sb = size(obj)
            sb = obj.count;
        end
        
        %% Function that extracts a sample batch from the buffer:
        function [v_batch,d_batch] = sample_batch(obj,batch_size)
            % Ensure matching of doubles and boolean variables:
            if obj.count < batch_size
                ind = randperm(obj.count);
            else
                ind = randperm(batch_size);
            end
            
            % Return the sample batch:
            v_batch = obj.buffer(ind,:);
            d_batch = obj.buffer_terminal(ind,:);
        end
    end
end