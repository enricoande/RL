% DQNLearn.m     E.Anderlini@ucl.ac.uk     26/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is designed to test the DQNLearn and CartPole classes.
%
% This script has been modified from Girish Joshi's original file.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
close all;

%% Training:
% Initialise training:
CartPoleQlearn = DQNLearn(20,0.99,0.8,false);

% Start training:
CartPoleQlearn.QLearningTrain();

%% Testing the DQN Network
test_simSteps = 100;
CartPoleQlearn.DQNTest(test_simSteps);