% cleanup.m     E.Anderlini@ucl.ac.uk     28/01/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script is used to clean up the work space after running
% reinforcement learning problems simulations.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Root directory of this running .m file:
projectRootDir = fileparts(mfilename('fullpath'));

%% Remove project directories from path:
rmpath(fullfile(projectRootDir,'algorithms'));
rmpath(fullfile(projectRootDir,'environments'));
rmpath(fullfile(projectRootDir,'scripts'));

%% leave no trace...
clear projectRootDir;
clear;