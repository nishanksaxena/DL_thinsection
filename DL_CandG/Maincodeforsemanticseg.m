clear all;clc;close all;

%% Set up directories
cd('XYZ\DL_CandG');
addpath('XYZ\DL_CandG\Functions');
pathnamefortraining = 'XYZ\DL_CandG\Testingset10class';

%% Load label definitions
Labeldefinition;

%% Parameters for deep learning
MaxEopchs = 100;
Networktype = 18; %A
weight_type = 0; %X -> 0,1,2,3,4
batchsize10class = 8; %Y -> 4,8
initiallearningrate = 1e-2; %Z -> 1e-2,1e-4
L2Reg = 5e-3; %M -> 5e-2, 5e-3, 5e-4, 5e-6 (5e-1 is too regularized)
Augmentflag = 1; %N -> 1 or 0

%% Run deep learning
[net10class] = Deeplearningproduction(MaxEopchs,Networktype,batchsize10class,classes10class,classeslabelIDs10class,pathnamefortraining,weight_type,initiallearningrate,L2Reg,Augmentflag);

%% Save network as Matlab data
cd('XYZ\DL_CandG\DIRAC'); save net10class.mat net10class;

%% Test code
I_inp = imread('test_image.tif');
[S_cat] = semanticseg(I_inp,net10class);
