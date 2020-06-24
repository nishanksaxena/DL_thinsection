function [net] = Deeplearningproduction(MaxEopchs,Networktype,batchsize,classes,classeslabelIDs,pathnamefortraining,weight_type,initiallearningrate,L2Reg,Augmentflag)
%[net] = Deeplearningtraining(classes,classeslabelIDs,pathname)
% Train SegNet based CNN using gray images and labeled images
% Written by Nishank Saxena (Shell, 2018)
% Inputs: 
%     MaxEopchs: Epochs for deep learning
%     Networktype: 18 (ResNet), 19 (VGG19), or 16 (VGG16)
%     batchsize: batch size for deep learning
%     classes: list of pore or mineral classes
%     classeslabelIDs: label ID (integers) corresponding to classes
%     pathnamefortraining: pathname for training folder containing
%     subfolders of "images" and "labels"
%     weight_type: weight type for class balancing
%     initiallearningrate: learning rate
%     L2Reg: L2 regularization
%     Augmentflag: Augment training data (1 for yes, 0 for no)
%
% Outputs:
%     net: trained network

imdsTrain = imageDatastore(fullfile(pathnamefortraining,'images'));
pxdsTrain = pixelLabelDatastore(fullfile(pathnamefortraining,'labels'),classes,classeslabelIDs);

I = readimage(imdsTrain,1);

if Networktype == 16
vgg16();
lgraph = segnetLayers(size(I),numel(classes),'vgg16');
end
if Networktype == 19
vgg19();
lgraph = segnetLayers(size(I),numel(classes),'vgg19');
end
if Networktype == 18
resnet18();
lgraph = helperDeeplabv3PlusResnet18(size(I), numel(classes));
end

if Networktype == 20
lgraph = unetLayers(size(I),numel(classes),'EncoderDepth',4);
end

tbl = countEachLabel(pxdsTrain);

if weight_type == 0
classWeights = 'none';
end

if weight_type == 1
%Inverse frequency balancing weights each class such that underrepresented classes are given higher weight:
totalNumberOfPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalNumberOfPixels;
classWeights = 1./frequency;
end

if weight_type == 2
%Median frequency balancing weights each class using the median frequency. The weight for each class is defined as median(imageFreq)/imageFreq(c), where imageFreq(c) represents the number of pixels of the class divided by the total number of pixels in images that had an instance of the class (c):
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq)./ imageFreq;
end

if weight_type == 3
imageFreq = tbl.PixelCount ./ tbl.ImagePixelCount;
classWeights = median(imageFreq) ./ imageFreq;
classWeights = classWeights.^0.2; 
end

pxLayer = pixelClassificationLayer('Name','labels','ClassNames',tbl.Name,'ClassWeights',classWeights);

if Networktype == 18
lgraph = replaceLayer(lgraph,"classification",pxLayer);
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9, ...
    'InitialLearnRate',initiallearningrate, ...
    'L2Regularization',L2Reg, ...
    'MaxEpochs',MaxEopchs, ...  
    'MiniBatchSize',batchsize, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2, ...
    'ExecutionEnvironment','multi-gpu', ...
    'Plots','training-progress');

end

if Networktype == 19 || Networktype == 16
lgraph = removeLayers(lgraph,'pixelLabels');
lgraph = addLayers(lgraph, pxLayer);
lgraph = connectLayers(lgraph,'softmax','labels');
options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise',...
    'LearnRateDropPeriod',5,...
    'LearnRateDropFactor',0.1,...
    'Momentum',0.9, ...
    'InitialLearnRate',initiallearningrate, ...
    'L2Regularization',L2Reg, ...
    'MaxEpochs',MaxEopchs, ...  
    'MiniBatchSize',batchsize, ...
    'Shuffle','every-epoch', ...
    'CheckpointPath', tempdir, ...
    'VerboseFrequency',2, ...
    'ExecutionEnvironment','multi-gpu', ...
        'Plots','training-progress');
end


augmenter = imageDataAugmenter('RandXReflection',true,'RandYReflection',true);

if Augmentflag == 1
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain,'DataAugmentation',augmenter);
else
pximds = pixelLabelImageDatastore(imdsTrain,pxdsTrain);    
end
[net, info] = trainNetwork(pximds,lgraph,options);
end

