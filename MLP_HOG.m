close all;clear all;%clc;
%% Load all images in the Faces folder
Facedata = imageSet('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\coursework_folder\cropped_images','recursive');

%% Split Database into Training and Test Sets
rng(1);
[training,test] = partition(Facedata,[0.7 0.3]);

%% Extract HOG Features for Training Set
keySet=[];
valueSet=[];
trainingSets = numel(training);        % number of categories 
trainingSetSize = sum([training.Count]);      % total number of training images

trainingLabels = zeros(trainingSets, trainingSetSize);      
featureCount = 1;

% loop through each image, extract and add HOG Features to trainingFeatures
% matrix, and populate labels matrix by setting a 1 at the index of the
% label number
for i=1:trainingSets
    Label = training(i).Description;
    Label = str2num(Label);
    valueSet=[valueSet Label];
    keySet = [keySet i];
    for j = 1:training(i).Count
        trainingFeatures(featureCount,:) = extractHOGFeatures(read(training(i),j));
        trainingLabels(i, featureCount) = 1;
        featureCount = featureCount + 1;
    end         
end
M = containers.Map(keySet,valueSet);
%% Set up and train feedforward neural network

net = feedforwardnet(40, 'trainscg');
%net = feedforwardnet(15, 'trainscg');
net = configure(net,trainingFeatures',trainingLabels);
net = train(net,trainingFeatures', trainingLabels);

% net = feedforwardnet(10, 'trainlm');
% net = train(net,trainingFeatures, trainingLabels);

%% Extract HOG Features for Test Set

testSets = numel(test);         % number of categories 
testSetSize = sum([test.Count]);        % total number of test images

testLabelsMatrix = zeros(testSets, testSetSize);      % zeros matrix for labels
testFeatureCount = 1;

% loop through each image in the test set, extract the HOG Features and the
% labels
for i=1:testSets
%     actualLabel = test(i).Description;
    for j=1:test(i).Count
        testFeatures(testFeatureCount,:) = extractHOGFeatures(read(test(i),j));
        testLabelsMatrix(i, testFeatureCount) = 1;
        testFeatureCount = testFeatureCount + 1;
        %trainingLabel{featureCount} = label;
%         check_actualTestLabels(testFeatureCount, :) = str2num(actualLabel);
    end
end

%% Predict matching labels for all images in the test set
testOutputs = net(testFeatures');

% loop through the output from the network - the closest match is the index
% where the maximum value is per column. At the same time, get the actual
% labels for the test dataset.
for i = 1 : testSetSize
    [value testLabels(1,i)] = max(testOutputs(:,i));
    testLabels2(1,i)=M(testLabels(1,i));
    actualTestLabels(i) = find(testLabelsMatrix(:,i));
    actualTestLabels2(i)=M(actualTestLabels(i));
end

% actualTestLabelsT=actualTestLabels';
% check_actualTestLabelsT=check_actualTestLabels(2:end,:);
% final = [testLabels' actualTestLabelsT];
% final2 = [actualTestLabels2' testLabels2'];

% tab=[actualTestLabels2' actualTestLabels'];
 

%% Calculate accuracy of test imageset
accuracy = (sum(testLabels == actualTestLabels) / testSetSize)*100;
accuracy_mlp = (sum(actualTestLabels2==testLabels2)/testSetSize)*100
%% save model
save('MLP_HOG.mat','net','M')