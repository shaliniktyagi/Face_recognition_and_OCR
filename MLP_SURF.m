clear all;clc;close all;
rng(8);
%% Load all images in the Faces folder
faceDatabase = imageSet('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\coursework_folder\cropped_images','recursive');

%% Split Database into Training & Test Sets
rng(1)
[training,test] = partition(faceDatabase,[0.7 0.3]);

%% Returns a bag of features object for the training set
bag = bagOfFeatures(training);

%% Extract Bag-of-Words Features for Training Set
valueSet=[]
keySet=[]
training_sets = numel(training);    % number of categories 
training_set_size = sum([training.Count]);       % total number of training images

trainingFeatures = encode(bag,training).';      
trainingLabels = zeros(training_sets, training_set_size);   % zeros matrix for labels
featureCount = 1;

% loop through each image, and populates labels matrix by setting a 1 at 
% the index of the label number
for i=1:training_sets
    Label = training(i).Description;
    Label = str2num(Label);
    valueSet=[valueSet Label];
    keySet = [keySet i];
    for j = 1:training(i).Count
        trainingLabels(i, featureCount) = 1;
        featureCount = featureCount + 1;
    end
end
M = containers.Map(keySet,valueSet);
%% Set up and train feedforward neural network

net = feedforwardnet(100, 'trainscg');
net = configure(net,trainingFeatures,trainingLabels);
net = train(net,trainingFeatures, trainingLabels);

%% Extract HOG Features for Test Set

testSets = numel(test);         % number of categories 
testSetSize = sum([test.Count]);        % total number of test images

testLabelsMatrix = zeros(testSets, testSetSize);      % zeros matrix for labels
testFeatureCount = 1;

% creates feature vector that represents a histogram of visual word 
% occurrences from the test set
testFeatures = encode(bag,test).';

% loop through each image in the test set and record the labels
for i=1:testSets
    for j=1:test(i).Count
        testLabelsMatrix(i, testFeatureCount) = 1;
        testFeatureCount = testFeatureCount + 1;
    end
end

%% Predict matching labels for all images in the test set
testOutputs = net(testFeatures);

% loop through the output from the network - the closest match is the index
% where the maximum value is per column. At the same time, get the actual
% labels for the test dataset.
for i = 1 : testSetSize
    [value testLabels(1,i)] = max(testOutputs(:,i));
    testLabels2(1,i)=M(testLabels(1,i));
    actualTestLabels(i) = find(testLabelsMatrix(:,i));
    actualTestLabels2(i)=M(actualTestLabels(i));
end
disp('Done');
%% Calculate accuracy of test imageset
accuracy = sum(testLabels == actualTestLabels) / testSetSize *100;
accuracy2 = (sum(actualTestLabels2==testLabels2)/testSetSize)*100

 save('MLP_SURF.mat','net','bag','M');