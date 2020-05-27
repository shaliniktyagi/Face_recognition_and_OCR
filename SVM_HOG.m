clear all;close all;clc;
faceDatabase = imageSet('C:\Users\Shali\Documents\MSC_DS_SHALU\Computer_Vision\computer_vision_coursework\coursework_folder\cropped_images','recursive');
%montage(faceDatabase(4).ImageLocation);
rng(8);
[training,test] = partition(faceDatabase,[0.7 0.3]);
% Extract HOG Features for Training Set
trainingSets = numel(training);         % number of categories 
trainingSetSize = sum([training.Count]);        % total number of training images
featureCount = 1;
 
 % loop through each image, extract and add HOG Features to trainingFeatures
 % matrix, and create vector holding label data per image
 for i=1:trainingSets
     label = training(i).Description;
%      label = str2num(label);
     for j = 1:training(i).Count
         %disp(j);
         I=read(training(i),j);
         %I=imresize(I,[100,100]);
         trainingFeatures(featureCount,:) = extractHOGFeatures(I);
         trainingLabel{featureCount} = label;
         %disp(trainingLabel);
         featureCount = featureCount + 1;
     end
 end
% 
% %% Train SVM using extracted HOG features and class labels 
 faceClassifier = fitcecoc(trainingFeatures,trainingLabel);
% 
% %% Extract HOG Features for Test Set
% 
 testSets = numel(test);         % number of categories 
 testSetSize = sum([test.Count]);        % total number of test images
 testFeatureCount = 1;
% 
% % loop through each image in the test set, extract the HOG Features and the
% % labels
disp('starting creating test sets');
 for i=1:testSets
     actualLabel = test(i).Description;
     for j=1:test(i).Count
         %disp(j);
         J=read(test(i),j);
         %J=imresize(J,[100,100]);
         testFeatures(testFeatureCount,:) = extractHOGFeatures(J);
         actualTestLabels{testFeatureCount} = actualLabel;
         %disp(actualTestLabels(testFeatureCount, :));
         testFeatureCount = testFeatureCount + 1;
     end
 end
% 
% %% Predict matching labels for all images in the test set
% 
 testLabels = predict(faceClassifier, testFeatures);
% 
% %% Calculate accuracy of test imageset
% 
 correctMatches = 0;
% 
% % Check whether predicted label matches actual label for each image, and
% % count how many are equivalent
 for i=1:testSetSize
     if strcmp(testLabels{i}, actualTestLabels{i})
        correctMatches = correctMatches + 1;
     end
 end
% 
% % calculate accuracy
 accuracy = correctMatches/testSetSize;
 
 CM_mdl = confusionmat(actualTestLabels, testLabels);
 Accuracy2 = 100*sum(diag(CM_mdl))./sum(CM_mdl(:))
 %% save model
  save('SVM_HOG.mat','faceClassifier')
