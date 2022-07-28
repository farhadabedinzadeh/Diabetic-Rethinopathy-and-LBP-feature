clc
clear all;
close all;

%% '================ Written by Farhad AbedinZadeh ================'
%                                                                 %
% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %                                                                
%% Healthy Data
path='.\healthy\*.jpg' ;
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    im =imread(fn);
    Im1=rgb2gray(im);
    
    wname='sym4';
    [cA,cH,cV,cD] = dwt2(Im1,wname);
    
    %features
    feature1 = extractLBPFeatures(cA);
    feature2 = extractLBPFeatures(cH);
    feature3 = extractLBPFeatures(cV);
    feature4 = extractLBPFeatures(cD);
    
    feature_healthy(i,:)=[feature1 feature2 feature3 feature4];
    
end

%% Diabeteic Data

path='.\diabetic\*.jpg' ;
files=dir(path);

for i = 1:length(files)
    fn = [path(1:end-5) files(i,1).name];
    Im =imread(fn);
    Im1=rgb2gray(Im);
    
    wname='sym4';
    [cA,cH,cV,cD] = dwt2(Im1,wname);
    
    %features
    feature5 = extractLBPFeatures(cA);
    feature6 = extractLBPFeatures(cH);
    feature7 = extractLBPFeatures(cV);
    feature8 = extractLBPFeatures(cD);
    
    feature_diabetic(i,:)=[feature5 feature6 feature7 feature8];
    
end


%% Classification

features=[feature_healthy;feature_diabetic];
output=[zeros(15,1);ones(15,1)];


%classiation learner
mix=[features,output];

trainingData=mix;

%% knn
[trainedClassifier, validationAccuracy] = allknn(trainingData)

T2=features;
yfit = trainedClassifier.predictFcn(T2);

plotconfusion(output' , yfit' )
%% SVM
[Accuracy,Sensitivity,Specificity,Precision,F1] = mSVM_opt(features,output,5) 