%% Load data
clear;
clc;
load('Data_Train.mat');
load('Label_Train.mat');
load('Data_Test.mat');
%% Test Bayes decisio rule
%train accuracy
pred_label_train = Bayes_decesion_rule(Data_Train,Label_Train,Data_Train);
right = 0;
for i = 1:size(Label_Train,1)
    if pred_label_train(i) == Label_Train(i)
        right=right+1;
    end
end
train_accuracy = right/size(Label_Train,1);
fprintf('Training accuracy of Bayes decesion rule is %6.4f\n',train_accuracy)
%predicted label
pred_label_test = Bayes_decesion_rule(Data_Train,Label_Train,Data_test);
save('pred_label_Bayes.mat','pred_label_test')

%% Test Fisher discriminant analysis
pred_label_train = Fisher_discriminant_analysis(Data_Train,Label_Train,Data_Train);
right = 0;
for i = 1:size(Label_Train,1)
    if pred_label_train(i) == Label_Train(i)
        right=right+1;
    end
end
train_accuracy = right/size(Label_Train,1);
fprintf('Training accuracy of Fisher discriminant analysis is %6.4f\n',train_accuracy)
%predicted label
pred_label_test = Fisher_discriminant_analysis(Data_Train,Label_Train,Data_test);
save('pred_label_Fisher.mat','pred_label_test')
