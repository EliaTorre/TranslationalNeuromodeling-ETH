%Training and validation classifier accuracy on generative embedding from
%Neural HGF
numFolds = 4;
seed = 25;
rng(seed);
%%%%%%%%%%%%%%%%%%%%

%Process eeg data by Neural Net and save the generative embedding for each
%fold
%First we load data, preprocess, load networks, process data and save
%embedding

disp('Saving Neural Embedding');
disp('');
for k=1:numFolds
    disp('====================');
    disp('Fold number');
    disp(k);
    fold_data_filename = sprintf('hgfdata_%d_%d.mat',k,numFolds);
    data_loaded = load(fold_data_filename);

    full_eeg = zeros([139,63,2,size(data_loaded.data,2)]);
    labels = cell(size(data_loaded.data,2),1);
    for i=1:size(data_loaded.data,2)
        full_eeg(:,:,:,i) = data_loaded.data(i).eeg_data;
        labels(i) = {data_loaded.data(i).drug_label};
    end
    full_eeg = dlarray(full_eeg, 'SSCB');
    meanfe = mean(full_eeg, 1);
    stdfe = std(full_eeg, 1);
    full_eeg = (full_eeg-meanfe)./(stdfe);
    full_eeg = reshape(full_eeg,139,63,1,[]);
    
    
    dsTest = arrayDatastore(full_eeg,IterationDimension=4);
    numOutputs = 1;
    
    mbqtest = minibatchqueue(dsTest,numOutputs, ...
        MiniBatchSize = 1, ...
        MiniBatchFcn=@preprocessMiniBatch, ...
        MiniBatchFormat="SSCB", ...
        PartialMiniBatch="return");

    net_filename = sprintf('neural_hgf_trained_foldnr_%d_of_%d.mat',k,numFolds);
    Nets = load(net_filename);
    net_Z2_X = Nets.net_Z2_X;
    net_Z1_Z2X = Nets.net_Z1_Z2X;
    net_Xhat_Z1 = Nets.net_Xhat_Z1;

    params_pred = modelPredictions_params(net_Z2_X, net_Z1_Z2X, net_Xhat_Z1, mbqtest);
    params_pred = reshape(params_pred,139,4,2,[]);
    params_pred = reshape(params_pred,139*4*2,[]);

    data = struct;
    for i=1:size(data_loaded.data,2)
        data(i).subject_id = data_loaded.data(i).subject_id;
        data(i).train_or_val = data_loaded.data(i).train_or_val;
        data(i).drug_label = data_loaded.data(i).drug_label;
        data(i).embedding = params_pred(:,i);
    end
    
    data_table = struct2table(data);
    save_processed = sprintf('5Drugs_nhgf_%d_%d.csv',k,numFolds);
    writetable(data_table, save_processed);
end
%%%%%%%%%%%%%%%%%%%%%%%%%
%Training Classifiers (Cross validated) on generative embedding using same
%train/validation splits as above.

clearvars -except seed numFolds num_permutations
rng(seed);
accuracies_rf = cell(numFolds);
accuracies_adb = cell(numFolds);
accuracies_knn = cell(numFolds);
accuracies_svm = cell(numFolds);
disp('Cross-validating for accuaracy in classification')
disp('')
for k=1:numFolds
    disp('==============');
    disp('Fold number');
    disp(k)

    filename = sprintf('5Drugs_nhgf_%d_%d.csv',k,numFolds);
    data_drugs = readtable(filename, 'ReadVariableNames', true);
    X = table2array(data_drugs(:,4:end));
    Y = table2array(data_drugs(:,3));
    idx = table2array(data_drugs(:,2));

    train_idxs = ~idx;
    test_idxs = ~train_idxs;
    
    y_cat = categorical(Y);
    placebo_idx_global = (y_cat == 'Placebo');
    amisulpride_idx_global = (y_cat == 'Amisulpride');
    biperdine_idx_global = (y_cat == 'Biperdine');
    galantamine_idx_global = (y_cat == 'Galantamine');
    levodopa_idx_global = (y_cat == 'Levodopa');
    
    X_galantamine_placebo_train = vertcat(X((placebo_idx_global&train_idxs),:), X((galantamine_idx_global&train_idxs),:));
    y_galantamine_placebo_train = vertcat(Y((placebo_idx_global&train_idxs),:), Y((galantamine_idx_global&train_idxs),:));
    X_galantamine_placebo_test = vertcat(X((placebo_idx_global&test_idxs),:), X((galantamine_idx_global&test_idxs),:));
    y_galantamine_placebo_test = vertcat(Y((placebo_idx_global&test_idxs),:), Y((galantamine_idx_global&test_idxs),:));
    X_galantamine_placebo_global = vertcat(X_galantamine_placebo_test, X_galantamine_placebo_train);
    y_galantamine_placebo_global = vertcat(y_galantamine_placebo_test, y_galantamine_placebo_train);
    idx_galantamine_placebo_test = (1:size(y_galantamine_placebo_test,1)).';
    
    X_amisulpride_placebo_train = vertcat(X((placebo_idx_global&train_idxs),:), X((amisulpride_idx_global&train_idxs),:));
    y_amisulpride_placebo_train = vertcat(Y((placebo_idx_global&train_idxs),:), Y((amisulpride_idx_global&train_idxs),:));
    X_amisulpride_placebo_test = vertcat(X((placebo_idx_global&test_idxs),:), X((amisulpride_idx_global&test_idxs),:));
    y_amisulpride_placebo_test = vertcat(Y((placebo_idx_global&test_idxs),:), Y((amisulpride_idx_global&test_idxs),:));
    X_amisulpride_placebo_global = vertcat(X_amisulpride_placebo_test, X_amisulpride_placebo_train);
    y_amisulpride_placebo_global = vertcat(y_amisulpride_placebo_test, y_amisulpride_placebo_train);
    idx_amisulpride_placebo_test = (1:size(y_amisulpride_placebo_test,1)).';
    
    X_levodopa_placebo_train = vertcat(X((placebo_idx_global&train_idxs),:), X((levodopa_idx_global&train_idxs),:));
    y_levodopa_placebo_train = vertcat(Y((placebo_idx_global&train_idxs),:), Y((levodopa_idx_global&train_idxs),:));
    X_levodopa_placebo_test = vertcat(X((placebo_idx_global&test_idxs),:), X((levodopa_idx_global&test_idxs),:));
    y_levodopa_placebo_test = vertcat(Y((placebo_idx_global&test_idxs),:), Y((levodopa_idx_global&test_idxs),:));
    X_levodopa_placebo_global = vertcat(X_levodopa_placebo_test, X_levodopa_placebo_train);
    y_levodopa_placebo_global = vertcat(y_levodopa_placebo_test, y_levodopa_placebo_train);
    idx_levodopa_placebo_test = (1:size(y_levodopa_placebo_test,1)).';
    
    X_biperdine_placebo_train = vertcat(X((placebo_idx_global&train_idxs),:), X((biperdine_idx_global&train_idxs),:));
    y_biperdine_placebo_train = vertcat(Y((placebo_idx_global&train_idxs),:), Y((biperdine_idx_global&train_idxs),:));
    X_biperdine_placebo_test = vertcat(X((placebo_idx_global&test_idxs),:), X((biperdine_idx_global&test_idxs),:));
    y_biperdine_placebo_test = vertcat(Y((placebo_idx_global&test_idxs),:), Y((biperdine_idx_global&test_idxs),:));
    X_biperdine_placebo_global = vertcat(X_biperdine_placebo_test, X_biperdine_placebo_train);
    y_biperdine_placebo_global = vertcat(y_biperdine_placebo_test, y_biperdine_placebo_train);
    idx_biperdine_placebo_test = (1:size(y_biperdine_placebo_test,1)).';
    
    X_biperdine_galantamine_train = vertcat(X((galantamine_idx_global&train_idxs),:), X((biperdine_idx_global&train_idxs),:));
    y_biperdine_galantamine_train = vertcat(Y((galantamine_idx_global&train_idxs),:), Y((biperdine_idx_global&train_idxs),:));
    X_biperdine_galantamine_test = vertcat(X((galantamine_idx_global&test_idxs),:), X((biperdine_idx_global&test_idxs),:));
    y_biperdine_galantamine_test = vertcat(Y((galantamine_idx_global&test_idxs),:), Y((biperdine_idx_global&test_idxs),:));
    X_biperdine_galantamine_global = vertcat(X_biperdine_galantamine_test, X_biperdine_galantamine_train);
    y_biperdine_galantamine_global = vertcat(y_biperdine_galantamine_test, y_biperdine_galantamine_train);
    idx_biperdine_galantamine_test = (1:size(y_biperdine_galantamine_test,1)).';
    
    X_levodopa_amisulpride_train = vertcat(X((amisulpride_idx_global&train_idxs),:), X((levodopa_idx_global&train_idxs),:));
    y_levodopa_amisulpride_train = vertcat(Y((amisulpride_idx_global&train_idxs),:), Y((levodopa_idx_global&train_idxs),:));
    X_levodopa_amisulpride_test = vertcat(X((amisulpride_idx_global&test_idxs),:), X((levodopa_idx_global&test_idxs),:));
    y_levodopa_amisulpride_test = vertcat(Y((amisulpride_idx_global&test_idxs),:), Y((levodopa_idx_global&test_idxs),:));
    X_levodopa_amisulpride_global = vertcat(X_levodopa_amisulpride_test, X_levodopa_amisulpride_train);
    y_levodopa_amisulpride_global = vertcat(y_levodopa_amisulpride_test, y_levodopa_amisulpride_train);
    idx_levodopa_amisulpride_test = (1:size(y_levodopa_amisulpride_test,1)).';
    
    X_train = X(train_idxs,:);
    Y_train = Y(train_idxs,:);
    X_test = X(test_idxs,:);
    Y_test = Y(test_idxs,:);
    X_global = vertcat(X_test, X_train);
    Y_global = vertcat(Y_test, Y_train);
    idx_global_test = (1:size(Y_test)).';
    
    Xs_train = {X_galantamine_placebo_train, X_amisulpride_placebo_train, X_levodopa_placebo_train, X_biperdine_placebo_train, X_biperdine_galantamine_train, X_levodopa_amisulpride_train, X_train};
    ys_train = {y_galantamine_placebo_train, y_amisulpride_placebo_train, y_levodopa_placebo_train, y_biperdine_placebo_train, y_biperdine_galantamine_train, y_levodopa_amisulpride_train, Y_train};
    
    Xs_test = {X_galantamine_placebo_test, X_amisulpride_placebo_test, X_levodopa_placebo_test, X_biperdine_placebo_test, X_biperdine_galantamine_test, X_levodopa_amisulpride_test, X_test};
    ys_test = {y_galantamine_placebo_test, y_amisulpride_placebo_test, y_levodopa_placebo_test, y_biperdine_placebo_test, y_biperdine_galantamine_test, y_levodopa_amisulpride_test, Y_test};
    
    Xs = {X_galantamine_placebo_global, X_amisulpride_placebo_global, X_levodopa_placebo_global, X_biperdine_placebo_global, X_biperdine_galantamine_global, X_levodopa_amisulpride_global, X_global};
    ys = {y_galantamine_placebo_global, y_amisulpride_placebo_global, y_levodopa_placebo_global, y_biperdine_placebo_global, y_biperdine_galantamine_global, y_levodopa_amisulpride_global, Y_global};
    
    idx_test = {idx_galantamine_placebo_test, idx_amisulpride_placebo_test, idx_levodopa_placebo_test, idx_biperdine_placebo_test, idx_biperdine_galantamine_test, idx_levodopa_amisulpride_test, idx_global_test};
    
    % Define the number of trees in the ensemble
    num_trees = 50;
    min_leaf_size = 15;
    max_num_splits = 18;
    
    % Define SVM Hyperparameters.
    t_svm = templateSVM('BoxConstraint', 0.0013119, 'KernelScale', 7.99);
    t_adb = templateTree('MinLeafSize', 71, 'MaxNumSplits', 1);
    
    accuracies_rf_curr = {0,0,0,0,0,0,0};
    accuracies_adb_curr = {0,0,0,0,0,0,0};
    accuracies_knn_curr = {0,0,0,0,0,0,0};
    accuracies_svm_curr = {0,0,0,0,0,0,0};
    
    for drug = 1:7
        xtrain = Xs_train{drug};
        ytrain = ys_train{drug};
        xtest = Xs_test{drug};
        ytest = ys_test{drug};
        x = Xs{drug};
        y = ys{drug};
        idxs = idx_test{drug};
        
        % Initialize class performance objects
        cp_rf = classperf(y);
        cp_knn = classperf(y);
        cp_svm = classperf(y);
        cp_adb = classperf(y);
        
        % TreeBagger
        rf = TreeBagger(num_trees, xtrain, ytrain, 'Method', 'classification', 'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
        class_rf = predict(rf, xtest);
        
        % AdaBoost
        if drug == 7
            % AdaBoostM2 for Multi-Class
            adb = fitcensemble(xtrain, ytrain, 'Method', 'AdaBoostM2', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
            class_adb = predict(adb, xtest);
        else
            % AdaBoostM1 for Binary-Class
            adb = fitcensemble(xtrain, ytrain, 'Method', 'AdaBoostM1', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
            class_adb = predict(adb, xtest);
        end
    
        % KNN
        knn = fitcknn(xtrain, ytrain, 'NumNeighbors', 9, 'Distance', 'cosine');
        class_knn = predict(knn, xtest);
    
        % SVM
        svm = fitcecoc(xtrain, ytrain, 'Coding', 'onevsone', 'Learners', t_svm);
        class_svm = predict(svm, xtest);
        
        % Update the class performance objects
        classperf(cp_rf, class_rf, idxs);
        classperf(cp_knn, class_knn, idxs);
        classperf(cp_svm, class_svm, idxs);
        classperf(cp_adb, class_adb, idxs);
    
        accuracy_rf = 1 - cp_rf.ErrorRate;
        accuracy_knn = 1 - cp_knn.ErrorRate;
        accuracy_svm = 1 - cp_svm.ErrorRate;
        accuracy_adb = 1 - cp_adb.ErrorRate;
        accuracies_rf_curr{drug} = accuracy_rf;
        accuracies_knn_curr{drug} = accuracy_knn;
        accuracies_svm_curr{drug} = accuracy_svm;
        accuracies_adb_curr{drug} = accuracy_adb;
    end

    accuracies_rf{k} = accuracies_rf_curr;
    accuracies_knn{k} = accuracies_knn_curr;
    accuracies_svm{k} = accuracies_svm_curr;
    accuracies_adb{k} = accuracies_adb_curr;

end

accuracies_rf_mat = zeros(numFolds,7);
accuracies_knn_mat = zeros(numFolds,7);
accuracies_svm_mat = zeros(numFolds,7);
accuracies_adb_mat = zeros(numFolds,7);

for k=1:numFolds
    accuracies_rf_mat(k,:) = cell2mat(accuracies_rf{k});
    accuracies_knn_mat(k,:) = cell2mat(accuracies_knn{k});
    accuracies_svm_mat(k,:) = cell2mat(accuracies_svm{k});
    accuracies_adb_mat(k,:) = cell2mat(accuracies_adb{k});
end

acc_rf = mean(accuracies_rf_mat,1);
acc_knn = mean(accuracies_knn_mat,1);
acc_svm = mean(accuracies_svm_mat, 1);
acc_adb = mean(accuracies_adb_mat,1);

cv_results = struct;

cv_results(1).method = 'RandomForest';
cv_results(1).galantamine_placebo = acc_rf(1,1);
cv_results(1).amisulpride_placebo = acc_rf(1,2);
cv_results(1).levodopa_placebo = acc_rf(1,3);
cv_results(1).biperdine_placebo = acc_rf(1,4);
cv_results(1).biperdine_galantamine = acc_rf(1,5);
cv_results(1).levodopa_amisulpride = acc_rf(1,6);
cv_results(1).global = acc_rf(1,7);

cv_results(2).method = 'KNN';
cv_results(2).galantamine_placebo = acc_knn(1,1);
cv_results(2).amisulpride_placebo = acc_knn(1,2);
cv_results(2).levodopa_placebo = acc_knn(1,3);
cv_results(2).biperdine_placebo = acc_knn(1,4);
cv_results(2).biperdine_galantamine = acc_knn(1,5);
cv_results(2).levodopa_amisulpride = acc_knn(1,6);
cv_results(2).global = acc_knn(1,7);

cv_results(3).method = 'SVM';
cv_results(3).galantamine_placebo = acc_svm(1,1);
cv_results(3).amisulpride_placebo = acc_svm(1,2);
cv_results(3).levodopa_placebo = acc_svm(1,3);
cv_results(3).biperdine_placebo = acc_svm(1,4);
cv_results(3).biperdine_galantamine = acc_svm(1,5);
cv_results(3).levodopa_amisulpride = acc_svm(1,6);
cv_results(3).global = acc_svm(1,7);

cv_results(4).method = 'AdaBoost';
cv_results(4).galantamine_placebo = acc_adb(1,1);
cv_results(4).amisulpride_placebo = acc_adb(1,2);
cv_results(4).levodopa_placebo = acc_adb(1,3);
cv_results(4).biperdine_placebo = acc_adb(1,4);
cv_results(4).biperdine_galantamine = acc_adb(1,5);
cv_results(4).levodopa_amisulpride = acc_adb(1,6);
cv_results(4).global = acc_adb(1,7);

cv_results_table = struct2table(cv_results);
filename = sprintf('5Drugs_classifier_results_neuralhgf_nrfolds_%d.csv',numFolds);
writetable(cv_results_table, filename);

%%%%%%%%%%%%%%%%%%%%%%%
%Helper functions
function params = modelPredictions_params(netE,netE2,netD,mbq)

params = [];

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);
    % Forward through encoder.
    [Z2_tilde, Z2, mu2, l2] = predict(netE,X);
    [Z1, mu1, l1] = predict(netE2,Z2_tilde);

    % Forward through dencoder.
    XGenerated = predict(netD,Z1);

    params_temp = cat(2,extractdata(mu1),exp(extractdata(l1)),extractdata(mu2),exp(extractdata(l2)));

    % Extract and concatenate predictions.
    params = cat(3,params, params_temp);
end

end

function X = preprocessMiniBatch(dataX)

% Concatenate.
X = cat(4,dataX{:});

end