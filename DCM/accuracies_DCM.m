% Input to the function: DCM Output CSV file and a Random Seed value.
% Output: 4 lists of accuracies for: RandomForest Tuned, AdaBoost Tuned,
% KNN Tuned, Support Vector Machine Tuned. 
% Each list contains Binary and Multi-Class Accuracies (in the following order):
% Placebo vs Galantamine, Placebo vs Amisulpride, Placebo vs Levodopa,
% Placebo vs Biperdine, Biperdine vs Galantamine, Amisulpride vs Levodopa,
% Placebo and 4 Drugs (5 Class) Classification.

function [accuracies_rf_DCM, accuracies_adb_DCM, accuracies_knn_DCM, accuracies_svm_DCM] = accuracies_DCM(data, seed, drugs)
    if nargin<3
        drugs = 1:7;
    end
    % Extracting Features and Labels
    X_DCM = data.dcm_features;
    y_DCM = data.drug_label;
    y_cat_DCM = categorical(y_DCM);
    X = X_DCM;
    y = y_DCM;

    placebo_idx = find(y_cat_DCM == 'Placebo');
    amisulpride_idx = find(y_cat_DCM == 'Amisulpride');
    biperdine_idx = find(y_cat_DCM == 'Biperdine');
    galantamine_idx = find(y_cat_DCM == 'Galantamine');
    levodopa_idx = find(y_cat_DCM == 'Levodopa');

    X_galantamine_placebo = vertcat(X(placebo_idx,:), X(galantamine_idx,:));
    y_galantamine_placebo = vertcat(y(placebo_idx,:), y(galantamine_idx,:));
    
    X_amisulpride_placebo = vertcat(X(placebo_idx,:), X(amisulpride_idx,:));
    y_amisulpride_placebo = vertcat(y(placebo_idx,:), y(amisulpride_idx,:));
    
    X_levodopa_placebo = vertcat(X(placebo_idx,:), X(levodopa_idx,:));
    y_levodopa_placebo = vertcat(y(placebo_idx,:), y(levodopa_idx,:));
    
    X_biperdine_placebo = vertcat(X(placebo_idx,:), X(biperdine_idx,:));
    y_biperdine_placebo = vertcat(y(placebo_idx,:), y(biperdine_idx,:));
    
    X_biperdine_galantamine = vertcat(X(galantamine_idx,:), X(biperdine_idx,:));
    y_biperdine_galantamine = vertcat(y(galantamine_idx,:), y(biperdine_idx,:));
    
    X_amisulpride_levodopa = vertcat(X(amisulpride_idx,:), X(levodopa_idx,:));
    y_amisulpride_levodopa = vertcat(y(amisulpride_idx,:), y(levodopa_idx,:));
    
    Xs = {X_galantamine_placebo, X_amisulpride_placebo, X_levodopa_placebo, X_biperdine_placebo, X_biperdine_galantamine, X_amisulpride_levodopa, X};
    ys = {y_galantamine_placebo, y_amisulpride_placebo, y_levodopa_placebo, y_biperdine_placebo, y_biperdine_galantamine, y_amisulpride_levodopa, y};

    rng(seed);

    % Define TreeBagger Hyperparams
    num_trees = 50;
    min_leaf_size = 5;
    max_num_splits = 10;
    num_PTS = 2;
    
    % Define SVM Hyperparameters.
    t_svm = templateSVM('BoxConstraint', 0.045284, 'KernelScale', 0.0019785);
    
    % Define Adaboost Hyperparams
    t_adb = templateTree('MinLeafSize', 71, 'MaxNumSplits', 3);
    
    accuracies_rf_DCM = zeros(1,7);
    accuracies_adb_DCM = zeros(1,7);
    accuracies_knn_DCM = zeros(1,7);
    accuracies_svm_DCM = zeros(1,7);

    for drug = drugs
        X = Xs{drug};
        y = ys{drug};
        
        cv = cvpartition(y,'Kfold', 10, 'Stratify', true);
    
        % Initialize class performance objects
        cp_rf_DCM = classperf(y);
        cp_adb_DCM = classperf(y);
        cp_knn_DCM = classperf(y);
        cp_svm_DCM = classperf(y);
    
        % Perform 10-fold cross-validation
        for i = 1:10
            train = cv.training(i); 
            test = cv.test(i);
            
            % TreeBagger
            rf_DCM = TreeBagger(num_trees, X(train,:), y(train,:), 'Method', 'classification', 'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
            class_rf_DCM = predict(rf_DCM, X(test,:));
    
            % % AdaBoost
            % if drug == 7
            %     % AdaBoostM2 for Multi-Class
            %     adb_DCM = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM2', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.008179);
            %     class_adb_DCM = predict(adb_DCM, X(test,:));
            % else
            %     % AdaBoostM1 for Binary-Class
            %     adb_DCM = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM1', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.008179);
            %     class_adb_DCM = predict(adb_DCM, X(test,:));
            % end
            % 
            % % KNN
            % knn_DCM = fitcknn(X(train,:), y(train,:), 'NumNeighbors', 50, 'Distance', 'minkowski');
            % class_knn_DCM = predict(knn_DCM, X(test,:));
    
            % SVM
            svm_DCM = fitcecoc(X(train,:), y(train,:), 'Coding', 'onevsone', 'Learners', t_svm);
            class_svm_DCM = predict(svm_DCM, X(test,:));
            
            % Update the class performance objects
            classperf(cp_rf_DCM, class_rf_DCM, test);
            %classperf(cp_knn_DCM, class_knn_DCM, test);
            classperf(cp_svm_DCM, class_svm_DCM, test);
            %classperf(cp_adb_DCM, class_adb_DCM, test);
        end
    
        accuracy_rf_DCM = 1 - cp_rf_DCM.ErrorRate;
        %accuracy_knn_DCM = 1 - cp_knn_DCM.ErrorRate;
        accuracy_svm_DCM = 1 - cp_svm_DCM.ErrorRate;
        %accuracy_adb_DCM = 1 - cp_adb_DCM.ErrorRate;
        accuracies_rf_DCM(drug) = accuracy_rf_DCM;
        %accuracies_knn_DCM(drug) = accuracy_knn_DCM;
        accuracies_svm_DCM(drug) = accuracy_svm_DCM;
        %accuracies_adb_DCM(drug) = accuracy_adb_DCM;
    end
end