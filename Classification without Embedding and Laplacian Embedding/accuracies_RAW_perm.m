% Input to the function: data_drugs table, number of permutations and a Random Seed value.
% Output: 4 matrices (permutations x 7) of permuted label accuracies for: RandomForest Tuned, AdaBoost Tuned,
% KNN Tuned, Support Vector Machine Tuned. 
% Each column contains Binary and Multi-Class Accuracies (in the following order):
% Placebo vs Galantamine, Placebo vs Amisulpride, Placebo vs Levodopa,
% Placebo vs Biperdine, Biperdine vs Galantamine, Amisulpride vs Levodopa,
% Placebo and 4 Drugs (5 Class) Classification.

function [accuracies_rf_perm, accuracies_adb_perm, accuracies_knn_perm, accuracies_svm_perm] = accuracies_RAW_perm(data_drugs, permutations, seed)
    X = table2array(data_drugs(:,3:end));
    y = table2array(data_drugs(:,2));
    y_cat = categorical(y);
    
    placebo_idx = find(y_cat == 'Placebo');
    amisulpride_idx = find(y_cat == 'Amisulpride');
    biperdine_idx = find(y_cat == 'Biperdine');
    galantamine_idx = find(y_cat == 'Galantamine');
    levodopa_idx = find(y_cat == 'Levodopa');
    
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

    % Define the number of trees in the ensemble
    num_trees = 50;
    min_leaf_size = 15;
    max_num_splits = 18;
    
    % Define SVM Hyperparameters.
    t_svm = templateSVM('BoxConstraint', 0.0013119, 'KernelFunction', 'rbf', 'KernelScale', 703.99);
    t_adb = templateTree('MinLeafSize', 71, 'MaxNumSplits', 1);
    
    accuracies_rf_perm = zeros(permutations,7);
    accuracies_adb_perm = zeros(permutations,7);
    accuracies_knn_perm = zeros(permutations,7);
    accuracies_svm_perm = zeros(permutations,7);
    
    for perm = 1:permutations
        for drug = 1:7
            X = Xs{drug};
            y = ys{drug};
            y_perm = y(randperm(length(y)));
            y = y_perm;
            
            cv = cvpartition(y_perm,'Kfold', 10, 'Stratify', true);
            
            % Initialize class performance objects
            cp_rf_perm = classperf(y);
            cp_knn_perm = classperf(y);
            cp_svm_perm = classperf(y);
            cp_adb_perm = classperf(y);
            
            % Perform 10-fold cross-validation
            for i = 1:10
                train = cv.training(i); 
                test = cv.test(i);
                
                % TreeBagger
                rf = TreeBagger(num_trees, X(train,:), y(train,:), 'Method', 'classification', 'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
                class_rf_perm = predict(rf, X(test,:));
                
                % AdaBoost
                if drug == 7
                    % AdaBoostM2 for Multi-Class
                    adb = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM2', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
                    class_adb_perm = predict(adb, X(test,:));
                else
                    % AdaBoostM1 for Binary-Class
                    adb = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM1', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
                    class_adb_perm = predict(adb, X(test,:));
                end
        
                % KNN
                knn = fitcknn(X(train,:), y(train,:), 'NumNeighbors', 9, 'Distance', 'cosine');
                class_knn_perm = predict(knn, X(test,:));
        
                % SVM
                svm = fitcecoc(X(train,:), y(train,:), 'Coding', 'onevsone', 'Learners', t_svm);
                class_svm_perm = predict(svm, X(test,:));
                
                % Update the class performance objects
                classperf(cp_rf_perm, class_rf_perm, test);
                classperf(cp_knn_perm, class_knn_perm, test);
                classperf(cp_svm_perm, class_svm_perm, test);
                classperf(cp_adb_perm, class_adb_perm, test);
            end
        
            accuracy_rf_perm = 1 - cp_rf_perm.ErrorRate;
            accuracy_knn_perm = 1 - cp_knn_perm.ErrorRate;
            accuracy_svm_perm = 1 - cp_svm_perm.ErrorRate;
            accuracy_adb_perm = 1 - cp_adb_perm.ErrorRate;
            accuracies_rf_perm(perm, drug) = accuracy_rf_perm;
            accuracies_knn_perm(perm, drug) = accuracy_knn_perm;
            accuracies_svm_perm(perm, drug) = accuracy_svm_perm;
            accuracies_adb_perm(perm, drug) = accuracy_adb_perm;
        end
    end
end