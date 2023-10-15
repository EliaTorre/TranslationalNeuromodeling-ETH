% Input to the function: data_drugs table, number of permutations and a Random Seed value.
% Output: 4 matrices (permutations x 7) of permuted label accuracies for: RandomForest Tuned, AdaBoost Tuned,
% KNN Tuned, Support Vector Machine Tuned. 
% Each column contains Binary and Multi-Class Accuracies (in the following order):
% Placebo vs Galantamine, Placebo vs Amisulpride, Placebo vs Levodopa,
% Placebo vs Biperdine, Biperdine vs Galantamine, Amisulpride vs Levodopa,
% Placebo and 4 Drugs (5 Class) Classification.

function [accuracies_rf_DCM_perm, accuracies_adb_DCM_perm, accuracies_knn_DCM_perm, accuracies_svm_DCM_perm] = accuracies_DCM_perm(data, permutations, seed, drugs)
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

    % Define the number of trees in the ensemble
    num_trees = 50;
    min_leaf_size = 15;
    max_num_splits = 18;
    
    % Define SVM Hyperparameters.
    t_svm = templateSVM('BoxConstraint', 0.0013119, 'KernelScale', 703.99);
    t_adb = templateTree('MinLeafSize', 71, 'MaxNumSplits', 1);
    
    % Set options for embedding
    opts = {'Distance', 'Euclidean', 'Sigma', 1, 'NumNeighbors', 10, 'NumDimensions', 2, 'Verbose', 1};
    
    accuracies_rf_DCM_perm = zeros(permutations,7);
    accuracies_adb_DCM_perm = zeros(permutations,7);
    accuracies_knn_DCM_perm = zeros(permutations,7);
    accuracies_svm_DCM_perm = zeros(permutations,7);
    
    for perm = 1:permutations
        for drug = drugs
            X = Xs{drug};
            y = ys{drug};
            y_perm = y(randperm(length(y)));
            y = y_perm;
            
            cv = cvpartition(y_perm,'Kfold', 10, 'Stratify', true);
            
            % Initialize class performance objects
            cp_rf_DCM_perm = classperf(y);
            cp_knn_DCM_perm = classperf(y);
            cp_svm_DCM_perm = classperf(y);
            cp_adb_DCM_perm = classperf(y);
            
            % Perform 10-fold cross-validation
            for i = 1:10
                train = cv.training(i); 
                test = cv.test(i);
                
                % TreeBagger
                rf = TreeBagger(num_trees, X(train,:), y(train,:), 'Method', 'classification', 'MinLeafSize', min_leaf_size, 'MaxNumSplits', max_num_splits);
                class_rf_DCM_perm = predict(rf, X(test,:));
                
                % % AdaBoost
                % if drug == 7
                %     % AdaBoostM2 for Multi-Class
                %     adb = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM2', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
                %     class_adb_DCM_perm = predict(adb, X(test,:));
                % else
                %     % AdaBoostM1 for Binary-Class
                %     adb = fitcensemble(X(train,:), y(train,:), 'Method', 'AdaBoostM1', 'NumLearningCycles', 13, 'Learners', t_adb, 'LearnRate', 0.0010282);
                %     class_adb_DCM_perm = predict(adb, X(test,:));
                % end
                % 
                % % KNN
                % knn = fitcknn(X(train,:), y(train,:), 'NumNeighbors', 9, 'Distance', 'cosine');
                % class_knn_DCM_perm = predict(knn, X(test,:));
        
                % SVM
                svm = fitcecoc(X(train,:), y(train,:), 'Coding', 'onevsone', 'Learners', t_svm);
                class_svm_DCM_perm = predict(svm, X(test,:));
                
                % Update the class performance objects
                classperf(cp_rf_DCM_perm, class_rf_DCM_perm, test);
                %classperf(cp_knn_DCM_perm, class_knn_DCM_perm, test);
                classperf(cp_svm_DCM_perm, class_svm_DCM_perm, test);
                %classperf(cp_adb_DCM_perm, class_adb_DCM_perm, test);
            end
        
            accuracy_rf_DCM_perm = 1 - cp_rf_DCM_perm.ErrorRate;
            %accuracy_knn_DCM_perm = 1 - cp_knn_DCM_perm.ErrorRate;
            accuracy_svm_DCM_perm = 1 - cp_svm_DCM_perm.ErrorRate;
            %accuracy_adb_DCM_perm = 1 - cp_adb_DCM_perm.ErrorRate;
            accuracies_rf_DCM_perm(perm, drug) = accuracy_rf_DCM_perm;
            %accuracies_knn_DCM_perm(perm, drug) = accuracy_knn_DCM_perm;
            accuracies_svm_DCM_perm(perm, drug) = accuracy_svm_DCM_perm;
            %accuracies_adb_DCM_perm(perm, drug) = accuracy_adb_DCM_perm;
        end
    end
end