%% load data

output_dir = "C:\Users\linus\Google Drive\Tmp\dcmoutput"; %contains dcm output files

plot_out_dir = "C:\Users\linus\Repos\translational-neuromodelling-23\dcm_perm_plots";

load(fullfile(output_dir, "model_defs.mat"));

for i_ttd=1:length(trial_type_defs)
    for i_dm = 1:length(dcm_models)
        for i_cm = 1:length(connectivity_models)
            fn = fullfile(output_dir, sprintf("%s_%s_%s.mat", trial_type_defs(i_ttd).name, dcm_models(i_dm).name, connectivity_models(i_cm).name));
            if exist(fn,"file");
                load(fn);
                disp(fn);
                data_all{i_ttd, i_dm, i_cm} = data;
            else
                data_all{i_ttd, i_dm, i_cm} = [];
            end
        end
    end
end

%% get accuracies
classifier_models = ["RF", "SVM"];

drug_label = categorical(data_all{1, 1, 1}.drug_label);
drug_names = string(categories(drug_label));
for i =1:length(drug_names)
    class_counts.(drug_names(i)) = sum(drug_label==drug_names(i));
end


nShuffles = 10;
ii = 0;
ii = length(results_struct);
for i_ttd=1:length(trial_type_defs)
    for i_dm = 1:length(dcm_models)
        for i_cm = 1:length(connectivity_models)  
            if isempty(data_all{i_ttd, i_dm, i_cm})
                continue;
            else
                if any(string({results_struct.model_name})==string(data_all{i_ttd, i_dm, i_cm}.Properties.Description))
                    continue;
                end
                ii = ii+1;
            end

            results_struct(ii).model_name = data_all{i_ttd, i_dm, i_cm}.Properties.Description;
            %results_struct(ii).F = [data_all{i_ttd, i_dm, i_cm}.dcm.F];
            results_struct(ii).F_sum = sum([data_all{i_ttd, i_dm, i_cm}.dcm.F]);

            for iS = 1:nShuffles
                seed= iS+20;
                %[results_struct2(ii).acc_rf_DCM{1}(iS,:), results_struct2(ii).acc_adb_DCM{1}(iS,:), results_struct2(ii).acc_knn_DCM{1}(iS,:), results_struct2(ii).acc_svm_DCM{1}(iS,:)] = accuracies_DCM(data_all{i_ttd, i_dm, i_cm}, seed);
                [results_struct2(ii).acc_rf_DCM{1}(iS,:), ~, ~, results_struct2(ii).acc_svm_DCM{1}(iS,:)] = accuracies_DCM(data_all{i_ttd, i_dm, i_cm}, seed);
                
            end

            [~, iMaxRf] = max(sum(results_struct2(ii).acc_rf_DCM{1},2));
            [~, iMaxSvm] = max(sum(results_struct2(ii).acc_svm_DCM{1},2));

            

            results_struct(ii).acc_rf_DCM_mean = mean(results_struct2(ii).acc_rf_DCM{1});
            %results_struct(ii).acc_adb_DCM_mean = mean(results_struct2(ii).acc_adb_DCM{1})-chance_levels;
            %results_struct(ii).acc_knn_DCM_mean = mean(results_struct2(ii).acc_knn_DCM{1})-chance_levels;
            results_struct(ii).acc_svm_DCM_mean = mean(results_struct2(ii).acc_svm_DCM{1});

            results_struct(ii).acc_rf_DCM_max = results_struct2(ii).acc_rf_DCM{1}(iMaxRf,:);
            results_struct(ii).acc_svm_DCM_max = results_struct2(ii).acc_svm_DCM{1}(iMaxSvm,:);
            results_struct(ii).acc_rf_DCM_iMaxRf = iMaxRf;
            results_struct(ii).acc_rf_DCM_iMaxSvm = iMaxSvm;

            [results_struct(ii).acc_max_DCM_mean, results_struct(ii).acc_imax_DCM_mean] = max(vertcat(results_struct(ii).acc_rf_DCM_mean, results_struct(ii).acc_svm_DCM_mean));
            [results_struct(ii).acc_max_DCM_max, results_struct(ii).acc_imax_DCM_max] = max(vertcat(results_struct(ii).acc_rf_DCM_max, results_struct(ii).acc_svm_DCM_max));

            [results_struct(ii).acc_rf_DCM_perm, ~, ~, results_struct(ii).acc_svm_DCM_perm] = accuracies_DCM_perm(data_all{i_ttd, i_dm, i_cm}, 30, 21);

            results_struct(ii).i_ttd = i_ttd;
            results_struct(ii).i_dm = i_dm;
            results_struct(ii).i_cm = i_cm;
        end
    end
end

results = struct2table(results_struct);
results_details = struct2table(results_struct2);

%%


[~, iBestModel_mean] = max(results.acc_max_DCM_mean);
iBestClassifierBestModel_mean = diag(results.acc_imax_DCM_mean(iBestModel_mean,:));
bestClassifierBestModel_mean = classifier_models(iBestClassifierBestModel_mean);
bestClassifierBestModel_mean = bestClassifierBestModel_mean(:);
bestModel_mean = string(results.model_name(iBestModel_mean));
%bestModel_mean = bestModel_mean(:);

[~, iBestModel_max] = max(results.acc_max_DCM_max);
iBestClassifierBestModel_max = diag(results.acc_imax_DCM_mean(iBestModel_max,:));
bestClassifierBestModel_max = classifier_models(iBestClassifierBestModel_max);
bestClassifierBestModel_max = bestClassifierBestModel_max(:);
bestModel_max = string(results.model_name(iBestModel_max));
%bestModel_max(:);

classifier = string({'Galantamine/Placebo', "Amisulpride/Placebo", "Levodopa/Placebo", "Biperdine/Placebo", "Biperdine/Galantamine", "Amisulpride/Levodopa", "All Drugs"})';

for i=1:7
    accuracies_model_mean(i,1) = results.acc_max_DCM_mean(iBestModel_mean(i),i);

    accuracies_model_perm_mean(:,i) = results.acc_rf_DCM_perm{iBestModel_mean(i)}(:,i); 
    % if iBestClassifierBestModel_mean(i)>=1 %RF
    %     accuracies_model_perm_mean(:,i) = results.acc_rf_DCM_perm{iBestModel_mean(i)}(:,i);
    % else
    %     accuracies_model_perm_mean(:,i) = results.acc_svm_DCM_perm{iBestModel_mean(i)}(:,i);
    % end
end
[mu_mean, sigma_mean, pval_mean] = plotting_DCM(bestModel_mean, accuracies_model_perm_mean, accuracies_model_mean, fullfile(plot_out_dir, "mean"));
overview_table_mean = table(classifier, bestModel_mean, bestClassifierBestModel_mean, accuracies_model_mean, mu_mean, sigma_mean, pval_mean);

for i=1:7
    accuracies_model_max(i,1) = results.acc_max_DCM_max(iBestModel_max(i),i);
    
    accuracies_model_perm_max(:,i) = results.acc_rf_DCM_perm{iBestModel_max(i)}(:,i);
    % if iBestClassifierBestModel_max(i)==1 %RF
    %     accuracies_model_perm_max(:,i) = results.acc_rf_DCM_perm{iBestModel_max(i)}(:,i);
    % else
    %     accuracies_model_perm_max(:,i) = results.acc_svm_DCM_perm{iBestModel_max(i)}(:,i);
    % end
end
[mu_max, sigma_max, pval_max] = plotting_DCM(bestModel_max, accuracies_model_perm_max, accuracies_model_max, fullfile(plot_out_dir, "max"));
overview_table_max = table(classifier, bestModel_max, bestClassifierBestModel_max, accuracies_model_max, mu_max, sigma_max, pval_max);

save(fullfile(output_dir, "results_withPerm"), "results", "results_details", "overview_table_max", "overview_table_mean");


%%

for i=1:10
    for j=1:length(results_struct2)
        rr(j,:,i) = max([results_struct2(j).acc_rf_DCM{1}(i,:); results_struct2(j).acc_svm_DCM{1}(i,:)]);
    end
end

mrr = max(rr, [], 1);
[~, imrr] = max(sum(mrr, 2), [], 3);

mrr(1,:,imrr)
