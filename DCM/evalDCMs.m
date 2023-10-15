%% load data

if ~exist(projectDir, "var")
    projectDir = pwd();
end

output_dir = fullfile(projectDir, 'output'); %dir where dcm output is loaded from
plot_out_dir = fullfile(projectDir, 'dcm_eval'); %dir where plots are saved

load(fullfile(output_dir, "model_defs.mat"));

if ~exist(plot_out_dir, "dir")
    mkdir(plot_out_dir);
end


trial_type_defs(1).report_name = "StdVsDev";
trial_type_defs(2).report_name = "StdVsDevStb";
trial_type_defs(3).report_name = "ExtStdVsDev";

connectivity_models(1).report_name = "Garrido2008";
connectivity_models(2).report_name = connectivity_models(2).name;
connectivity_models(3).report_name = connectivity_models(3).name;


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

seed = 25;
nShuffles = 1;
ii = 0;
%ii = length(results_struct);
for i_ttd=1:length(trial_type_defs)
    for i_dm = 1:length(dcm_models)
        for i_cm = 1:length(connectivity_models)  
            if isempty(data_all{i_ttd, i_dm, i_cm})
                continue;
            else
                % if any(string({results_struct.model_name})==string(data_all{i_ttd, i_dm, i_cm}.Properties.Description))
                %     continue;
                % end
                ii = ii+1;
            end

            results_struct(ii).model_name = data_all{i_ttd, i_dm, i_cm}.Properties.Description;
            results_struct(ii).report_name = sprintf("%s+%s+%s", trial_type_defs(i_ttd).report_name, dcm_models(i_dm).name, connectivity_models(i_cm).report_name);

            
            %results_struct(ii).F = [data_all{i_ttd, i_dm, i_cm}.dcm.F];
            results_struct(ii).F_sum = sum([data_all{i_ttd, i_dm, i_cm}.dcm.F]);

            [results_struct(ii).acc_rf_DCM, ~, ~, results_struct(ii).acc_svm_DCM] = accuracies_DCM(data_all{i_ttd, i_dm, i_cm}, seed);

            [results_struct(ii).acc_max_DCM, results_struct(ii).acc_imax_DCM] = max(vertcat(results_struct(ii).acc_rf_DCM, results_struct(ii).acc_svm_DCM));

            [results_struct(ii).acc_rf_DCM_perm, ~, ~, results_struct(ii).acc_svm_DCM_perm] = accuracies_DCM_perm(data_all{i_ttd, i_dm, i_cm}, 30, seed);


            results_struct(ii).i_ttd = i_ttd;
            results_struct(ii).i_dm = i_dm;
            results_struct(ii).i_cm = i_cm;
        end
    end
end

results = struct2table(results_struct);

%%


[~, iBestModel] = max(results.acc_max_DCM);
iBestClassifierBestModel = diag(results.acc_imax_DCM(iBestModel,:));
bestClassifierBestModel = classifier_models(iBestClassifierBestModel);
bestClassifierBestModel = bestClassifierBestModel(:);
%bestModel = string(results.model_name(iBestModel));
bestModel = string(results.report_name(iBestModel))+"+"+bestClassifierBestModel;


classifier = string({'Galantamine/Placebo', "Amisulpride/Placebo", "Levodopa/Placebo", "Biperdine/Placebo", "Biperdine/Galantamine", "Amisulpride/Levodopa", "All Drugs"})';

for i=1:7
    accuracies_model(i,1) = results.acc_max_DCM(iBestModel(i),i);

    accuracies_model_perm(:,i) = results.acc_rf_DCM_perm{iBestModel(i)}(:,i); 
    % if iBestClassifierBestModel(i)>=1 %RF
    %     accuracies_model_perm(:,i) = results.acc_rf_DCM_perm{iBestModel(i)}(:,i);
    % else
    %     accuracies_model_perm(:,i) = results.acc_svm_DCM_perm{iBestModel(i)}(:,i);
    % end
end
[mu, sigma, pval] = plotting_DCM(bestModel, accuracies_model_perm, accuracies_model, plot_out_dir);
final_table = table(classifier, bestModel, accuracies_model, mu, sigma, pval);

iBestOverallModel = 13;
accuracies_model = results.acc_rf_DCM(iBestOverallModel,:)';
[mu_b, sigma_b, pval_b] = plotting_DCM(repmat(bestModel(1), 7, 1), results.acc_rf_DCM_perm{iBestOverallModel}, accuracies_model, fullfile(plot_out_dir, "best_overall_model"));
best_overal_model_table = table(classifier, accuracies_model, mu_b, sigma_b, pval_b)


save(fullfile(plot_out_dir, "results_withPerm"), "results", "final_table", "best_overal_model_table");


%create latex table
for ic =1:height(final_table)
    if final_table.pval(ic) <= 0.001
        pvaltxt = sprintf("\\textbf{<0.001}");
    elseif final_table.pval(ic) <= 0.05
        pvaltxt = sprintf("\\textbf{%0.3f}", final_table.pval(ic));
    else
        pvaltxt = sprintf("%0.3f", final_table.pval(ic));
    end
    tabletext(ic) = sprintf("%s & %s & %0.3f & %0.3f & %0.3f & %s \\\\ \\hline", final_table.classifier(ic), final_table.bestModel(ic), final_table.accuracies_model(ic), final_table.mu(ic), final_table.sigma(ic), pvaltxt);
end
writelines(tabletext, fullfile(plot_out_dir, "latex_results_table.txt"));



