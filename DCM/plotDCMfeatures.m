%% load data

output_dir = "C:\Users\linus\Google Drive\Tmp\dcmoutput"; %contains dcm output files

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


%%
i_ttd = 2;
i_dm = 2;
i_cm = 1;

data_all{i_ttd, i_dm, i_cm}.dcm(1).Ep

nSubjects = height(data_all{i_ttd, i_dm, i_cm});
drug_label = categorical(data_all{i_ttd, i_dm, i_cm}.drug_label);

clear param_means_vec param_std_vec
for iSubject = 1:nSubjects
    param_means = data_all{i_ttd, i_dm, i_cm}.dcm(iSubject).Ep;

    param_means_vec(iSubject, :) = full(spm_vec(param_means))';
end

[~, param_names] = spm_vec_with_names(param_means);
param_names = string(param_names);

which_params = find(var(param_means_vec,1)~=0);
%which_params = which_params(1:2);
param_means_vec = param_means_vec(:, which_params);
%param_std_vec = param_std_vec(:, which_params);
param_names = param_names(which_params);

nParams = size(param_means_vec,2);

which_subjects = 1:nSubjects;
%which_subjects = ismember(string(drug_label),["Biperdine", "Galantamine", "Placebo"]);

p_anova = zeros(1,size(param_means_vec,2));
for i=1:size(param_means_vec,2)
    p_anova(i) = anova1(param_means_vec(which_subjects,i),drug_label(which_subjects),'off');
end
which_params2 = p_anova<0.05;
param_means_vec = param_means_vec(:, which_params2);
nParams_sig = size(param_means_vec,2);





T = table(drug_label, param_means_vec);
T = T(which_subjects,:);

G = groupsummary(T,"drug_label",["mean", "std"],"param_means_vec");

G.sem_param_means_vec = G.std_param_means_vec./sqrt(G.GroupCount);
f=figure(4);
clf;
hold on;
for i=1:height(G)
    errorbar((1:nParams_sig)+(i-3)/8, G.mean_param_means_vec(i,:), G.sem_param_means_vec(i,:), "square");
end
xticks((1:nParams_sig));

% ss = spiny stellate
% sp = superficial pyramidal
% dp = deep pyramidal
% ii = inhibitory interneurons

xt = ["lA1 (sp) \rightarrow lSTG (ss)" "rA1 (sp) \rightarrow rSTG (ss)" "rSTG (sp) \rightarrow rIFG (ss)" "lA1 (sp) \rightarrow lSTG (dp)" "rA1 (sp) \rightarrow rSTG (dp)" "rSTG (sp) \rightarrow rIFG (dp)" "lSTG (dp) \rightarrow lA1 (sp)"];
xticklabels(xt);
xticklabels(param_names);
ax = gca;
%ax.TickLabelInterpreter = "none";
xtickangle(45);
xlim([0 nParams_sig+1]);
legend(G.drug_label);
ylabel("posterior mean estimate");
xlabel("parameter")

folderName = "C:\Users\linus\Repos\translational-neuromodelling-23\dcm_eval";
f.PaperPosition =  [0,0,6,7]; %[left bottom width height] (left and bottom are ignored for picture formats)
%print(f, fullfile(folderName, "dcm_params.png"), '-dpng');