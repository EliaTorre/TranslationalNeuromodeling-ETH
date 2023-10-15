% this scripts inverts length(trial_type_defs)*length(dcm_models)*length(connectivity_models) DCMs
% (this script takes many days to run)

%% paths (set paths here before running)
%projectDir = 'C:\data\translationalNeuromodeling';
projectDir = pwd();

data_dir = fullfile(projectDir, 'data'); %dir with pre-proecessed eeg data
information_dir = fullfile(projectDir, 'Information'); %dir with pharma and paradigm dir


tmp_dir = fullfile(projectDir, 'tmp'); %dir for temporary data that is generated
output_dir = fullfile(projectDir, 'output'); %dir where dcm output is saved


% init spm and load labels and stimuli
anta_labels = load(fullfile(information_dir, 'pharma\dprst_anta_druglabels_anonym.mat'));
agon_labels = load(fullfile(information_dir, 'pharma\dprst_agon_druglabels_anonym.mat'));
load(fullfile(information_dir, 'paradigm\stimulus_sequence.mat'));

spm('defaults','EEG');
global defaults
defaults.cmdline = 1;

%% definition of different DCMs
trial_type_defs(1).name = "classical";
trial_type_defs(1).trial_type = getTrialtype(tones, 5, false);
trial_type_defs(1).desc = "deviant: every change after at least 5 repetitions, standard: every 6th repetition of a tone (2conditions)";

trial_type_defs(2).name = "classicalWithStability";
trial_type_defs(2).trial_type = getTrialtype(tones, 5, true);
trial_type_defs(2).desc = "like classical but with added stable vs volatile type as defined in Weber2021 (4conditions)";

trial_type_defs(3).name = "2deviantAnd2StandardConds";
trial_type_defs(3).trial_type = getTrialtype2(tones);
trial_type_defs(3).desc = "deviant3-5: every change after 3-5 repetitions, deviant6+: every change after at least 6 repetitions, standard5: every 6th repetition of a tone, standard12+: every 13th or higher reptition of a tone (4conditions)";


dcm_models(1).name = "ERP";
dcm_models(1).model = 'ERP';
dcm_models(1).desc = "3-population convolution-based NMM";

dcm_models(2).name = "CMC";
dcm_models(2).model = 'CMC';
dcm_models(2).desc = "4-population convolution-based NMM";

dcm_models(3).name = "NMDA";
dcm_models(3).model = 'NMDA';
dcm_models(3).desc = "3-population conductance-based NMM that includes the NMDA receptor";


connectivity_models(1).name = "Garrido2009";
connectivity_models(1).Lpos  = [[-42; -22; 7] [46; -14; 8] [-61; -32; 8] [59; -25; 8] [46; 20; 8]]; %coordinates from Garrido2009
connectivity_models(1).Sname = {'left A1', 'right A1', 'left STG', 'right STG', 'right IFG'};
A{1} = zeros(5,5);
A{1}(3,1) = 1;
A{1}(4,2) = 1;
A{1}(5,4) = 1;

A{2} = zeros(5,5);
A{2}(1,3) = 1;
A{2}(2,4) = 1;
A{2}(4,5) = 1;

A{3} = zeros(5,5);

B = A{1}+A{2};
B(1,1) = 1;
B(2,2) = 1;

connectivity_models(1).A = A;
connectivity_models(1).B = B;
connectivity_models(1).C = [1; 1; 0; 0; 0];
connectivity_models(1).desc = "As in Garrido2009 (no lateral connections)";

connectivity_models(2).name = "Ranlund2016";
connectivity_models(2).Lpos  = [[-42; -22; 7] [46; -14; 8] [-61; -32; 8] [59; -25; 8] [46; 20; 8]]; %coordinates from Garrido2009
connectivity_models(2).Sname = {'left A1', 'right A1', 'left STG', 'right STG', 'right IFG'};

A{3} = zeros(5,5);
A{3}(2,1) = 1;
A{3}(1,2) = 1;
A{3}(4,3) = 1;
A{3}(3,4) = 1;

connectivity_models(2).A = A;
connectivity_models(2).B = eye(5);
connectivity_models(2).C = [1; 1; 0; 0; 0];
connectivity_models(2).desc = "As in Ranlund2016 (with lateral connections and modulated self-inhibition)";


connectivity_models(3).name = "David2006";
connectivity_models(3).Lpos  = [[-42; -22; 7] [46; -14; 8] [-27; 42; -14] [32; 42; -12] [59; -25; 8]]; %OC coordinates from: https://www.pnas.org/doi/10.1073/pnas.1704831114
connectivity_models(3).Sname = {'left A1', 'right A1', 'left OC', 'right OC', 'right STG'};

A{1} = zeros(5,5);
A{1}(3,1) = 1;
A{1}(4,2) = 1;
A{1}(4,5) = 1;
A{1}(5,2) = 1;

A{2} = zeros(5,5);
A{2}(1,3) = 1;
A{2}(2,4) = 1;
A{2}(5,4) = 1;
A{2}(2,5) = 1;

A{3} = zeros(5,5);
A{3}(2,1) = 1;
A{3}(1,2) = 1;
A{3}(4,3) = 1;
A{3}(3,4) = 1;

B = A{1}+A{2};

connectivity_models(3).A = A;
connectivity_models(3).B = B;
connectivity_models(3).C = [1; 1; 0; 0; 0];
connectivity_models(3).desc = "As in David2006";

%connectivity_models = connectivity_models(3);


%% create data table, compute trial average over standard and deviant trials and save result in tmp_dir

save_mean_trials = true;

skip_list.anta = [6, 11, 25, 30, 31, 50, 55, 60, 62, 71];
skip_list.agon = [63, 77, 81];

%delete tmp dir and recreate tmp_dir
if exist(tmp_dir, "dir")
    rmdir(tmp_dir, 's');
end
mkdir(tmp_dir);

% loop through data
data_preprocess = cell(length(trial_type_defs),1);
for i_ttd=1:length(trial_type_defs)


    sub_tmp_dir = fullfile(tmp_dir, trial_type_defs(i_ttd).name);
    mkdir(sub_tmp_dir);


    data_preprocess{i_ttd} = struct();
    iSubject = 0;
    for study = ["anta" "agon"]
    
        subject_list = 1:81;
        subject_list(skip_list.(study)) = [];
        for i = subject_list
            iSubject = iSubject+1;
    
            if study == "anta"
                subject_id = sprintf('%04d', 100+i);
                if ismember(subject_id, anta_labels.subjACh.subjIDs)
                    drug_label = 'Biperdine' ; % set the drug label to 'Biperdine = 1' if the subject received Biperdine
                elseif ismember(subject_id, anta_labels.subjDA.subjIDs)
                    drug_label = 'Amisulpride'; % set the drug label to 'Amisulpride = 2' if the subject received Amisulpride
                elseif ismember(subject_id, anta_labels.subjPla.subjIDs)
                    drug_label = 'Placebo'; % set the drug label to 'Placebo = 0' if the subject received Placebo
                else
                    drug_label = 'Unknown'; % set the drug label to 'Unknown' if the subject ID is not found in any of the drug label structs
                end
            else
                subject_id = sprintf('%04d', 200+i);
                if ismember(subject_id, agon_labels.subjACh.subjIDs)
                    drug_label = 'Galantamine' ; % set the drug label to 'Biperdine = 1' if the subject received Biperdine
                elseif ismember(subject_id, agon_labels.subjDA.subjIDs)
                    drug_label = 'Levodopa'; % set the drug label to 'Amisulpride = 2' if the subject received Amisulpride
                elseif ismember(subject_id, agon_labels.subjPla.subjIDs)
                    drug_label = 'Placebo'; % set the drug label to 'Placebo = 0' if the subject received Placebo
                else
                    drug_label = 'Unknown'; % set the drug label to 'Unknown' if the subject ID is not found in any of the drug label structs
                end
            end
            data_preprocess{i_ttd}.subject_id(iSubject,1) = string(subject_id);
            data_preprocess{i_ttd}.drug_label(iSubject,1) = string(drug_label);
            data_preprocess{i_ttd}.study(iSubject,1) = string(study);
    
            data_preprocess{i_ttd}.data_fn(iSubject,1) = fullfile(data_dir, study, sprintf("DPRST_%s", subject_id), sprintf("DPRST_%s_MMN_preproc", subject_id));
            data_preprocess{i_ttd}.tmp_data_fn(iSubject,1) = fullfile(sub_tmp_dir, sprintf("DPRST_%s_MMN_preproc", subject_id));

            conditions = string(categories(trial_type_defs(i_ttd).trial_type));
            conditions = conditions(conditions~="undefined");
            nConditions = length(conditions);
            trial_type_defs(i_ttd).conditions = conditions;
            trial_type_defs(i_ttd).nConditions = nConditions;

    
            D  = spm_eeg_load(char(data_preprocess{i_ttd}.data_fn(iSubject)));
            D2 = D.clone(char(data_preprocess{i_ttd}.tmp_data_fn(iSubject)), [D.nchannels, D.nsamples, nConditions]);
    
            goodtrials = true(1,D.ntrials);
            goodtrials(D.badtrials) = false;
    
            
            data_preprocess{i_ttd}.nTrialsPerCondition{iSubject,1} = zeros(1,nConditions);
            for iC = 1:nConditions
                trialIds = find(trial_type_defs(i_ttd).trial_type == conditions(iC) & goodtrials);
                data_preprocess{i_ttd}.nTrialsPerCondition{iSubject,1}(iC) = length(trialIds);
                D2(:,:,iC) = mean(D(:,:,trialIds),3);
                D2 = D2.conditions(iC,char(conditions(iC)));
            end
    

            data_preprocess{i_ttd}.mean_eeg_signals{iSubject,1} = D2(:,:,:);
    
            % the following we do to prevent spm_dcm_erp_dipfit to open a gui
            % (on line 118ff). 
            val = D2.val;
            sMRI = []; %template head model
            Msize = 2; %normal size
            D2.inv{val}.mesh = spm_eeg_inv_mesh(sMRI, Msize); %line 77 from spm_eeg_inv_mesh_ui.m
            D2 = spm_eeg_inv_datareg_ui(D2);
            D2.inv{val}.forward(1).voltype = 'EEG BEM'; % alternative: '3-Shell Sphere'
            D2 = spm_eeg_inv_forward(D2); % line 55 from spm_eeg_inv_forward_ui.m
            %-----------------------------------------------------------------------
    
            D2 = D2.save();
        end
    
    end
    
    data_preprocess{i_ttd} = struct2table(data_preprocess{i_ttd});
end

nSubjects = height(data_preprocess{i_ttd});

save(fullfile(output_dir, "model_defs"), "connectivity_models", "dcm_models", "trial_type_defs", "data_preprocess");

%% loop through all dcm and fit them all
if ~exist(output_dir, "dir")
    mkdir(output_dir);
end


data_all = cell(length(trial_type_defs), length(dcm_models), length(connectivity_models));
for i_ttd=1:length(trial_type_defs)
    for i_dm = 1:length(dcm_models)
        for i_cm = 1:length(connectivity_models)
% setup dcm (see example file spm12\man\example_scripts\DCM_ERP_subject1.m)

% Parameters and options used for setting up model
%--------------------------------------------------------------------------
DCM.options.analysis = 'ERP'; % analyze evoked responses
%DCM.options.model    = 'ERP'; % ERP model (see: spm_fx_erp.m and spm_erp_priors.m for model infos)
DCM.options.model    = dcm_models(i_dm).model;
DCM.options.spatial  = 'ECD'; %'IMG'; % spatial model
DCM.options.trials   = 1:trial_type_defs(i_ttd).nConditions; % index of ERPs within ERP/ERF file
DCM.options.Tdcm(1)  = 0;     % start of peri-stimulus time to be modelled
DCM.options.Tdcm(2)  = 452;   % end of peri-stimulus time to be modelled
DCM.options.Nmodes   = 8;     % nr of modes for data selection
DCM.options.h        = 1;     % nr of DCT components
DCM.options.onset    = 60;    % selection of onset (prior mean)
DCM.options.D        = 1;     % downsampling

%--------------------------------------------------------------------------
% Location priors for dipoles
%--------------------------------------------------------------------------
DCM.Lpos  = connectivity_models(i_cm).Lpos;
DCM.Sname = connectivity_models(i_cm).Sname;

%--------------------------------------------------------------------------
% Specify connectivity model
%--------------------------------------------------------------------------

DCM.A = connectivity_models(i_cm).A;
DCM.B = repmat({connectivity_models(i_cm).B},1,trial_type_defs(i_ttd).nConditions-1);
DCM.C = connectivity_models(i_cm).C;

%--------------------------------------------------------------------------
% Between trial effects
%--------------------------------------------------------------------------
DCM.xU.X = [zeros(1, trial_type_defs(i_ttd).nConditions-1); eye(trial_type_defs(i_ttd).nConditions-1)]; %see spm_gen_erp.m spm_gen_Q.m 
%DCM.xU.name = {'deviant'};

DCM.name = sprintf('%s_%s_%s', trial_type_defs(i_ttd).name, dcm_models(i_dm).name, connectivity_models(i_cm).name);


if ~exist(output_dir, "dir")
    mkdir(output_dir);
end


% do dcm
data = data_preprocess{i_ttd};
data.Properties.Description = DCM.name;

dcm_cell = cell(nSubjects,1);

for iSubject = 1:nSubjects
    DCM.xY.Dfile = char(data.tmp_data_fn(iSubject));
    %DCM2  = spm_dcm_erp_data(DCM); % Data and spatial model
    %DCM2 = spm_dcm_erp_dipfit(DCM2); % Spatial model

    DCM2 = spm_dcm_erp(DCM); % invert dcm (spm_dcm_erp_data and spm_dcm_erp_dipfit are called in spm_dcm_erp, hence no need to call them before)

    dcm_cell{iSubject} = DCM2;
end
data.dcm = cell2mat(dcm_cell); 


dcm_features = zeros(height(data),length(spm_vec(DCM2.Ep)));
for iSubject = 1:height(data)
    dcm_features(iSubject,:) = spm_vec(data.dcm(iSubject).Ep)'; % save means of all posterior parameter distributions (in log space) as vector
end
dcm_features(:,var(dcm_features,1)==0) = []; %remove constant params

data.dcm_features = dcm_features;

data_all{i_ttd, i_dm, i_cm} = data;

% save data table
%writetable(removevars(data,"dcm"),strcat(output_table_fn, ".csv"));
save(fullfile(output_dir, data.Properties.Description), "data");


        end
    end
end

save(fullfile(output_dir, "data_all"), "data_all");


