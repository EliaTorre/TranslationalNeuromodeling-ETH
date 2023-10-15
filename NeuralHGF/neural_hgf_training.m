clearvars
%Global hyperparameters
seed = 999;
rng(seed);
numFolds = 10; %kfold cross validation

%Neural network training hyperparameters
miniBatchSize = 60;
numEpochs = 1000;
learnRate = 1e-3;
temp_filter_size = 32; %Convolution kernel in time axis
spatial_filter_size = 32; %Convolution kernel in sensor axis


%Code for extracting and reshaping eeg data, which will be saved in this
%folder for later use.
addpath('../')

anta_labels = load('../Information/pharma/dprst_anta_druglabels_anonym.mat');
agon_labels = load('../Information/pharma/dprst_agon_druglabels_anonym.mat');
tones = load('../Information/paradigm/stimulus_sequence.mat');
tones = struct2cell(tones);
tones = [tones{:}];
trial_type = getTrialtype(tones);
standard_idx = find(trial_type == 'standard');
deviant_idx = find(trial_type == 'deviant');

data = struct; % initialize an empty struct to store the data for matlab analysis
list_to_skip = [6, 11, 25, 30, 31, 50, 55, 60, 62, 71];
data_idx = 1;
for i = 1:81
    if ismember(i, list_to_skip)
        continue
    else
        folder_name = sprintf('DPRST_%04d', i+100); % create the folder name string
        file_name = sprintf('%s_MMN_preproc.mat', folder_name); % create the EEG data file name string
        file_path = fullfile('../anta', folder_name, file_name); % create the full file path
        eeg_data = spm_eeg_load(file_path); % load the EEG data file
        eeg_chan_idx = eeg_data.indchantype('EEG');
        eeg_data = eeg_data(eeg_chan_idx, :, :);
        standards = eeg_data(:,:,standard_idx);
        deviants = eeg_data(:,:,deviant_idx);
        standards_avg = mean(standards, 3);
        deviants_avg = mean(deviants, 3);
        standards_avg = reshape(standards_avg, [1,size(standards_avg,1),size(standards_avg,2)]);
        deviants_avg = reshape(deviants_avg, [1,size(deviants,1),size(deviants_avg,2)]);
        eeg_averaged = vertcat(standards_avg, deviants_avg);
        eeg_averaged = permute(eeg_averaged,[3,2,1]);
        
        subject_id = folder_name(end-3:end); % extract the subject ID from the folder name
        if ismember(subject_id, anta_labels.subjACh.subjIDs)
            drug_label = 'Biperdine' ; % set the drug label to 'Biperdine = 1' if the subject received Biperdine
        elseif ismember(subject_id, anta_labels.subjDA.subjIDs)
            drug_label = 'Amisulpride'; % set the drug label to 'Amisulpride = 2' if the subject received Amisulpride
        elseif ismember(subject_id, anta_labels.subjPla.subjIDs)
            drug_label = 'Placebo'; % set the drug label to 'Placebo = 0' if the subject received Placebo
        else
            drug_label = 'Unknown'; % set the drug label to 'Unknown' if the subject ID is not found in any of the drug label structs
        end
        % add the EEG data, subject ID, and drug label to the data struct
        data(data_idx).subject_id = subject_id;
        data(data_idx).drug_label = drug_label;
        data(data_idx).eeg_data = eeg_averaged;
        data_idx=data_idx+1;
    end
end

list_to_skip2 = [144, 158];
for i = 82:161
    if ismember(i, list_to_skip2)
        continue
    else
        folder_name = sprintf('DPRST_%04d', i+119); % create the folder name string
        file_name = sprintf('%s_MMN_preproc.mat', folder_name); % create the EEG data file name string
        file_path = fullfile('../agon', folder_name, file_name); % create the full file path
        eeg_data = spm_eeg_load(file_path); % load the EEG data file
        eeg_chan_idx = eeg_data.indchantype('EEG');
        eeg_data = eeg_data(eeg_chan_idx, :, :);
        standards = eeg_data(:,:,standard_idx);
        deviants = eeg_data(:,:,deviant_idx);
        standards_avg = mean(standards, 3);
        deviants_avg = mean(deviants, 3);
        standards_avg = reshape(standards_avg, [1,size(standards_avg,1),size(standards_avg,2)]);
        deviants_avg = reshape(deviants_avg, [1,size(deviants,1),size(deviants_avg,2)]);
        eeg_averaged = vertcat(standards_avg, deviants_avg);
        eeg_averaged = permute(eeg_averaged,[3,2,1]);
        
        subject_id = folder_name(end-3:end); % extract the subject ID from the folder name
        if ismember(subject_id, agon_labels.subjACh.subjIDs)
            drug_label = 'Galantamine' ; % set the drug label to 'Biperdine = 1' if the subject received Biperdine
        elseif ismember(subject_id, agon_labels.subjDA.subjIDs)
            drug_label = 'Levodopa'; % set the drug label to 'Amisulpride = 2' if the subject received Amisulpride
        elseif ismember(subject_id, agon_labels.subjPla.subjIDs)
            drug_label = 'Placebo'; % set the drug label to 'Placebo = 0' if the subject received Placebo
        else
            drug_label = 'Unknown'; % set the drug label to 'Unknown' if the subject ID is not found in any of the drug label structs
        end
        % add the EEG data, subject ID, and drug label to the data struct
        data(data_idx).subject_id = subject_id;
        data(data_idx).drug_label = drug_label;
        data(data_idx).eeg_data = eeg_averaged;
        data_idx=data_idx+1;
    end
end
save('agonandantadata.mat', "data")


%Now we load the code from the above saved .mat file and perform
%cross-validated training with neural network.
clearvars -except temp_filter_size spatial_filter_size numEpochs numFolds learnRate miniBatchSize seed rng
%Code for loading the data from .mat files
full_data = load('agonandantadata.mat');
full_eeg = zeros([139,63,2,size(full_data.data,2)]);
labels = cell(size(full_data.data,2),1);
for i=1:size(full_data.data,2)
    full_eeg(:,:,:,i) = full_data.data(i).eeg_data;
    labels(i) = {full_data.data(i).drug_label};
end
full_eeg = dlarray(full_eeg, 'SSCB');
meanfe = mean(full_eeg, 1);
stdfe = std(full_eeg, 1);
full_eeg = (full_eeg-meanfe)./(stdfe);

cv = cvpartition(labels, 'KFold',numFolds,'Stratify', true);

for k=1:numFolds
    full_data = load('agonandantadata.mat');
    full_eeg = zeros([139,63,2,size(full_data.data,2)]);
    labels = cell(size(full_data.data,2),1);
    for i=1:size(full_data.data,2)
        full_eeg(:,:,:,i) = full_data.data(i).eeg_data;
        labels(i) = {full_data.data(i).drug_label};
    end
    full_eeg = dlarray(full_eeg, 'SSCB');
    meanfe = mean(full_eeg, 1);
    stdfe = std(full_eeg, 1);
    full_eeg = (full_eeg-meanfe)./(stdfe);

    disp('===========================')
    disp('Fold number')
    disp(k)
    disp('')
    test_idx = cv.test(k);
    data = full_data.data;

    for i=1:size(test_idx,1)
        data(i).train_or_val = test_idx(i);
    end
    filename = sprintf('hgfdata_%d_%d.mat',k,numFolds);
    save(filename, "data")
    clear data

    % Separate to training and test data
    eegTrain = full_eeg(:,:,:,~test_idx);
    eegTest  = full_eeg(:,:,:,test_idx);
    
    clear full_eeg meanfe stdfe labels full_data

    eegTrain = reshape(eegTrain,139,63,1,[]);
    eegTest = reshape(eegTest,139,63,1,[]);

    %Dataloader code for the neural network (minibatch)
    dsTrain = arrayDatastore(eegTrain,IterationDimension=4);
    numOutputs = 1;
    
    mbqtrain = minibatchqueue(dsTrain,numOutputs, ...
        MiniBatchSize = miniBatchSize, ...
        MiniBatchFcn=@preprocessMiniBatch, ...
        MiniBatchFormat="SSCB", ...
        PartialMiniBatch="return");
    
    dsTest = arrayDatastore(eegTest,IterationDimension=4);
    numOutputs = 1;
    
    mbqtest = minibatchqueue(dsTest,numOutputs, ...
        MiniBatchSize = 1, ...
        MiniBatchFcn=@preprocessMiniBatch, ...
        MiniBatchFormat="SSCB", ...
        PartialMiniBatch="return");
    
    
    % % % % % % % % % %% % % %% % % % % % %% %
    %Load networks
    [layer_Z2_X,layer_Z1_Z2X, layer_Xhat_Z1] = NeuralHGF_residual(temp_filter_size, spatial_filter_size);
    net_Z2_X = dlnetwork(layer_Z2_X);
    net_Z1_Z2X = dlnetwork(layer_Z1_Z2X);
    net_Xhat_Z1 = dlnetwork(layer_Xhat_Z1);
    
    %%%%%%%%%%%%%%%%%%%%%
    %Training loop
    trailingAvgE = [];
    trailingAvgSqE = [];
    trailingAvgE2 = [];
    trailingAvgSqE2 = [];
    trailingAvgD = [];
    trailingAvgSqD = [];
    
    numObservationsTrain = size(eegTrain,4);
    numIterationsPerEpoch = ceil(numObservationsTrain / miniBatchSize);
    numIterations = numEpochs * numIterationsPerEpoch;
    
    monitor = trainingProgressMonitor( ...
        Metrics=["Training_Loss", "KL_Div"], ...
        Info="Epoch", ...
        XLabel="Iteration");
    
    epoch = 0;
    iteration = 0;
    
    % Loop over epochs.
    while epoch < numEpochs && ~monitor.Stop
        if mod(epoch, 50)==0
            disp(epoch);
        end
        epoch = epoch + 1;
    
        % Shuffle data.
        shuffle(mbqtrain);
    
        % Loop over mini-batches.
        while hasdata(mbqtrain) && ~monitor.Stop
            iteration = iteration + 1;
    
            % Read mini-batch of data.
            X = next(mbqtrain);
    
            % Evaluate loss and gradients.
            [loss,KL,gradientsE,gradientsE2,gradientsD] = dlfeval(@modelLoss,net_Z2_X,net_Z1_Z2X,net_Xhat_Z1,X);
            %[validation_loss] = validationLoss(net_Z2_X, net_Z1_Z2X, net_Xhat_Z1, mbqtest, eegTest);
    
            % Update learnable parameters.
            [net_Z2_X, trailingAvgE,trailingAvgSqE] = adamupdate(net_Z2_X, ...
                gradientsE,trailingAvgE,trailingAvgSqE,iteration,learnRate);
            [net_Z1_Z2X,trailingAvgE2,trailingAvgSqE2] = adamupdate(net_Z1_Z2X, ...
                gradientsE2,trailingAvgE2,trailingAvgSqE2,iteration,learnRate);
            [net_Xhat_Z1, trailingAvgD, trailingAvgSqD] = adamupdate(net_Xhat_Z1, ...
                gradientsD,trailingAvgD,trailingAvgSqD,iteration,learnRate);
    
            % Update the training progress monitor. 
            recordMetrics(monitor,iteration,Training_Loss=loss/(139*63),KL_Div=KL/139);
            updateInfo(monitor,Epoch=epoch + " of " + numEpochs);
            monitor.Progress = 100*iteration/numIterations;
        end
    end
    
    YTest = modelPredictions(net_Z2_X,net_Z1_Z2X,net_Xhat_Z1,mbqtest);
    err = mean((eegTest-YTest).^2,[1 2 3 4]);
    disp('Validation Error');
    disp(err)
    
    net_filename = sprintf('neural_hgf_trained_foldnr_%d_of_%d.mat',k,numFolds);
    save(net_filename,'net_Xhat_Z1', 'net_Z1_Z2X', 'net_Z2_X')
end


function [validation_loss] = validationLoss(netE,netE2,netD,mbq, XTest)
    Y = modelPredictions(netE, netE2, netD, mbq);
    validation_loss = mean((Y-XTest).^2,[1 2 3 4]);
end


function [loss,KL,gradientsE,gradientsE2,gradientsD] = modelLoss(netE,netE2,netD,X)

% Forward through encoder.
[Z2_tilde, Z2, mu2, logSigmaSq2] = forward(netE,X);
[Z1, mu1, logSigmaSq1] = forward(netE2, Z2_tilde);

% Forward through decoder.
Y = forward(netD,Z1);

% Calculate loss and gradients.
[loss, KL] = elboLoss(Y,X,mu1,logSigmaSq1, mu2, logSigmaSq2, Z2);
[gradientsE,gradientsE2,gradientsD] = dlgradient(loss,netE.Learnables,netE2.Learnables,netD.Learnables);

end

function [loss, KL] = elboLoss(Y,T,mu1,logSigmaSq1, mu2, logSigmaSq2, Z2)
% Reconstruction loss.
reconstructionLoss = l1loss(Y,T);

% KL divergence.
KL2 = -0.5 * sum(1 + logSigmaSq2 - mu2.^2 - exp(logSigmaSq2),1);
KL2 = mean(KL2);
KL1 = -0.5 * sum(1 + logSigmaSq1-Z2 - (mu1.^2)./exp(Z2) - exp(logSigmaSq1-Z2),1);
KL1 = mean(KL1);
KL = KL1+KL2;
% Combined loss.
loss = reconstructionLoss+KL;

end

function Y = modelPredictions(netE,netE2,netD,mbq)

Y = [];

% Loop over mini-batches.
while hasdata(mbq)
    X = next(mbq);

    % Forward through encoder.
    [Z2_tilde, Z2, mu2, l2] = predict(netE,X);
    [Z1, mu1, l1] = predict(netE2,Z2_tilde);

    % Forward through dencoder.
    XGenerated = predict(netD,Z1);

    % Extract and concatenate predictions.
    Y = cat(4,Y,extractdata(XGenerated));
end

end

function X = preprocessMiniBatch(dataX)

% Concatenate.
X = cat(4,dataX{:});

end