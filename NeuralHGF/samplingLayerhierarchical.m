classdef samplingLayerhierarchical < nnet.layer.Layer
    properties
        % Layer properties.
        OutputSize
    end

    methods       
        function layer = samplingLayerhierarchical(OutputSize)
            % layer = samplingLayer creates a sampling layer for VAEs.
            %
            % layer = samplingLayer(Name=name) also specifies the layer 
            % name.

            % Parse input arguments.
            layer.OutputSize = OutputSize;
            % Layer properties.
            layer.Type = "Sampling";
            layer.Description = "Mean and log-variance sampling";
            layer.NumOutputs = 4;
        end

        function [Z2_tilde, Z2, mu2, logSigmaSq2] = predict(~,X)
            % [Z,mu,logSigmaSq] = predict(~,Z) Forwards input data through
            % the layer at prediction and training time and output the
            % result.
            %
            % Inputs:
            %         X - Concatenated input data where X(1:K,:) and 
            %             X(K+1:end,:) correspond to the mean and 
            %             log-variances, respectively, and K is the number 
            %             of latent channels.
            % Outputs:
            %         Z          - Sampled output
            %         mu         - Mean vector.
            %         logSigmaSq - Log-variance vector

            % Data dimensions.
            numLatentChannels = size(X,1)/4;
            miniBatchSize = size(X,2);
            
            % Split statistics.
            mu1 = X(1:numLatentChannels,:);
            logSigmaSq1 = X(numLatentChannels+1:2*numLatentChannels,:);
            mu2 = X(2*numLatentChannels+1:3*numLatentChannels,:);
            logSigmaSq2 = X(3*numLatentChannels+1:end,:);

            epsilon2 = randn(numLatentChannels,miniBatchSize,"like",X);
            sigma2 = exp(.5 * logSigmaSq2);
            Z2 = epsilon2 .* sigma2 + mu2;

            Z2_ = reshape(Z2,139,1,1,[]);
            logSigmaSq1 = reshape(logSigmaSq1,139,1,1,[]);
            mu1 = reshape(mu1, 139,1,1,[]);
            
            Z2_tilde = cat(3,mu1,Z2_);
            
            Z2_tilde = reshape(Z2_tilde,(139*1*2),[]);

            %Z2_tilde = Z2;

        end
    end
end