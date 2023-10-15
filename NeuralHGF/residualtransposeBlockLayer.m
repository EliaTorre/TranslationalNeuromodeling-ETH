classdef residualtransposeBlockLayer < nnet.layer.Layer & nnet.layer.Formattable
    % Example custom residual block layer.
    
    properties (Learnable)
        % Layer learnable parameters.

        % Residual block.
        Network
    end
    
    methods
        function layer = residualtransposeBlockLayer(kernel1, kernel2,numFilters,NameValueArgs)
            % layer = residualBlockLayer(numFilters) creates a residual
            % block layer with the specified number of filters.
            %
            % layer = residualBlockLayer(numFilters,Name=Value) specifies
            % additional options using one or more name-value arguments:
            % 
            %     Stride                 - Stride of convolution operation 
            %                              (default 1)
            %
            %     IncludeSkipConvolution - Flag to include convolution in
            %                              skip connection
            %                              (default false)
            %
            %     Name                   - Layer name
            %                              (default '')
            
            % Parse input arguments.
            arguments
                kernel1
                kernel2
                numFilters
                NameValueArgs.Stride = 1
                NameValueArgs.IncludeSkipConvolution = true
                NameValueArgs.Name = ''
            end
            
            stride = NameValueArgs.Stride;
            includeSkipConvolution = NameValueArgs.IncludeSkipConvolution;
            name = NameValueArgs.Name;
            
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            description = "Residual block with " + numFilters + " filters, stride " + stride;
            if includeSkipConvolution
                description = description + ", and skip convolution";
            end
            layer.Description = description;
            
            % Set layer type.
            layer.Type = "Residual Block";
            
            % Define nested layer graph.
            layers = [
                transposedConv2dLayer([kernel1, kernel2],numFilters,Cropping="same",Stride=stride)
                %groupNormalizationLayer("all-channels")
                reluLayer
                transposedConv2dLayer([kernel1, kernel2],numFilters,Cropping="same")
                %groupNormalizationLayer("channel-wise")
                additionLayer(2,Name="add")
                reluLayer];
            
            lgraph = layerGraph(layers);
            
            % Add skip connection.
            if includeSkipConvolution
                layers = [
                    transposedConv2dLayer(1,numFilters,Stride=stride,Name="gnSkip")
                    %groupNormalizationLayer("all-channels",Name="gnSkip")
                    ];
                
                lgraph = addLayers(lgraph,layers);
                lgraph = connectLayers(lgraph,"gnSkip","add/in2"); 
            end 
                            
            % Convert to dlnetwork.
            net = dlnetwork(lgraph,Initialize=false);
            
            % Set Network property.
            layer.Network = net;
        end
        
        function Z = predict(layer, X)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer - Layer to forward propagate through
            %         X     - Input data
            % Outputs:
            %         Z - Output of layer forward function
            
            % Predict using nested network.
            net = layer.Network;
            Z = predict(net,X,X);
        end
    end
end