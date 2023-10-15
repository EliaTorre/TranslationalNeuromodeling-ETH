function [enc_Z2_X, enc_Z1_Z2, dec_Xhat_Z1] = NeuralHGF_residual(temp_filter_size, spatial_filter_size)

%Input assumed to be (T=139, C=63, 2, N) where T is number of time series points, C
%is number of EEG channels and 2 because we have time series for standards
%and deviants and N is batch size (MATLAB needs H W C N, opposite of
%PyTorch)

%X->enc_layers_base->Xp
%Xp->enc_Z2->mu2,logsigma2,Z2
%Xp,Z2->enc_Z2_Z1->mu1,logsigma1,Z1
%Z1->dec_Z1->Xhat


imageSize = [139, 63, 1];


enc_Z2_X = [imageInputLayer(imageSize,Normalization="none")
    residualBlockLayer(temp_filter_size,1,4)
    residualBlockLayer(1,spatial_filter_size,4)
    averagePooling2dLayer([1,4],Stride=[1,4])
    eluLayer%%%%%%%
    residualBlockLayer(temp_filter_size,1,16)
    residualBlockLayer(1,spatial_filter_size/4,16)
    averagePooling2dLayer([1,4],Stride=[1,4])
    eluLayer%%%%%%%
    convolution2dLayer([1, 1], 32,Padding="same",Stride=1) %(8,3,128, N)
    convolution2dLayer([1, 3], 32) %(8,1,128, N)
    eluLayer
    residualBlockLayer(temp_filter_size,1,16)
    residualBlockLayer(temp_filter_size,1,8)
    residualBlockLayer(temp_filter_size,1,4)
    convolution2dLayer([temp_filter_size, 1], 4,Padding="same",Stride=1) %(8,1,256, N)
    FlattenLayer(139*1*4)
    samplingLayerhierarchical([139,1,4])
    ];

enc_Z1_Z2 = [featureInputLayer(139*1*2,Normalization="none")
    ReshapeLayer([139,1,2])
    convolution2dLayer([1, 1], 2,Padding="same",Stride=1) %(8,1,256, N)
    FlattenLayer(139*1*2)
    samplingLayer];

dec_Xhat_Z1 = [featureInputLayer(139*1,Normalization="none")
    ReshapeLayer([139,1,1])
    transposedConv2dLayer([temp_filter_size,1],4,Cropping="same")
    eluLayer
    residualtransposeBlockLayer(temp_filter_size,1,4)
    residualtransposeBlockLayer(temp_filter_size,1,8)
    residualtransposeBlockLayer(temp_filter_size,1,16)
    transposedConv2dLayer([1,3],32)
    transposedConv2dLayer([1, 1],32)
    eluLayer%%%%%%
    resize2dLayer('OutputSize',[139,15])
    residualtransposeBlockLayer(1,spatial_filter_size/4,16)
    residualtransposeBlockLayer(temp_filter_size,1,16)
    resize2dLayer('OutputSize',[139,63])
    residualtransposeBlockLayer(1,spatial_filter_size,4)
    residualtransposeBlockLayer(temp_filter_size,1,4)
    transposedConv2dLayer([1,spatial_filter_size], 4, Cropping="same")
    transposedConv2dLayer([temp_filter_size,1], 1, Cropping="same")
    ];