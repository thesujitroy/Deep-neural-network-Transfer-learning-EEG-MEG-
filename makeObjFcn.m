function ObjFcn = makeObjFcn(XTrain,YTrain,XValidation,YValidation)
ObjFcn = @valErrorFun;
    function [valError,cons,VE] = valErrorFun(optVars)
        imageSize = [size(XTrain,1) size(XTrain,2) 3];
        numClasses = numel(unique(YTrain));
        numF = round(16/sqrt(optVars.SectionDepth));
        layers = [
            imageInputLayer(imageSize)
            
            % The spatial input and output sizes of these convolutional
            % layers are 32-by-32, and the following max pooling layer
            % reduces this to 16-by-16.
            convBlock(5,numF,optVars.SectionDepth)
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            % The spatial input and output sizes of these convolutional
            % layers are 16-by-16, and the following max pooling layer
            % reduces this to 8-by-8.
            convBlock(3,2*numF,optVars.SectionDepth)
            maxPooling2dLayer(3,'Stride',2,'Padding','same')
            
            % The spatial input and output sizes of these convolutional
            % layers are 8-by-8. The global average pooling layer averages
            % over the 8-by-8 inputs, giving an output of size
            % 1-by-1-by-4*initialNumFilters. With a global average
            % pooling layer, the final classification output is only
            % sensitive to the total amount of each feature present in the
            % input image, but insensitive to the spatial positions of the
            % features.
            convBlock(3,4*numF,optVars.SectionDepth)
            averagePooling2dLayer(8)
            
            % Add the fully connected layer and the final softmax and
            % classification layers.
            fullyConnectedLayer(numClasses)
            softmaxLayer
            classificationLayer];
        miniBatchSize = 64;
        validationFrequency = floor(numel(YTrain)/miniBatchSize);
        options = trainingOptions('sgdm', ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
             'Momentum',optVars.Momentum, ...
            'MaxEpochs',50, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',40, ...
            'LearnRateDropFactor',0.1, ...
            'MiniBatchSize',miniBatchSize, ...
            'L2Regularization',optVars.L2Regularization, ...
            'Shuffle','every-epoch', ...
            'Plots','training-progress', ...
            'ValidationData',{XValidation,YValidation}, ...
            'ValidationFrequency',validationFrequency);
        pixelRange = [-4 4];
        imageAugmenter = imageDataAugmenter( ...
            'RandXReflection',true, ...
            'RandXTranslation',pixelRange, ...
            'RandYTranslation',pixelRange);
        datasource = augmentedImageDatastore(imageSize,XTrain,YTrain,'DataAugmentation',imageAugmenter);
        trainedNet = trainNetwork(datasource,layers,options);
        close(findall(groot,'Tag','NNET_CNN_TRAININGPLOT_FIGURE'));
        YPredicted = classify(trainedNet,XValidation);
        valError = 1 - mean(YPredicted == YValidation);
        VE = num2str(valError) + ".mat";
        save(VE,'trainedNet','valError','options')
        cons = [];
        
    end
end
