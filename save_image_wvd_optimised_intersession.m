%% dictionary of dataset and label
clc;
clear all;
dirname = 'U:\MEG-BCI\main\Deep Learning\BCICIV_2b_gdf';
dirname2 ='U:\MEG-BCI\main\Deep Learning\true_labels';

%%%%%% get data of each subject
XTrain = [];
YTrain = [];
categ =[];
nn = 0;
oo = 0;
for kn=1:9
    
    p(kn) = 0; % get the number of none value
    sampleRate = 250; % sample frequency
    
    % to enrich data, set window size to 2 s with an overlap of 90%
    delayTime = 50;
    timeScale = 2;
    
    total_data_idx = 1; % combinitation of data from all session; for optimal frequency band selection
    for i = 1:5 % set 5 to also get evalution dataset
        if(i<4)
            dataName = ['B0',num2str(kn),'0',num2str(i),'T.gdf'];
            labelName = ['B0',num2str(kn),'0',num2str(i),'T.mat'];
        else
            dataName = ['B0',num2str(kn),'0',num2str(i),'E.gdf'];
            labelName = ['B0',num2str(kn),'0',num2str(i),'E.mat'];
        end
        [signal{i},H{i}] = mexSLOAD(fullfile(dirname,dataName));
        
        test_EVENT = H{i}.EVENT;
%         indx_trig = sort([find(test_EVENT.TYP==769);find(test_EVENT.TYP==770)]);
        indx_trig = sort(find(test_EVENT.TYP==768));
        H{i}.TRIG = H{i}.EVENT.POS(indx_trig);
        
        load(fullfile(dirname2,labelName));
        trueLabels{i}= classlabel;
        
        % get the label of each segments
        CIV2b_S{kn}.D{i}.raw = signal{i}(:,1:3); % original signals from c3,c4 and cz
        trial = length(H{i}.TRIG);
        
        CIV2b_S{kn}.D{i}.labels = trueLabels{i}; % get label
        
        n = 1;
        for j = 1:trial % extract MI signals from 3-7s
            meanValue = mean( CIV2b_S{kn}.D{i}.raw(H{i}.TRIG(j) : H{i}.TRIG(j) + sampleRate*2 ,:));
            if( any( any( isnan( CIV2b_S{kn}.D{i}.raw( H{i}.TRIG(j) : H{i}.TRIG(j) + sampleRate*7 ,: ) ) ) ) )
                p(kn) = p(kn) + 1;
                continue;
            end
            temp = CIV2b_S{kn}.D{i}.raw( H{i}.TRIG(j) + sampleRate*3 : H{i}.TRIG(j) + sampleRate*7 ,: );
            CIV2b_S{kn}.D{i}.MI{j}= [temp(:,1) - meanValue(1),temp(:,2) - meanValue(2),temp(:,3) - meanValue(3)];
            m = 0;
            while ( m*delayTime <= 500) % get the 2s length segment
                CIV2b_S{kn}.D{i}.tra{n}.data{m+1} =  CIV2b_S{kn}.D{i}.MI{j}(m * delayTime + 1 : m * delayTime + timeScale * sampleRate,:);
                CIV2b_S{kn}.D{i}.tra{n}.labels{m+1} =  CIV2b_S{kn}.D{i}.labels(j);
                
                % data combination
                CIV2b_S{kn}.all_data{total_data_idx} = CIV2b_S{kn}.D{i}.tra{n}.data{m+1};
                CIV2b_S{kn}.all_data_label(total_data_idx) = CIV2b_S{kn}.D{i}.tra{n}.labels{m+1};
                total_data_idx = total_data_idx + 1;
                
                m = m+1;
            end
            CIV2b_S{kn}.D{i}.Labels(n) = CIV2b_S{kn}.D{i}.labels(j);
            n = n+1;
        end
    end
    
    
    % clearvars -EXCEPT CIV2b_S p
    %%
    %**********************************************************
    % 2) preprocessing and form of input image of CNN
    % for kn = 4
    
    %subject optimal frequency selection BP FDA-F-Score
    %     [muBand, betaBand] = BPFeatureBandSelection(CIV2b_S{kn}.all_data, CIV2b_S{kn}.all_data_label, 250);
    
    % subject optimal frequency selection AR PSD FDA-F-Score
    %     [muBand, betaBand] = ARFeatureBandSelection(CIV2b_S{kn}.all_data, CIV2b_S{kn}.all_data_label, 250);
    cnt = 0;
    for sess = 1:5 % 5 for contains evaluation session
        % get training data and test data
        disp(sess)
        CIV2b_Data_S{kn}.se{sess}.Labels = CIV2b_S{kn}.D{sess}.labels;
        
        % extend frequency bands (better performance)
        muBand = [4,13];
        betaBand = [13,32];
        
        % get the all input images and labelsof CNN
        for i = 1 : length(CIV2b_S{kn}.D{sess}.tra)
            fprintf ('trial number: %d\n', i)
            for kk = 1:length(CIV2b_S{kn}.D{sess}.tra{i}.data)
                for j = 1:3
                    Cx{j} = CIV2b_S{kn}.D{sess}.tra{i}.data{kk}(:,j);
                    
                    % short time Fourier transform
                    fs =250;
                
                    %[d, f, t] = wvd(Cx{j},duration(0,0,1/fs,'Format','hh:mm:ss.SSS'), 'NumFrequencyPoints',201,'NumTimePoints',256, 'smoothedPseudo');
                    
                    
                    [d, f, t] = wvd(Cx{j},fs, 'smoothedPseudo');
                    Mu{j} = abs( d( (find(f<muBand(1),1,'last') ) : (find(f<muBand(2),1,'last')) +1 ,:) );
                    Beta = abs( d( (find(f<betaBand(1),1,'last') ) : (find(f<betaBand(2),1,'last')) +1,: ) );%%和paper不一样
                    
                    % beta band cubic interpolation
                    interNum = size(Mu{j},1);
                    fBeta = betaBand(1) : (betaBand(2)-betaBand(1))/(interNum-1) : betaBand(2);
                    [X,Y] = meshgrid(t,f);
                    [X1,Y1] = meshgrid(t,fBeta);
                    Beta_intrp{j} = interp2( X,Y,abs( d ),X1,Y1,'cubic');
                    
                    % normalization
                    Mu{j} = NorValue(Mu{j},1);
                    Beta_intrp{j} = NorValue(Beta_intrp{j}, 1);
                end
                
                CIV2b_Data_S{kn}.se{sess}.tra{i}.C3{kk} = [Beta_intrp{1}; Mu{1}];
                CIV2b_Data_S{kn}.se{sess}.tra{i}.Cz{kk} = [Beta_intrp{2}; Mu{2}];
                CIV2b_Data_S{kn}.se{sess}.tra{i}.C4{kk} = [Beta_intrp{3}; Mu{3}];
                
                X =  [CIV2b_Data_S{kn}.se{sess}.tra{i}.C4{kk};CIV2b_Data_S{kn}.se{sess}.tra{i}.Cz{kk};CIV2b_Data_S{kn}.se{sess}.tra{i}.C3{kk}];
                
                switch CIV2b_S{kn}.D{sess}.tra{i}.labels{kk} % for each label
                    case 1
                        ilabel = 1;
                    case 2
                        ilabel = 2;
                    otherwise
                        ilabel = 3;
                end
                cnt = cnt + 1;
                if (sess < 4)
                    fname = fullfile('H:\Sujit Roy\stacked\CompIVdata\Train', sprintf('%d',ilabel), sprintf('subj%02dfeat%05d', kn, cnt));
                else
                    fname = fullfile('H:\Sujit Roy\stacked\CompIVdata\Test', sprintf('%d',ilabel), sprintf('subj%02dfeat%05d', kn, cnt));
                end
                
                
                %fname = fullfile('\\scis_cl2FS\home$\SB00747428\MEG-BCI\main\Deep_Learning\compIVdata\Test', sprintf('%d',ilabel), sprintf('subj%02dfeat%05d', kn, cnt));
                save(fname, 'X');
            end
        end
        
        CIV2b_Data_S{kn}.band = [muBand,betaBand];
    end
    
clearvars CIV2b_S CIV2b_Data_S
   
end
%