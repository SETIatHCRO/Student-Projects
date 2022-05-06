close all; clc; clear;

% Load binary file
file = fopen('/mnt/local_data/sobernberger/bfconversion/ts_decim_full_2.sigmf-data');
in = fread(file, 'float32');

% Configure sample rate
sr = 2e6;

% Deinterleave
waveform = in(1:2:end) + 1j*in(2:2:end);

% Initialize Constellation Diagramm
scat = comm.ConstellationDiagram;
scat.ShowGrid = true;
scat.Position = [1350 100 400 400];
scat.XLimits = [-3 3];
scat.YLimits = [-3 3];
% Initialize Spectrum Analyzer object
sA = dsp.SpectrumAnalyzer();
sA.Position = [0 500 320 200];
sA.SampleRate = sr;

% Initialize object for correlation plots
corrplt = dsp.ArrayPlot();
corrplt.XLabel = 'Timing Offset';
corrplt.YLabel = 'Correlation';
corrplt.ShowGrid = true;
corrplt.Position = [0 1000-410 320 200];
corrplt.PlotType = 'Line';

% Display size of waveform
WaveformSize = whos('waveform');
fprintf('Decoding of provided LTE Waveform starts. Waveform size is %4.2fGb\n', WaveformSize.bytes/1024^3)

% Initialization of enb structure. This structure serves as a central point, to save system information in.
% Many MATLAB LTE Toolbox require some information about the basestation. enb provides these informations:
% CyclicPrefix contains cyclic prefix length (either normal, or extended)
% NDLRB: Number of ressource blocks
% Duplex Mode: FDD or TDD
% NCellID: Physical Cell ID
% NCellRefP: Number of antenna ports
enb = struct;           
enb.NDLRB = 6;     
enb.CyclicPrefix = 'Normal';
% lteOFDMInfo calculates some information about the signals structure, based on the information provided in enb
% sampling rate, which matches NDLRB
% samples for each cyclic prefix
% FFT length of OFDM demodulation
ofdmInfo = lteOFDMInfo(enb);

sep = repmat('*',1,50);
% Calculate an aproximation for the number of total frames
NumSamplesPerFrame = 10e-3*sr;
ApproxNum2Frames = floor(length(waveform)/NumSamplesPerFrame);

fprintf('\nPlotting received signal spectrum...\n');
% Print spectrum of received waveform
sA(waveform);


% Set parameters to use for channel estimation
cec.PilotAverage = 'UserDefined';    
cec.FreqWindow = 13;                 
cec.TimeWindow = 9;                
cec.InterpType = 'cubic';          
cec.InterpWindow = 'Centered';     
cec.InterpWinSize = 1;   
% Initialize structures to save successfuly decoded MIB and found Cell IDs in    
MIBDecoded = {};
CellIds = double.empty(0, 3);
MIBFLAG = 0;
% Loop through datasets each containing two frames worth of samples (20ms)
for frameNum = 1:(floor(ApproxNum2Frames/2)-1)
    % Initialize flags 
    CellIdOk = 1;
    MIBok = 1;
    % Set NDLRB to 6, as this is the number of ressource blocks, which contain PBCH and Synchronization Signals
    enb.NDLRB = 6;
    % Check if signal has to be resampled (MATLAB functions expect certain sampling rates)
    if (sr~=ofdmInfo.SamplingRate)
        if (sr < ofdmInfo.SamplingRate)
            warning('The received signal sampling rate (%0.3fMS/s) is lower than the desired sampling rate for cell search / MIB decoding (%0.3fMs/s); cell search / MIB decoding may fail.',sr/1e6,ofdmInfo.SamplingRate/1e6);
        end
        fprintf('\nResampling from %0.3fMs/s to %0.3fMS/s for cell search / MIB decoding for Sample Block nr. %d (2 Frames worth of Samples)...\n',sr/1e6,ofdmInfo.SamplingRate/1e6, frameNum);
    else
        fprintf('\nResampling not required; received signal is at desired sampling rate for cell search / MIB decoding (%0.3fMs/s).\n',sr/1e6);
    end
    % Extract dataset from entire time domain signal
    sampIdxStart = (frameNum-1)*NumSamplesPerFrame*2 + 1;
    Samples = waveform(sampIdxStart:sampIdxStart+NumSamplesPerFrame*2);

    % Resample 
    nSamples = ceil(ofdmInfo.SamplingRate/round(sr)*size(Samples,1));
    nRxAnts = size(Samples, 2);
    downsampled = zeros(nSamples, nRxAnts);
    for i=1:nRxAnts
         downsampled(:,i) = resample(Samples(:,i), ofdmInfo.SamplingRate, round(sr));
    end
    

    %% CELL SEARCH
    fprintf('\nPerforming cell search for Sample Block nr. %d: \n', frameNum);
    duplexModes = {'TDD' 'FDD'};
    cyclicPrefixes = {'Normal' 'Extended'};
    
   
    % Set parameters for Cell Search Algorithm
    % Search for only one Cell ID
    searchalg.MaxCellCount = 1;
    % Perfrorm SSS Detection in time domain
    searchalg.SSSDetection = 'PostFFT';
    % Best Cell is chosen by comparing peaks of cross-correlations. Initialize peakMax variable.
    peakMax = -Inf;
    
    % Loop over all duplex modes and cyclic prfix lengths to find the combination of these two which deliver the best correlation
    for duplexMode = duplexModes
        for cyclicPrefix = cyclicPrefixes
            enb.DuplexMode = duplexMode{1};
            enb.CyclicPrefix = cyclicPrefix{1};
            % Perform Cell Search. Save found Cell ID, time offset untill beginning of next frame and found peak
            [enb.NCellID, offset, peak] = lteCellSearch(enb, downsampled, searchalg);
            enb.NCellID = enb.NCellID(1);
            offset = offset(1);
            peak = peak(1);
            % if peak is larger than previous peaks, the found parameters are saved
            if (peak>peakMax)
                enbMax = enb;
                offsetMax = offset;
                peakMax = peak;
            end
        end
    end

    enb = enbMax;
    offset = offsetMax;
    % Save found Cell ID and add peaks for averaging later
    if ismember(enb.NCellID, CellIds(:,1)) % If Cell ID has been found before
        [~, idx] = ismember(enb.NCellID, CellIds(:,1));
        CellIds(idx, 2) = CellIds(idx, 2) + peakMax;
        CellIds(idx, 3) = CellIds(idx, 3) + 1;
    else 
       CellIds(end+1, 1) = enb.NCellID;
       CellIds(end, 2) = peakMax;
       CellIds(end, 3) = 1;
    end

    % Calculate Correlation again, to check if it exceeds a thershold
    % If it doesn't, Cell Search is considered failed
    corr = cell(1,3);
    idGroup = floor(enbMax.NCellID/3);
    for i = 0:2
        enb.NCellID = idGroup*3 + mod(enbMax.NCellID + i,3);
        [~,corr{i+1}] = lteDLFrameOffset(enb, downsampled);
        corr{i+1} = sum(corr{i+1},2);
    end
    threshold = 1.3 * max([corr{2}; corr{3}]);
    if (max(corr{1})<threshold)    
        warning('Synchronization signal correlation was weak; detected CellID may be incorrect.');
        CellIdOk = 0;
    end
    
    % Return to originally detected cell identity
    enb.NCellID = enbMax.NCellID;

    corrplt.YLimits = [0 max([corr{1}; threshold])*1.1];
    corrplt([corr{1} threshold*ones(size(corr{1}))]);
     
    if offset < 0
        fprintf('Offset value of %d is invalid. Skipping to next dataset.', offset);
        continue
    end
    % Delete all samples before beginning of new frame for time sync
    fprintf('Timing offset to frame start: %d samples\n',offset);
    downsampled = downsampled(1+offset:end,:); 
    enb.NSubframe = 0;
    % Display enb after Cell Search
    fprintf('Cell-wide settings after cell search:\n');
    disp(enb);
    
    fprintf('\nPerforming frequency offset estimation...\n');
    % Set TDD configurations, if detected duplex type is TDD
    if (strcmpi(enb.DuplexMode,'TDD'))
        enb.TDDConfig = 0;
        enb.SSC = 0;
    end
    % Perform frequency Synchronization
    delta_f = lteFrequencyOffset(enb, downsampled);
    fprintf('Frequency offset: %0.3fHz\n',delta_f);
    downsampled = lteFrequencyCorrect(enb, downsampled, delta_f); 

    % Assume 4 cell-specific reference signals for initial decoding attempt
    enb.CellRefP = 4;   
                        
    fprintf('Performing OFDM demodulation...\n\n');
    % Retruns 3 element vector [number of subcarriers, number of symbols in one subframe, numbe of antenna ports]
    griddims = lteResourceGridSize(enb); 
    L = griddims(2);     

    % OFDM demodulate signal 
    rxgrid = lteOFDMDemodulate(enb, downsampled);    
    
    if (isempty(rxgrid))
        fprintf('After timing synchronization, signal is shorter than one subframe so no further demodulation will be performed.\n');
        return;
    end
    % Perform channel estimation (only for first subframe, as PBCH is transmitted here)
    [hest, nestsf] = lteDLChannelEstimate(enb, cec, rxgrid(:,1:L,:));

    fprintf('Performing MIB decoding...\n');
    % get indices of PBCH
    pbchIndices = ltePBCHIndices(enb);
    % Extract symbols of PBCH
    [pbchRx, pbchHest] = lteExtractResources(pbchIndices, rxgrid(:,1:L,:), hest(:,1:L,:,:));
    
    % Decode PBCH
    [bchBits, pbchSymbols, nfmod4, mib, enb.CellRefP] = ltePBCHDecode(enb, pbchRx, pbchHest, nestsf); 
    % Update Constellation Diagramm
    release(scat)
    step(scat, pbchSymbols);
    % Add MIB information to enb structure
    enb = lteMIB(mib, enb); 
    % System frame number (SFN) is stored as floor(SFN/4). Therefore this calculates the actual SFN
    enb.NFrame = enb.NFrame+nfmod4;
    % Display system settings after MIB decoding
    fprintf('Cell-wide settings after MIB decoding:\n');
    disp(enb);
    % lteMIB returns either CellRefP = 0 or NDLRB = 0, if MIB decoding hasn't been successful
    if (enb.CellRefP==0)
        warning('MIB decoding failed (enb.CellRefP=0).\n\n');
        MIBok=0;
    end
    if (enb.NDLRB==0)
        warning('MIB decoding failed (enb.NDLRB=0).\n\n');
        MIBok=0;
        
    end
    
    % Display system settings again, if Cell Search and MIB decoding have been successful
    if CellIdOk && MIBok
        fprintf('\n   CellID has been found, and MIB has been successfuly decoded. \n')
        fprintf(strcat('\n', sep, '\n'))
        disp(enb)
        MIBFLAG = MIBFLAG + 1;
        MIBDecoded{end+1} = enb;
    end
end


% Store Cell IDs, Correlation Peaks, number of occurences of the Cell IDs and frequency
Correlations = CellIds(:, 2)./CellIds(:, 3);
Result(:, 1) =CellIds(:, 1);
Result(:, 2) = Correlations;
Result(:, 3) = CellIds(:, 3);
Result = sortrows(Result, 3, 'descend');

% Display Cell IDs which occured the most
if length(Result) >= 15
    disp(Result(1:15, :))
else
    disp(Result)
end


% Set up strings for output file name
c = clock;
if ~exist('center_freq', 'var')
    center_freq = 0;
end
c = string(c);
for i = 2:length(c)-1
    if strlength(c(i)) < 2
        c(i) = strcat('0', c(i));
        
    end
end

% Add indicator to filename to show if MIB has been decoded
if MIBFLAG > 0
    flag = '*';
else
    flag = '';
end
% Create folder to save file in, and write found Cell IDs
folder = strcat(c(1), c(2), c(3), '/');
strcat(c(1), c(2), c(3), '/');
mkdir(strcat('RecordedCellIdCorrelations/', folder));
filename = strcat('RecordedCellIdCorrelations/', folder, num2str(center_freq/1e6, ...
    '%.1f'),'__', c(1), '_', c(2), '_', c(3), '__', c(4), '_', c(5), flag, '.xlsx');
writematrix(Result, filename);

% Write enb structure after MIB decoding to another sheet of the same xlsx file
CellSize = size(MIBDecoded);
if MIBFLAG > 0
    writecell(fieldnames(MIBDecoded{1})', filename, WriteMode="append")
    for i = 1:CellSize(2)
        writetable(struct2table(MIBDecoded{i}), filename, WriteMode="append");
    end
    fprintf('   MIB was decoded %d time(s).\n', MIBFLAG);
end
