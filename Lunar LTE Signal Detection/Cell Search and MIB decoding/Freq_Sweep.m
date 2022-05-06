clear all; close all; clc;
% Start Timer
tic
% Discover pluged in USRPs
a = findsdru %#ok<NOPTS>'
% Set up Burst Mode
frameTime = 10e-3;
sampinframe = frameTime*sr;
numframes = 100*3;
% Bandwidth
sr = 20e6;

% Initialize SDRuReceiver object for communication with USRP
rx = comm.SDRuReceiver(...
          'Platform',a.Platform, ...
          'SerialNum',a.SerialNum, ...
          'CenterFrequency',1e9, ...
          'MasterClockRate', sr, ...   
          'DecimationFactor', 1, ...
          'OutputDataType', 'double', ...
          'Gain', 76, ...
          'EnableBurstMode', true, ...
          'NumFramesInBurst', numframes, ...
          'SamplesPerFrame', sampinframe,  ...  
          'LocalOscillatorOffset', 0 ...
          ); 

% Define possibly interesting frequencies and bundle in bands          
custom = [624.5, 1982.5]*1e6;
custom2 = [762.8:0.1:763.2]*1e6;
B25 = [1930:0.1:1995]*1e6;
B66 = [2110:0.1:2200]*1e6;
B26 = [859:0.1:894]*1e6;
B12 = [729:0.1:746]*1e6;
B13 = [746:0.1:756]*1e6;
B14 = [758:0.1:768]*1e6;
B30 = [2350:0.1:2360]*1e6;
B41 = [2496:0.1:2690]*1e6;
B71 = [617:0.1:652]*1e6;
B29 = [717:0.1:728]*1e6;
B46 = [5150:0.1:5925]*1e6;
B48 = [3550:0.1:3700]*1e6;



% Narrow bands, around previous findings
B25p = [1957.5:0.1:1962.5]*1e6;        
B66p = [2132.5:0.1:2137.5]*1e6;        
B26p = [882.6:0.1:887.6]*1e6;    
B12p = [740:0.1:745]*1e6;
B13p = [748.5:0.1:753.5]*1e6;
B14p = [760.5:0.1:765.5]*1e6;
B30p = [2352.5:0.1:2357.5]*1e6;
B71p = [622:0.1:627]*1e6;

one = 742.5e6;

% Frequencies at which Cell Searches have been successful
best = [1960, 1982.5, 1980, 1947.5, 1967.5, 1950, 1942.5, 2135, 2120, ...
    2125, 2145, 2147.5, 885.1, 887.5, 763, 742.5, 751, 624.5]*1e6;

% Configure date strings for naming of output files
c = clock;
c = string(c);
timestart = c;
for i = 2:length(c)-1
    if strlength(c(i)) < 2
        c(i) = strcat('0', c(i));
        
    end
end

% Cell arrays to bundle different previously defined bands
important = {'B26p', 'B25p', 'B66p', 'B13p', 'B71p'};
all = {'B25', 'B66', 'B26', 'B12', 'B13', 'B14', 'B30', 'B71', 'B29'};
oneband = {'one'};
bestfreqs = {'best'};
customfreqs = {'custom'};


% The value of '"bands" actually determines on which frequencies Cell Search will be performed
bands = bestfreqs; 


% Creating folder for output files
folder = strcat(c(1), c(2), c(3));
flg = 1;
cnt = 1;
fld = folder;
while flg
    
    folder = strcat(fld, '_', int2str(cnt), '/');

    if ~exist(folder, "dir")
        status = mkdir(folder);
        flg = 0;
    else
        cnt = cnt+1;
    end
end


% Loop through the selected bands
for m = 1:length(bands)
    
    whichfreqs = bands{m};

    if ~status
        error('Folder for result storage could not be created');
    end

    % Name for output file for the current band
    filename = strcat(folder, whichfreqs, '__', c(1), '_', c(2), '_', c(3), '.xlsx');
    
    % Selection of frequencies for loop through the frequencies of one band
    switch whichfreqs
        case 'custom'
            freqs = custom;
        case 'custom2'
            freqs = custom2;
        case 'freqs'
            freqs = [624.5 742.5 751 763 885.1 887.5 ...
            1947.5 1960 1967.5 1980 1982.5 2120 2125 2135 2147.5 2175]*1e6; 
    
        case 'wikifreqs'
            freqs = [634.5 881.5 1960]*1e6;
    
        case 'bandcenters'
            freqs = [2140 1960 1842.5 2132.5 881.5 2655 942.5 1862.5 2140 1486 737.5 ...
            751 763 740 867.5 882.5 806 1503.5 3550 1542 1962.5 876.5 860.5 780.5 ...
            722.5 2355 465 1474 1910 2017.5 1880 1960 1920 2595 1900 2350 2593 3500 ...
            3700 753 1457 5537.5 5890 3625 1474.5 1429.5 3350 2489.5 2155 ...
            748 768 2595 2007.5 634.5 463.5 462.5 1496.5 1474.5 1429.5 737 422.5 424.5]*1e6;
        
        
            bands = [1 2 3 4 5 7 8 9 10 11 12 13 14 17 18 19 20 21 22 24 25 26 27 28 29 ...
            30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 50 51 52 53 66 ...
            67 68 69 70 71 72 73 74 75 76 85 87 88];
        case 'B25'
            freqs = B25;
        case 'B66'
            freqs = B66;
        case 'B26'
            freqs = B26;
        case 'B12'
            freqs = B12;
        case 'B13'
            freqs = B13;
        case 'B14'
            freqs = B14;
        case 'B30'
            freqs = B30;
        case 'B41'
            freqs = B41;
        case 'B71'
            freqs = B71;
        case 'B29'
            freqs = B29;
        case 'B46'
            freqs = B46;
        case 'B48'
            freqs = B48;


        case 'B25p'
            freqs = B25p;
        case 'B66p'
            freqs = B66p;
        case 'B26p'
            freqs = B26p;
        case 'B12p'
            freqs = B12p;
        case 'B13p'
            freqs = B13p;
        case 'B14p'
            freqs = B14p;
        case 'B30p'
            freqs = B30p;
        case 'B71p'
            freqs = B71p;
        case 'one'
            freqs = one;
        case 'best'
            freqs = best;
    end
    
    length(freqs)

    sep = repmat('*',1,50);

    % Loop through all frequencies within one band
    for n = 1:length(freqs)
        try
        center_freq = freqs(n);
        fprintf(strcat('Current carrier frequency: ', num2str(center_freq/1e6, '%.1f'), ' MHz. \n'));
        % Set SDRuReceiver object to the current center frequency. This will tune the LO frequency of USRP
        rx.CenterFrequency = center_freq;
        % Initialize dsp.SignalSink object for data collection
        rxLog = dsp.SignalSink;
        oversum = 0;
        counter = 1;
        % Collect data, untill the requested amount of LTE frames is received
        while counter <= numframes
            
            try
                [dat, ~, overrun] = rx();
                oversum = oversum + overrun;
                
            catch
               
                warning('Data collection interrupted')
               
                continue
            end
            if ~overrun
                rxLog(dat);
                counter = counter+1;
            end
        end

        % Extract samples from dsp.SignalSink object and realease it
        waveform = rxLog.Buffer;
        release(rxLog)

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
        % Save center frequency and band in enb for housekeeping
        enb.center_freq = num2str(center_freq/1e6, '%.1f');
        enb.band = whichfreqs;

        % Calculate an aproximation for the number of total frames
        NumSamplesPerFrame = 10e-3*sr;
        ApproxNum2Frames = floor(length(waveform)/NumSamplesPerFrame);
        
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
            
            % Extract dataset from entire time domain signal
            sampIdxStart = (frameNum-1)*NumSamplesPerFrame*2 + 1;
            Samples = waveform(sampIdxStart:sampIdxStart+NumSamplesPerFrame*2);

            % Downsample 
            nSamples = ceil(ofdmInfo.SamplingRate/round(sr)*size(Samples,1));
            nRxAnts = size(Samples, 2);
            downsampled = zeros(nSamples, nRxAnts);
            for i=1:nRxAnts
                downsampled(:,i) = resample(Samples(:,i), ofdmInfo.SamplingRate, round(sr));
            end
        
            %% CELL SEARCH

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
            if (max(corr{1})<threshold) || max(corr{1})==0

                CellIdOk = 0;

            end
            
            % Return to originally detected cell identity
            enb.NCellID = enbMax.NCellID;

            if offset < 0
                fprintf('Offset value of %d is invalid. Skipping to next dataset. \n', offset);
                continue
            end

            % Delete all samples before beginning of new frame for time sync
            downsampled = downsampled(1+offset:end,:); 
            enb.NSubframe = 0;
            % Set TDD configurations, if detected duplex type is TDD
            if (strcmpi(enb.DuplexMode,'TDD'))
                enb.TDDConfig = 0;
                enb.SSC = 0;
            end
            % Perform frequency Synchronization
            delta_f = lteFrequencyOffset(enb, downsampled);
            downsampled = lteFrequencyCorrect(enb, downsampled, delta_f); 
        
            % Assume 4 cell-specific reference signals for initial decoding attempt
            enb.CellRefP = 4;   
            
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
            
            % get indices of PBCH
            pbchIndices = ltePBCHIndices(enb);
            % Extract symbols of PBCH
            [pbchRx, pbchHest] = lteExtractResources(pbchIndices, rxgrid(:,1:L,:), hest(:,1:L,:,:));
            
            % Decode PBCH
            [bchBits, pbchSymbols, nfmod4, mib, enb.CellRefP] = ltePBCHDecode(enb, pbchRx, pbchHest, nestsf); 

            % Add MIB information to enb structure
            enb = lteMIB(mib, enb); 
            % System frame number (SFN) is stored as floor(SFN/4). Therefore this calculates the actual SFN
            enb.NFrame = enb.NFrame+nfmod4;
            
            % lteMIB returns either CellRefP = 0 or NDLRB = 0, if MIB decoding hasn't been successful
            if (enb.CellRefP==0)
                MIBok=0;
            end
            if (enb.NDLRB==0)

                MIBok=0;
                 
            end
            % If Cell ID has been found, and MIB has been decoded, display system information in terminal
            if CellIdOk && MIBok
                fprintf('\n   CellID has been found, and MIB has been successfuly decoded. \n')
                fprintf(strcat('\n', sep, '\n'))
                disp(enb)
                MIBFLAG = MIBFLAG + 1;
                MIBDecoded{end+1} = enb;
                MIBok = 0;
                CellIdOk = 0;
            end
        end
        
        
        % Delete Result variable
        clear Result
        % Store Cell IDs, Correlation Peaks, number of occurences of the Cell IDs and frequency
        Correlations = CellIds(:, 2)./CellIds(:, 3);
        Result(:, 1) = CellIds(:, 1);
        Result(:, 2) = Correlations;
        Result(:, 3) = CellIds(:, 3);
        Result(:, 4) = center_freq./1e6.*ones(size(Result, 1), 1);
        % Display Cell IDs which occured the most
        Result = sortrows(Result, 3, 'descend');
        if size(Result, 1) >= 2
            disp(Result(1:2, :))
        else
            disp(Result)
        end

        % Add indicator to filename to show if MIB has been decoded
        if MIBFLAG > 0
            flag = '*';
        else
            flag = '';
        end
        
        % As every Cell Search produces a Cell ID, regardless of correlation peaks, this eliminates Cell IDs which only occured very rarely
        % and writes them to file
        cellsfound = sum(Result(:,3));
        for numphyid = 1:size(Result, 1)  
            if Result(numphyid, 3) > cellsfound/30
                fprintf('\n Writing to file CellId. \n')
                writematrix(Result(numphyid,:), filename, 'Sheet', 'Data', WriteMode="append")
            end
        end
        
        % Write enb structure after MIB decoding to another sheet of the same xlsx file
        CellSize = size(MIBDecoded);
        if MIBFLAG > 0
            fprintf('\n Writing to file MIB. \n')
            writecell(fieldnames(MIBDecoded{1})', filename, 'Sheet', 'enb', WriteMode="append")
            for i = 1:CellSize(2)
                writetable(struct2table(MIBDecoded{i}), filename, 'Sheet', 'enb', WriteMode="append");
            end
            fprintf('\n   MIB was decoded %d time(s).\n', MIBFLAG);
            MIBFLAG = 0;
        end
        fprintf('\nCell search for center frequency %.1f MHz concluded\n', center_freq/1e6)

        % Handles any errors which may occur during data collection, Cell Search and writing to file
        % Write error message to file, but let's programm continue
        catch me
            % WRITE ERROR MESSAGE FOR FREQUENCY IN FILE 
            fprintf('\n Writing error message \n')
            format = '%.1f';
            cfstr = num2str(center_freq/1e6, format);
            errorlogname = strcat(folder, 'Error', cfstr, '.txt');
            file = fopen(errorlogname, 'w');
            errormsg = strcat('Error for carrier frequency', 32, cfstr, [' MHz.' ...
                ' Consider searching for that cell again.'], '\n', me.message, '\n line: ', int2str(me.stack.line));
            fprintf(file, errormsg);
            fclose(file);
            CellIdOk = 0;
            MIBok = 0;
            continue
        end
    end

    % Tidies up output file
    header = {'Cell ID', 'Correlation level', 'Number of events', 'Frequency'};
    
    if exist(filename, 'file')
        resultfile = readmatrix(filename);
        resultfile = sortrows(resultfile, 3, 'descend');
        writecell(header, filename, 'Sheet', 'Data', WriteMode='overwritesheet')
        writematrix(resultfile, filename, 'Sheet', 'Data', WriteMode="append")
    end
end

% Write timelog
timestop = clock;
timestop = string(timestop);
time = toc;
timelog = fopen(strcat(folder, 'timelog.txt'), 'w');
timetext = strcat('Programm has been running for', 32, num2str(time/60/60), 32, ' hours. \n');
timetext = strcat(timetext, '\nStart:', timestart(2), '/', timestart(3), '/', ...
    timestart(1), '\t', timestart(4), ':', timestart(5), '\t', 'UTC\n', 'End:', timestop(2), ...
    '/', timestop(3), '/', timestop(1), '\t', timestop(4), ':', timestop(5), '\t', 'UTC\n');
fprintf(timelog, timetext);
fprintf(timetext)
fclose(timelog);
release(rx)
fprintf(sep);
fprintf('\n Have a nice day!\n')