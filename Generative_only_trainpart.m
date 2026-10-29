    %% CVAE
    ReconOptions.reconMethod = 1;
    % 1: generative Mutitasking
    % 2: conventional Multitasking
    % GMT
    disp('Using Generative Multitasking');
    % Pass variables to python for training

    ExtractToPython = extractVariablesToPython(Params, DataArray, TemporalBasis);
    % ExtractToPython.wallClock = mod(ExtractToPython.wallClock - 1, 5) + 1;
    % normalize
    % ExtractToPython.Phi_rt_small = ExtractToPython.Phi_rt_small / std(ExtractToPython.Phi_rt_small(:,1), 0); %normalize by first time point (should this be last time point or by first component in the future?)
    % ExtractToPython.Segidx_norm = (ExtractToPython.Segidx - mean(ExtractToPython.Segidx(:))) / std(ExtractToPython.Segidx(:));
    % ExtractToPython.wallClock_norm = (ExtractToPython.wallClock - mean(ExtractToPython.wallClock(:))) / std(ExtractToPython.wallClock(:));
    % ExtractToPython.thetas_norm = (ExtractToPython.thetas - mean(ExtractToPython.thetas(:))) / std(ExtractToPython.thetas(:));
    % ExtractToPython.theta0 = zeros(size(ExtractToPython.thetas_norm));
    % ExtractToPython.flipAngle_norm = (ExtractToPython.flipAngle - mean(ExtractToPython.flipAngle(:))) / std(ExtractToPython.flipAngle(:));
    ExtractToPython.Segidx_norm = (ExtractToPython.Segidx - mean(ExtractToPython.Segidx(:))) / std(ExtractToPython.Segidx(:), 1);
    ExtractToPython.wallClock_norm = (ExtractToPython.wallClock - mean(ExtractToPython.wallClock(:))) / std(ExtractToPython.wallClock(:), 1);
    ExtractToPython.thetas_norm = (ExtractToPython.thetas - mean(ExtractToPython.thetas(:))) / std(ExtractToPython.thetas(:), 1);
    ExtractToPython.theta0 = zeros(size(ExtractToPython.thetas_norm));
    ExtractToPython.flipAngle_norm = (ExtractToPython.flipAngle - mean(ExtractToPython.flipAngle(:))) / std(ExtractToPython.flipAngle(:), 1);

    %split real/imag
    % ExtractToPython.Phi_rt_com = [real(ExtractToPython.Phi_rt_small); imag(ExtractToPython.Phi_rt_small)].'; %later: rename so it's not "init"

    % NETWORK VARIABLES
    ExtractToPython.KLDindex = 1e-9; % change per sequence?
    ExtractToPython.LPF = 0.67; % Low pass filter frequency bound
    ExtractToPython.BPFlb = 0.67; % band pass filter lower bound
    ExtractToPython.BPFub = 2.5; % band pass filter upper bound
    ExtractToPython.seed = 42; % initialization seed

    % CONDITIONAL INPUTS
    ExtractToPython.NumofCon = 1; % number of conditional inputs
    ExtractToPython.NumofLatent = 2; % number of latent vectors


    % Assumes all necessary variables (Segidx_norm, wallClock_norm, etc.) are already defined.
    switch ExtractToPython.NumofCon
        case 1
            % If NumofCon is 1, use only thetas_norm
            ExtractToPython.TrainingCondition = ExtractToPython.thetas_norm;
            ExtractToPython.InferenceCondition = ExtractToPython.theta0;
        case 2
            % If NumofCon is 2, use wallClock and thetas
            ExtractToPython.TrainingCondition = squeeze(cat(3, ExtractToPython.wallClock_norm, ExtractToPython.thetas_norm));
            ExtractToPython.InferenceCondition = squeeze(cat(3, ExtractToPython.wallClock_norm, ExtractToPython.theta0));
        case 3
            % If NumofCon is 3, use Segidx, wallClock and thetas
            ExtractToPython.TrainingCondition = squeeze(cat(3, ExtractToPython.Segidx_norm, ExtractToPython.wallClock_norm, ExtractToPython.thetas_norm));
            ExtractToPython.InferenceCondition = squeeze(cat(3, ExtractToPython.Segidx_norm, ExtractToPython.wallClock_norm, ExtractToPython.theta0));
        case 4
            % If NumofCon is 4 (the original case)
            ExtractToPython.TrainingCondition = squeeze(cat(3, ExtractToPython.Segidx_norm, ExtractToPython.wallClock_norm, ExtractToPython.flipAngle_norm, ExtractToPython.thetas_norm));
            ExtractToPython.InferenceCondition = squeeze(cat(3, ExtractToPython.Segidx_norm, ExtractToPython.wallClock_norm, ExtractToPython.flipAngle_norm, ExtractToPython.theta0));
        otherwise
            % Optional: handle other cases or provide an error message
            error('Unsupported number of conditional inputs.');
    end
    % TRAINING PARAMETERS
    ExtractToPython.lr = 1e-4;  % learning rate
    ExtractToPython.NumofEpochs = 60000;   % number of epochs

    saveFileName = fullfile(Params.filePath,'ExtractToPython.mat');
    save(saveFileName, 'ExtractToPython', '-v7.3');
    disp(['Saved extracted variables to ', saveFileName]);

    outputFilePath = fullfile(Params.filePath, 'OutputFromPython.mat');

    %% Training CVAE
    cmd = sprintf('python3 /home/lingcengma/CODE/vae_temporal_analysis/train_network.py "%s" "%s"', saveFileName, outputFilePath);
    status = system(cmd);
    if status ~= 0
        error("Python training script failed to run.");
    end
    %% passing latent vectors and trained model path to python
    training_result = load(outputFilePath);
    model_path = string(training_result.model_path); % Convert Python str to MATLAB string
    latent_space_mu = double(training_result.latent_space_mu);
    latent_space_logvar = double(training_result.latent_space_logvar);
    realtime_output = double(training_result.recon_x);
    realtime_ec = double(training_result.recon_ec); % after eddy current correction

    %% save temporary workspace
    WorkspaceTempPath = fullfile(Params.filePath, 'Workspacetemp.mat');

   % save(WorkspaceTempPath);

    %% latent space plots
    figure;
    subplot(311);
    plot(latent_space_mu(:,1));
    subplot(312);
    plot(latent_space_mu(:,2));
    subplot(313);
    plot(latent_space_mu(:,3));
