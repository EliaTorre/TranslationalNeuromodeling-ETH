function [mu, sigma, pval] = plotting_DCM(model, accuracies_model_perm, accuracies_model, folderName)
    if nargin < 4
        folderName = 'plot_DCM';
    end

    cl = {'Galantamine/Placebo', "Amisulpride/Placebo", "Levodopa/Placebo", "Biperdine/Placebo", "Biperdine/Galantamine", "Amisulpride/Levodopa", "All Drugs"};
    numColumns = 7;
    numBins = 10;  % Number of bins for the histogram
    

    if ~isfolder(folderName)
        mkdir(folderName);
    end

    f = figure(1);  % Create a new figure for the histograms
    t = tiledlayout(3,3);

    for col = 1:numColumns
        nexttile;

        hold on;
        
        % Create the histogram for the current column with density on the y-axis
        histogram(accuracies_model_perm(:, col), numBins, 'Normalization', 'pdf', 'FaceAlpha', 0.3, 'FaceColor', "#A9A9A9");
        
        % Fit a Gaussian curve to the histogram
        pd = fitdist(accuracies_model_perm(:, col), 'Normal');
        if col == 1
            x = linspace(0.55, 0.75, 100);
        elseif col == 2
            x = linspace(0.6, 0.75, 100);
        elseif col == 3
            x = linspace(0.45, 0.75, 100);
        elseif col == 4
            x = linspace(0.5, 0.8, 100);
        elseif col == 5
            x = linspace(0.3, 0.8, 100);
        elseif col == 6
            x = linspace(0.2, 0.8, 100);
        else
            x = linspace(0.25, 0.4, 100);
        end
    
        y = pdf(pd, x);


        mu(col,1) = pd.mu;
        sigma(col,1) = pd.sigma;

        %p value
        pval(col,1) = 1 - cdf(pd,accuracies_model(col));
        
        % Plot the Gaussian curve
        plot(x, y, 'LineWidth', 2);
        
        % Add a dotted line with a point for the corresponding accuracy
        accuracy = accuracies_model(col);
        yMax = max(y);
        plot([accuracy, accuracy], [0, yMax], '--r');  % Dotted line
        plot(accuracy, yMax, 'ro', 'MarkerSize', 10);  % Point at the top
        
        hold off;
        
        % Add labels, title, and legend to the plot
        xlabel('Accuracy');
        ylabel('Density');
        title(cl{col});  % Set the title using cl{col}
        
        % Add text displaying the accuracy value in the top-right corner
        textX = max(x);
        textY = max(y);
        text(textX, textY, {sprintf('Accuracy: %.3f', accuracy), sprintf('P-value: %.3f', pval(col))}, 'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');

        % Create a folder named "plot_RAW" if it doesn't exist
        %folderName = 'plot_DCM';

        
        % Save the current figure as a PNG file with a dynamic filename
        % filename = sprintf('plot_%s_%d.png', model(col), col);
        % fullFilePath = fullfile(folderName, filename);
        % saveas(gcf, fullFilePath);

        
  
    end
    f.PaperPositionMode = 'manual';
    f.PaperUnits = 'centimeters';
    %f3.PaperPosition = [0,0,20,5.5]; %[left bottom width height] (left and bottom are ignored for picture formats)
    f.PaperPosition =  1.5*[0,0,24,20]; %[left bottom width height] (left and bottom are ignored for picture formats)
    print(f, fullfile(folderName, "permTest.png"), '-dpng');
end