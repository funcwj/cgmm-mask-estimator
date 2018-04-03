function cgmm_mask_visualization(mask_path)
screensize = get( groot, 'Screensize' );
set(gcf, 'position', screensize);
mask = load(mask_path);
lambda_noise = transpose(fliplr(abs(mask.lambda_noise)));
lambda_clean = transpose(fliplr(1 - abs(mask.lambda_noise)));
[num_bins, ~] = size(lambda_clean);
freq_ticks = linspace(0, num_bins - 1, 9);
colormap gray
subplot(1, 2, 1), imagesc(lambda_noise);
yticks(freq_ticks);
yticklabels(fliplr(freq_ticks) / (num_bins - 1) * 8);
ylabel('Frequency(kHz)');
xlabel('Frames');
title('noise mask');
colorbar
subplot(1, 2, 2), imagesc(lambda_clean);
yticks(freq_ticks);
yticklabels(fliplr(freq_ticks) / (num_bins - 1) * 8);
ylabel('Frequency(kHz)');
xlabel('Frames');
title('clean mask');
colorbar
saveas(gcf, [mask_path '.jpg']);
end
