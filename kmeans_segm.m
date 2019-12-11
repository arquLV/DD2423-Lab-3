function [segmentation, centers] = kmeans_segm(image, K, L, seed)
% image : (H,W,3)
if ndims(image) > 2
    [H,W,~] = size(image);
    pixels = double(reshape(image, W*H, 3));
else
    pixels = image;
end
num_pixels = size(pixels, 1);
% disp(size(pixels));

if nargin > 3
    rng(seed); % seed the random generator
end

%centers = randi([0,255], K, 3); % K random RGB centers
disp(num_pixels);
centers = pixels(randperm(num_pixels,K),:);

for i = 1:L
    distances = pdist2(pixels, centers); % (W*H, K)
    
    [dist, segmentation] = min(distances,[],2); % find smallest distance among columns (centers) in reach row (pixel)
%     disp(size(segmentation));
    
    % update each cluster
    for k = 1:K
        pixel_indices = find(segmentation == k); % indices of pixels belonging to cluster k
        cluster_mean = mean(pixels(pixel_indices,:)); % 1x3 (mean values for each channel)
        centers(k,:) = cluster_mean;
    end
end

if ndims(image) > 2
    segmentation = reshape(segmentation, H, W);
end
end