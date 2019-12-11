function prob = mixture_prob(image, K, L, mask)

[H,W,~] = size(image);
vector_image = double(reshape(image, W*H, 3));
vector_mask = double(reshape(mask, W*H, 1));

% Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).

% Store all pixels for which mask=1 in a Nx3 matrix
%pixels = vector_image .* vector_mask;
pixels = vector_image(find(vector_mask),:);
num_pixels = size(pixels, 1);

% Randomly initialize the K components using masked pixels
[segm, centers] = kmeans_segm(pixels, K, L);

% covariances 3x3 (3 channels)
cov = cell(K,1);
w = zeros(1,K);
for k = 1:K
    cov{k} = eye(3);
    w(:,k) = nnz(segm == k) / num_pixels; % fraction of pixels belonging to k
end

g = zeros(num_pixels, K);
p = zeros(num_pixels, K);
% Iterate L times
for i = 1:L
    % Expectation: Compute probabilities P_ik using masked pixels
    for k = 1:K
        diff = bsxfun(@minus, pixels, centers(k,:));
        g(:,k) = w(:,k) .* (1 / sqrt(det(cov{k}) * (2 * pi)^3) * exp(-0.5 * sum((diff * inv(cov{k}) .* diff), 2)));
    end
    p = bsxfun(@rdivide, g, sum(g,2));
    
    % Maximization: Update weights, means and covariances using masked pixels
    for k = 1:K
       pixel_sum = sum(p(:,k), 1);
       w(:,k) = pixel_sum / num_pixels;
       centers(k,:) = (p(:,k)' * pixels) / pixel_sum;
       
       diff = bsxfun(@minus, pixels, centers(k,:)); % updated diff
       cov{k} = (diff' * bsxfun(@times, diff, p(:,k))) / pixel_sum;
    end
end

% Compute probabilities p(c_i) in Eq.(3) for all pixels I.
prob = zeros(W*H, K);
for k = 1:K
    diff = bsxfun(@minus, vector_image, centers(k,:));
    prob(:,k) = w(:,k) .* (1 / sqrt(det(cov{k}) * (2 * pi)^3) * exp(-0.5 * sum((diff * inv(cov{k}) .* diff), 2)));
end
prob = sum(prob,2);
prob = reshape(prob, H, W, 1);

end