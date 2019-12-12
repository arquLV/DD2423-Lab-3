function prob = mixture_prob(image, K, L, mask)

[H,W,~] = size(image);
vector_image = im2double(reshape(image, W*H, 3));
vector_mask = reshape(mask, W*H, 1);

% Let I be a set of pixels and V be a set of K Gaussian components in 3D (R,G,B).

% Store all pixels for which mask=1 in a Nx3 matrix
%pixels = vector_image .* vector_mask;
pixels = vector_image(find(vector_mask),:);
num_pixels = size(pixels, 1);

% Randomly initialize the K components using masked pixels
[segm, centers] = kmeans_segm(pixels, K, L);
cov = cell(K, 1);

w = zeros(1, K);
for k = 1:K
    cov{k} = eye(3);
    w(k) = nnz(segm == k) / num_pixels;
end

g = zeros(num_pixels, K);
p = zeros(num_pixels, K);
% Iterate L times
for i = 1:L
    % Expectation: Compute probabilities P_ik using masked pixels
    for k = 1:K
        g(:,k) = w(:,k) .* mvnpdf(pixels,centers(k,:),cov{k});
    end
    p = bsxfun(@rdivide, g, sum(g,2));
    
    % Maximization: Update weights, means and covariances using masked pixels
    w = sum(p, 1) / size(p, 1);
    for k = 1 : K
        w(k) = sum(p(:,k),1) / num_pixels;
        pixel_sum = sum(p(:,k),1); % over all pixels for each k
        centers(k,:) = p(:,k)' * pixels / pixel_sum;

        diff = bsxfun(@minus, pixels, centers(k,:));
        cov{k} = (diff' * bsxfun(@times, diff, p(:,k))) / pixel_sum;
    end
end

% Compute probabilities p(c_i) in Eq.(3) for all pixels I.
prob_clusters = zeros(W*H, K);
for k = 1:K
    prob_clusters(:,k) = w(:,k) .* mvnpdf(vector_image,centers(k,:),cov{k});
end

prob = sum(prob_clusters, 2);
prob = reshape(prob, H, W, 1);

end