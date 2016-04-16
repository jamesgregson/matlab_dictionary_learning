% Dictionary learning driver code
clear all;
close all;

% load an image, convert it to grayscale, linearize it
filename = 'lena.jpg';
I = im2double( rgb2gray( imread( filename ) ) ).^2.4;
width  = size( I, 2 );
height = size( I, 1 );

% regularization scale factor
patch_size      = 11;    % size of patch to compute dictionary for
dict_size_ratio = 1.5;  % ratio of dictionary atoms to patch DOF count

reg_weight      = 0.05;  % regularizer weight (relative to 1.2/sqrt(DOF))
batch_size      = 512;  % size of batches for dictionary update
iterations      = 100;   % number of dictionary update iterations

% extract some patches
win    = round((patch_size-1)/2);
X      = extract_patches( I, batch_size, win );
pwidth = win+win+1;

% generate an initial dictionary
Nd = round( pwidth*dict_size_ratio );
D = extract_patches( I, Nd*Nd, win );
if sum(sum(abs(isnan(D)))) > 1e-4 || sum(sum(abs(isinf(D))))
    error('Detected invalid values in initial dictionary');
end

% main dictionary computation loop, extract a new set of patches
% and refine the current dictionary estimate
A = zeros( size(D,2), size(D,2) );
B = zeros( size(D,1), size(D,2) );
for i=1:iterations,
    if mod( i, iterations/20 ) == 0
        fprintf( 'iteration [%d/%d]\n', i, iterations );
    end
    X = extract_patches( I, batch_size, win );
    %[D, alpha ] = refine_dictionary( D, X, reg_weight*1.2/patch_size );
    [D, alpha, A, B ] = online_dictionary_learning( i, D, A, B, X, reg_weight*1.2/patch_size );
end

fprintf( 'minimum sparse coding value: %f\n', min(alpha(:)) );
fprintf( 'maximum sparse coding value: %f\n', max(alpha(:)) );
fprintf( 'minimum dictionary value:    %f\n', min(D(:)) );
fprintf( 'maximum dictionary value:    %f\n', max(D(:)) );

% predict the final set of patches
pX = max( D*alpha, 0 );

% rescale the patches to the range [0,1] for display
for i=1:size(D,2),
    D(:,i) = D(:,i) - min(D(:,i));
    D(:,i) = D(:,i)/max( D(:,i) );
end
subplot( 2, 2, 1 );
dD = tile_patches( D, Nd, Nd );
imshow( dD );
title('Computed Dictionary');
imwrite( dD, 'dictionary.png' );

% plots the sparse coefficents used with the dictionary
subplot( 2, 2, 2 );
alpha = abs(alpha);
imshow( (alpha/max(alpha(:))).^0.33 );
title('Sparse Coding');
imwrite( (alpha/max(alpha(:))).^(1.0/2.4), 'sparse_codes.png');

subplot( 2, 2, 3 );
dpX = tile_patches( X, 10, 10 );
imshow( dpX.^(1.0/2.4) );
title('Target Patches');
imwrite( dpX.^(1.0/2.4), 'target_patches.png' );

subplot( 2, 2, 4 );
dpX = tile_patches( pX, 10, 10 );
imshow( dpX.^(1.0/2.4) );
title('Coded Patches');
imwrite( dpX.^(1.0/2.4), 'coded_patches.png' );

