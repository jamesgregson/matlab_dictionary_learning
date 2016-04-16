function [ P ] = extract_patches( I, Np, win )
    % get image dimensions
    width  = size( I, 2 );
    height = size( I, 1 );

    % window offset from central pixel, patch will be win pixels either side of
    % the central pixel. pwidth is the width of the patch, in pixels and N
    % is the number of pixels in the patch
    pwidth = 2*win+1;
    N      = pwidth^2;

    % patch center location, here assuming an MxM array of output patches, we
    % generate a Np=M*M patch center location randomly, constraining the
    % centers to be at least win pixels from the image border to avoid edge
    % cases
    px = randi( [win+1,width-win-1],  Np, 1 );
    py = randi( [win+1,height-win-1], Np, 1 );

    % Y will store the patch dictionary, with each patch packed as a column in
    % the N*Np matrix. Each patch pixel is looped over and the appropriate row
    % of Y is generated via the interp2 function using nearest neighbor
    % interpolation
    P = zeros( N, Np );
    id = 1;
    for i=-win:win,
        for j=-win:win,
            P( id, : ) = interp2( I, px+i, py+j, 'nearest' );
            id = id+1;
        end
    end
    
    % ugly hack here...
    %P(isnan(P)) = 0.0;
    %P(isinf(P)) = 0.0;
end