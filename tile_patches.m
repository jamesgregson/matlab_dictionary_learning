function [ I ] = tile_patches( P, nx, ny )
    pwidth = round( size( P, 1 )^0.5 );
    I = zeros( ny*pwidth, nx*pwidth );
    id = 1;
    for i=1:nx,
       for j=1:ny,
          if id <= size( P, 2 )
              I( (i-1)*pwidth+1:i*pwidth, (j-1)*pwidth+1:j*pwidth ) = reshape( P(:,id), [pwidth,pwidth] );
              id = id+1;
          end
       end
    end
end