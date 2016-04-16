function [ D, alpha ] = refine_dictionary( D, X, wl1 )
    
    % define the problem dimensions
    N  = size( X, 1 );  % length of sample vector 
    M  = size( X, 2 );  % number of sample vectors
    Nd = size( D, 2 );  % number of dictionary atoms

    % =====================================================================
    % first step, perform sparse coding of the columns (patches) of X 
    % w.r.t. the columns (atoms) of the dictionary D. Use proximal form 
    % of ADMM to perform the sparse coding task with an LLT factorization
    % of the system arising in the data term for efficiency
    % =====================================================================
    
    lambda     = 1.0;   % ADMM splitting penalty weight
    gamma      = wl1;   % L1 penalty weight on sparse coding
    admm_iters = 100;    % number of ADMM iterations to perform
    
    % objective function for sparse coding us
    % alpha = argmin (1/2) || D alpha - X ||_2^2 + gamma || alpha ||_1
    % f(alpha) = (1/(2*gamma))|| D alpha - X ||_2^2
    % g(alpha) = || alpha ||_1
    
    % intialize the sparse coding coefficients
    alpha = randn( Nd, M );
    
    % pre-factorize the regularized problem for efficiency
    pI = pinv( (lambda/gamma)*((D')*D) + eye( Nd ) );
    
    % defind the proximal operators for the ADMM sparse coding operation
    prox_f = @( v ) pI*((lambda/gamma)*((D')*X) + v);
    prox_g = @( v ) max( v - lambda, 0 ) - max( -v - lambda, 0 );

    % initialize sparse coding solution, splitting variable and Lagrange
    % multipliers
    Z = alpha;
    U = alpha-Z;

    % perform the ADMM algorithm
    for iter=1:admm_iters,
       % update the sparse coding vector
       alpha = prox_f( Z - U );

       % update the splitting variable
       Z = prox_g( alpha + U );

       % update the Lagrange multipliers
       U = U + alpha - Z;
    end
   
    
    % =====================================================================
    % second step, update the dictionary
    % =====================================================================
    % the input set of patches may be rank-deficient, so compute the 
    % svd and solve the least squares problem using the pseudo-inverse
    [ U, S, V ] = svd( alpha' );
    for i=1:min(size(S)),
        S(i,i) = 1.0/max( S(i,i), 1e-3);
    end
    
    % form the dictionary from the SVD and then re-normalize the columns
    % of D.  These are highly unlikely to be zero norm, but the paranoid
    % could add a check
    D = ((V*(S')*(U'))*(X'))';
    for i=1:size(D,2),
       D(:,i) = D(:,i) / norm( D(:,i) ); 
    end
    
end

