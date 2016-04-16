function [ D, alpha, A, B ] = online_dictionary_learning( t, D, A, B, X, wl1 )
    
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
    admm_iters = 200;   % number of ADMM iterations to perform
    
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
    % update A and B
    if t < size(X,2),
        theta = t*size(X,2);
    else
        theta = size(X,2)^2+t-size(X,2);
    end
    beta  = (theta+1-size(X,2))/(theta+1);
    A = beta*A + alpha*alpha';
    B = beta*B + X*alpha';
    
    % update each column of the dictionary sequentially
    for j=1:size( D, 2 ),
        u = (B(:,j) - D*alpha(:,j))/A(j,j) + D(:,j);
        D(:,j) = u/max( norm(u), 1 );
    end
    

end