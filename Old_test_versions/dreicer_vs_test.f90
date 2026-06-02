program test_get_vc
    use physconst
    use HT_funcs
    implicit none

    real(dp) :: T0, Tf, t_char
    real(dp) :: n0, E0
    real(dp) :: vc
    real(dp) :: lnA
    real(dp) :: col_log

   
    
    T0     = 3.1d3
    Tf     = 0.01d0 * T0
    t_char = 3.d-4
    n0     = 2.8d19
    E0     = 0.01d0
    lnA    = 15.0_dp

    
    vc = get_vc(T0, E0, n0)
    col_log = getDreicerField(n0, lnA, T0 )

    print *, "vc = ", vc, " m/s"
    print *, "Columb Log = ", col_log
    
    

end program test_get_vc