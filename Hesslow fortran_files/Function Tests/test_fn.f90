program test_fn
    use physconst
    use nu_module
    use hesslow_module
    implicit none

    integer, parameter :: n = 100
    real(dp) :: pvals(n), yvals(n)
    real(dp) :: pmin, pmax, dp_step
    real(dp) :: E_star, T
    real(dp), dimension(2) :: Z, Z0, nj
    integer :: i
    real(dp) :: pstar
    character(len=*), parameter :: element = "Ar"

    ! ------------------------------
    ! Set up parameters (matches Python)
    ! ------------------------------
    Z  = [1.0_dp, 10.0_dp]
    Z0 = [1.0_dp,  1.0_dp]
    nj = [1.0e20_dp, 1.0e20_dp]

    T      = 10.0_dp
    E_star = 5.0_dp

    pmin = 0.0_dp
    pmax = 3.0_dp
    dp_step = (pmax - pmin) / real(n-1, dp)

    ! ------------------------------
    ! Compute fn(p) over the interval
    ! ------------------------------
    do i = 1, n
        pvals(i) = pmin + dp_step * real(i-1, dp)
        yvals(i) = fn(pvals(i), E_star, Z, Z0, nj, 2, T, element)
    end do

    ! ------------------------------
    ! Compute pstar using get_pstar
    ! ------------------------------
    pstar = get_pstar(E_star, Z, Z0, nj, 2, T, element)
    print *, "Computed pstar = ", pstar

    ! ------------------------------
    ! Write CSV (including pstar)
    ! ------------------------------
    open(unit=10, file="fn_output.csv", status="replace", action="write")
    write(10,'(A)') "p,fn"

    do i = 1, n
        write(10,'(F10.6,1x,ES14.6)') pvals(i), yvals(i)
    end do

    ! Append pstar point as (pstar,0)
    write(10,'(F10.6,1x,ES14.6)') pstar, 0.0_dp

    close(10)


    print *, "CSV written to fn_output.csv"

end program test_fn
