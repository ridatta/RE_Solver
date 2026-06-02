program test_nuD_nuS
    use nu_module
    implicit none

    integer, parameter :: nSpecies = 2
    integer, parameter :: N = 1000    ! match python 1000 points

    real(dp) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    real(dp) :: T
    character(len=*), parameter :: element = "Ar"

    real(dp) :: p_values(N), log_min, log_max
    real(dp) :: nuD, nuS
    integer  :: i, unit

    ! Open CSV file
    unit = 30
    open(unit, file="nuD_nuS_test.csv", status="replace", action="write")
    write(unit,'(A)') "p,nuD,nuS"

    ! --- MATCH PYTHON INPUTS ---
    Z  = (/ 1.0_dp, 18.0_dp /)
    Z0 = (/ 1.0_dp,  1.0_dp /)
    nj = (/ 0.0_dp, 1.0e20_dp /)
    T  = 10.0_dp

    ! Generate Fortran logspace(-1,2,1000)
    log_min = -1.0_dp
    log_max =  2.0_dp

    do i = 1, N
        p_values(i) = 10.0_dp ** ( log_min + (log_max-log_min) * real(i-1,dp) / real(N-1,dp) )
    end do

    ! --- Compute nuD, nuS ---
    do i = 1, N
        call get_nuD_nuS(p_values(i), Z, Z0, nj, nSpecies, T, element, nuD, nuS)
        write(unit,'(E15.8,1x,E15.8,1x,E15.8)') p_values(i), nuD, nuS
    end do

    close(unit)
    print *, "SUCCESS: wrote nuD_nuS_test.csv"

end program test_nuD_nuS
