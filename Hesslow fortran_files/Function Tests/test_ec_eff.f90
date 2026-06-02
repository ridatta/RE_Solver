program test_ec_eff
    use ec_module
    implicit none

    integer, parameter :: nSpecies = 2
    integer, parameter :: N = 100

    real(dp) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    real(dp) :: T, B
    real(dp) :: nz_values(N), log_min, log_max
    real(dp) :: Eceff, Ectot, ratio
    integer  :: i

    character(len=*), parameter :: element = "Ne"

    ! CSV file
    integer :: unit
    unit = 20
    open(unit, file="ec_eff_test.csv", status="replace", action="write")

    ! Write CSV header
    write(unit, '(A)') "nz,Eceff_over_Ectot"

    ! ---- MATCH PYTHON VALUES ----
    Z  = (/ 1.0_dp, 10.0_dp /)
    Z0 = (/ 1.0_dp, 2.0_dp /)
    T  = 10.0_dp
    B  = 0.0_dp

    ! Generate Fortran-equivalent logspace(19,21,100)
    log_min = 19.0_dp
    log_max = 21.0_dp
    do i = 1, N
        nz_values(i) = 10.0_dp ** ( log_min + (log_max-log_min)*(real(i-1,dp)/real(N-1,dp)) )
    end do

    ! ---- Compute Ec_eff/Ec_tot sweep ----
    do i = 1, N
        nj(1) = 1.0e20_dp      ! main species density
        nj(2) = nz_values(i)   ! impurity density

        call getEc_eff(Z, Z0, nj, nSpecies, T, B, element, Eceff, Ectot)

        ratio = Eceff / Ectot

        write(unit,'(E15.8,1x,E15.8)') nz_values(i), ratio
    end do

    close(unit)
    print *, "SUCCESS: wrote ec_eff_test.csv"

end program test_ec_eff
