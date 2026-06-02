program test_hesslow_rate
    use physconst
    use hesslow_module
    implicit none

    integer, parameter :: nSpecies = 2
    real(dp) :: Z(nSpecies), Z0(nSpecies), nj(nSpecies)
    real(dp) :: T, B
    real(dp) :: ne_free, lnA, Ec
    real(dp), allocatable :: Epar(:), gamma(:)
    real(dp), allocatable :: Estar(:)
    real(dp) :: Eceff, Ectmp
    integer :: i, npts

   
    Z  = (/ 1.0_dp, 18.0_dp /)
    Z0 = (/ 1.0_dp,  1.0_dp /)
    nj = (/ 1.0e20_dp, 1.0e20_dp /)

    T = 10.0_dp
    B = 0.0_dp

    ne_free = sum(Z0 * nj)

    lnA = 14.6_dp + 0.5_dp * log( T / (1.0_dp) )

    Ec = getConnorHastieField(ne_free, lnA)

    call getEc_eff(Z, Z0, nj, nSpecies, T, B, "Ar", Eceff, Ectmp)

    print *, "Ec     = ", Ec
    print *, "Eceff  = ", Eceff

    ! ------------------------------
    ! Create Epar array
    ! ------------------------------
    npts = 100
    allocate(Epar(npts), Estar(npts), gamma(npts))

    do i = 1, npts
        Epar(i) = 1.0_dp + (100.0_dp - 1.0_dp) * real(i-1,dp) / real(npts-1,dp)
        Estar(i) = Epar(i) / Ec
    end do

    ! ------------------------------
    ! Compute gamma(E*)
    ! ------------------------------
    do i = 1, npts
        gamma(i) = getHesslowAvalanchingRate(Estar(i), Z, Z0, nj, nSpecies, T, "Ar", B)
    end do

    ! ------------------------------
    ! Write CSV
    ! ------------------------------
    open(10, file="hesslow_rate.csv", status="replace", action="write")
    write(10,'(A)') "Epar,gamma_ms"

    do i = 1, npts
        write(10,'(F10.4,1x,ES14.6)') Epar(i), gamma(i)*1.0e-3_dp
    end do

    close(10)

    print *, "CSV written: hesslow_rate.csv"

end program test_hesslow_rate
