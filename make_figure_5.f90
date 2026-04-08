program make_HT_time_scan
    use physconst
    use HT_funcs
    implicit none

    integer, parameter :: N = 50

    real(dp) :: T0, Tf, t_char, n0, E0
    real(dp) :: t(N), nht(N), Temp
    real(dp) :: tmin, tmax, dt
    integer :: i

    ! parameters
    T0 = 3.1d3
    Tf = 0.01d0*T0
    t_char = 3.d-4
    n0 = 2.8d19
    E0 = 0.01d0

    ! time grid
    tmin = 1.d-5
    tmax = 2.d-3
    dt = (tmax - tmin)/real(N-1,dp)

    do i=1,N
        t(i) = tmin + dt*real(i-1,dp)
    end do

    nht = 0.d0

    ! main loop (same logic as Python)
    do i=2,N

        Temp = temperature(t(i), T0, Tf, t_char)

        nht(i) = getHT(n0, T0, E0, Temp, t(i))

        ! enforce monotonic growth
        if (nht(i) < nht(i-1)) then
            nht(i) = nht(i-1)
        end if

    end do
    !print *, nht

    ! write CSV
    open(unit=10,file="figure_5.csv",status="replace")

    write(10,*) "t_ms,nHT_over_n0,Temperature"

    do i=1,N
        Temp = temperature(t(i), T0, Tf, t_char)
        print *, nht(i)/n0

        write(10,'(E18.10, ",", E18.10, ",", E18.10)') t(i)*1.d3, nht(i)/n0, Temp 
    end do

    close(10)

    print *, "CSV file written: figure_5.csv"

end program make_HT_time_scan